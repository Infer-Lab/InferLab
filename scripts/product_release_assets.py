#!/usr/bin/env python3
"""Prepare canonical workspace-side wheels for one product release."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import cast

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHA256 = re.compile(r"^[0-9a-f]{64}$")
NETWORK_TIMEOUT_SECONDS = 30


class ReleaseAssetError(Exception):
    """A product release asset could not be established exactly."""


@dataclasses.dataclass(frozen=True)
class PackageIdentity:
    name: str
    version: str


JsonReader = Callable[[str], object]
ByteReader = Callable[[str], bytes]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReleaseAssetError(f"{context} must be an object")
    return value


def _string(mapping: Mapping[str, object], field: str, context: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ReleaseAssetError(f"{context}.{field} must be a nonempty string")
    return value


def _workspace_packages(root: pathlib.Path) -> list[PackageIdentity]:
    try:
        inventory = subprocess.run(
            [str(root / "scripts/python-package-inventory.sh"), "workspace-side"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ReleaseAssetError(f"could not read workspace package inventory: {error}") from error
    if inventory.returncode != 0:
        detail = inventory.stderr.strip() or inventory.stdout.strip()
        raise ReleaseAssetError(f"workspace package inventory failed: {detail}")

    packages: list[PackageIdentity] = []
    for package in inventory.stdout.splitlines():
        pyproject = root / "python" / package / "pyproject.toml"
        try:
            with pyproject.open("rb") as source:
                document = tomllib.load(source)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ReleaseAssetError(
                f"could not read {pyproject.relative_to(root)}: {error}"
            ) from error
        project = _mapping(document.get("project"), f"{pyproject.relative_to(root)}.project")
        identity = PackageIdentity(
            name=_string(project, "name", str(pyproject.relative_to(root))),
            version=_string(project, "version", str(pyproject.relative_to(root))),
        )
        if identity.name != package:
            raise ReleaseAssetError(
                f"inventory entry {package} resolves to distribution {identity.name}"
            )
        packages.append(identity)
    return packages


def _product_version(root: pathlib.Path) -> str:
    manifest = root / "Cargo.toml"
    try:
        with manifest.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseAssetError(f"could not read Cargo.toml: {error}") from error
    workspace = _mapping(document.get("workspace"), "Cargo.toml.workspace")
    package = _mapping(workspace.get("package"), "Cargo.toml.workspace.package")
    return _string(package, "version", "Cargo.toml.workspace.package")


def _read_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "inferlab-release-preparation"})
    try:
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseAssetError(f"could not read package-index metadata {url}: {error}") from error


def _read_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "inferlab-release-preparation"})
    try:
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            return cast(bytes, response.read())
    except urllib.error.URLError as error:
        raise ReleaseAssetError(f"could not download package-index wheel {url}: {error}") from error


def _wheel_release(package: PackageIdentity, payload: object) -> tuple[str, str, str]:
    release = _mapping(payload, f"package-index release {package.name} {package.version}")
    info = _mapping(release.get("info"), f"package-index info {package.name}")
    if _string(info, "name", f"package-index info {package.name}") != package.name:
        raise ReleaseAssetError(f"package index returned the wrong identity for {package.name}")
    if _string(info, "version", f"package-index info {package.name}") != package.version:
        raise ReleaseAssetError(f"package index returned the wrong version for {package.name}")

    urls = release.get("urls")
    if not isinstance(urls, list) or not all(isinstance(value, Mapping) for value in urls):
        raise ReleaseAssetError(f"package-index urls for {package.name} must be an object list")
    wheels = [value for value in urls if value.get("packagetype") == "bdist_wheel"]
    if len(wheels) != 1:
        raise ReleaseAssetError(
            f"expected exactly one package-index wheel for {package.name} {package.version}, "
            f"found {len(wheels)}"
        )
    wheel = wheels[0]
    if wheel.get("yanked") is not False:
        raise ReleaseAssetError(f"package-index wheel for {package.name} is yanked")
    filename = _string(wheel, "filename", f"package-index wheel {package.name}")
    wheel_name = re.sub(r"[-_.]+", "_", package.name)
    if not filename.startswith(f"{wheel_name}-{package.version}-") or not filename.endswith(".whl"):
        raise ReleaseAssetError(
            f"package-index wheel {filename} does not match {package.name} {package.version}"
        )
    url = _string(wheel, "url", f"package-index wheel {package.name}")
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ReleaseAssetError(f"package-index wheel URL for {package.name} is not HTTPS")
    digests = _mapping(wheel.get("digests"), f"package-index wheel digests {package.name}")
    digest = _string(digests, "sha256", f"package-index wheel digests {package.name}")
    if SHA256.fullmatch(digest) is None:
        raise ReleaseAssetError(f"package-index SHA-256 for {package.name} is invalid")
    return filename, url, digest


def prepare_assets(
    packages: list[PackageIdentity],
    output: pathlib.Path,
    read_json: JsonReader = _read_json,
    read_bytes: ByteReader = _read_bytes,
) -> list[pathlib.Path]:
    if not packages:
        raise ReleaseAssetError("workspace package inventory is empty")
    identities = [(package.name, package.version) for package in packages]
    if len(set(identities)) != len(identities):
        raise ReleaseAssetError("workspace package inventory contains a duplicate identity")
    if output.exists():
        raise ReleaseAssetError(f"release asset directory already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    names: list[str] = []
    try:
        for package in packages:
            metadata_url = (
                "https://pypi.org/pypi/"
                f"{urllib.parse.quote(package.name, safe='')}/"
                f"{urllib.parse.quote(package.version, safe='')}/json"
            )
            filename, wheel_url, index_digest = _wheel_release(package, read_json(metadata_url))
            if filename in names:
                raise ReleaseAssetError(f"release wheel filename is duplicated: {filename}")
            content = read_bytes(wheel_url)
            actual_digest = hashlib.sha256(content).hexdigest()
            if actual_digest != index_digest:
                raise ReleaseAssetError(
                    f"downloaded wheel digest for {package.name} is {actual_digest}, "
                    f"package index reports {index_digest}"
                )
            (stage / filename).write_bytes(content)
            checksum_name = f"{filename}.sha256"
            (stage / checksum_name).write_text(
                f"{actual_digest}  {filename}\n",
                encoding="ascii",
            )
            names.extend((filename, checksum_name))
        stage.rename(output)
    except (OSError, ReleaseAssetError):
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return [output / name for name in names]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {pathlib.Path(sys.argv[0]).name} TAG OUTPUT")
    tag = sys.argv[1]
    output = pathlib.Path(sys.argv[2]).resolve()
    try:
        version = _product_version(ROOT)
        if tag != f"v{version}":
            raise ReleaseAssetError(f"tag {tag} != product version v{version}")
        prepared = prepare_assets(_workspace_packages(ROOT), output)
    except ReleaseAssetError as error:
        raise SystemExit(f"prepare-product-release-assets: {error}") from error
    for path in prepared:
        print(path)


if __name__ == "__main__":
    main()
