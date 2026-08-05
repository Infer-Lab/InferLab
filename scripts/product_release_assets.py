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
REPOSITORY_CHECKSUMMED_ASSETS = (
    "inferlab-x86_64-linux",
    "inferlab-aarch64-linux",
    "install.sh",
    "inferlab-plugin.tar.gz",
)
REPOSITORY_UNCHECKSUMMED_ASSETS = ("LICENSE",)


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


def _wheel_filename_matches(package: PackageIdentity, filename: str) -> bool:
    wheel_name = re.sub(r"[-_.]+", "_", package.name)
    return filename.startswith(f"{wheel_name}-{package.version}-") and filename.endswith(".whl")


def _verified_local_digest(asset: pathlib.Path, checksum: pathlib.Path) -> str:
    try:
        content = asset.read_bytes()
        checksum_text = checksum.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ReleaseAssetError(f"could not read local release asset: {error}") from error
    digest = hashlib.sha256(content).hexdigest()
    expected = f"{digest}  {asset.name}\n"
    if checksum_text != expected:
        raise ReleaseAssetError(f"checksum sidecar {checksum.name} does not match {asset.name}")
    return digest


def _wheel_assets(packages: list[PackageIdentity], directory: pathlib.Path) -> list[pathlib.Path]:
    if not packages:
        raise ReleaseAssetError("workspace package inventory is empty")
    if not directory.is_dir():
        raise ReleaseAssetError(f"release asset directory does not exist: {directory}")

    prepared: list[pathlib.Path] = []
    for package in packages:
        matches = [
            path for path in directory.glob("*.whl") if _wheel_filename_matches(package, path.name)
        ]
        if len(matches) != 1:
            raise ReleaseAssetError(
                f"expected one wheel for {package.name} {package.version}, found {len(matches)}"
            )
        wheel = matches[0]
        checksum = directory / f"{wheel.name}.sha256"
        if not checksum.is_file():
            raise ReleaseAssetError(f"release wheel has no checksum sidecar: {wheel.name}")
        _verified_local_digest(wheel, checksum)
        prepared.extend((wheel, checksum))
    return prepared


def _repository_assets(directory: pathlib.Path) -> list[pathlib.Path]:
    if not directory.is_dir():
        raise ReleaseAssetError(f"release asset directory does not exist: {directory}")
    prepared: list[pathlib.Path] = []
    for filename in REPOSITORY_CHECKSUMMED_ASSETS:
        asset = directory / filename
        checksum = directory / f"{filename}.sha256"
        if not asset.is_file() or not checksum.is_file():
            raise ReleaseAssetError(f"repository release asset is incomplete: {filename}")
        _verified_local_digest(asset, checksum)
        prepared.extend((asset, checksum))
    for filename in REPOSITORY_UNCHECKSUMMED_ASSETS:
        asset = directory / filename
        if not asset.is_file():
            raise ReleaseAssetError(f"repository release asset is missing: {filename}")
        prepared.append(asset)
    return prepared


def _verify_exact_inventory(
    directory: pathlib.Path, prepared: list[pathlib.Path]
) -> list[pathlib.Path]:
    expected = set(prepared)

    try:
        observed = set(directory.iterdir())
    except OSError as error:
        raise ReleaseAssetError(f"could not inspect release assets: {error}") from error
    unexpected = sorted(path.name for path in observed - expected)
    if unexpected:
        raise ReleaseAssetError(
            f"release asset directory contains unexpected entries: {', '.join(unexpected)}"
        )
    return prepared


def verify_wheel_assets(
    packages: list[PackageIdentity], directory: pathlib.Path
) -> list[pathlib.Path]:
    return _verify_exact_inventory(directory, _wheel_assets(packages, directory))


def verify_repository_assets(directory: pathlib.Path) -> list[pathlib.Path]:
    return _verify_exact_inventory(directory, _repository_assets(directory))


def verify_aggregate_assets(
    packages: list[PackageIdentity], directory: pathlib.Path
) -> list[pathlib.Path]:
    prepared = _repository_assets(directory) + _wheel_assets(packages, directory)
    return _verify_exact_inventory(directory, prepared)


def prepare_assets(
    packages: list[PackageIdentity],
    output: pathlib.Path,
    candidates: pathlib.Path,
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
    if not candidates.is_dir():
        shutil.rmtree(stage, ignore_errors=True)
        raise ReleaseAssetError(f"candidate directory does not exist: {candidates}")
    try:
        candidate_entries = set(candidates.iterdir())
    except OSError as error:
        shutil.rmtree(stage, ignore_errors=True)
        raise ReleaseAssetError(f"could not inspect candidate directory: {error}") from error
    used_candidates: set[pathlib.Path] = set()
    try:
        for package in packages:
            local_matches = [
                path for path in candidate_entries if _wheel_filename_matches(package, path.name)
            ]
            if len(local_matches) > 1:
                raise ReleaseAssetError(
                    f"found multiple local candidates for {package.name} {package.version}"
                )
            if local_matches:
                candidate = local_matches[0]
                candidate_checksum = candidate.with_name(f"{candidate.name}.sha256")
                if candidate_checksum not in candidate_entries:
                    raise ReleaseAssetError(
                        f"local candidate has no checksum sidecar: {candidate.name}"
                    )
                actual_digest = _verified_local_digest(candidate, candidate_checksum)
                filename = candidate.name
                shutil.copyfile(candidate, stage / filename)
                used_candidates.update((candidate, candidate_checksum))
            else:
                metadata_url = (
                    "https://pypi.org/pypi/"
                    f"{urllib.parse.quote(package.name, safe='')}/"
                    f"{urllib.parse.quote(package.version, safe='')}/json"
                )
                filename, wheel_url, index_digest = _wheel_release(package, read_json(metadata_url))
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
        unexpected_candidates = sorted(path.name for path in candidate_entries - used_candidates)
        if unexpected_candidates:
            raise ReleaseAssetError(
                "candidate directory contains unexpected entries: "
                + ", ".join(unexpected_candidates)
            )
        prepared = verify_wheel_assets(packages, stage)
        stage.rename(output)
    except (OSError, ReleaseAssetError):
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return [output / path.name for path in prepared]


def main() -> None:
    usage = f"usage: {pathlib.Path(sys.argv[0]).name} OPERATION TAG DIRECTORY [CANDIDATES]"
    operations = ("prepare", "verify-wheels", "verify-repository", "verify-aggregate")
    if len(sys.argv) not in (4, 5) or sys.argv[1] not in operations:
        raise SystemExit(usage)
    operation = sys.argv[1]
    tag = sys.argv[2]
    directory = pathlib.Path(sys.argv[3]).resolve()
    try:
        version = _product_version(ROOT)
        if tag != f"v{version}":
            raise ReleaseAssetError(f"tag {tag} != product version v{version}")
        packages = _workspace_packages(ROOT)
        if operation == "prepare":
            if len(sys.argv) != 5:
                raise ReleaseAssetError("prepare requires a candidate directory")
            candidates = pathlib.Path(sys.argv[4]).resolve()
            prepared = prepare_assets(packages, directory, candidates)
        elif len(sys.argv) != 4:
            raise ReleaseAssetError(f"{operation} does not accept a candidate directory")
        elif operation == "verify-wheels":
            prepared = verify_wheel_assets(packages, directory)
        elif operation == "verify-repository":
            prepared = verify_repository_assets(directory)
        else:
            prepared = verify_aggregate_assets(packages, directory)
    except ReleaseAssetError as error:
        raise SystemExit(f"product-release-assets: {error}") from error
    for path in prepared:
        print(path)


if __name__ == "__main__":
    main()
