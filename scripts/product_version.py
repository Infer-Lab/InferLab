#!/usr/bin/env python3
"""Validate and update InferLab product-version projections structurally."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import sys
import tempfile
from collections.abc import Mapping, MutableMapping
from typing import cast

import tomlkit
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from python_package_inventory import InventoryError, inventory
from tomlkit import TOMLDocument

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
DEPENDENCY_TABLES = frozenset({"dependencies", "dev-dependencies", "build-dependencies"})


class ProductVersionError(Exception):
    """Product-owned version metadata is incomplete or inconsistent."""


@dataclasses.dataclass(frozen=True)
class Projection:
    path: pathlib.Path
    content: str


def _mapping(value: object, context: str) -> MutableMapping[str, object]:
    if not isinstance(value, MutableMapping):
        raise ProductVersionError(f"{context} must be a table")
    return cast(MutableMapping[str, object], value)


def _string(mapping: Mapping[str, object], field: str, context: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ProductVersionError(f"{context}.{field} must be a nonempty string")
    return value


def _load_toml(path: pathlib.Path) -> TOMLDocument:
    try:
        with path.open("rb") as source:
            return tomlkit.load(source)
    except (OSError, tomlkit.exceptions.ParseError) as error:
        raise ProductVersionError(f"could not read {path.relative_to(ROOT)}: {error}") from error


def _load_json(path: pathlib.Path) -> MutableMapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductVersionError(f"could not read {path.relative_to(ROOT)}: {error}") from error
    return _mapping(value, str(path.relative_to(ROOT)))


def _product_version(document: TOMLDocument) -> str:
    workspace = _mapping(document.get("workspace"), "Cargo.toml.workspace")
    package = _mapping(workspace.get("package"), "Cargo.toml.workspace.package")
    version = _string(package, "version", "Cargo.toml.workspace.package")
    if SEMVER.fullmatch(version) is None:
        raise ProductVersionError(
            f"Cargo.toml.workspace.package.version is not strict semver: {version}"
        )
    return version


def _crate_requirement(version: str) -> str:
    major, minor, _ = version.split(".")
    return f"{major}.{minor}" if major == "0" else major


def _workspace_manifests(
    root_document: TOMLDocument,
) -> tuple[dict[str, pathlib.Path], list[tuple[pathlib.Path, TOMLDocument]]]:
    workspace = _mapping(root_document.get("workspace"), "Cargo.toml.workspace")
    members_value = workspace.get("members")
    if not isinstance(members_value, list) or not all(
        isinstance(member, str) for member in members_value
    ):
        raise ProductVersionError("Cargo.toml.workspace.members must be a string array")

    names: dict[str, pathlib.Path] = {}
    manifests: list[tuple[pathlib.Path, TOMLDocument]] = []
    for member in members_value:
        manifest = ROOT / member / "Cargo.toml"
        document = _load_toml(manifest)
        package = _mapping(document.get("package"), f"{manifest.relative_to(ROOT)}.package")
        name = _string(package, "name", f"{manifest.relative_to(ROOT)}.package")
        version = _mapping(package.get("version"), f"{manifest.relative_to(ROOT)}.package.version")
        if version.get("workspace") is not True:
            raise ProductVersionError(
                f"{manifest.relative_to(ROOT)}.package.version must inherit the workspace version"
            )
        if name in names:
            raise ProductVersionError(f"duplicate Cargo workspace package name: {name}")
        names[name] = manifest
        manifests.append((manifest, document))
    return names, manifests


def _update_cargo_dependencies(
    value: object,
    context: str,
    package_names: frozenset[str],
    old_requirement: str,
    new_requirement: str,
) -> None:
    if not isinstance(value, MutableMapping):
        return
    table = cast(MutableMapping[str, object], value)
    for key, child in list(table.items()):
        child_context = f"{context}.{key}"
        if key in DEPENDENCY_TABLES:
            dependencies = _mapping(child, child_context)
            for dependency_name, dependency_value in dependencies.items():
                if not isinstance(dependency_value, MutableMapping):
                    continue
                dependency = cast(MutableMapping[str, object], dependency_value)
                actual_name = dependency.get("package", dependency_name)
                if not isinstance(actual_name, str):
                    raise ProductVersionError(
                        f"{child_context}.{dependency_name}.package must be a string"
                    )
                if "path" not in dependency or actual_name not in package_names:
                    continue
                requirement = _string(dependency, "version", f"{child_context}.{dependency_name}")
                if requirement != old_requirement:
                    raise ProductVersionError(
                        f"{child_context}.{dependency_name}.version {requirement} "
                        f"does not match product requirement {old_requirement}"
                    )
                dependency["version"] = new_requirement
        else:
            _update_cargo_dependencies(
                child,
                child_context,
                package_names,
                old_requirement,
                new_requirement,
            )


def _replace_exact_product_requirement(
    dependency: str,
    product_packages: frozenset[str],
    old_version: str,
    new_version: str,
    context: str,
) -> str:
    try:
        requirement = Requirement(dependency)
    except InvalidRequirement as error:
        raise ProductVersionError(f"{context} is not a valid requirement: {error}") from error
    if canonicalize_name(requirement.name) not in product_packages:
        return dependency
    if requirement.url is not None or str(requirement.specifier) != f"=={old_version}":
        raise ProductVersionError(
            f"{context} must select its product-owned dependency exactly at {old_version}"
        )

    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    marker = f"; {requirement.marker}" if requirement.marker is not None else ""
    return f"{requirement.name}{extras}=={new_version}{marker}"


def _python_projections(old_version: str, new_version: str) -> list[Projection]:
    try:
        product_packages = frozenset(inventory("release-owned"))
    except InventoryError as error:
        raise ProductVersionError(str(error)) from error
    normalized_product_packages = frozenset(map(canonicalize_name, product_packages))
    projections: list[Projection] = []
    for package_name in sorted(product_packages):
        path = ROOT / "python" / package_name / "pyproject.toml"
        document = _load_toml(path)
        project = _mapping(document.get("project"), f"{path.relative_to(ROOT)}.project")
        name = _string(project, "name", f"{path.relative_to(ROOT)}.project")
        if name != package_name:
            raise ProductVersionError(
                f"{path.relative_to(ROOT)}.project.name {name} does not match {package_name}"
            )
        version = _string(project, "version", f"{path.relative_to(ROOT)}.project")
        if version != old_version:
            raise ProductVersionError(
                f"{path.relative_to(ROOT)}.project.version {version} "
                f"does not match product version {old_version}"
            )
        project["version"] = new_version

        dependencies = project.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) for dependency in dependencies
        ):
            raise ProductVersionError(
                f"{path.relative_to(ROOT)}.project.dependencies must be a string array"
            )
        for index, dependency in enumerate(dependencies):
            try:
                requirement = Requirement(dependency)
            except InvalidRequirement as error:
                raise ProductVersionError(
                    f"{path.relative_to(ROOT)}.project.dependencies[{index}] "
                    f"is not a valid requirement: {error}"
                ) from error
            if canonicalize_name(requirement.name) == "inferlab-adapter-sdk":
                raise ProductVersionError(
                    f"{path.relative_to(ROOT)}: product-owned packages must not depend "
                    "on the independently released inferlab-adapter-sdk"
                )
            dependencies[index] = _replace_exact_product_requirement(
                dependency,
                normalized_product_packages,
                old_version,
                new_version,
                f"{path.relative_to(ROOT)}.project.dependencies[{index}]",
            )
        projections.append(Projection(path, tomlkit.dumps(document)))
    return projections


def _plugin_projections(old_version: str, new_version: str) -> list[Projection]:
    projections: list[Projection] = []
    marketplace_path = ROOT / ".claude-plugin/marketplace.json"
    marketplace = _load_json(marketplace_path)
    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list):
        raise ProductVersionError(".claude-plugin/marketplace.json.plugins must be an array")
    matches = [
        plugin
        for plugin in plugins
        if isinstance(plugin, MutableMapping) and plugin.get("name") == "inferlab"
    ]
    if len(matches) != 1:
        raise ProductVersionError(
            ".claude-plugin/marketplace.json must declare exactly one inferlab plugin"
        )
    marketplace_plugin = cast(MutableMapping[str, object], matches[0])
    if marketplace_plugin.get("version") != old_version:
        raise ProductVersionError(
            ".claude-plugin/marketplace.json inferlab version does not match "
            f"product version {old_version}"
        )
    marketplace_plugin["version"] = new_version
    projections.append(Projection(marketplace_path, json.dumps(marketplace, indent=2) + "\n"))

    for relative in (
        "plugins/inferlab/.claude-plugin/plugin.json",
        "plugins/inferlab/.codex-plugin/plugin.json",
    ):
        path = ROOT / relative
        manifest = _load_json(path)
        if manifest.get("name") != "inferlab":
            raise ProductVersionError(f"{relative}.name must be inferlab")
        if manifest.get("version") != old_version:
            raise ProductVersionError(
                f"{relative}.version does not match product version {old_version}"
            )
        manifest["version"] = new_version
        projections.append(Projection(path, json.dumps(manifest, indent=2) + "\n"))
    return projections


def _prepare_projections(new_version: str) -> tuple[str, list[Projection]]:
    if SEMVER.fullmatch(new_version) is None:
        raise ProductVersionError(f"VERSION must be strict semver (X.Y.Z), got: {new_version}")
    root_path = ROOT / "Cargo.toml"
    root_document = _load_toml(root_path)
    old_version = _product_version(root_document)
    package_paths, cargo_manifests = _workspace_manifests(root_document)
    old_requirement = _crate_requirement(old_version)
    new_requirement = _crate_requirement(new_version)

    workspace = _mapping(root_document.get("workspace"), "Cargo.toml.workspace")
    package = _mapping(workspace.get("package"), "Cargo.toml.workspace.package")
    package["version"] = new_version
    projections = [Projection(root_path, tomlkit.dumps(root_document))]
    for path, document in cargo_manifests:
        _update_cargo_dependencies(
            document,
            str(path.relative_to(ROOT)),
            frozenset(package_paths),
            old_requirement,
            new_requirement,
        )
        projections.append(Projection(path, tomlkit.dumps(document)))
    projections.extend(_python_projections(old_version, new_version))
    projections.extend(_plugin_projections(old_version, new_version))
    return old_version, projections


def _write_projection(projection: Projection) -> None:
    temporary: pathlib.Path | None = None
    try:
        mode = projection.path.stat().st_mode
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=projection.path.parent,
            prefix=f".{projection.path.name}.",
            delete=False,
        ) as destination:
            temporary = pathlib.Path(destination.name)
            destination.write(projection.content)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.chmod(mode)
        temporary.replace(projection.path)
    except OSError as error:
        raise ProductVersionError(
            f"could not update {projection.path.relative_to(ROOT)}: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bump(new_version: str) -> None:
    old_version, projections = _prepare_projections(new_version)
    changed = [
        projection
        for projection in projections
        if projection.path.read_text(encoding="utf-8") != projection.content
    ]
    for projection in changed:
        _write_projection(projection)
    print(f"product version: {old_version} -> {new_version}")
    print(f"updated projections: {len(changed)}")


def check(tag: str) -> None:
    root_document = _load_toml(ROOT / "Cargo.toml")
    version = _product_version(root_document)
    if tag != f"v{version}":
        raise ProductVersionError(f"tag {tag} != product version v{version}")
    _prepare_projections(version)
    print(f"product version projections agree at {version}")


def show() -> None:
    print(_product_version(_load_toml(ROOT / "Cargo.toml")))


def main() -> int:
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "show":
            show()
        elif len(sys.argv) == 3 and sys.argv[1] == "bump":
            bump(sys.argv[2])
        elif len(sys.argv) == 3 and sys.argv[1] == "check":
            check(sys.argv[2])
        else:
            raise ProductVersionError("usage: product_version.py {show|bump VERSION|check TAG}")
    except ProductVersionError as error:
        print(f"product-version: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
