#!/usr/bin/env python3
"""Read Python package release ownership from each package's metadata."""

from __future__ import annotations

import pathlib
import sys
import tomllib
from collections.abc import Mapping

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIFECYCLES = frozenset({"product", "workspace-side"})
MEASUREMENT_SDK = "inferlab-measurement-sdk"


class InventoryError(Exception):
    """The package inventory is incomplete or internally inconsistent."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InventoryError(f"{context} must be a table")
    return value


def _string(mapping: Mapping[str, object], field: str, context: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise InventoryError(f"{context}.{field} must be a nonempty string")
    return value


def _package_metadata(path: pathlib.Path) -> tuple[str, str, tuple[str, ...]]:
    relative = path.relative_to(ROOT)
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise InventoryError(f"could not read {relative}: {error}") from error

    project = _mapping(document.get("project"), f"{relative}.project")
    name = _string(project, "name", f"{relative}.project")
    dependencies_value = project.get("dependencies", [])
    if not isinstance(dependencies_value, list) or not all(
        isinstance(value, str) for value in dependencies_value
    ):
        raise InventoryError(f"{relative}.project.dependencies must be a string array")

    tool = _mapping(document.get("tool"), f"{relative}.tool")
    inferlab = _mapping(tool.get("inferlab"), f"{relative}.tool.inferlab")
    release = _mapping(inferlab.get("release"), f"{relative}.tool.inferlab.release")
    lifecycle = _string(release, "lifecycle", f"{relative}.tool.inferlab.release")
    if lifecycle not in LIFECYCLES:
        expected = ", ".join(sorted(LIFECYCLES))
        raise InventoryError(
            f"{relative}.tool.inferlab.release.lifecycle must be one of: {expected}"
        )
    return name, lifecycle, tuple(dependencies_value)


def inventory(scope: str) -> list[str]:
    packages: list[tuple[str, str, tuple[str, ...]]] = []
    for path in sorted((ROOT / "python").glob("*/pyproject.toml")):
        package = _package_metadata(path)
        directory_name = path.parent.name
        if package[0] != directory_name:
            raise InventoryError(
                f"{path.relative_to(ROOT)}: project name {package[0]} "
                f"does not match directory {directory_name}"
            )
        packages.append(package)
    if not packages:
        raise InventoryError("python package inventory is empty")

    if scope == "all":
        selected = [name for name, _, _ in packages]
    elif scope == "workspace-side":
        selected = [name for name, lifecycle, _ in packages if lifecycle == scope]
    elif scope == "release-owned":
        selected = [name for name, lifecycle, _ in packages if lifecycle == "product"]
    elif scope == "release-runners":
        selected = [
            name
            for name, lifecycle, dependencies in packages
            if lifecycle == "product"
            and any(
                dependency == MEASUREMENT_SDK or dependency.startswith(f"{MEASUREMENT_SDK}==")
                for dependency in dependencies
            )
        ]
    else:
        raise InventoryError(
            "usage: python-package-inventory.sh {all|workspace-side|release-owned|release-runners}"
        )

    if not selected:
        raise InventoryError(f"python package inventory is empty for scope: {scope}")
    return sorted(selected)


def main() -> int:
    try:
        if len(sys.argv) != 2:
            raise InventoryError(
                "usage: python-package-inventory.sh "
                "{all|workspace-side|release-owned|release-runners}"
            )
        print("\n".join(inventory(sys.argv[1])))
    except InventoryError as error:
        print(f"python-package-inventory: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
