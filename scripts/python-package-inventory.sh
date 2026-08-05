#!/bin/sh
set -eu

root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec python3 "${root}/scripts/python_package_inventory.py" "$@"
