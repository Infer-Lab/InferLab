#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="${1:-}"

fail() {
  echo "bump-product-version: $1" >&2
  exit 1
}

test "$#" -eq 1 || fail "usage: $0 VERSION"

cd "${root}"

# [[ADR-0021]] The structured updater discovers product ownership from Cargo
# workspace metadata and each Python package's tool.inferlab.release table.
# Workspace-side package versions remain untouched.
pixi run python scripts/product_version.py bump "${version}"

cargo build --workspace
pixi run build-python

echo
echo "== product development line opened at ${version}; review the diff, then =="
echo "just verify"
echo "== when ${version} is ready to release =="
echo "govctl release ${version}"
echo "Promote the curated public Unreleased notes to [${version}] without exposing internal WI text."
