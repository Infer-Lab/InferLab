#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="${1:-}"

fail() {
  echo "check-product-release-version: $1" >&2
  exit 1
}

test "$#" -eq 1 || fail "usage: $0 TAG"

cd "${root}"
pixi run python scripts/product_version.py check "${tag}"
