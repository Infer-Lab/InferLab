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
product_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' Cargo.toml | head -1)"
test -n "${product_version}" || fail "Cargo.toml: no workspace package version found"
test "${tag}" = "v${product_version}" \
  || fail "tag ${tag} != product version v${product_version}"

# [[ADR-0021]] Product tags cover only product-owned package metadata. Adapter
# SDK and framework-integration versions are deliberately absent from this gate.
release_owned_inventory="$(scripts/python-package-inventory.sh release-owned)"
while IFS= read -r package; do
  pyproject="python/${package}/pyproject.toml"
  package_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "${pyproject}" | head -1)"
  test -n "${package_version}" || fail "${pyproject}: no project version found"
  test "${package_version}" = "${product_version}" \
    || fail "${pyproject}: version ${package_version} != product version ${product_version}"
done <<< "${release_owned_inventory}"

for runner in inferlab-bench-runner inferlab-eval-runner; do
  pyproject="python/${runner}/pyproject.toml"
  grep -Fq "inferlab-measurement-sdk==${product_version}" "${pyproject}" \
    || fail "${pyproject}: does not depend exactly on product-owned inferlab-measurement-sdk ${product_version}"
  ! grep -Fq "inferlab-adapter-sdk" "${pyproject}" \
    || fail "${pyproject}: release-owned runner depends on the public adapter SDK"
done
