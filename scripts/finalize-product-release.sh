#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="${1:-}"
wheels="${2:-}"
candidates="${3:-}"
repository="${4:-}"

fail() {
  echo "finalize-product-release: $1" >&2
  exit 1
}

test "$#" -eq 4 || fail "usage: $0 TAG WHEELS CANDIDATES REPOSITORY"
test -d "${candidates}" || fail "candidate directory does not exist: ${candidates}"
"${root}/scripts/check-product-release-version.sh" "${tag}"
python3 "${root}/scripts/product_release_assets.py" \
  verify-wheels "${tag}" "${wheels}" > /dev/null

candidate_wheels=()
while IFS= read -r -d '' candidate; do
  test -f "${candidate}" || fail "candidate entry is not a file: ${candidate}"
  case "${candidate}" in
    *.whl.sha256)
      test -f "${candidate%.sha256}" \
        || fail "candidate checksum has no wheel: ${candidate}"
      ;;
    *.whl)
      candidate_wheels+=("${candidate}")
      ;;
    *)
      fail "candidate directory contains an unexpected entry: ${candidate}"
      ;;
  esac
done < <(find "${candidates}" -mindepth 1 -maxdepth 1 -print0)

for candidate in "${candidate_wheels[@]}"; do
  filename="$(basename "${candidate}")"
  checksum="${candidate}.sha256"
  test -f "${checksum}" || fail "candidate wheel has no checksum: ${candidate}"
  (cd "${candidates}" && sha256sum --check --strict "${filename}.sha256") > /dev/null
  aggregate_wheel="${wheels}/${filename}"
  test -f "${aggregate_wheel}" \
    || fail "candidate ${filename} is absent from the aggregate wheel inventory"
  cmp -s "${candidate}" "${aggregate_wheel}" \
    || fail "candidate ${filename} does not match aggregate wheel bytes"
  cmp -s "${checksum}" "${aggregate_wheel}.sha256" \
    || fail "candidate ${filename} checksum does not match aggregate wheel checksum"
done

"${root}/scripts/verify-product-release.sh" \
  "${tag}" "${repository}" draft repository

wheel_assets=()
while IFS= read -r -d '' asset; do
  wheel_assets+=("${asset}")
done < <(find "${wheels}" -mindepth 1 -maxdepth 1 -type f -print0 | sort -z)
gh release upload "${tag}" "${wheel_assets[@]}" \
  --repo "${repository}" --clobber

"${root}/scripts/verify-product-release.sh" \
  "${tag}" "${repository}" draft aggregate
gh release edit "${tag}" --repo "${repository}" --draft=false
"${root}/scripts/verify-product-release.sh" \
  "${tag}" "${repository}" published aggregate

echo
echo "== registry publication is now unlocked (ADR-0033) =="
echo "# crates.io:"
printf 'just release-crates %q\n' "${tag}"
if [ "${#candidate_wheels[@]}" -gt 0 ]; then
  echo "# Python package index, using the wheel bytes attached above:"
  for candidate in "${candidate_wheels[@]}"; do
    printf 'twine upload %q\n' "${candidate}"
  done
else
  echo "# no new workspace-side wheel requires a Python package-index upload"
fi
