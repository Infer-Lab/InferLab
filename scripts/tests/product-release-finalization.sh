#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

fail() {
  echo "product-release-finalization test: $1" >&2
  exit 1
}

mkdir -p "${temporary}/bin" "${temporary}/wheels" "${temporary}/candidates"
cp "${root}/scripts/tests/fixtures/gh-release.sh" "${temporary}/bin/gh"
chmod +x "${temporary}/bin/gh"

version="$(sed -n 's/^version = "\(.*\)"$/\1/p' "${root}/Cargo.toml" | head -1)"
tag="v${version}"
"${root}/scripts/tests/fixtures/product-release-assets.sh" \
  "${root}" - "${temporary}/wheels"
candidate_path="$(find "${temporary}/wheels" -name '*.whl' -print -quit)"
candidate_wheel="$(basename "${candidate_path}")"
test -n "${candidate_wheel}" || fail "workspace package inventory is empty"
cp "${candidate_path}" "${temporary}/candidates/"
cp "${candidate_path}.sha256" "${temporary}/candidates/"

reset_release() {
  rm -rf "${temporary}/release"
  "${root}/scripts/tests/fixtures/product-release-assets.sh" \
    "${root}" "${temporary}/release" -
  printf 'draft\n' > "${temporary}/state"
  : > "${temporary}/gh.calls"
}

export INFERLAB_TEST_RELEASE_DIR="${temporary}/release"
export INFERLAB_TEST_RELEASE_STATE="${temporary}/state"
export INFERLAB_TEST_GH_CALLS="${temporary}/gh.calls"

reset_release
PATH="${temporary}/bin:${PATH}" \
  "${root}/scripts/finalize-product-release.sh" \
    "${tag}" "${temporary}/wheels" "${temporary}/candidates" example/inferlab \
    > "${temporary}/finalize.out"
test "$(< "${temporary}/state")" = published \
  || fail "successful finalization did not publish the draft"
grep -Fq "twine upload ${temporary}/candidates/${candidate_wheel}" \
  "${temporary}/finalize.out" \
  || fail "successful finalization omitted the qualified candidate upload command"
python3 "${root}/scripts/product_release_assets.py" \
  verify-aggregate "${tag}" "${temporary}/release" > /dev/null
upload_line="$(grep -n '^release upload ' "${temporary}/gh.calls" | cut -d: -f1)"
publish_line="$(grep -n '^release edit ' "${temporary}/gh.calls" | cut -d: -f1)"
test -n "${upload_line}" && test -n "${publish_line}" && test "${upload_line}" -lt "${publish_line}" \
  || fail "draft publication preceded wheel upload"

reset_release
printf 'published\n' > "${temporary}/state"
if PATH="${temporary}/bin:${PATH}" \
  "${root}/scripts/finalize-product-release.sh" \
    "${tag}" "${temporary}/wheels" "${temporary}/candidates" example/inferlab \
    > "${temporary}/non-draft.out" 2>&1; then
  fail "finalization accepted a non-draft Release"
fi
grep -Fq 'expected draft Release' "${temporary}/non-draft.out" \
  || fail "non-draft failure did not identify the required Release state"
! grep -q '^release upload ' "${temporary}/gh.calls" \
  || fail "non-draft finalization uploaded assets"

reset_release
missing_checksum="$(find "${temporary}/wheels" -name '*.whl.sha256' -print -quit)"
mv "${missing_checksum}" "${missing_checksum}.missing"
if PATH="${temporary}/bin:${PATH}" \
  "${root}/scripts/finalize-product-release.sh" \
    "${tag}" "${temporary}/wheels" "${temporary}/candidates" example/inferlab \
    > "${temporary}/incomplete.out" 2>&1; then
  fail "finalization accepted an incomplete wheel inventory"
fi
test ! -s "${temporary}/gh.calls" \
  || fail "incomplete inventory reached GitHub"
mv "${missing_checksum}.missing" "${missing_checksum}"

reset_release
printf 'different qualified bytes\n' > "${temporary}/candidates/${candidate_wheel}"
(cd "${temporary}/candidates" && \
  sha256sum "${candidate_wheel}" > "${candidate_wheel}.sha256")
if PATH="${temporary}/bin:${PATH}" \
  "${root}/scripts/finalize-product-release.sh" \
    "${tag}" "${temporary}/wheels" "${temporary}/candidates" example/inferlab \
    > "${temporary}/mismatch.out" 2>&1; then
  fail "finalization accepted candidate bytes that differ from the aggregate inventory"
fi
grep -Fq 'does not match aggregate wheel' "${temporary}/mismatch.out" \
  || fail "candidate mismatch failure did not identify the byte-identity boundary"
test ! -s "${temporary}/gh.calls" \
  || fail "candidate mismatch reached GitHub"

printf 'product release finalization tests passed\n'
