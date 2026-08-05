#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="${1:-}"
repository="${2:-}"
expected_state="${3:-}"
inventory="${4:-}"

fail() {
  echo "verify-product-release: $1" >&2
  exit 1
}

test "$#" -eq 4 \
  || fail "usage: $0 TAG REPOSITORY {draft|published} {repository|aggregate}"
case "${expected_state}" in
  draft) expected_draft=true ;;
  published) expected_draft=false ;;
  *) fail "state must be draft or published, got: ${expected_state}" ;;
esac
case "${inventory}" in
  repository) verification=verify-repository ;;
  aggregate) verification=verify-aggregate ;;
  *) fail "inventory must be repository or aggregate, got: ${inventory}" ;;
esac

"${root}/scripts/check-product-release-version.sh" "${tag}"
actual_draft="$(
  gh release view "${tag}" --repo "${repository}" \
    --json isDraft --jq '.isDraft'
)"
if [ "${actual_draft}" != "${expected_draft}" ]; then
  fail "expected ${expected_state} Release ${tag}, but isDraft=${actual_draft}"
fi

download="$(mktemp -d)"
trap 'rm -rf "${download}"' EXIT
gh release download "${tag}" --repo "${repository}" --dir "${download}"
python3 "${root}/scripts/product_release_assets.py" \
  "${verification}" "${tag}" "${download}" > /dev/null
printf 'verified %s %s Release: %s\n' "${expected_state}" "${inventory}" "${tag}"
