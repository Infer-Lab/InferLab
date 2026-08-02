#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_root="${root}/python"
scope="${1:-}"

workspace_side_packages=(
  inferlab-adapter-sdk
  inferlab-integration-sglang
  inferlab-integration-specialized-engine
  inferlab-integration-tensorrt-llm
  inferlab-integration-tokenspeed
  inferlab-integration-vllm
)
release_runner_packages=(
  inferlab-bench-runner
  inferlab-eval-runner
)
release_owned_packages=(
  "${release_runner_packages[@]}"
  inferlab-measurement-sdk
)
classified_packages=("${workspace_side_packages[@]}" "${release_owned_packages[@]}")
mapfile -t discovered_packages < <(
  for pyproject in "${python_root}"/*/pyproject.toml; do
    basename "$(dirname "${pyproject}")"
  done | LC_ALL=C sort
)
mapfile -t classified_packages < <(printf '%s\n' "${classified_packages[@]}" | LC_ALL=C sort)
test "${discovered_packages[*]}" = "${classified_packages[*]}" || {
  echo "python package ownership inventory does not classify every pyproject.toml" >&2
  exit 1
}

case "${scope}" in
  all)
    packages=("${classified_packages[@]}")
    ;;
  workspace-side)
    packages=("${workspace_side_packages[@]}")
    ;;
  release-owned)
    packages=("${release_owned_packages[@]}")
    ;;
  release-runners)
    packages=("${release_runner_packages[@]}")
    ;;
  *)
    echo "usage: $0 {all|workspace-side|release-owned|release-runners}" >&2
    exit 2
    ;;
esac

test "${#packages[@]}" -gt 0 || {
  echo "python package inventory is empty for scope: ${scope}" >&2
  exit 1
}

for package in "${packages[@]}"; do
  pyproject="${python_root}/${package}/pyproject.toml"
  test -f "${pyproject}" || {
    echo "python package inventory entry has no pyproject.toml: ${pyproject}" >&2
    exit 1
  }
  printf '%s\n' "${package}"
done | LC_ALL=C sort
