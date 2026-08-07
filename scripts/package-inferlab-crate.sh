#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${root}/target/package}"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT
stage="$("${root}/scripts/stage-inferlab-crate.sh" "${temporary}/source")"

mkdir -p "${output}"

# This local preflight proves the packaged payload before the current product
# version's internal crates are available from the registry. The operator
# publication phase performs full verification from a retained stage without
# these path patches.
CARGO_TARGET_DIR="${stage}/target" cargo package \
  --manifest-path "${stage}/Cargo.toml" \
  --locked \
  --offline \
  --allow-dirty \
  --no-verify \
  --config "patch.crates-io.inferlab-runtime.path='${stage}/crates/inferlab-runtime'" \
  --config "patch.crates-io.inferlab-profiler.path='${stage}/crates/inferlab-profiler'" \
  --config "patch.crates-io.inferlab-protocol.path='${stage}/crates/inferlab-protocol'" \
  --config "patch.crates-io.inferlab-proxy.path='${stage}/crates/inferlab-proxy'" \
  --config "patch.crates-io.inferlab-serve-domain.path='${stage}/crates/inferlab-serve-domain'" \
  -p inferlab

mapfile -t artifacts < <(
  find "${stage}/target/package" -maxdepth 1 -type f -name 'inferlab-*.crate' -print
)
if [ "${#artifacts[@]}" -ne 1 ]; then
  echo "expected one staged inferlab crate, found ${#artifacts[@]}" >&2
  exit 1
fi

artifact="${artifacts[0]}"
archive_list="$(tar -tzf "${artifact}")"
grep -q '/resources/bench-agentic-sources.toml$' <<< "${archive_list}"
grep -q '/resources/toolchain-python/inferlab_eval_runner/__init__.py$' <<< "${archive_list}"
grep -q '/resources/toolchain-python/inferlab_bench_runner/__init__.py$' <<< "${archive_list}"
grep -q '/resources/toolchain-python/inferlab_measurement_sdk/__init__.py$' <<< "${archive_list}"
grep -q '/resources/plugin/plugins/inferlab/skills/inferlab/SKILL.md$' <<< "${archive_list}"
grep -q '/resources/plugin/plugins/inferlab/skills/inferlab/references/capability-map.md$' <<< "${archive_list}"
grep -q '/resources/plugin/plugins/inferlab/skills/inferlab/references/workspace-authoring.md$' <<< "${archive_list}"
! grep -q '/resources/plugin/docs/workspace-authoring.md$' <<< "${archive_list}"
grep -q '/resources/plugin/docs/backend-support.md$' <<< "${archive_list}"
grep -q '/resources/plugin/LICENSE$' <<< "${archive_list}"

cp "${artifact}" "${output}/"
printf '%s\n' "${output}/$(basename "${artifact}")"
