#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${root}/target/package}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

mkdir -p "${output}"

# Stage the current source view without ignored caches, build products, or
# local bindings. This admits reviewed working-copy changes during release
# preparation while retaining Git as the public package inventory authority.
if [ -d "${root}/.jj" ]; then
  file_inventory=(jj -R "${root}" file list -T 'path ++ "\0"')
else
  file_inventory=(git -C "${root}" ls-files --cached --others --exclude-standard -z)
fi

"${file_inventory[@]}" \
  | while IFS= read -r -d '' path; do
      if [ -e "${root}/${path}" ] || [ -L "${root}/${path}" ]; then
        printf '%s\0' "${path}"
      fi
    done \
  | tar -C "${root}" --null --files-from=- -cf - \
  | tar -C "${stage}" -xf -

copy_tree() {
  local source="$1"
  local destination="$2"
  mkdir -p "${destination}"
  tar -C "${source}" -cf - . | tar -C "${destination}" -xf -
}

copy_python_tree() {
  local source="$1"
  local destination="$2"
  mkdir -p "${destination}"
  tar -C "${source}" --exclude='__pycache__' --exclude='*.pyc' -cf - . \
    | tar -C "${destination}" -xf -
}

payload="${stage}/crates/inferlab/resources"
copy_python_tree \
  "${root}/python/inferlab-eval-runner/src/inferlab_eval_runner" \
  "${payload}/toolchain-python/inferlab_eval_runner"
copy_python_tree \
  "${root}/python/inferlab-bench-runner/src/inferlab_bench_runner" \
  "${payload}/toolchain-python/inferlab_bench_runner"
copy_python_tree \
  "${root}/python/inferlab-measurement-sdk/src/inferlab_measurement_sdk" \
  "${payload}/toolchain-python/inferlab_measurement_sdk"

mkdir -p "${payload}/plugin"
cp "${root}/LICENSE" "${payload}/plugin/LICENSE"
for directory in .claude-plugin .agents plugins; do
  copy_tree "${root}/${directory}" "${payload}/plugin/${directory}"
done

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
grep -q '/resources/toolchain-python/inferlab_eval_runner/__init__.py$' <<< "${archive_list}"
grep -q '/resources/toolchain-python/inferlab_bench_runner/__init__.py$' <<< "${archive_list}"
grep -q '/resources/toolchain-python/inferlab_measurement_sdk/__init__.py$' <<< "${archive_list}"
grep -q '/resources/plugin/plugins/inferlab/skills/inferlab/SKILL.md$' <<< "${archive_list}"
grep -q '/resources/plugin/LICENSE$' <<< "${archive_list}"

cp "${artifact}" "${output}/"
printf '%s\n' "${output}/$(basename "${artifact}")"
