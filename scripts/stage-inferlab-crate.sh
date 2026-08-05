#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:-}"

test "$#" -eq 1 || {
  echo "stage-inferlab-crate: usage: $0 STAGE" >&2
  exit 2
}
test ! -e "${stage}" || {
  echo "stage-inferlab-crate: staging path already exists: ${stage}" >&2
  exit 1
}

mkdir -p "$(dirname "${stage}")"
mkdir "${stage}"
stage="$(cd "${stage}" && pwd)"

# Copy the current reviewed source inventory without caches, build products,
# or local bindings. The retained tree becomes the one package/publish source.
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
  "${stage}/python/inferlab-eval-runner/src/inferlab_eval_runner" \
  "${payload}/toolchain-python/inferlab_eval_runner"
copy_python_tree \
  "${stage}/python/inferlab-bench-runner/src/inferlab_bench_runner" \
  "${payload}/toolchain-python/inferlab_bench_runner"
copy_python_tree \
  "${stage}/python/inferlab-measurement-sdk/src/inferlab_measurement_sdk" \
  "${payload}/toolchain-python/inferlab_measurement_sdk"

mkdir -p "${payload}/plugin"
cp "${stage}/LICENSE" "${payload}/plugin/LICENSE"
mkdir -p "${payload}/plugin/docs"
cp \
  "${stage}/docs/workspace-authoring.md" \
  "${stage}/docs/backend-support.md" \
  "${payload}/plugin/docs/"
for directory in .claude-plugin .agents plugins; do
  copy_tree "${stage}/${directory}" "${payload}/plugin/${directory}"
done

printf '%s\n' "${stage}"
