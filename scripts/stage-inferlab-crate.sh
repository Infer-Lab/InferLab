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

# Both payload trees copy with the same cache excludes the crate build script
# applies, so no producer can ship or omit caches differently.
copy_payload_tree() {
  local source="$1"
  local destination="$2"
  mkdir -p "${destination}"
  tar -C "${source}" --exclude='__pycache__' --exclude='*.pyc' -cf - . \
    | tar -C "${destination}" -xf -
}

payload="${stage}/crates/inferlab/resources"
# The member set has one manifest, shared with the crate build script and the
# packaging test (scripts/toolchain-python-members.txt).
while IFS= read -r member; do
  [ -n "${member}" ] || continue
  source="${member%% *}"
  package="${member##* }"
  copy_payload_tree "${stage}/${source}" "${payload}/toolchain-python/${package}"
done < <(sed 's/[[:space:]]*$//' "${stage}/scripts/toolchain-python-members.txt")

# The member set has one manifest, shared with the crate build script and the
# release tarball script (scripts/plugin-package-members.txt).
mkdir -p "${payload}/plugin"
while IFS= read -r member; do
  [ -n "${member}" ] || continue
  if [ -d "${stage}/${member}" ]; then
    copy_payload_tree "${stage}/${member}" "${payload}/plugin/${member}"
  else
    mkdir -p "$(dirname "${payload}/plugin/${member}")"
    cp "${stage}/${member}" "${payload}/plugin/${member}"
  fi
done < <(sed 's/[[:space:]]*$//' "${stage}/scripts/plugin-package-members.txt")

printf '%s\n' "${stage}"
