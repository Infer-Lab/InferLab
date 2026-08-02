#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="${1:-}"

fail() {
  echo "bump-product-version: $1" >&2
  exit 1
}

test "$#" -eq 1 || fail "usage: $0 VERSION"
echo "${version}" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' \
  || fail "VERSION must be strict semver (X.Y.Z), got: ${version}"

cd "${root}"

IFS=. read -r major minor _patch <<< "${version}"
if [ "${major}" = 0 ]; then
  crate_requirement="${major}.${minor}"
else
  crate_requirement="${major}"
fi

# [[ADR-0021]] Product bumps own the Cargo workspace, embedded plugin, and
# internal measurement package metadata and dependency closure. Workspace-side
# integrations retain their independently released package versions and SDK
# requirements.
sed -i "s/^version = \"[^\"]*\"/version = \"${version}\"/" Cargo.toml
sed -E -i "/^inferlab-runtime = / s/version = \"[^\"]+\"/version = \"${crate_requirement}\"/" \
  crates/inferlab/Cargo.toml
sed -E -i "/^inferlab-profiler = / s/version = \"[^\"]+\"/version = \"${crate_requirement}\"/" \
  crates/inferlab/Cargo.toml
sed -E -i "/^inferlab-protocol = / s/version = \"[^\"]+\"/version = \"${crate_requirement}\"/" \
  crates/inferlab/Cargo.toml
sed -E -i "/^inferlab-proxy = / s/version = \"[^\"]+\"/version = \"${crate_requirement}\"/" \
  crates/inferlab/Cargo.toml
sed -E -i "/^inferlab-serve-domain = / s/version = \"[^\"]+\"/version = \"${crate_requirement}\"/" \
  crates/inferlab/Cargo.toml

release_owned_inventory="$(scripts/python-package-inventory.sh release-owned)"
while IFS= read -r package; do
  pyproject="python/${package}/pyproject.toml"
  sed -i "0,/^version = \"[^\"]*\"/s//version = \"${version}\"/" "${pyproject}"
done <<< "${release_owned_inventory}"
release_runner_inventory="$(scripts/python-package-inventory.sh release-runners)"
while IFS= read -r runner; do
  pyproject="python/${runner}/pyproject.toml"
  sed -E -i \
    "s/inferlab-measurement-sdk==[0-9]+\\.[0-9]+\\.[0-9]+/inferlab-measurement-sdk==${version}/" \
    "${pyproject}"
done <<< "${release_runner_inventory}"

for manifest in \
  .claude-plugin/marketplace.json \
  plugins/inferlab/.claude-plugin/plugin.json \
  plugins/inferlab/.codex-plugin/plugin.json; do
  sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${version}\"/" "${manifest}"
done

for skill in plugins/inferlab/skills/inferlab/SKILL.md; do
  grep -qE '\[[0-9]+\.[0-9]+\.[0-9]+ workspace authoring guide\]' "${skill}" \
    || fail "${skill}: no versioned workspace-authoring link label found"
  grep -qE '/blob/v[0-9]+\.[0-9]+\.[0-9]+/docs/workspace-authoring\.md' "${skill}" \
    || fail "${skill}: no versioned workspace-authoring link found"
  sed -E -i \
    "s|\\[[0-9]+\\.[0-9]+\\.[0-9]+ workspace authoring guide\\]|[${version} workspace authoring guide]|" \
    "${skill}"
  sed -E -i \
    "s|/blob/v[0-9]+\\.[0-9]+\\.[0-9]+/docs/workspace-authoring\\.md|/blob/v${version}/docs/workspace-authoring.md|" \
    "${skill}"
done

cargo build --workspace
pixi run build-python

echo
echo "== product development line opened at ${version}; review the diff, then =="
echo "just verify"
echo "== when ${version} is ready to release =="
echo "govctl release ${version} && govctl render changelog"
