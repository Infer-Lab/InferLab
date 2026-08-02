#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

fail() {
  echo "release-versioning test: $1" >&2
  exit 1
}

fixture="${temporary}/repo"
mkdir -p "${fixture}/scripts" "${fixture}/bin" "${fixture}/dist"
cp "${root}/scripts/python-package-inventory.sh" "${fixture}/scripts/"
cp "${root}/scripts/bump-product-version.sh" "${fixture}/scripts/"
cp "${root}/scripts/check-product-release-version.sh" "${fixture}/scripts/"
cp "${root}/scripts/prepare-python-package-release.sh" "${fixture}/scripts/"
cp "${root}/scripts/python-package-release-metadata.py" "${fixture}/scripts/"
cp "${root}/Cargo.toml" "${fixture}/"
cp "${root}/LICENSE" "${fixture}/"

all_packages="$("${root}/scripts/python-package-inventory.sh" all)"
while IFS= read -r package; do
  path="python/${package}/pyproject.toml"
  mkdir -p "${fixture}/$(dirname "${path}")"
  cp "${root}/${path}" "${fixture}/${path}"
done <<< "${all_packages}"

while IFS= read -r path; do
  mkdir -p "${fixture}/$(dirname "${path}")"
  cp "${root}/${path}" "${fixture}/${path}"
done <<'EOF'
crates/inferlab/Cargo.toml
.claude-plugin/marketplace.json
plugins/inferlab/.claude-plugin/plugin.json
plugins/inferlab/.codex-plugin/plugin.json
plugins/inferlab/skills/inferlab/SKILL.md
protocol/fixtures/valid/plan-serve-response.json
protocol/fixtures/valid/render-serve-response.json
protocol/fixtures/valid/render-serve-response-launch-file.json
EOF

true_path="$(type -P true)"
ln -s "${true_path}" "${fixture}/bin/cargo"
cp "${root}/scripts/tests/fixtures/pixi-release.sh" "${fixture}/bin/pixi"
chmod +x "${fixture}/bin/pixi"

workspace_inventory="$("${fixture}/scripts/python-package-inventory.sh" workspace-side)"
release_owned_inventory="$("${fixture}/scripts/python-package-inventory.sh" release-owned)"
release_runner_inventory="$("${fixture}/scripts/python-package-inventory.sh" release-runners)"
workspace_projects=()
while IFS= read -r package; do
  workspace_projects+=("${fixture}/python/${package}/pyproject.toml")
done <<< "${workspace_inventory}"
protocol_fixtures=("${fixture}"/protocol/fixtures/valid/*-response*.json)
workspace_hashes="$(sha256sum "${workspace_projects[@]}")"
protocol_hashes="$(sha256sum "${protocol_fixtures[@]}")"
sdk_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' \
  "${fixture}/python/inferlab-adapter-sdk/pyproject.toml")"
vllm_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' \
  "${fixture}/python/inferlab-integration-vllm/pyproject.toml")"
target_product_version="9.8.7"

PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/bump-product-version.sh" "${target_product_version}" \
  > "${temporary}/bump.out"

grep -Eq "^version = \"${target_product_version}\"$" "${fixture}/Cargo.toml" \
  || fail "product version was not updated"
for dependency in inferlab-runtime inferlab-profiler inferlab-protocol inferlab-proxy inferlab-serve-domain; do
  grep -Eq "^${dependency} = .*version = \"9\"" "${fixture}/crates/inferlab/Cargo.toml" \
    || fail "${dependency} requirement did not follow the product major version"
done
while IFS= read -r package; do
  grep -Eq "^version = \"${target_product_version}\"$" \
    "${fixture}/python/${package}/pyproject.toml" \
    || fail "${package} did not follow the product version"
done <<< "${release_owned_inventory}"
while IFS= read -r package; do
  grep -Fq "inferlab-measurement-sdk==${target_product_version}" \
    "${fixture}/python/${package}/pyproject.toml" \
    || fail "${package} measurement SDK dependency did not follow the product bump"
done <<< "${release_runner_inventory}"
while IFS= read -r package; do
  ! grep -Fq "inferlab-adapter-sdk" \
    "${fixture}/python/${package}/pyproject.toml" \
    || fail "release-owned package ${package} depends on the public adapter SDK"
done <<< "${release_owned_inventory}"
printf '%s\n' "${workspace_hashes}" | sha256sum --check --quiet \
  || fail "a workspace-side package changed during a product bump"
printf '%s\n' "${protocol_hashes}" | sha256sum --check --quiet \
  || fail "adapter fixture identity changed during a product bump"
grep -Eq "\"version\": \"${target_product_version}\"" \
  "${fixture}/plugins/inferlab/.codex-plugin/plugin.json" \
  || fail "embedded plugin did not follow the product version"
for skill in plugins/inferlab/skills/inferlab/SKILL.md; do
  grep -Fq "[${target_product_version} workspace authoring guide]" \
    "${fixture}/${skill}" \
    || fail "${skill} documentation link label did not follow the product version"
  grep -Fq "/blob/v${target_product_version}/docs/workspace-authoring.md" \
    "${fixture}/${skill}" \
    || fail "${skill} documentation link did not follow the product version"
done
"${fixture}/scripts/check-product-release-version.sh" "v${target_product_version}"

vllm_wheel="inferlab_integration_vllm-${vllm_version}-py3-none-any.whl"
sdk_wheel="inferlab_adapter_sdk-${sdk_version}-py3-none-any.whl"
touch "${fixture}/dist/${vllm_wheel}"
touch "${fixture}/dist/${sdk_wheel}"

vllm_pyproject="${fixture}/python/inferlab-integration-vllm/pyproject.toml"
cp "${vllm_pyproject}" "${temporary}/vllm-pyproject.toml"
sed -i \
  "s/inferlab-adapter-sdk==${sdk_version}/inferlab-adapter-sdk>=${sdk_version}/" \
  "${vllm_pyproject}"
printf '\n[project.optional-dependencies]\ntest = ["inferlab-adapter-sdk==%s"]\n' \
  "${sdk_version}" >> "${vllm_pyproject}"
if PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/prepare-python-package-release.sh" inferlab-integration-vllm \
  > "${temporary}/non-exact.out" 2>&1; then
  fail "publication preparation accepted a non-exact runtime SDK dependency"
fi
grep -q 'exact inferlab-adapter-sdk runtime dependency' \
  "${temporary}/non-exact.out" \
  || {
    cat "${temporary}/non-exact.out" >&2
    fail "non-exact SDK dependency failure did not identify the runtime requirement"
  }
cp "${temporary}/vllm-pyproject.toml" "${vllm_pyproject}"

PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/prepare-python-package-release.sh" inferlab-integration-vllm \
  > "${temporary}/release.out"

grep -Fq "twine upload dist/${vllm_wheel}" "${temporary}/release.out" \
  || fail "publication output did not select the requested wheel"
! grep -Fq 'gh release create' "${temporary}/release.out" \
  || fail "package-only publication still emits a package-scoped GitHub release"
! grep -Fq "inferlab-integration-vllm-v${vllm_version}" "${temporary}/release.out" \
  || fail "package-only publication still derives a package-scoped tag"
! grep -Fq "${sdk_wheel}" "${temporary}/release.out" \
  || fail "publication output included an unrelated wheel"
test -f "${fixture}/dist/${vllm_wheel}.sha256" \
  || fail "publication preparation did not write the selected wheel checksum"

if PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/prepare-python-package-release.sh" inferlab-measurement-sdk \
  > "${temporary}/invalid.out" 2>&1; then
  fail "publication preparation accepted the internal measurement SDK"
fi
grep -q 'not a workspace-side package' "${temporary}/invalid.out" \
  || fail "internal measurement SDK failure did not name the ownership boundary"
