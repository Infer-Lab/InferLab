#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

fail() {
  echo "release-versioning test: $1" >&2
  exit 1
}

test ! -e "${root}/docs/workspace-authoring.md" \
  || fail "the obsolete aggregate workspace-authoring document still exists"

fixture="${temporary}/repo"
mkdir -p "${fixture}/scripts" "${fixture}/bin" "${fixture}/dist"
cp "${root}/scripts/python-package-inventory.sh" "${fixture}/scripts/"
cp "${root}/scripts/python_package_inventory.py" "${fixture}/scripts/"
cp "${root}/scripts/bump-product-version.sh" "${fixture}/scripts/"
cp "${root}/scripts/product_version.py" "${fixture}/scripts/"
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

while IFS= read -r manifest; do
  mkdir -p "${fixture}/$(dirname "${manifest}")"
  cp "${root}/${manifest}" "${fixture}/${manifest}"
done < <(cd "${root}" && find crates -mindepth 2 -maxdepth 2 -name Cargo.toml -print | sort)

while IFS= read -r path; do
  mkdir -p "${fixture}/$(dirname "${path}")"
  cp "${root}/${path}" "${fixture}/${path}"
done <<'EOF'
.claude-plugin/marketplace.json
plugins/inferlab/.claude-plugin/plugin.json
plugins/inferlab/.codex-plugin/plugin.json
plugins/inferlab/skills/inferlab/SKILL.md
docs/backend-support.md
protocol/fixtures/valid/plan-serve-response.json
protocol/fixtures/valid/render-serve-response.json
protocol/fixtures/valid/render-serve-response-launch-file.json
EOF

while IFS= read -r path; do
  mkdir -p "${fixture}/$(dirname "${path}")"
  cp "${root}/${path}" "${fixture}/${path}"
done < <(cd "${root}" && find plugins/inferlab/skills/inferlab/references -type f -name '*.md' -print | sort)

true_path="$(type -P true)"
ln -s "${true_path}" "${fixture}/bin/cargo"
cp "${root}/scripts/tests/fixtures/pixi-release.sh" "${fixture}/bin/pixi"
chmod +x "${fixture}/bin/pixi"
export INFERLAB_TEST_PIXI_PYTHON
INFERLAB_TEST_PIXI_PYTHON="$(pixi run python -c 'import sys; print(sys.executable)')"

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
product_projection_files=(
  "${fixture}/Cargo.toml"
  "${fixture}/.claude-plugin/marketplace.json"
  "${fixture}/plugins/inferlab/.claude-plugin/plugin.json"
  "${fixture}/plugins/inferlab/.codex-plugin/plugin.json"
)
while IFS= read -r manifest; do
  product_projection_files+=("${fixture}/${manifest}")
done < <(cd "${root}" && find crates -mindepth 2 -maxdepth 2 -name Cargo.toml -print | sort)
while IFS= read -r package; do
  product_projection_files+=("${fixture}/python/${package}/pyproject.toml")
done <<< "${release_owned_inventory}"
product_projection_hashes="$(sha256sum "${product_projection_files[@]}")"
product_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "${fixture}/Cargo.toml" | head -1)"
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
while IFS= read -r dependency; do
  echo "${dependency}" | grep -Eq 'version = "9"' \
    || fail "internal Cargo requirement did not follow the product major version: ${dependency}"
done < <(rg '^inferlab-[a-z-]+ = \{ path = .*version = ' "${fixture}/crates" -g Cargo.toml)
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
skill="plugins/inferlab/skills/inferlab/SKILL.md"
grep -Fq '[Workspace authoring](references/workspace-authoring.md)' \
  "${fixture}/${skill}" \
  || fail "${skill} does not use the bundled workspace-authoring guide"
authoring_index="${fixture}/plugins/inferlab/skills/inferlab/references/workspace-authoring.md"
for reference in workspace-definition.md execution-authoring.md eval-authoring.md bench-authoring.md; do
  test -f "${fixture}/plugins/inferlab/skills/inferlab/references/${reference}" \
    || fail "the bundled workspace-authoring guide is missing ${reference}"
  grep -Fq "(${reference})" "${authoring_index}" \
    || fail "the bundled workspace-authoring index does not route to ${reference}"
done
grep -Fq '[backend support matrix](../../../../docs/backend-support.md)' \
  "${fixture}/${skill}" \
  || fail "${skill} does not use the bundled backend-support matrix"
! grep -Eq '/blob/v[0-9]+\.[0-9]+\.[0-9]+/docs/' "${fixture}/${skill}" \
  || fail "${skill} still makes documentation URLs a version projection"
python3 - "${fixture}/plugins/inferlab/skills/inferlab" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
links = re.compile(r"\[[^]]+\]\(([^)]+)\)")
files = [root / "SKILL.md", *sorted((root / "references").glob("*.md"))]
for source in files:
    for target in links.findall(source.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#")):
            continue
        path = (source.parent / target.split("#", 1)[0]).resolve()
        if not path.is_file():
            raise SystemExit(f"{source}: local skill reference does not exist: {target}")
PY
PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/check-product-release-version.sh" "v${target_product_version}"

PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/bump-product-version.sh" "${product_version}" \
  > "${temporary}/round-trip.out"
printf '%s\n' "${product_projection_hashes}" | sha256sum --check --quiet \
  || fail "a structured product-version projection did not round-trip exactly"

bench_pyproject="${fixture}/python/inferlab-bench-runner/pyproject.toml"
cp "${bench_pyproject}" "${temporary}/bench-pyproject.toml"
sed -i \
  "s/inferlab-measurement-sdk==${product_version}/inferlab-measurement-sdk==7.7.7/" \
  "${bench_pyproject}"
inconsistent_projection_hashes="$(sha256sum "${product_projection_files[@]}")"
if PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/bump-product-version.sh" "${target_product_version}" \
  > "${temporary}/inconsistent.out" 2>&1; then
  fail "product bump accepted inconsistent product-owned dependency metadata"
fi
grep -Fq "must select its product-owned dependency exactly at ${product_version}" \
  "${temporary}/inconsistent.out" \
  || fail "inconsistent product-owned dependency failure was not actionable"
printf '%s\n' "${inconsistent_projection_hashes}" | sha256sum --check --quiet \
  || fail "a failed product-version preflight partially updated a projection"
cp "${temporary}/bench-pyproject.toml" "${bench_pyproject}"

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
  "${fixture}/scripts/prepare-python-package-release.sh" \
    inferlab-integration-vllm "${temporary}/candidates" \
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
  "${fixture}/scripts/prepare-python-package-release.sh" \
    inferlab-integration-vllm "${temporary}/candidates" \
  > "${temporary}/release.out"

test -f "${temporary}/candidates/${vllm_wheel}" \
  || fail "candidate staging omitted the requested wheel"
test -f "${temporary}/candidates/${vllm_wheel}.sha256" \
  || fail "candidate staging omitted the requested wheel checksum"
grep -Fq 'registry upload is emitted only after aggregate Release finalization' \
  "${temporary}/release.out" \
  || fail "candidate staging did not preserve registry-last ordering"
! grep -Fq 'twine upload' "${temporary}/release.out" \
  || fail "candidate staging emitted a premature package-index command"
! grep -Fq 'gh release create' "${temporary}/release.out" \
  || fail "package-only publication still emits a package-scoped GitHub release"
! grep -Fq "inferlab-integration-vllm-v${vllm_version}" "${temporary}/release.out" \
  || fail "package-only publication still derives a package-scoped tag"
! grep -Fq "${sdk_wheel}" "${temporary}/release.out" \
  || fail "publication output included an unrelated wheel"
test -f "${fixture}/dist/${vllm_wheel}.sha256" \
  || fail "publication preparation did not write the selected wheel checksum"

if PATH="${fixture}/bin:${PATH}" \
  "${fixture}/scripts/prepare-python-package-release.sh" \
    inferlab-measurement-sdk "${temporary}/candidates" \
  > "${temporary}/invalid.out" 2>&1; then
  fail "publication preparation accepted the internal measurement SDK"
fi
grep -q 'not a workspace-side package' "${temporary}/invalid.out" \
  || fail "internal measurement SDK failure did not name the ownership boundary"
