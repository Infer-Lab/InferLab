#!/usr/bin/env bash
set -euo pipefail

root="${1:-}"
repository_assets="${2:-}"
wheel_assets="${3:-}"
test "$#" -eq 3 \
  || { echo "product-release-assets fixture: usage: $0 ROOT REPOSITORY_ASSETS WHEEL_ASSETS" >&2; exit 1; }

if [ "${repository_assets}" != - ]; then
  mkdir -p "${repository_assets}"
  for asset in inferlab-x86_64-linux inferlab-aarch64-linux install.sh inferlab-plugin.tar.gz; do
    printf 'qualified bytes for %s\n' "${asset}" > "${repository_assets}/${asset}"
    (cd "${repository_assets}" && sha256sum "${asset}" > "${asset}.sha256")
  done
  cp "${root}/LICENSE" "${repository_assets}/LICENSE"
fi

if [ "${wheel_assets}" != - ]; then
  mkdir -p "${wheel_assets}"
  while IFS= read -r package; do
    package_version="$(
      sed -n 's/^version = "\(.*\)"$/\1/p' \
        "${root}/python/${package}/pyproject.toml" | head -1
    )"
    wheel="${package//-/_}-${package_version}-py3-none-any.whl"
    printf 'qualified bytes for %s\n' "${wheel}" > "${wheel_assets}/${wheel}"
    (cd "${wheel_assets}" && sha256sum "${wheel}" > "${wheel}.sha256")
  done < <("${root}/scripts/python-package-inventory.sh" workspace-side)
fi
