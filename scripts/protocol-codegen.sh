#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-check}"
adapter_schema="${root}/protocol/schema/adapter-protocol-v7.schema.json"
measurement_schema="${root}/protocol/schema/measurement-protocol-v1.schema.json"
adapter_models="${root}/python/inferlab-adapter-sdk/src/inferlab_adapter_sdk/_generated.py"
measurement_models="${root}/python/inferlab-measurement-sdk/src/inferlab_measurement_sdk/_generated.py"
resource_measurement_models="${root}/crates/inferlab/resources/toolchain-python/inferlab_measurement_sdk/_generated.py"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

mkdir -p "${temporary}/schema"
cargo run --quiet --locked --manifest-path "${root}/Cargo.toml" \
  -p inferlab-protocol --example generate_schema -- \
  "${temporary}/schema/adapter-protocol-v7.schema.json" \
  "${temporary}/schema/measurement-protocol-v1.schema.json"

generate_models() {
  local schema="$1"
  local models="$2"
  datamodel-codegen \
    --input "${schema}" \
    --input-file-type jsonschema \
    --output "${models}" \
    --output-model-type pydantic_v2.BaseModel \
    --target-python-version 3.12 \
    --disable-future-imports \
    --enable-generated-header-marker \
    --use-standard-collections \
    --use-union-operator \
    --use-annotated \
    --strict-nullable \
    --enum-field-as-literal one \
    --infer-union-variant-names \
    --use-one-literal-as-default \
    --extra-fields forbid \
    --formatters black isort \
    --disable-timestamp

  sed -i '1i# Rust wire source: crates/inferlab-protocol/src/wire.rs' "${models}"
}

generate_models \
  "${temporary}/schema/adapter-protocol-v7.schema.json" \
  "${temporary}/adapter_generated.py"
generate_models \
  "${temporary}/schema/measurement-protocol-v1.schema.json" \
  "${temporary}/measurement_generated.py"

case "${mode}" in
  generate)
    mkdir -p \
      "$(dirname "${adapter_schema}")" \
      "$(dirname "${measurement_schema}")" \
      "$(dirname "${adapter_models}")" \
      "$(dirname "${measurement_models}")" \
      "$(dirname "${resource_measurement_models}")"
    cp "${temporary}/schema/adapter-protocol-v7.schema.json" "${adapter_schema}"
    cp "${temporary}/schema/measurement-protocol-v1.schema.json" "${measurement_schema}"
    cp "${temporary}/adapter_generated.py" "${adapter_models}"
    cp "${temporary}/measurement_generated.py" "${measurement_models}"
    cp "${temporary}/measurement_generated.py" "${resource_measurement_models}"
    ;;
  check)
    cmp "${temporary}/schema/adapter-protocol-v7.schema.json" "${adapter_schema}"
    cmp "${temporary}/schema/measurement-protocol-v1.schema.json" "${measurement_schema}"
    cmp "${temporary}/adapter_generated.py" "${adapter_models}"
    cmp "${temporary}/measurement_generated.py" "${measurement_models}"
    cmp "${temporary}/measurement_generated.py" "${resource_measurement_models}"
    ;;
  *)
    printf 'usage: %s [generate|check]\n' "$0" >&2
    exit 2
    ;;
esac
