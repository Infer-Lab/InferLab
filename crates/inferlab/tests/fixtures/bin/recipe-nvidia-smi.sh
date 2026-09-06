#!/bin/sh
case " $* " in
  *" --query-compute-apps="*)
    if [ -n "${FIXTURE_NVIDIA_SMI_COMPUTE_APPS:-}" ]; then
      printf '%s\n' "$FIXTURE_NVIDIA_SMI_COMPUTE_APPS"
    fi
    exit 0
    ;;
esac
ids="0,1,2,3,4,5,6,7"
while [ $# -gt 0 ]; do
  case "$1" in
    -i) ids="$2"; shift 2 ;;
    *) shift ;;
  esac
done
IFS=,
for id in $ids; do
  printf '%s, Fixture GPU, 97871, GPU-fixture-000%s, 580.65.06\n' "$id" "$id"
done
