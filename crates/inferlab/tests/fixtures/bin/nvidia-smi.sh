#!/bin/sh
# Fake `nvidia-smi` for every test harness. Mirrors the CSV rows of
# `--query-gpu=index,name,memory.total,uuid,driver_version` and
# `--query-compute-apps=pid,used_memory` with `--format=csv,noheader,nounits`
# (driver 580 and 595 print the same shape) for eight fixture GPUs.
# Switches: FIXTURE_NVIDIA_SMI_ERROR fails every call; FIXTURE_NVIDIA_SMI_ERROR_ON
# fails on one ssh target (FAKE_SSH_TARGET); FIXTURE_NVIDIA_SMI_COMPUTE_APPS
# rows answer compute-app queries, optionally only for device
# FIXTURE_NVIDIA_SMI_COMPUTE_APPS_DEVICE; FIXTURE_NVIDIA_SMI_PROBE_LOG counts
# compute-app probes, after FIXTURE_NVIDIA_SMI_HELD_PROBES of which devices
# read free and after FIXTURE_NVIDIA_SMI_HANG_AFTER_PROBES of which a probe hangs.
if [ -n "${FIXTURE_NVIDIA_SMI_ERROR_ON:-}" ] && [ "${FAKE_SSH_TARGET:-}" = "$FIXTURE_NVIDIA_SMI_ERROR_ON" ]; then
  printf 'fixture probe failure on %s\n' "$FAKE_SSH_TARGET" >&2
  exit 9
fi
if [ -n "${FIXTURE_NVIDIA_SMI_ERROR:-}" ]; then
  printf '%s\n' "$FIXTURE_NVIDIA_SMI_ERROR" >&2
  exit 9
fi
case " $* " in
  *" --query-compute-apps="*)
    if [ -n "${FIXTURE_NVIDIA_SMI_PROBE_LOG:-}" ]; then
      printf 'probe\n' >> "$FIXTURE_NVIDIA_SMI_PROBE_LOG"
      probes="$(wc -l < "$FIXTURE_NVIDIA_SMI_PROBE_LOG")"
    fi
    if [ -n "${FIXTURE_NVIDIA_SMI_HANG_AFTER_PROBES:-}" ] && [ "$probes" -gt "$FIXTURE_NVIDIA_SMI_HANG_AFTER_PROBES" ]; then
      sleep 3600
    fi
    if [ -n "${FIXTURE_NVIDIA_SMI_HELD_PROBES:-}" ] && [ "$probes" -gt "$FIXTURE_NVIDIA_SMI_HELD_PROBES" ]; then
      exit 0
    fi
    device=""
    previous=""
    for argument in "$@"; do
      if [ "$previous" = "-i" ]; then device="$argument"; fi
      previous="$argument"
    done
    if [ -n "${FIXTURE_NVIDIA_SMI_COMPUTE_APPS_DEVICE:-}" ] && [ "$device" != "$FIXTURE_NVIDIA_SMI_COMPUTE_APPS_DEVICE" ]; then
      exit 0
    fi
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
