#!/usr/bin/env bash
set -euo pipefail

test -n "${INFERLAB_CARGO_CALLS:-}"
printf '%q ' "$@" >> "${INFERLAB_CARGO_CALLS}"
printf '\n' >> "${INFERLAB_CARGO_CALLS}"
