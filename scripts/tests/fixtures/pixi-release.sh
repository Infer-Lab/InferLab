#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = run ] && [ "${2:-}" = python ]; then
  shift 2
  test -n "${INFERLAB_TEST_PIXI_PYTHON:-}" || {
    echo "fake pixi has no locked Python interpreter" >&2
    exit 1
  }
  exec "${INFERLAB_TEST_PIXI_PYTHON}" "$@"
fi

if [ "${1:-}" = run ] && [ "${2:-}" = build-python ]; then
  exit 0
fi

echo "unexpected fake pixi invocation: $*" >&2
exit 1
