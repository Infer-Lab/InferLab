#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "gh-release fixture: $1" >&2
  exit 1
}

test -n "${INFERLAB_TEST_RELEASE_DIR:-}" || fail "INFERLAB_TEST_RELEASE_DIR is required"
test -n "${INFERLAB_TEST_RELEASE_STATE:-}" || fail "INFERLAB_TEST_RELEASE_STATE is required"
test -n "${INFERLAB_TEST_GH_CALLS:-}" || fail "INFERLAB_TEST_GH_CALLS is required"

printf '%q ' "$@" >> "${INFERLAB_TEST_GH_CALLS}"
printf '\n' >> "${INFERLAB_TEST_GH_CALLS}"

test "${1:-}" = release || fail "expected release command"
operation="${2:-}"
shift 2

case "${operation}" in
  view)
    if [[ " $* " = *" --json isDraft "* ]]; then
      if [ "$(< "${INFERLAB_TEST_RELEASE_STATE}")" = draft ]; then
        echo true
      else
        echo false
      fi
    elif [[ " $* " = *" --json assets "* ]]; then
      find "${INFERLAB_TEST_RELEASE_DIR}" -mindepth 1 -maxdepth 1 -type f \
        -printf '%f\n' | sort
    else
      fail "unsupported release view"
    fi
    ;;
  upload)
    shift
    while [ "$#" -gt 0 ] && [ "$1" != --repo ] && [ "$1" != --clobber ]; do
      cp "$1" "${INFERLAB_TEST_RELEASE_DIR}/"
      shift
    done
    ;;
  download)
    destination=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --dir ]; then
        destination="$2"
        shift 2
      else
        shift
      fi
    done
    test -n "${destination}" || fail "download destination is required"
    cp "${INFERLAB_TEST_RELEASE_DIR}"/* "${destination}/"
    ;;
  edit)
    [[ " $* " = *" --draft=false "* ]] || fail "edit must publish the draft"
    printf 'published\n' > "${INFERLAB_TEST_RELEASE_STATE}"
    ;;
  *)
    fail "unsupported operation: ${operation}"
    ;;
esac
