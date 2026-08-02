#!/bin/sh
set -eu
operation="$1"
shift
if [ "$operation" = launch ]; then
  # Escape options splice ahead of the managed tail; tests declare them in
  # =-form, so any leading option token is skippable and --session-new is
  # the only separate-value option.
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --session-new) shift 2 ;;
      -*) shift ;;
      *) exec "$@" ;;
    esac
  done
elif [ "$operation" = start ]; then
  output=
  count=1
  session=
  for argument in "$@"; do
    case "$argument" in
      --output=*) output="${argument#--output=}" ;;
      --capture-range-end=repeat:*) count="${argument#--capture-range-end=repeat:}"; count="${count%%:*}" ;;
      --session=*) session="${argument#--session=}" ;;
    esac
  done
  mkdir -p "$(dirname "$output")"
  printf '%s\t%s\t0\t%s' "$output" "$count" "$session" > "$FIXTURE_NSYS_STATE"
elif [ "$operation" = sessions ]; then
  subcommand="$1"
  shift
  [ "$subcommand" = list ]
  if [ ! -f "$FIXTURE_NSYS_STATE" ]; then
    exit 0
  fi
  old_ifs="$IFS"
  IFS="$(printf '\t')"
  read -r output count index session < "$FIXTURE_NSYS_STATE" || true
  IFS="$old_ifs"
  state=StartRange
  if [ "$index" -ge "$count" ]; then
    state=Launched
  fi
  printf '[{"id":"fixture","duration":"0:00:01","state":"%s","launch":"1","name":"%s","accessible":true}]\n' "$state" "$session"
elif [ "$operation" = stop ]; then
  old_ifs="$IFS"
  IFS="$(printf '\t')"
  read -r output count index session < "$FIXTURE_NSYS_STATE" || true
  IFS="$old_ifs"
  if [ "$index" -ge "$count" ]; then
    printf 'Collection stop is not allowed in this state.\n' >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s' "$output" "$count" "$count" "$session" > "$FIXTURE_NSYS_STATE"
  printf 'nsys_stop\n' >> "$FIXTURE_CAPTURE_EVENTS"
else
  exit 2
fi
