#!/bin/sh
# Fake `getent` for every test harness. Answers `getent ahosts NAME` in the
# row layout of glibc 2.39 (address, socket type, and on the first row the
# canonical name; one row per socket type), and exits 2 for an unknown name
# as getent does. Addresses are documentation ranges (RFC 5737, RFC 3849).
#   *.example        one IPv4 address per name
#   loopback.example 127.0.1.1, like a hostname mapped in /etc/hosts
#   dual.example     an IPv6 address first, then an IPv4 address
if [ "$1" != ahosts ] || [ "$#" -ne 2 ]; then
  printf 'unexpected getent fixture arguments: %s\n' "$*" >&2
  exit 1
fi
rows() {
  printf '%-15s STREAM %s\n%-15s DGRAM  \n%-15s RAW    \n' "$1" "$2" "$1" "$1"
}
case "$2" in
  node-a.example) rows 192.0.2.11 "$2" ;;
  node-b.example) rows 192.0.2.12 "$2" ;;
  prefill-a.example) rows 192.0.2.13 "$2" ;;
  prefill-b.example) rows 192.0.2.14 "$2" ;;
  decode.example) rows 192.0.2.15 "$2" ;;
  expired.example) rows 192.0.2.16 "$2" ;;
  loopback.example) rows 127.0.1.1 "$2" ;;
  dual.example) rows 2001:db8::11 "$2"; rows 192.0.2.17 "$2" ;;
  *) exit 2 ;;
esac
