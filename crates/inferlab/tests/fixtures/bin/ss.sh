#!/bin/sh
# Fake `ss` for every test harness. Mirrors `ss -Hltnp` (iproute2 6.1): one
# LISTEN row per port named in FIXTURE_LISTENING_PORTS, so resolution observes
# no host sockets unless a test declares them: the suite's leases already own
# its ports, and the host's other sockets are not test state. Like the real
# tool, a row carries `users:(("name",pid=N,fd=M))` only for processes the
# caller may inspect; FIXTURE_LISTENING_PROCESS="name,pid" supplies one.
if [ "$*" != -Hltnp ]; then
  printf 'unexpected ss fixture arguments: %s\n' "$*" >&2
  exit 2
fi
users=
if [ -n "${FIXTURE_LISTENING_PROCESS:-}" ]; then
  users=" users:((\"${FIXTURE_LISTENING_PROCESS%,*}\",pid=${FIXTURE_LISTENING_PROCESS#*,},fd=5))"
fi
for port in ${FIXTURE_LISTENING_PORTS:-}; do
  printf 'LISTEN 0      4096                                       0.0.0.0:%s 0.0.0.0:*%s\n' "$port" "$users"
done
