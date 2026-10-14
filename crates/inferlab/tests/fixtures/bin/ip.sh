#!/bin/sh
# Fake `ip` for every test harness. Answers `ip route get` and `ip -o -4 addr`
# with the fields InferLab reads, in iproute2 6.1 order; real rows also carry
# fields such as `via`, `uid`, `brd`, `scope`, and lifetimes, which InferLab
# ignores. FAKE_NETWORK_MODE selects the host:
#   link-local (default)  default route over a link-local NIC, plus an RDMA NIC
#   link-local-only       only the link-local NIC
#   rdma                  default route and only address on the RDMA NIC
mode="${FAKE_NETWORK_MODE:-link-local}"
if [ "$1" = route ] && [ "$2" = get ]; then
  if [ "$mode" = rdma ]; then
    printf '8.8.8.8 dev ens-rdma src 192.0.2.10\n'
  else
    printf '8.8.8.8 dev enx-link-local src 169.254.3.1\n'
  fi
  exit 0
fi
if [ "$1" = -o ] && [ "$2" = -4 ] && [ "$3" = addr ]; then
  if [ "$mode" = rdma ]; then
    printf '1: ens-rdma inet 192.0.2.10/24\n'
    exit 0
  fi
  printf '1: enx-link-local inet 169.254.3.1/24\n'
  if [ "$mode" != link-local-only ]; then
    printf '2: ens-rdma inet 192.0.2.10/24\n'
  fi
  exit 0
fi
printf 'unexpected ip fixture arguments: %s\n' "$*" >&2
exit 2
