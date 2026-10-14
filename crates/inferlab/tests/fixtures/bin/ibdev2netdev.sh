#!/bin/sh
# Fake `ibdev2netdev` for every test harness, mirroring Mellanox OFED rows;
# FAKE_NETWORK_MODE=link-local-only reports no RDMA device (see ip.sh).
if [ "${FAKE_NETWORK_MODE:-link-local}" != link-local-only ]; then
  printf 'mlx5_0 port 1 ==> ens-rdma (Up)\n'
fi
