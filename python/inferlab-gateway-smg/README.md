# inferlab-gateway-smg

The SMG frontend lowering that InferLab integrations share when they run SMG
as their Gateway, or as a fused Gateway and P/D Router: the backend name `smg`,
the implementation identity and installed version, the `smg launch` command for
routed-single and prefill/decode targets, the Gateway readiness probe, and the
P/D Router worker-registry readiness.

Integrations keep what their own qualification decides: worker commands,
routing policies, and the public endpoint capabilities they declare. The SMG
Gateway itself comes from the workspace environment's `tokenspeed-smg`
distribution, whose version this package records.
