# inferlab-gateway-dynamo

The Dynamo lowering that InferLab integrations share when they run Dynamo as
their Gateway, or as a fused Gateway and P/D Router: the backend name `dynamo`,
the implementation identity and installed version, the `dynamo.frontend`
command, the etcd discovery process, the worker discovery and listener
environment, and the three readiness layers — discovery health, worker
generation-endpoint health, and frontend registry membership.

Integrations keep what is a fact of their framework: the Dynamo worker module,
its engine arguments, and the disaggregation and KV-transfer spelling. The
Dynamo frontend comes from the workspace environment's `ai-dynamo`
distribution, whose version this package records; etcd comes from the same
environment.
