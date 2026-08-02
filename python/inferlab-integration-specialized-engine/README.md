# Inferlab Integration for Specialized Engines

This package connects Inferlab to a hardware-by-model Specialized Engine that
implements the canonical `inferlab-token-engine smg-worker` command. The Engine
accepts prompt token IDs and returns generated token IDs over SMG's worker
protocol. TokenSpeed SMG remains responsible for the public HTTP API,
tokenization, chat templates, detokenization, and response formatting.

The integration contains no model-, architecture-, hardware-, or
Engine-implementation-specific lowering. A downstream workspace supplies the
Rust Engine binary, SMG, model intent, source revision, locked environment, and
private placement bindings. Consequently, a new conforming Engine does not
need another Inferlab integration package.

The supported shape is deliberately closed: one `single` Engine replica in one
process behind one SMG Gateway. The process owns an arbitrary nonzero pure-TP
device set; attention, expert, and dense-expert tensor parallelism all equal
the outer TP width, while pipeline, data, context, and expert parallelism stay
at one. The contract remains serial and has no P/D Router, KV-transfer,
batching, or Engine-local profiling surface. InferLab can profile the Engine
process tree while TokenSpeed SMG exposes the capture-window
`POST /start_profile` and `POST /stop_profile` actions on the Gateway.

The Gateway exposes server metrics on its separately allocated `prometheus`
port. The integration selects SMG's single-target `least_load` policy so its
canonical worker monitor polls Engine load fields while retaining the same sole
routing target. In the current downstream validation, that endpoint exported
SMG-owned metric families but no `smg_engine_*` families, so Engine-series
re-export remains a downstream qualification gap.
