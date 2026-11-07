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

InferLab renders only what it allocates: the listen address, model, served
model name, prefill/decode role with its transfer mechanism and bootstrap port,
and a `filesystem` KV tier's bound directory, with the Engine's devices as its
`CUDA_VISIBLE_DEVICES`. Layout, memory, and every other Engine option travel
verbatim in the role's `extra_args` and `extra_env`. The Engine reports its
data-parallel rank count and, with a tier, its tier access and publication
bound at readiness. Servers run one or more Engine replicas behind one SMG
Gateway, or prefill and decode Engines behind one fused SMG Gateway and P/D
Router. InferLab can profile a single Engine's process tree while SMG exposes
the capture-window `POST /start_profile` and `POST /stop_profile` actions on the
Gateway.

The Gateway exposes server metrics on its separately allocated `prometheus`
port. The integration selects SMG's single-target `least_load` policy so its
canonical worker monitor polls Engine load fields while retaining the same sole
routing target. In the current downstream validation, that endpoint exported
SMG-owned metric families but no `smg_engine_*` families, so Engine-series
re-export remains a downstream qualification gap.
