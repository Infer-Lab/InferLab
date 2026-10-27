# Specialized Engines

A Specialized Engine is an inference engine built for a particular hardware,
architecture, or model outside the general-purpose frameworks. Every such
Engine that implements the token-worker contract below runs through one
reusable `specialized-engine` integration. Adding an Engine adds no InferLab
integration package; the Engine and its source stay in its own repository and
downstream workspace.

The contract splits the work. The Engine executes the model on token IDs: it
receives prompt token IDs and returns generated token IDs, and owns scheduling,
batching, sampling, and its kernels. InferLab and the SMG Gateway supply the
rest of the serving system:

- the SMG Gateway owns the OpenAI-compatible HTTP API, tokenizer, chat template,
  detokenizer, response formatting, and routing across replicas;
- InferLab owns placement, process lifecycle, readiness, cache resets, an
  optional KV store, Bench evidence, profiling, records, and cleanup.

## A served Engine

1. InferLab resolves the server into Engine replicas, each one rank process
   owning its devices — `serve` replicas, or `prefill` and `decode` replicas
   for a prefill/decode server; one zero-device Gateway process, which also
   routes prefill and decode; and, when the server declares a KV store, one
   zero-device store process.
2. It starts the store, then every Engine with the stack's `engine_command`
   and the contract arguments below, then the Gateway.
3. The server is ready when the Gateway lists every Engine healthy and, with a
   store, every Engine reports the store engaged.
4. Workloads use the Gateway's public routes; the Engine sees only token IDs.
5. Cleanup stops the processes in reverse order, and the record keeps every
   rendered command, process outcome, and cleanup result.

## What an Engine implements

| The Engine provides | When | InferLab capability |
| --- | --- | --- |
| An executable accepting the contract arguments, the gRPC worker service, and gRPC health | Always | Serving, readiness, routing |
| Layout arguments for data, context, and expert parallelism | For a layout that uses them | Multi-device layouts in one process |
| Cached-token counts in every completion | Always | Observed prefix reuse |
| `FlushCache` that clears every prefix-cache entry the Engine could read | Always | Cold Bench starts and resets between cases |
| Speculative-token counts in every completion | When it decodes speculatively | Draft acceptance metrics |
| `StartProfile` and `StopProfile` | When it is to be profiled | Workload-attached profiling |
| The `--kv-tier-*` options and the tier state in `GetServerInfo` | When it uses a KV store | A KV store shared by every Engine |
| The `--disaggregation-*` options and the bootstrap rendezvous | When it serves prefill/decode | A prefill/decode server |

### Command

Each implementation ships its executable under its own name. The stack that
selects the implementation names it in the stack setting `engine_command`: a
bare executable name on the stack environment's `PATH`, then any fixed
arguments the implementation needs, such as a subcommand. Every server and role
using the stack shares it.

```toml
[stacks.engine]
integration = "specialized-engine"
pixi_environment = "serving"

[stacks.engine.settings]
engine_command = ["example-token-engine", "serve"]
```

A stack that declares no `engine_command` gets
`["inferlab-token-engine", "smg-worker"]`, and the plan records the value in
force either way. A path or an entry naming a contract-owned option fails
planning. InferLab renders `engine_command` followed by the contract arguments:

```text
example-token-engine serve \
  --listen <host:port> \
  --model <model-locator> \
  --served-model-name <public-name> \
  --tensor-parallel-size <attention-TP>
```

A layout with data, context, or expert parallelism adds these arguments, each
only when its value is not 1, so a pure tensor-parallel Engine never sees them:

```text
  --data-parallel-size <DP>
  --context-parallel-size <CP>
  --expert-parallel-size <EP>
```

It also accepts these token-budget, memory, and prefix-cache options. InferLab
passes each one only when the serve role declares it, so an omitted setting
leaves the Engine's own default in force rather than restating it:

```text
  --default-max-output-tokens <count>
  --max-num-batched-tokens <count>
  --gpu-memory-utilization-percent <1-100>
  --workspace-reserve-mib <count>
  --prefix-cache-gpu-entries <count>            # per rank
  --prefix-cache-host-memory-percent <1-100>
  --prefix-cache-cpu-bytes-per-rank <bytes>     # repeated once per rank
  --prefix-cache-numa-node-per-rank <node>      # repeated once per rank
```

`max_num_batched_tokens` is the maximum aggregate token work admitted to one
model iteration. An Engine uses it to materialize request-shaped execution
workspaces before reporting healthy and keeps its own scheduling and admission
authority.

The two per-rank options pair by occurrence order, so the first occurrence of
each describes rank 0. They are supplied together or not at all, each appears
exactly once per device. They and
`--prefix-cache-host-memory-percent` are alternative host-cache sizing
authorities: declaring both `prefix_cache_ranks` and
`prefix_cache_host_memory_percent` is rejected while planning. A workspace
declares the per-rank options as one list of rank entries, one per device of the
Engine's world, so the two argument lists cannot drift apart:

```toml
[servers.engine.roles.serve.settings]
gpu_memory_utilization_percent = 90
prefix_cache_gpu_entries = 16
prefix_cache_ranks = [
  { cpu_bytes = 100, numa_node = 3 },
  { cpu_bytes = 200, numa_node = 4 },
]
```

### Implementation-specific options

An Engine implementation may define additional command-line and environment
knobs of its own. The serve role carries them through the `extra_args` and
`extra_env` escape-hatch settings, which exist for engine bring-up and
implementation-specific experiments rather than for contract behavior:

```toml
[servers.engine.roles.serve.settings]
extra_args = ["--example-engine-knob", "4"]
extra_env = { EXAMPLE_ENGINE_FLAG = "1" }
```

Extra arguments precede the managed arguments in the rendered command, so the
managed values win ordinary last-wins parsing. Entries naming an InferLab-owned
option — the contract command flags and every flag rendered from the typed
settings above — are rejected while planning. Tokens after a bare `--`
sentinel pass through verbatim after the managed arguments as a deliberate
override; the sentinel itself is never rendered. The exact effective hatch
contents are returned in the plan's effective settings and recorded with the
rendered command, so a run exercised through the hatch demonstrates that exact
configuration rather than the typed contract alone.

### Worker service

The listener serves the gRPC service `tokenspeed.grpc.scheduler.TokenSpeedScheduler`
as defined by the SMG Gateway distribution in the stack, plus the standard gRPC
health service, which reports `SERVING` once every rank can admit requests;
SMG uses both during worker registration
([RFC-0003:C-SPECIALIZED-ENGINE](rfc/RFC-0003.md)). An Engine implements
`Generate` (server streaming), `HealthCheck`, `Abort`, `GetModelInfo`,
`GetServerInfo`, `GetLoads`, and `FlushCache`. An Engine that is to be
profiled also implements `StartProfile` and `StopProfile`, which the Gateway's
`POST /start_profile` and `POST /stop_profile` actions invoke. Other RPCs —
`GetTokenizer` and `SubscribeKvEvents` among them — may answer
`UNIMPLEMENTED`; a Gateway feature built on one is then unavailable for that
Engine. InferLab defines no protocol version of its own: the SMG
distribution owns the wire contract (the `smg-grpc-proto` Python package and
the `smg-grpc-client` Rust crate publish it), field additions stay compatible,
and the Gateway's worker registration is the compatibility check. The recorded
worker protocol is that service name.

Request execution requires tokenized input. Prompt text is transport metadata
and must not enter the model core; the model core returns token IDs without
tokenizing or detokenizing them. The Engine may batch concurrent requests and
implement any sampling field the protocol carries; unsupported sampling and
request fields are rejected rather than silently reinterpreted.

A completion carries the prompt, completion, and cached token counts, with
zero cached tokens reported explicitly. An Engine that decodes speculatively
reports each completion's accepted and drafted speculative-token counts in
`spec_accepted_tokens` and `spec_draft_tokens`, which the Gateway reads and
Bench reports as draft acceptance.

The contract has no multimodal input, and the Engine needs no HTTP or
profiling endpoint of its own. An Engine may emit
stream chunks only after generation completes, so the contract makes no
low-latency online-streaming claim.

## Layout and devices

Each replica is one rank process that owns all of its allocated devices and
serves every data-parallel rank behind its one listen address. The device
count is `devices`, or `outer.tensor_parallel_size` when `devices` is not
declared. Attention tensor x data x context parallelism equals the device
count, expert parallelism divides it, expert tensor parallelism is the device
count divided by expert parallelism, and dense-expert tensor parallelism equals
attention tensor parallelism. There is no pipeline or expert data parallelism.
For example, an 8-device data x expert parallel Engine declares:

```toml
[servers.engine]
devices = 8

[servers.engine.parallelism.attention]
data_parallel_size = 8

[servers.engine.parallelism.experts]
expert_parallel_size = 8
```

and receives `--tensor-parallel-size 1 --data-parallel-size 8
--expert-parallel-size 8`. InferLab checks only this arithmetic; which layouts
an Engine supports is its own rule. An Engine that does not support a layout
exits before it reports `SERVING`, naming the option.

## Replicas and routing

A server may run several Engine replicas behind its one Gateway by declaring
`replicas` on the `serve` role. Each replica owns its own devices, and the
Gateway routes across every replica's listen address with the policy declared
in the typed Gateway settings, `least_load` by default:

```toml
[servers.engine.roles.serve]
replicas = 2

[servers.engine.gateway_settings]
policy = "cache_aware"
```

The accepted policies are `least_load`, `power_of_two`, `round_robin`,
`random`, `cache_aware`, `prefix_hash`, and `consistent_hashing`. SMG runs
`power_of_two` only over two or more workers, so that policy over one replica
— or one prefill or decode replica for the P/D Router policies below — fails
planning. The server
becomes ready once every Engine process runs and the Gateway's `/workers`
registry lists every replica as healthy; SMG's own `/readiness` turns ready
after the first healthy worker, so it is not the signal.

## KV store

When the server declares a `mooncake` KV store, each Engine also receives
`--kv-tier-store <host:port>`, the store address, and the contract's tier
settings when declared:

```text
  --kv-tier-capacity <bytes>           # kv_tier_capacity_bytes
  --kv-tier-snapshot-interval <count>  # kv_tier_snapshot_interval
  --kv-tier-timeout <seconds>          # kv_tier_timeout_seconds
```

Tier settings without a store fail planning. An Engine given a store may serve
before it reaches the store and keeps serving if it loses it, so readiness alone
does not show the tier engaged. Its `GetServerInfo` `scheduler_info` therefore
reports `kv_tier_state` (`available` once it has completed a store operation,
`unavailable` otherwise) and `kv_tier_store`, the address it uses. InferLab
reads both from every Engine through the integration: the server becomes ready
only when every Engine is `available` at the rendered store address, every
Bench case records each Engine's state at its profiling window's start and end,
and a case fails as having lost its store if any Engine is not engaged at
either boundary.

## Prefill and decode

A `prefill_decode` server splits each request between a `prefill` Engine and
a `decode` Engine behind one SMG process that is both Gateway and P/D Router.
It declares both frontend backends and the KV-transfer mechanism, and its
routing policies are typed P/D Router settings:

```toml
[servers.engine-pd]
stack = "engine"
model = "example-moe"
topology = "prefill_decode"
gateway_backend = "smg"
pd_router_backend = "smg"
kv_transfer = "mooncake"

[servers.engine-pd.roles.prefill]
devices = 4
[servers.engine-pd.roles.decode]
devices = 4

[servers.engine-pd.pd_router_settings]
prefill_policy = "cache_aware"   # default round_robin
decode_policy = "least_load"     # default round_robin
```

Each Engine receives, after its usual arguments:

```text
  --disaggregation-mode prefill|decode
  --disaggregation-transfer-backend <kv_transfer>
  --disaggregation-bootstrap-port <port>     # prefill only
```

The bootstrap port is allocated by InferLab from the machine's port list. A
`prefill` Engine accepts its `decode` counterparts on that port at its listen
host before it reports `SERVING`; a `decode` Engine learns its counterpart
from each request. An Engine that does not support a mode or the declared
mechanism exits before it reports `SERVING`, naming the option, so the
recorded mechanism is one the Engine accepted. Transfer and bootstrap
timeouts are the Engine's own options and go in `extra_args`.

The server is ready when the Gateway's `/workers` registry lists every
`prefill` replica as a healthy prefill worker with its bootstrap port and every
`decode` replica as a healthy decode worker. Prefill and decode replica counts
are independent. A prefill/decode server cannot be profiled yet, and routing
policies go in `pd_router_settings`, not `gateway_settings`.

A prefill/decode server may declare a KV store. Both roles then receive
`--kv-tier-store` and their declared tier settings: a `prefill` Engine reads
and writes the tier, and a `decode` Engine writes the blocks it generates so a
later request can reuse them. KV still moves from prefill to decode over the
declared transfer mechanism; the store is not the transfer path. Readiness and
every profiling window probe both roles, and the prefill Engines share one
prefix cache through the tier, so a primed start sends one conditioning
request.

## Cache control

The Gateway resets prefix caches through `POST /flush_cache`, which calls
`FlushCache` on every registered Engine, prefill and decode alike; the reset succeeds only when SMG
reports every Engine flushed. With a KV store, `FlushCache` must also clear
every tier entry the Engine could read, so a cold start is cold.

An Engine may make a prefix it wrote readable from the tier only some bounded
time after the request completes. It declares that bound in the stack
setting `kv_tier_publication_seconds` (default 0); a primed start on the
shared tier waits it after its conditioning request and before profiling, and
the workload record keeps the wait as a preparation step:

```toml
[stacks.engine.settings]
kv_tier_publication_seconds = 1.0
```

The Gateway has no prefix-cache conditioning fan-out, so a primed Bench start
conditions through ordinary requests. That primes the server only when it has
one cache-owning target: either a single replica — on a prefill/decode
server, a single `prefill` replica — with one data-parallel rank, or Engines that share one prefix cache through a KV store, which take one
conditioning request for the whole server whatever the replica and
data-parallel counts. A primed start on any other layout is rejected at
planning.

## Profiling

Workload-attached Nsight Systems profiling is supported for a single replica
without adding an HTTP surface to the Engine; a server with several replicas
cannot be profiled yet, because each capture target would open its window
through the one Gateway. InferLab wraps the Engine rank process as the capture
target, so the trace covers the Engine process tree, while it opens and closes
the framework range through the Gateway's `POST /start_profile` and
`POST /stop_profile` actions. The resolved plan and record preserve both the
logical Gateway binding and the Gateway process's effective endpoint. Managed
capture traces CUDA and NVTX by default. Operators who need OS runtime data can
opt in through the typed `profiler.nsys.trace` override.

## Ownership and identity

The common `inferlab-integration-specialized-engine` package owns only planning,
validation, and rendering for this contract. It contains no Engine-, model-,
GPU-architecture-, or kernel-specific branches. Its Gateway is SMG, lowered by
the shared `inferlab-gateway-smg` package that the TokenSpeed integration also
uses: the recorded implementation is `smg`, and its version comes from the
`tokenspeed-smg` distribution in the downstream Pixi environment.

Each Engine implementation supplies its own executable, which its stack names
in `engine_command`. The downstream workspace owns the Engine's source revision, build lock, compiler
and CUDA closure, model intent, and private local bindings. InferLab records
the source snapshot, locked environment, rendered commands, effective model
locator and placement, process outcomes, and cleanup. The generic integration
identity therefore describes the contract; the workspace source evidence
identifies the concrete Engine. The backend support matrix tracks the present
qualification state.

## Failure and cleanup

An Engine bind or model-load failure prevents the Gateway from registering
the worker. A protocol or health mismatch keeps the Gateway unready. The
rendered SMG command defers its internal worker-registration timeout so it does
not race InferLab's operator-configured `readiness_timeout_seconds`; that
framework-neutral server field is the sole managed startup deadline. An
invalid request returns a typed gRPC failure that SMG maps to its public
response. In a managed run, InferLab retains lifecycle authority and attempts
reverse-order cleanup after launch, readiness, measurement, interruption, or
cancellation failure as well as after success.

Model paths, device selection, ports, and machine identities belong only in the
ignored `.inferlab/local.toml`. Shareable workspace definitions name the
generic integration and the concrete Engine source path without embedding
those local facts.
