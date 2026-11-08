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
- InferLab owns placement, process lifecycle, readiness, cache resets,
  an optional KV tier, Bench evidence, profiling, records, and cleanup.

## A served Engine

1. InferLab resolves the server into Engine replicas, each one rank process
   owning its devices — `serve` replicas, or `prefill` and `decode` replicas
   for a prefill/decode server; and one zero-device Gateway process, which
   also routes prefill and decode.
2. It starts every Engine with the stack's `engine_command`, the role's Engine
   arguments, and the contract arguments below, then the Gateway.
3. The server is ready when the Gateway lists every Engine healthy with its
   data-parallel rank count and, with a KV tier, every Engine reports the tier
   engaged with its access and publication bound.
4. Workloads use the Gateway's public routes; the Engine sees only token IDs.
5. Cleanup stops the processes in reverse order, and the record keeps every
   rendered command, process outcome, and cleanup result.

## What an Engine implements

| The Engine provides | When | InferLab capability |
| --- | --- | --- |
| An executable accepting the contract arguments, the gRPC worker service, and gRPC health | Always | Serving, readiness, routing |
| `data_parallel_size` in `GetServerInfo` `server_args` | Always | Cache-owning targets and records |
| Cached-token counts in every completion | Always | Observed prefix reuse |
| `FlushCache` that clears every prefix-cache entry the Engine could read | Always | Cold Bench starts and resets between cases |
| Speculative-token counts in every completion | When it decodes speculatively | Draft acceptance metrics |
| `StartProfile` and `StopProfile` | When it is to be profiled | Workload-attached profiling |
| `--kv-tier-backend`, `--kv-tier-store`, and the tier report in `GetServerInfo` | When it uses a KV tier | A tier shared by every Engine |
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
force either way. A path or an entry naming a contract argument fails
planning.

InferLab renders only what it allocates. Each Engine runs as
`engine_command`, then the role's Engine arguments, then these contract
arguments:

```text
example-token-engine serve <extra_args …> \
  --listen <host:port> \
  --model <model-locator> \
  --served-model-name <public-name>
```

The Engine's devices arrive only through `CUDA_VISIBLE_DEVICES`, with
`CUDA_DEVICE_ORDER=PCI_BUS_ID` so the ordinals are the `nvidia-smi` indices
InferLab allocated. A
prefill/decode Engine and an Engine given a KV tier receive the further
contract arguments described in [Prefill and decode](#prefill-and-decode) and
[KV tier](#kv-tier). InferLab renders no layout, memory, token-budget, or
prefix-cache option: those are the Engine's own.

### Engine arguments

Every option that is not a contract argument is an Engine argument, spelled the
way the Engine spells it, in the role's `extra_args` and `extra_env` settings.
They carry the layout, memory and admission limits, context length,
speculative decoding, tier tuning, and transfer timeouts:

```toml
[servers.engine.roles.serve.settings]
extra_args = ["--data-parallel-size", "8", "--max-num-batched-tokens", "16384"]
extra_env = { EXAMPLE_ENGINE_FLAG = "1" }
```

Engine arguments precede the contract arguments in the rendered command.
Tokens after a bare `--` sentinel are rendered after the contract arguments;
the sentinel itself is never rendered. An entry naming a contract argument is
rejected while planning, before or after the sentinel, and `extra_env` may not
set `CUDA_VISIBLE_DEVICES` or `CUDA_DEVICE_ORDER`. The effective contents are returned in the plan's
effective settings and recorded with the rendered command.

The integration takes no typed Engine settings. Declaring a retired one —
`default_max_output_tokens`, `max_num_batched_tokens`,
`gpu_memory_utilization_percent`, `workspace_reserve_mib`, the
`prefix_cache_*` settings, the `kv_tier_*` tier settings, or the stack setting
`kv_tier_publication_seconds` — fails planning with a diagnostic naming its
replacement: the Engine's own flag in `extra_args`, or, for the publication
bound, the Engine's report.

### Worker service

The listener serves the gRPC service
`tokenspeed.grpc.scheduler.TokenSpeedScheduler` as defined by the SMG Gateway
distribution in the stack, plus the standard gRPC health service, which reports
`SERVING` once every rank can admit requests; SMG uses both during worker
registration ([RFC-0003:C-SPECIALIZED-ENGINE](rfc/RFC-0003.md)). An Engine
implements `Generate` (server streaming), `HealthCheck`, `Abort`,
`GetModelInfo`, `GetServerInfo`, `GetLoads`, and `FlushCache`. An Engine that
is to be profiled also implements `StartProfile` and `StopProfile`, which the
Gateway's `POST /start_profile` and `POST /stop_profile` actions invoke. Other
RPCs — `GetTokenizer` and `SubscribeKvEvents` among them — may answer
`UNIMPLEMENTED`; a Gateway feature built on one is then unavailable for that
Engine. InferLab defines no protocol version of its own: the SMG distribution
owns the wire contract (the `smg-grpc-proto` Python package and the
`smg-grpc-client` Rust crate publish it), field additions stay compatible, and
the Gateway's worker registration is the compatibility check. The recorded
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
profiling endpoint of its own. An Engine may emit stream chunks only after
generation completes, so the contract makes no low-latency online-streaming
claim.

## Layout and devices

Each replica is one rank process that owns all of its allocated devices and
serves every data-parallel rank behind its one listen address. A role declares
its device count as `devices` and no parallelism component; a declared
component, or a role without `devices`, fails planning. The layout within the
replica is the Engine's: it takes its devices from `CUDA_VISIBLE_DEVICES` and
any further layout from its Engine arguments. For example, an 8-device data x
expert parallel Engine declares:

```toml
[servers.engine]
devices = 8

[servers.engine.roles.serve.settings]
extra_args = ["--data-parallel-size", "8", "--expert-parallel-size", "8"]
```

An Engine that does not support a layout, or whose layout disagrees with its
device count, exits before it reports `SERVING`, naming the option.

At readiness InferLab reads each Engine's data-parallel rank count from the
Gateway's `/workers` registry, which relays the `data_parallel_size` the Engine
reports in `GetServerInfo` `server_args`. Readiness waits until every Engine's
count is a positive integer, the record keeps each count, and Bench derives
the server's cache-owning targets from them.

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
planning. The server becomes ready once every Engine process runs and the
Gateway's `/workers` registry lists every replica as healthy; SMG's own
`/readiness` turns ready after the first healthy worker, so it is not the
signal.

Other SMG options go in `extra_args` of the Gateway settings, or of the P/D
Router settings on a prefill/decode server. They follow every option InferLab
renders and are recorded verbatim:

```toml
[servers.engine.gateway_settings]
policy = "cache_aware"
extra_args = ["--worker-startup-check-interval", "1"]
```

An entry naming an option InferLab renders — listen address and port,
Prometheus port, worker-startup timeout, model and tokenizer paths, worker
URLs, prefill/decode wiring, policies, retries, or the circuit breaker — or a
bare `--` fails planning.

## Replicas across machines

A `serve` replica can span machines when its layout needs more devices than one
machine has. The portable server stays the same; the split is a placement fact,
written in `.inferlab/local.toml` as an explicit rank list, one entry per
machine. A machine pool never splits a replica on its own.

```toml
[servers.engine]
devices = 16

[placements.two-hosts.roles.serve]
ranks = [
  { machine = "host-a", devices = [0, 1, 2, 3, 4, 5, 6, 7] },
  { machine = "host-b", devices = [0, 1, 2, 3, 4, 5, 6, 7] },
]
```

The server's `devices` counts the whole replica across every host: 16 for two
8-device machines, not 8. Each rank runs one Engine on its machine, with that machine's devices as its
`CUDA_VISIBLE_DEVICES` and that machine's model locator. Rank 0 is host 0, the
replica's only endpoint:

```text
host 0 : … --listen <host:port> --model <locator> --served-model-name <name> \
           --host-count 2 --rendezvous-listen <host:port>
host 1 : … --model <locator> --served-model-name <name> \
           --host-count 2 --host-index 1 --rendezvous <host 0 rendezvous>
```

With a KV tier, every host also receives the tier arguments for its own machine.
The rendezvous port comes from host 0's port list and must be reachable from
every other host. Members get no `--listen`; the endpoint port the placement
reserves for them stays unused.

Host 0 speaks for the whole replica. It reports `SERVING` only after every
member has joined, its `data_parallel_size` counts the ranks of every host, its
`FlushCache` clears every host, and with a tier its tier report covers every
host. Each host's Engine exits before `SERVING` when its layout's world differs
from its visible devices times `--host-count`. The Gateway, readiness, and every
observation address host 0; the other hosts are ready while their processes
run.

Hosts with different device counts, a `prefill` or `decode` replica across
machines, and a profiled multi-host server are rejected before launch. Cleanup
stops the members first and host 0 last. Device leases or other per-host launch
policy belong in an `engine_command` wrapper on the stack environment's `PATH`.

## KV tier

A server may declare a KV tier with one of two backends. A `mooncake` tier is
a Mooncake Store master that InferLab runs as its own zero-device process
beside the Gateway: it starts before every Engine and stops after them, and
each Engine receives its address:

```toml
[servers.engine.kv_store]
backend = "mooncake"
```

```text
  --kv-tier-backend mooncake
  --kv-tier-store <host:port>
```

A `filesystem` tier is one directory on a file system that every machine of
the server mounts.

```toml
[servers.engine.kv_store]
backend = "filesystem"
```

Each machine binds its directory in `.inferlab/local.toml`, as it binds model
weights:

```toml
[machines.gpu-node]
kv_tier_directory = "/path/to/shared-fs/kv-tier"
```

Every Engine on the machine receives the directory unchanged:

```text
  --kv-tier-backend filesystem
  --kv-tier-store <kv_tier_directory>
```

InferLab runs no store process for the tier and does not create, inspect,
maintain, or clean the directory; the file system, its mounts, and any
reclaimer are infrastructure. The directory is not reserved for one server:
other servers, workspaces, or teams may share it, and keeping it undisturbed
during a measurement is the responsibility of whoever runs the measurement. A
server whose machine binds no directory fails resolution, naming the machine.
Tier tuning, such as a capacity, snapshot interval, or timeout, is an Engine
argument for either backend. An Engine that implements only one backend
rejects the other before it reports `SERVING`, naming the option.

An Engine given a tier may serve before it reaches the tier and keeps serving
if it loses it, so readiness alone does not show the tier engaged. Its
`GetServerInfo` `scheduler_info` therefore reports:

| Key | Meaning |
| --- | --- |
| `kv_tier_state` | `available` once it has completed a tier operation, `unavailable` before that and whenever the tier is lost |
| `kv_tier_store` | the location it uses: the store address or the directory |
| `kv_tier_access` | `read_write` when it reads matching prefixes from the tier and writes the prefixes it computes, `write` when it only writes |
| `kv_tier_publication_ms` | how long after a request completes a prefix it wrote may take to become readable, a non-negative integer |

The server becomes ready only when every Engine is `available` at the location
rendered for it and has reported its access and publication bound, and every
Engine of one role reports the same access. Every Bench case records each
Engine's report at its profiling window's start and end, and a case fails as
having lost its tier if any Engine is not engaged at either boundary.

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
from each request. A `decode` Engine must also admit a request that finishes on the
prefill Engine's first token: a primed Bench warms the cache with a single
output token. An Engine that does not support a mode or the declared
mechanism exits before it reports `SERVING`, naming the option, so the
recorded mechanism is one the Engine accepted. `kv_transfer` is the Engine's own
name for its transport — `mooncake`, or a transport the Engine implements
itself — written in lowercase letters, digits, `-`, and `_`. The integration
passes any name through unchanged, so a misspelled name is caught only when
the Engine refuses it after launch; check the rendered
`--disaggregation-transfer-backend` with `--dry-run` first. Transfer and bootstrap
timeouts are the Engine's own options and go in `extra_args`.

The server is ready when the Gateway's `/workers` registry lists every
`prefill` replica as a healthy prefill worker with its bootstrap port and every
`decode` replica as a healthy decode worker. Prefill and decode replica counts
are independent. A prefill/decode server cannot be profiled yet, and routing
policies go in `pd_router_settings`, not `gateway_settings`.

A prefill/decode server may declare a KV tier. Both roles then receive the
tier arguments, and each reports its own access: typically a `prefill` Engine
reads and writes the tier, and a `decode` Engine writes the blocks it generates
so a later request can reuse them. KV still moves from prefill to decode over
the declared transfer mechanism; the tier is not the transfer path. Readiness
and every profiling window probe both roles. When every `prefill` Engine
reports `read_write`, the prefill Engines share one prefix cache through the
tier, so a primed start sends one conditioning request.

## Cache control

The Gateway resets prefix caches through `POST /flush_cache`, which calls
`FlushCache` on every registered Engine, prefill and decode alike; the reset
succeeds only when SMG reports every Engine flushed. With a KV tier,
`FlushCache` must also clear every tier entry the Engine could read, so a cold
start is cold.

A primed start on a shared tier waits the largest publication bound any Engine
reported after its conditioning request and before profiling, and the workload
record keeps the wait as a preparation step.

The Gateway has no prefix-cache conditioning fan-out, so a primed Bench start
conditions through ordinary requests. That primes the server only when it has
one cache-owning target: either a single replica — on a prefill/decode
server, a single `prefill` replica — whose Engine reports one data-parallel
rank, or Engines that share one prefix cache through a KV tier, which take one
conditioning request for the whole server whatever the replica and
data-parallel counts. Because the rank counts and tier access are reported at
readiness, a primed start on any other layout fails after the server is ready
and before any reset or conditioning request, naming the observed counts.

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
in `engine_command`. The downstream workspace owns the Engine's source
revision, build lock, compiler and CUDA closure, model intent, and private
local bindings. InferLab records the source snapshot, locked environment,
rendered commands, effective model locator and placement, process outcomes, and
cleanup. The generic integration identity therefore describes the contract; the
workspace source evidence identifies the concrete Engine. The backend support
matrix tracks the present qualification state.

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

Model paths, tier directories, device selection, ports, and machine identities
belong only in the ignored `.inferlab/local.toml`. Shareable workspace
definitions name the generic integration and the concrete Engine source path
without embedding those local facts.