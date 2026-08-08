# Specialized Engines

InferLab exposes one reusable `specialized-engine` integration for Engines that
implement a closed token-worker contract. A hardware × architecture × model
Engine remains in its downstream workspace; adding one does not create another
InferLab integration package.

The first supported workflow is deliberately small:

1. InferLab resolves one `single` Engine replica as one rank process owning its
   pure-TP device set, plus one zero-device Gateway process.
2. It starts the Engine with the canonical command below.
3. TokenSpeed SMG owns the OpenAI-compatible HTTP API, tokenizer, chat
   template, detokenizer, and response formatting.
4. The Engine receives prompt token IDs and returns generated token IDs.
5. An InferLab recipe sends a deterministic public-route request, records the
   response and both process logs, and cleans up Gateway then Engine.

## Engine contract

Every implementation provides this command on the selected stack environment's
`PATH`:

```text
inferlab-token-engine smg-worker \
  --listen <host:port> \
  --model <model-locator> \
  --served-model-name <public-name> \
  --tensor-parallel-size <N> \
  --default-max-output-tokens <count> \
  --max-num-batched-tokens <count>
```

It also accepts these memory and prefix-cache options. InferLab passes each one
only when the serve role declares it, so an omitted setting leaves the Engine's
own default in force rather than restating it:

```text
  --gpu-memory-utilization-percent <1-100>      # default 100
  --workspace-reserve-mib <count>               # default 0
  --prefix-cache-gpu-entries <count>            # per rank, default 8
  --prefix-cache-host-memory-percent <1-100>    # default 75
  --prefix-cache-cpu-bytes-per-rank <bytes>     # repeated once per rank
  --prefix-cache-numa-node-per-rank <node>      # repeated once per rank
```

The two per-rank options pair by occurrence order, so the first occurrence of
each describes rank 0. They are supplied together or not at all, each appears
exactly `tensor-parallel-size` times, and their presence replaces
`--prefix-cache-host-memory-percent` as the host-cache sizing authority. A
workspace declares them as one list of rank entries so the two argument lists
cannot drift apart:

```toml
[servers.engine.roles.serve.settings]
gpu_memory_utilization_percent = 90
prefix_cache_gpu_entries = 16
prefix_cache_ranks = [
  { cpu_bytes = 100, numa_node = 3 },
  { cpu_bytes = 200, numa_node = 4 },
]
```

The listener serves the published TokenSpeed scheduler gRPC protocol and the
standard gRPC health service used by TokenSpeed SMG during worker registration.
Request execution requires tokenized input. Prompt text is transport metadata
and must not enter the model core; the model core returns token IDs without
tokenizing or detokenizing them.

`max_num_batched_tokens` is the maximum aggregate token work admitted to one
model iteration. A concrete Engine uses it to materialize request-shaped
execution workspaces before reporting healthy and retains its own scheduling
and admission authority.

The 0.2 contract is one replica and one rank process with an arbitrary nonzero
pure tensor-parallel width. That process owns all `N` allocated devices.
Attention tensor parallelism, expert tensor parallelism, and dense-expert
tensor parallelism equal the outer width; pipeline, data, context, and expert
parallelism remain one. The contract otherwise remains serial greedy generation
with no P/D Router, KV transfer, Engine-local profiling endpoint, log
probabilities, multimodal input, or request batching. Unsupported sampling and
request fields are rejected rather than silently reinterpreted. A worker may
emit SMG stream chunks only after generation completes, so this version makes
no low-latency online-streaming claim.

Workload-attached Nsight Systems profiling is supported without adding an HTTP
surface to the Engine. InferLab wraps the Engine rank process as the capture
target, so the trace covers the Engine process tree, while it opens and closes
the framework range through TokenSpeed SMG's Gateway
`POST /start_profile` and `POST /stop_profile` actions. The resolved plan and
record preserve both the logical Gateway binding and the Gateway process's
effective endpoint. Managed capture traces CUDA and NVTX by default. Operators
who need OS runtime data can opt in through the typed
`profiler.nsys.trace` override.

## Ownership and identity

The common `inferlab-integration-specialized-engine` package owns only planning,
validation, and rendering for this stable contract. It contains no Grout,
model, GPU-architecture, or kernel branches. Its Gateway implementation is
`tokenspeed-smg`, and its recorded version comes from the `tokenspeed-smg`
distribution in the downstream Pixi environment.

The downstream workspace owns the concrete Engine source revision, Cargo lock,
CUDA and compiler closure, model intent, and private local bindings. InferLab
records the source snapshot, locked environment, rendered commands, effective
model locator and placement, process outcomes, and cleanup. The generic
integration identity therefore describes the contract; the workspace source
evidence identifies its concrete implementation.

Grout Qwen3-4B on SM120 is the first real baseline. Grout supplies
`inferlab-token-engine`; there is intentionally no
`inferlab-integration-grout` package. The retained initial-contract record
qualifies its exact TP1 source baseline only. The `0.2.0` package candidate and
wider single-process TP widths remain unqualified until an exact downstream
route produces a real record.

## Failure and cleanup

An Engine bind or model-load failure prevents TokenSpeed SMG from registering
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
