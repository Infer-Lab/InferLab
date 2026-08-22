# Eval And Bench Operations

For Eval, Bench, dataset, session, prompt, metric, and SLO definition syntax,
read [Eval authoring](eval-authoring.md) or [Bench authoring](bench-authoring.md).
This reference covers
toolchain preparation, execution, runtime phases, and evidence inspection.

## Prepare The Measurement Runtime

Install the release-owned measurement runtime before the first lm-eval or
Bench execution:

```sh
inferlab toolchain install
```

The installed runtime is fixed by the InferLab product release. Serving
workspaces do not declare or install its internal measurement SDK.

## Run Eval And Bench Workloads

Use a recipe when the server lifecycle, ordered workload suite, gate, and
cleanup belong to one recorded experiment:

```sh
inferlab recipe run <RECIPE> --dry-run
inferlab recipe run <RECIPE>
```

Use a manual Bench only with an explicit managed server record:

```sh
inferlab bench <BENCH> --serve <SERVER_RECORD_ID> --dry-run
inferlab bench <BENCH> --serve <SERVER_RECORD_ID>
```

Both entry points consume the same resolved Bench definition. Invocation-scoped
changes use repeatable `--set PATH=VALUE`; read
[Invocation patches](execution-authoring.md#invocation-patches) for the owned
paths and restrictions.

The built-in smoke workload is the smallest completion-path correctness check.
lm-eval execution resolves the selected task and tokenizer before sending
requests. Serving Bench execution freezes its request population before
traffic, preserving the same seeded population basis across cases.

## Runtime Phases

Cache start defaults to uncontrolled. For a cold or primed start, native
warmup drains first, then InferLab resets the cache; primed additionally sends
the frozen maximum canonical prefix before profiling release. Under attention
data parallelism the conditioning fans out one `X-Data-Parallel-Rank`-pinned
request per prefill replica and rank — through `POST /prime_prefix_cache` on
the built-in vLLM Mooncake, vLLM NIXL, and SGLang prefill/decode proxies — and
preserves per-(replica, rank) evidence; any rank's failure fails the case. A
primed or prefix-geometry Bench against an endpoint without declared backend
cache-read capability fails at planning with the enable-reporting remediation,
and router-fronted pairs without primed capability reject a primed start at
planning. Population preparation, warmup, reset, and conditioning remain
outside normalized profiling counts and metrics. A default captured Bench opens
the framework window only after these preparation actions succeed.

Independent request populations and dependent linear sessions use separate
native phase identities. A session keeps each conversation live across its
inter-turn delays; one failed turn terminates that session rather than becoming
an unrelated request.

AgentX trace replay delegates source-tree materialization, snapshot warmup,
branch scheduling, and scenario validity to the release-pinned AIPerf runtime.
Its declared concurrency is root-tree lanes, while completed and failed counts
remain transport-request counts. Its 600-second warmup and at least 900 seconds
of profiling make the ordinary timeout examples too short; budget source
configuration, warmup grace, profiling, and result handling explicitly.

Adaptive Bench records every measured rate and selects the highest observed
feasible rate under its bounded search policy. It does not claim an unmeasured
optimum.

Read [Profiling](profiling.md) before attaching capture to a recipe workload or
manual Bench.

## Evidence Checks

The record freezes the resolved route, prompt authority, tokenizer identity,
template provenance, request population, prefix schedule, warmup/profiling
phase identity, request-body fragment, native commands, artifacts, and
normalized metrics.

For AgentX, also inspect source expected/observed revision and digest, the
native scenario verdict and invalidity reasons, warmup and profiling raw
records, source/runtime request coordinates, cache-bust markers, complete
`branch_stats`, the aggregate artifact, and the explicit unavailable scheduler
dimensions. Branch counters are not task-success or root-tree-throughput
metrics. `benchmark_lib.sh` is qualification evidence only and is never run,
parsed, or copied by InferLab.

Inspect backend-observed prompt-token evidence separately from configured
prompt geometry. Prefix sharing describes the request population; cache-read
metrics describe observed server behavior.

Read [Evidence and diagnosis](evidence-and-diagnosis.md) before comparing runs.
