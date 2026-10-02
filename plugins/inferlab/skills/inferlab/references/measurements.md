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

The installed runtime is fixed by the InferLab product release, and each
InferLab version installs its own, so rerun the install after upgrading.
Installation fetches the pinned measurement packages, including a
commit-pinned AIPerf build from GitHub, so the install host needs network
access to GitHub. Serving workspaces do not declare or install its internal
measurement SDK.

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

lm-eval execution resolves the selected task and tokenizer before sending
requests. Serving Bench execution freezes its request population before
traffic, preserving the same seeded population basis across cases.

## Runtime Phases

A cold or primed cache start prepares the cache between warmup and profiling;
[Bench authoring](bench-authoring.md#serving-bench-warmup-and-metrics) owns the
ordering, fan-out, evidence, and rejection detail.

Independent request populations and dependent linear sessions use separate
native phase identities. A session keeps each conversation live across its
inter-turn delays; one failed turn terminates that session rather than becoming
an unrelated request.

AgentX trace replay delegates source-tree materialization, snapshot warmup,
branch scheduling, and scenario validity to the release-pinned AIPerf runtime;
its completed and failed counts are transport-request counts. Trace
materialization at client start, the cache-pressure warmup, the default or
declared profiling duration, and result handling all consume the case timeout,
so the ordinary timeout examples are too short;
[Bench authoring](bench-authoring.md#semianalysis-agentx-trace-replay) owns the
AgentX semantics.

Adaptive Bench records every measured rate and selects the highest observed
feasible rate under its bounded search policy. It does not claim an unmeasured
optimum.

Read [Profiling](profiling.md) before attaching capture to a recipe workload or
manual Bench.

## Evidence Checks

[Evidence and diagnosis](evidence-and-diagnosis.md) owns record reading and
comparison. For a Bench, also check the frozen request population, prefix
schedule, and warmup/profiling phase identity.

For AgentX, also inspect source expected/observed revision and digest, the
native scenario verdict and invalidity reasons, `context_overflow_count`,
`ordinary_failure_count`, `warmup_error_records`, complete `branch_stats`, the
aggregate artifact, and the explicit unavailable scheduler dimensions. Raw
records, source/runtime request coordinates, and cache-bust markers exist only
at `artifact_level = "diagnostic"`; at `performance` they are recorded as
unavailable due to the artifact level. Branch counters are not task-success or
root-tree-throughput metrics. `benchmark_lib.sh` is qualification evidence only
and is never run, parsed, or copied by InferLab.

Inspect backend-observed prompt-token evidence separately from configured
prompt geometry. Prefix sharing describes the request population; cache-read
metrics describe observed server behavior.
