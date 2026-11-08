# Eval And Bench Operations

This reference covers toolchain preparation, execution, runtime phases, and
evidence inspection. For definition syntax (datasets, sessions, prompts,
metrics, SLOs), read [Eval authoring](eval-authoring.md) or
[Bench authoring](bench-authoring.md).

## Prepare The Measurement Runtime

Install the release-owned measurement runtime before the first lm-eval or
Bench execution:

```sh
inferlab toolchain install
```

The runtime is fixed by the InferLab release and each version installs its
own, so rerun the install after upgrading. Installation fetches the pinned
measurement packages, including a commit-pinned AIPerf build from GitHub, so
the install host needs GitHub access. Serving workspaces do not declare or
install the measurement SDK. Dry-run and execution never install the runtime;
when a selected lm-eval or Bench measurement needs a missing one, they fail
before the first effect and name `inferlab toolchain install`.

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

Both entry points consume the same resolved Bench definition. Every Eval
and every Bench case first checks that each server process is alive, and checks
again when it ends; a process found exited fails it, naming the process and the
`inferlab serve stop` command. A static matrix then skips its remaining cases,
and a recipe skips its remaining workloads and cleans up. Nothing watches an
idle server; `inferlab serve status` shows each process. `bench
--detach` runs the Bench as a job that outlives the session; see
[Detached runs](serving-and-recipes.md#detached-runs). Invocation-scoped
changes use repeatable `--set PATH=VALUE`; see
[Invocation patches](execution-authoring.md#invocation-patches) for the owned
paths and restrictions.

## Runtime Phases

lm-eval execution resolves the selected task and tokenizer before sending
requests. Serving Bench execution freezes its request population before
traffic, preserving the same seeded population basis across cases.

A cold or primed cache start prepares the cache between warmup and profiling;
[Bench authoring](bench-authoring.md#prefix-cache-start) owns the ordering,
conditioning, evidence, and rejection detail.

Independent request populations and dependent linear sessions use separate
native phase identities. A session keeps each conversation live across its
inter-turn delays; one failed turn terminates that session rather than becoming
an unrelated request.

AgentX trace replay delegates source-tree materialization, snapshot warmup,
branch scheduling, and scenario validity to the release-pinned AIPerf runtime;
its completed and failed counts are transport-request counts. Its case timeout
must cover trace materialization as well as warmup and profiling, so the
ordinary timeout examples are too short; see
[Bench authoring](bench-authoring.md#semianalysis-agentx-trace-replay).

Read [Profiling](profiling.md) before attaching capture to a recipe workload or
manual Bench.

## Evidence Checks

[Evidence and diagnosis](evidence-and-diagnosis.md) owns record reading and
comparison. For a Bench, also check the frozen request population, prefix
schedule, warmup/profiling phase identity, and, for a cold or primed start,
`cache_preparation` (reset attempts, absent for a namespaced generated
population, and per-rank conditioning).

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
