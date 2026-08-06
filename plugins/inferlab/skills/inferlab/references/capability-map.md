# InferLab Capability Map

Use this index to select an owning reference and to check that an operator task
has not fallen outside the supported product. Framework features absent from
this map are not implicitly exposed by InferLab.

## Public CLI

| Command | Purpose | Owning reference |
| --- | --- | --- |
| `tui` | View operations, records, declarations, metrics, logs, and journal state without mutation | [Evidence and diagnosis](evidence-and-diagnosis.md) |
| `workspace show`, `workspace lock` | Inspect the merged public catalog and produce its Pixi lock | [Workspaces and stacks](workspaces-and-stacks.md) |
| `stack status` | Check Pixi confirmation and declared realization checks | [Workspaces and stacks](workspaces-and-stacks.md) |
| `toolchain install` | Install the release-owned lm-eval and AIPerf runtimes | [Measurements](measurements.md) |
| `serve start`, `status`, `logs`, `stop` | Own a long-running named server lifecycle | [Serving and recipes](serving-and-recipes.md) |
| `recipe run` | Run serve, selected Eval/Bench suite, and cleanup as one closed loop | [Serving and recipes](serving-and-recipes.md) |
| `bench` | Run one named Bench against an explicit managed server | [Measurements](measurements.md) |
| `run` | Execute an unrecorded probe in a stack or image realization | [Images and ad-hoc execution](images-and-run.md) |
| `image build` | Assemble, inspect, optionally export, and validate a named runtime image | [Images and ad-hoc execution](images-and-run.md) |
| `scratchpad note`, `scratchpad show` | Maintain the append-only operator narrative | [Evidence and diagnosis](evidence-and-diagnosis.md) |
| `agent install`, `update`, `uninstall`, `doctor` | Manage the bundled agent plugin | [Agent plugin](agent-plugin.md) |
| `license` | Print the retained product license | [Agent plugin](agent-plugin.md) |

The global parser accepts `--workspace` throughout the command tree, but only
workspace-scoped commands read it; `agent` and `license` do not discover a
workspace. Stateful selection commands expose only their typed options; do not
invent generic config, endpoint, or engine-argument switches.

## Workspace Authoring Authority

| Surface | Covered facts | Owning reference |
| --- | --- | --- |
| `models`, `stacks`, `servers`, cases, `workload_suites`, `recipes` | Public definitions, topology, placement-independent server behavior, readiness, and composition | [Workspace definitions and placement](workspace-definition.md) |
| local `model_weights`, `machines`, `placements`, `adapter` | Private locators, launch targets, devices, ports, caches, deadlines, ranks, and replicas | [Workspace definitions and placement](workspace-definition.md) |
| profiling, `images`, `external_images`, local `builders`, machine `container` | Profiler settings, runtime-image inputs, builders, container grants, and invocation patches | [Execution authoring](execution-authoring.md) |
| `evals`, `benches` | Eval, Bench, load, sources, sessions, prompt authority, metrics, and SLOs | [Measurement authoring](measurement-authoring.md) |

The [workspace-authoring index](workspace-authoring.md) routes every authoring
task to one of these three authorities. Operational references describe how to
run and diagnose the resolved definitions; they do not define a second schema.

## Measurement Coverage

The [measurement-authoring reference](measurement-authoring.md) owns exact
definition semantics. Its supported areas are lm-eval and smoke workloads,
static and adaptive serving load, deterministic synthetic and pinned dataset
sources, dependent sessions, prompt authority and prefix geometry, normalized
metrics, server exports, SLOs, and the closed SemiAnalysis AgentX trace-replay
profiles. AgentX uses AIPerf's release-pinned tree scheduler; it does not add a
general InferLab DAG runtime.

Prefix geometry describes the frozen request population. Cache-read metrics
describe observed server behavior; neither substitutes for the other.

## Cross-Cutting Workflows

- Dry-run and typed overrides: [Serving and recipes](serving-and-recipes.md) and
  [Execution authoring](execution-authoring.md).
- Local, SSH, multi-rank, multi-replica, and prefill/decode placement:
  [Workspace definitions and placement](workspace-definition.md).
- Workload-attached capture definitions: [Execution authoring](execution-authoring.md).
  Capture execution and evidence: [Profiling](profiling.md).
- Runtime-image and container definitions: [Execution authoring](execution-authoring.md).
  Build, selection, and probes: [Images and ad-hoc execution](images-and-run.md).
- Source identity, effective-value records, cleanup, TUI observations, metric
  comparisons, scratchpad links, stable error codes, and privacy:
  [Evidence and diagnosis](evidence-and-diagnosis.md).

Consult the bundled backend support matrix before claiming a topology, endpoint,
profiling path, parallelism mode, or hardware/model combination is qualified.
“Supported” and “Qualified” are not interchangeable.
