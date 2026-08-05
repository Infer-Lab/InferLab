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

## Workspace And Local Authorities

| Surface | Covered facts | Owning reference |
| --- | --- | --- |
| `models`, `stacks` | Served names, integration, Pixi environment, source paths, checks, image postprocessing | [Workspaces and stacks](workspaces-and-stacks.md) |
| `servers`, server cases | Topology, frontend backends, KV transfer, roles, replicas, parallelism, settings, readiness, profiling | [Serving and recipes](serving-and-recipes.md), [Profiling](profiling.md) |
| `evals`, `benches` | Smoke and lm-eval, static and adaptive serving Bench, load, sources, metrics, SLOs | [Measurements](measurements.md) |
| `workload_suites`, `recipes` | Ordered measurement selection, gate, and server composition | [Serving and recipes](serving-and-recipes.md) |
| `images`, `external_images` | Generated runtime images and digest-pinned images InferLab did not build | [Images and ad-hoc execution](images-and-run.md) |
| local `model_weights`, `machines`, `placements` | Private locators, launch targets, devices, ports, caches, rank and replica placement | [Workspaces and stacks](workspaces-and-stacks.md) |
| local `builders`, `adapter`, machine `container` | Local Docker builder, adapter deadlines/device workaround, container environment and hardware grants | [Images and ad-hoc execution](images-and-run.md) |

## Measurement Coverage

The [measurement reference](measurements.md) covers all supported combinations:

- built-in smoke and lm-eval tasks from a pinned name, release-bundled task, or
  workspace YAML; request bodies, few-shot, limits, trials, seeds, concurrency,
  metrics, filters, thresholds, and task-directed completion/chat routes;
- static concurrency and request-rate cases, unbounded and stochastic load,
  request-count or duration bounds, warmup, prefix-cache reset, server metrics,
  aggregate SLOs, request SLOs, goodput, and adaptive rate search;
- deterministic synthetic fixed or inclusive-uniform ISL/OSL, weighted shape
  mixtures, ShareGPT independent requests, first-turn SPEED-Bench profiles, and
  dependent linear ShareGPT sessions with inter-turn delay controls;
- synthetic `flat`, `rendered_chat`, and `server_chat` prompt authority; custom
  local templates and kwargs; exact token- or ratio-based final-prompt prefix
  geometry including zero and full sharing; and server-chat-only pre-template
  shared system content; and
- normalized throughput, prompt-token, latency, TTFT, TPOT, cache-read,
  good-request, goodput, and SPEED acceptance evidence together with the native
  AIPerf artifacts.

`prefix_sharing` declares prompt geometry. It does not declare cache state or a
cache-hit percentage. Full sharing (`1.0` or all input tokens) is supported as
geometry, but a decode-only claim still requires observed cache-read evidence.
Controlled cache starts and required per-request cache-read observations are
not part of the execution surface carried by this plugin version.

## Cross-Cutting Workflows

- Dry-run and typed overrides: [Serving and recipes](serving-and-recipes.md) and
  [Measurements](measurements.md).
- Local, SSH, multi-rank, multi-replica, and prefill/decode placement:
  [Workspaces and stacks](workspaces-and-stacks.md) and
  [Serving and recipes](serving-and-recipes.md).
- Workload-attached Nsight Systems capture, profiler escapes, warmup boundary,
  framework range control, finalization, and report verification:
  [Profiling](profiling.md).
- Built-image and declared external-image launches, container grants, OCI
  export, and unrecorded probes: [Images and ad-hoc execution](images-and-run.md).
- Source identity, effective-value records, cleanup, TUI observations, metric
  comparisons, scratchpad links, stable error codes, and privacy:
  [Evidence and diagnosis](evidence-and-diagnosis.md).

Consult the release's backend support matrix before claiming a topology,
endpoint, profiling path, parallelism mode, or hardware/model combination is
qualified. “Supported” and “Qualified” are not interchangeable.
