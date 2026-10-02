# Profiling, runtime images, and invocation patches

## Workload profiling

Profiling is prepared on a server and requested on a selected Eval or Bench; it
is not a separate workload kind. Enable capture-target preparation on the
server, a server case, or an invocation patch. The three capture deadlines are
optional; omitted ones resolve to built-in defaults that dry-run renders, so
declare them only to size a particular capture:

```toml
[servers.example]
profiling = true
capture_arm_deadline_seconds = 120
capture_control_deadline_seconds = 120
capture_finalization_deadline_seconds = 1800
```

Select the capture mechanism on the server, a case, or an invocation patch
(`--set server.profiler.mechanism=engine_trace`); omission resolves to
managed collection:

```toml
[servers.example.profiler]
mechanism = "engine_trace"
```

Managed collection wraps each captured rank process tree with Nsight Systems.
Engine trace instead lets the framework's own profiler write per-rank traces
into a record-owned directory that InferLab assigns at planning. Integrations
that cannot honor `engine_trace` reject it while planning; the
[backend support matrix](backend-support.md)
lists which do. [Profiling](profiling.md#capture-lifecycle) owns the capture
lifecycle, placement limits, and coverage rules.

An engine-trace server declares no `profiler.nsys` escape inputs: combining
`mechanism = "engine_trace"` with nsys escapes in one server or role fails
workspace load, and a combination assembled across layers, such as a case or
`--set` selecting `engine_trace` for a server with escapes, fails resolution. A server
case may declare only `profiler.mechanism`; nsys escapes belong to the server
and its roles.

Declaring `profiler.mechanism` or nsys escape inputs on a server whose
profiling resolves off (no `profiling = true` and no requested `--capture`)
fails resolution with a typed error naming the declaration and the
enable-profiling remediation; profiler declarations never silently drop.

`capture_arm_deadline_seconds` is one budget for preparing and arming every
selected rank target. `capture_control_deadline_seconds` bounds each framework
range action that opens a window, and each managed-collection close.
`capture_finalization_deadline_seconds` is one budget for session inspection,
any required collection stop, asynchronous report completion, report coverage,
and every engine-trace window close. The undeclared finalization default
follows the resolved mechanism and is larger for engine trace; dry-run renders
the effective value. [Profiling](profiling.md#capture-lifecycle) describes how
these budgets combine during a capture.

A server may replace the dedicated Nsight Systems fields or add launch/start
options and environment; role declarations merge after the common layer:

```toml
[servers.example.profiler.nsys]
executable = "/opt/nsight-systems/bin/nsys"
trace = ["cuda", "nvtx", "osrt"]
launch_options = []
start_options = []
sampling = "none"
context_switch = "none"

[servers.example.profiler.nsys.env]
PATH = "/opt/nsight-systems/bin:/usr/bin"
```

`executable`, `trace`, `sampling`, and `context_switch` replace their common
values. Role option lists append after server lists, and role environment
entries win by key. The effective environment applies to launch, collection
start, session inspection, and collection stop; only the launch invocation
passes it onward to the wrapped server. InferLab rejects options that attempt
to replace managed session, report, range, export, overwrite, or launch-wait
facts.

Use OS runtime tracing when the experiment needs it; disabling `osrt` is not a
general cure for startup or finalization timing. Select it explicitly in
`trace` and size the lifecycle deadlines for its additional capture cost.

Request a capture as described in [Profiling](profiling.md#run-a-capture).

## Runtime images and ad-hoc execution

A runtime image definition selects one stack, a base image that InferLab
resolves to an immutable per-platform digest, one or more platforms, an
optional subset of the stack's `source_paths`, and recipe-referenced
validations:

```toml
[images.vllm-runtime]
stack = "vllm"
base_image = "example.com/micromamba:1.0"
platforms = ["linux/amd64"]
packages = ["upstream/vllm"]

[[images.vllm-runtime.validations]]
recipe = "smoke"
server_case = "tp1"
```

Omitting `packages` selects every stack source path. Each selected path builds
one wheel for every project the stack's Pixi environment installs from inside
it, at the directory the Pixi manifest declares. A submodule whose Python
project lives below its root (for example SGLang's `python/`) is selected by
its root path, while the manifest points the dependency at the project
directory. A selected path the environment installs nothing from fails image
resolution, so when a stack also lists source paths consumed only while
building another package, declare `packages` as the subset the environment
installs. Every locked registry package must carry a hash: a locked PyPI
package without one fails image resolution naming the package, because the
image could not pin it; take such packages from conda instead and relock. A
validation names only a recipe and optional server case; it does not restate
model, placement, server, or measurement facts. Builds require a clean
workspace. Local bindings currently expose one builder kind:

```toml
[builders.local]
kind = "local-docker"
```

[Images and ad-hoc execution](images-and-run.md#build-workflow) covers the
build workflow.

Portable contexts and image metadata exclude model locators, builder hosts,
workspace paths, placements, and other machine-private facts. Per-machine
container bindings apply to every server container on that machine, including
image validations, but not to `inferlab run`. They may pass environment values
by name, grant absolute device paths, lift the memlock limit, and add only
`IPC_LOCK`, `SYS_NICE`, or `SYS_PTRACE`; InferLab never requests privileged
mode. `pass_env` may not name variables InferLab manages in the container
(`HOME`, `USER`, `LOGNAME`, `CUDA_VISIBLE_DEVICES`, `CONDA_PREFIX`):

```toml
[machines.local.container]
pass_env = ["HF_TOKEN"]
devices = ["/dev/infiniband"]
memlock_unlimited = true
capabilities = ["IPC_LOCK", "SYS_NICE"]
```

A workspace may also declare a digest-pinned image it did not build:

```toml
[external_images.official]
reference = "example.com/vllm@sha256:<64-hex-digest>"
integration = "vllm"
```

The `integration` claim must equal the integration of every server stack the
image is selected for. InferLab lowers an external image's commands with the
workspace's own integration packages, so the workspace must commit and install
a framework-free Pixi environment named `adapter` that contains
`inferlab-adapter-sdk` and `inferlab-integration-<integration>`.

By default InferLab replaces an external image's entrypoint with the rendered
command, because an entrypoint may itself be a fixed serving command. When the
image's own entrypoint is the supported way to run commands in it, for example
an entrypoint that prepares the image's CUDA or NCCL runtime before executing
its arguments, declare `entrypoint = "image"`: serving processes and
`inferlab run` commands then keep that entrypoint and receive the command as
its arguments. The recorded server command shows which path ran.

Variables that settings or the integration set for a containerized process,
such as `extra_env`, reach the container verbatim and apply only inside it;
the host-side container client keeps its launch machine's own environment.

[Built and external image selection](images-and-run.md#built-and-external-image-selection)
and [ad-hoc probes](images-and-run.md#ad-hoc-probes) cover how a launch or an
`inferlab run` command selects an image.

## Invocation patches

Use repeatable `--set` for temporary typed changes. Values use TOML syntax and
later assignments win.

```sh
inferlab serve start example \
  --set server.readiness_timeout_seconds=1800 \
  --set server.settings.max_model_len=32768 \
  --set server.roles.serve.parallelism.outer.tensor_parallel_size=4 \
  --dry-run

inferlab recipe run qualify \
  --set evals.gsm8k.limit=100 \
  --set evals.gsm8k.trials=5 \
  --set evals.gsm8k.concurrency=8 \
  --set 'benches.random-8k1k.concurrency=[1, 8]' \
  --dry-run
```

Recipe measurement patches may name only Eval and Bench definitions selected
by that recipe's workload suite. They cannot change identities, kinds, suite
membership, the gate, or the selected server.

The `evals.gsm8k.trials` patch above repeats one lm-eval task; read
[Repeated trials](eval-authoring.md#repeated-trials) for which tasks accept it
and how trials are seeded.
