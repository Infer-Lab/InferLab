# Profiling, runtime images, and invocation patches

## Workload profiling

Profiling is prepared on a server and requested on a selected Eval or Bench; it
is not a separate workload kind. Enable capture-target preparation on the
server, a server case, or an invocation patch:

```toml
[servers.example]
profiling = true
capture_arm_deadline_seconds = 60
capture_control_deadline_seconds = 60
capture_finalization_deadline_seconds = 300
```

`capture_arm_deadline_seconds` is one budget for preparing and arming every
selected rank target. `capture_control_deadline_seconds` covers the complete
HTTP response for each framework range action.
`capture_finalization_deadline_seconds` is one budget for session inspection,
any required collection stop, asynchronous report completion, and report
coverage across all targets. Capture-armed readiness is unbounded overall but
retains `readiness_attempt_timeout_seconds` on every blocking attempt, so
process exit and operator interruption remain observable.

Managed Nsight Systems defaults use the `nsys` executable and the `cuda,nvtx`
trace set. A server may replace the dedicated fields or add launch/start
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

Request capture with repeatable `recipe run --capture <WORKLOAD_ID>` or with
`bench --capture` against a server started with profiling enabled. A positive
AIPerf warmup drains before the framework capture window opens. The window
closes at client completion, and complete report coverage may establish a
successful capture even when a framework stop action failed. Image-backed
server launches reject profiling because InferLab has no in-container profiler
contract.

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

Omitting `packages` selects every stack source path. A validation names only a
recipe and optional server case; it does not restate model, placement, server,
or measurement facts. Builds require a clean workspace. Local bindings
currently expose one builder kind:

```toml
[builders.local]
kind = "local-docker"
```

`inferlab image build <IMAGE>` resolves, assembles, inspects, optionally
exports unique OCI archives with `--export <DIR>`, and runs every eligible
validation as one recorded closed loop. `--builder`, `--placement`, `--local`,
and `--dry-run` retain their owning selection semantics. Built images remain in
local builder storage and are never pushed by this workflow.

Portable contexts and image metadata exclude model locators, builder hosts,
workspace paths, placements, and other machine-private facts. Per-machine
container bindings may pass environment values by name, grant absolute device
paths, lift the memlock limit, and add only `IPC_LOCK`, `SYS_NICE`, or
`SYS_PTRACE`; InferLab never requests privileged mode:

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

Select a successful build record with `--image` or the declared artifact with
`--external-image`, never both. External images are probed on every launch
machine and are not pulled automatically. Use `inferlab run` for unrecorded
stack or image probes; container mode exposes no mount or device implicitly,
so declare repeatable `--mount PATH[:rw]` and `--devices INDEX[,INDEX...]` as
needed. Never invoke a binary directly through `.pixi/envs/<env>/bin/`, which
would bypass the activation used by product launches.

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

An lm-eval definition may set `trials` for repeated evaluation of one resolved
single-sample `generate_until` task. Its default is `1`. The definition seed
is the repeated base seed; when it is absent, repeated evaluation uses `1234`.
Trial `i` uses `base_seed + i - 1`. The existing `concurrency` field controls
those requests, and `request_body.seed` is rejected because the definition owns
the seed schedule. Each trial repeats the complete resolved Eval; InferLab does
not rewrite task-owned response multiplicity, filters, or scorer behavior.
