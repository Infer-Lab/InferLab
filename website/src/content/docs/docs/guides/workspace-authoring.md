---
title: "Workspace authoring"
description: "Author versioned workspaces and machine-local bindings."
---

An InferLab workspace has two authorities:

- committed `.inferlab/workspace.toml` and `.inferlab/workspace.d/*.toml` files
  describe shareable models, stacks, servers, cases, measurements, and recipes;
- git-ignored `.inferlab/local.toml` binds those definitions to model weights,
  machines, devices, ports, and placement for one operator.

Run `inferlab workspace show` to validate and browse the committed authority.
It does not read local bindings or inspect a stack realization. Use
`inferlab workspace show --json` when another tool needs the canonical merged
definition.

## Upgrading to 0.8

InferLab 0.8 retains adapter protocol version 7. Existing workspaces must
update their exact package pins to `inferlab-adapter-sdk==0.6.1` and version
`0.5.2` of the selected vLLM, SGLang, TensorRT-LLM, or TokenSpeed integration.
A Specialized Engine workspace uses
`inferlab-integration-specialized-engine==0.2.2`. Update the SDK and selected
integration together, then run `inferlab workspace lock` so the committed Pixi
lock becomes the new workspace authority. The product-owned
`inferlab-measurement-sdk` remains internal to the installed measurement
toolchain and must not be added to a serving workspace.

Serving Bench continues to use a Bench definition rather than an
engine-specific benchmark configuration. New definitions may select variable
random shapes, release-catalog SPEED-Bench profiles, or linear sessions. Every
Bench request remains structured chat: an optional `chat_template` or
`chat_template_kwargs` is sent only as an ordinary `request_body` member for
the model server to interpret. InferLab may project that effective template
locally to make a synthetic complete-prompt ISL exact, but never sends the
projection as a completion prompt; an unavailable or unsatisfiable projection
is retained and identified as fallback in the record.

Workspaces upgrading from 0.5 or earlier must replace the former
`routing_backend` field because protocol version 7 does not interpret
protocol-version-6 requests or responses.
A direct `single` server declares neither frontend backend; a routed `single`
declares `gateway_backend`; and a `prefill_decode` server declares both
`gateway_backend` and `pd_router_backend`. The protocol-v7 control plane rejects
the old combined field rather than guessing how to divide its ownership.

## Minimal workspace

The root file owns the schema version. Definitions may live there or in
identifier-disjoint fragments under `workspace.d/`.

```toml
schema_version = 2

[models.example]
served_name = "example"

[stacks.vllm]
integration = "vllm"
pixi_environment = "vllm"
source_paths = []

[servers.example]
stack = "vllm"
model = "example"
topology = "single"
readiness_timeout_seconds = 900

[servers.example.settings]
max_model_len = 8192

[servers.example.cases.tp2.parallelism.outer]
tensor_parallel_size = 2

[evals.smoke]
kind = "openai-smoke"
prompt = "Hello"
max_tokens = 16
timeout_seconds = 60

[workload_suites.smoke]
evals = ["smoke"]
gate = "smoke"

[recipes.smoke]
server = "example"
workload_suite = "smoke"
```

The sole `tp2` case is selected automatically, so this server does not need a
`default_case`. A server with no cases uses its base behavior. A server with
multiple cases must declare `default_case`; the operator may always select a
different one with `--case`.

`readiness_timeout_seconds` owns the complete ordinary server-readiness wait.
Each blocking process-status or HTTP attempt within that wait is capped by
`readiness_attempt_timeout_seconds`, which defaults to 30 seconds. Profiled
servers use separate budgets for preparing and arming all targets, controlling
one framework capture window, and finalizing all reports:

```toml
[servers.example]
readiness_attempt_timeout_seconds = 30
capture_arm_deadline_seconds = 60
capture_control_deadline_seconds = 60
capture_finalization_deadline_seconds = 300
```

These values may be declared on the server, patched by a selected server case,
or overridden for one invocation with paths such as
`--set server.readiness_attempt_timeout_seconds=45`. Capture-armed readiness
remains unbounded overall but retains the bounded attempt deadline so process
exit and operator interruption can be observed. Cleanup grace and polling
cadence are product policy rather than workspace settings; SSH connection and
keepalive policy remain in the selected OpenSSH target configuration.

Framework settings belong under `settings`, either on the server or on a
canonical role. Integrations validate their typed fields. `extra_args` remains
the explicit backend escape hatch and is replaced as one complete array by a
case or invocation patch.

## Prefill/decode servers

A P/D server uses the canonical Engine roles `prefill` and `decode`. It selects
Gateway and P/D Router backends as two independent facts. Gateway owns the
public API boundary; P/D Router owns prefill/decode target selection and phase
orchestration. The currently supported pairs derive one fused, zero-device
`gateway` process for the two components, while InferLab retains placement,
lifecycle, cleanup, and record authority. Resolved evidence does not encode
that current implementation choice as a permanent limit: its closed
`frontend` section owns a `processes` collection, and each component names its
realizing process through `process_id`. A future qualified split pair can
therefore bind Gateway and P/D Router to distinct processes. Do not declare
`gateway`, `pd_router`, or `router` under public `roles`.

```toml
[servers.example-pd]
stack = "vllm"
model = "example"
topology = "prefill_decode"
gateway_backend = "builtin"
pd_router_backend = "builtin"
readiness_timeout_seconds = 1800
default_case = "builtin-nixl"

[servers.example-pd.roles.prefill]
replicas = 2

[servers.example-pd.roles.prefill.parallelism.outer]
tensor_parallel_size = 4

[servers.example-pd.roles.decode]
replicas = 2

[servers.example-pd.roles.decode.parallelism.outer]
tensor_parallel_size = 2

[servers.example-pd.cases.builtin-nixl]
kv_transfer = "nixl"

[servers.example-pd.cases.native-mooncake]
readiness_timeout_seconds = 900
gateway_backend = "vllm-router"
pd_router_backend = "vllm-router"
kv_transfer = "mooncake"

[servers.example-pd.cases.native-mooncake.roles.prefill.settings]
kv_transfer_protocol = "rdma"
```

Common settings apply to every model-serving role. Role settings apply after
the common layer. A selected case may patch common or role settings,
parallelism, replica counts, either frontend-backend identity, transfer,
profiling, and readiness. A P/D server base must declare both backend fields;
cases may replace either value but may not add or remove a component. Each
integration validates the selected Gateway/P/D Router pair, so independently
recorded fields do not imply that every cross-provider pair is supported.
Frontend component presence is fixed by the server base for every topology: a
direct `single` server cannot acquire a Gateway from a case or invocation
patch, while a routed `single` server declares `gateway_backend` on its base
and may replace only that identity later.

## Local bindings

For a single-machine TP2 server, a minimal `.inferlab/local.toml` is:

```toml
default_placement = "local"

[model_weights.example]
locator = "/models/example"

[machines.local]
host = "127.0.0.1"
devices = [0, 1]
ports = [8000]

[placements.local]
machines = ["local"]
```

Published workspaces should provide this shape as
`.inferlab/local.example.toml`; operators copy it to the ignored local file and
replace the generic values.

Adapter invocation deadlines are machine-local because process startup and
container startup costs vary by site. The two paths remain independent:

```toml
[adapter]
timeout_seconds = 30       # process-backed plan/render; default 30
image_timeout_seconds = 120 # image-backed plan/render; default 120
```

Both values must be positive when declared. `image_device` is a separate,
optional workaround for container runtimes that cannot create a device-less
adapter container; it does not affect process-backed lowering.

Use explicit rank placement when replicas span machines, roles use different
device counts, or the same model has different locators on each machine. This
example places two TP4 prefill replicas across pairs of machines, two TP2
decode replicas on individual machines, and a zero-device fused frontend on
the controller. The frontend is placed as `gateway` even though the same process
also realizes P/D Router, and it has no model locator, replica index, or rank:

```toml
default_placement = "cluster"

[model_weights.example.machine_locators]
prefill-a = "/models/example-a"
prefill-b = "/models/example-b"
prefill-c = "/models/example-c"
prefill-d = "/models/example-d"
decode-a = "/models/example-a"
decode-b = "/models/example-b"

[machines.controller]
host = "controller.example"
devices = []
ports = [7000]

[machines.prefill-a]
host = "prefill-a.example"
devices = [0, 1]
ports = [8000, 8001, 8100]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "prefill-a" }

[machines.prefill-b]
host = "prefill-b.example"
devices = [0, 1]
ports = [8000, 8001, 8100]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "prefill-b" }

[machines.prefill-c]
host = "prefill-c.example"
devices = [0, 1]
ports = [8000, 8001, 8100]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "prefill-c" }

[machines.prefill-d]
host = "prefill-d.example"
devices = [0, 1]
ports = [8000, 8001, 8100]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "prefill-d" }

[machines.decode-a]
host = "decode-a.example"
devices = [0, 1]
ports = [8000, 8001]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "decode-a" }

[machines.decode-b]
host = "decode-b.example"
devices = [0, 1]
ports = [8000, 8001]
workspace = "/srv/inferlab/example"
launch = { kind = "ssh", target = "decode-b" }

[[placements.cluster.roles.prefill.replicas]]
ranks = [
  { machine = "prefill-a", devices = [0, 1] },
  { machine = "prefill-b", devices = [0, 1] },
]

[[placements.cluster.roles.prefill.replicas]]
ranks = [
  { machine = "prefill-c", devices = [0, 1] },
  { machine = "prefill-d", devices = [0, 1] },
]

[placements.cluster.roles.decode]
replicas = [
  { machine = "decode-a", devices = [0, 1] },
  { machine = "decode-b", devices = [0, 1] },
]

[placements.cluster.roles.gateway]
machine = "controller"
devices = []
```

For model-serving roles, the role-level `machine` and `devices` form is one
replica at rank 0. Use a `ranks` list only when that one Engine replica spans
two or more machines; use a `replicas` list only when the role has two or more
replicas. Replica and rank numbers are derived from list order. The derived
`gateway` placement is instead one process-only binding and accepts only one
direct `machine`, an empty `devices` list, and an optional `endpoint_port`.

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

## lm-eval tasks and inference requests

An lm-eval definition selects exactly one task. Use a pinned lm-eval task name,
a release-bundled InferLab task, or a workspace-owned task YAML:

```toml
[evals.builtin]
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
metric_filter = "strict-match"
threshold = 0.90
timeout_seconds = 900

[evals.bundled]
kind = "lm-eval"
task = { bundled = "estonia" }
metric = "estonia_pass"
metric_filter = "strict-terminal-answer"
threshold = 0.50
timeout_seconds = 3600

[evals.workspace-task]
kind = "lm-eval"
task = { yaml = "evals/long-context.yaml" }
metric = "exact_match"
threshold = 0.80
timeout_seconds = 3600
```

The task, not a second InferLab dataset layer, owns `dataset_path`,
`dataset_name`, split selection, prompting, output type, filters, and scoring.
Workspace YAML paths must be workspace-relative tracked `.yaml` or `.yml`
files. InferLab resolves their YAML include closure, records the effective task
configuration and dataset selection, and includes that closure in source
identity. Release-bundled tasks are addressed only by their catalog name and
carry a release-owned closure digest.

InferLab uses the resolved model-weight locator as the Hugging Face tokenizer
locator. This follows the normal model-directory convention and avoids a
second tokenizer setting; the locator must contain a usable tokenizer.
`generate_until` tasks use chat completions. Tasks whose
resolved output type is `loglikelihood`, `loglikelihood_rolling`, or
`multiple_choice` use completions and first run a prompt-logprob/tokenizer
alignment probe. Dynamic Python tasks are probed conservatively as well. A
probe failure makes support inconclusive rather than silently removing the
task.

Use `request_body` for task-specific inference parameters such as sampling,
reasoning effort, logprobs, or chat-template arguments:

```toml
[evals.reasoning]
kind = "lm-eval"
task = { yaml = "evals/reasoning.yaml" }
metric = "exact_match"
threshold = 0.80
timeout_seconds = 1800

[evals.reasoning.request_body]
temperature = 1.0
reasoning_effort = "high"
logprobs = true

[evals.reasoning.request_body.chat_template_kwargs]
enable_thinking = true
```

The same nested values may be patched for one run, for example
`--set evals.reasoning.request_body.temperature=0.6`. `request_body` is a JSON
request fragment, not a replacement request: InferLab retains ownership of the
model, prompt or messages, streaming mode, one-completion policy, output bound,
and stop conditions. Eval also owns the repeated-trial seed schedule. Conflicts
with those fields fail during validation and the complete effective fragment is
preserved in dry-run and record evidence.

## Serving Bench warmup and metrics

A concurrency Bench may run a native warmup phase before its profiled phase:

```toml
[benches.random-8k1k]
kind = "serving"
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
concurrency = [1, 8]
prompts_per_concurrency = 4
warmup_prompts_per_concurrency = 2
request_body = { temperature = 1.0 }
timeout_seconds = 900
```

For concurrency `c`, the resolved warmup count is
`c * warmup_prompts_per_concurrency`. Warmup uses the same route, request
source, request body, and concurrency as profiling, but consumes a disjoint
prefix of the frozen request population. It is excluded from normalized
metrics and profiling request counts. A requested prefix-cache reset happens
once before warmup, and the case timeout covers reset, warmup, profiling, and
result handling; process cleanup retains its separate grace. When the Bench is
captured, InferLab opens the framework capture window only after native warmup
has drained and closes it at the existing client-completion boundary. A warmup
failure leaves the capture window unopened.

Every successful Bench reports `request_throughput`, `output_throughput`, and
`total_token_throughput`. For each of `request_latency_ms`, `ttft_ms`, and
`tpot_ms`, InferLab reports `mean`, `min`, `max`, `stddev`, `p50`, `p90`,
`p95`, and `p99` using names such as `p95_tpot_ms`. TPOT is not applicable to
an `output_tokens = 1` prefill-dominant workload and its TPOT metrics are then
omitted. `prompt_cache_read_ratio` is present only when AIPerf reports valid
cache-usage evidence. `good_request_ratio` and `goodput` are derived only when
a request SLO is configured.

Set `server_metrics = true` to ask AIPerf to collect the server's declared JSON
metrics export. This is accepted only when the selected integration/topology
declares a metrics endpoint and the Bench has zero warmup. The integration may
bind that endpoint to the public serving port or to one named port it already
requires; InferLab allocates the port and freezes the exact URL before launching
the measurement client. InferLab preserves framework routes such as `/metrics`
or `/v1/metrics` rather than substituting a framework-neutral default. Direct
SGLang declares the capability only when its server settings include
`enable_metrics = true`. The Specialized Engine integration binds SMG's
`prometheus` port and activates its canonical Engine-load polling. A successful
`speed_bench` case additionally runs AIPerf's pinned SPEED report twice and
publishes the CSV cells as `acceptance_length` and `acceptance_rate`; other
request sources retain the raw server metrics but do not publish those two
SPEED-specific scalars.

## Serving Bench SLOs

A static or adaptive serving Bench may constrain normalized aggregate metrics,
individual request latency, or both. Aggregate constraints are inclusive and
AND-composed. Request SLOs count a request as good only when every configured
latency bound passes, then gate the case with `minimum_good_request_ratio`.

```toml
[benches.saturation]
kind = "adaptive-serving"
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
initial_request_rates = [1.0, 4.0]
aggregate_slos = [
  { metric = "request_throughput", at_least = 1.0 },
  { metric = "p99_ttft_ms", at_most = 800.0 },
]
request_slo = { request_latency_ms = 5000.0, ttft_ms = 800.0, tpot_ms = 30.0, minimum_good_request_ratio = 0.99 }
max_search_steps = 6
min_rate_resolution = 0.25
duration_seconds = 60
timeout_seconds = 900
```

Adaptive Bench uses `highest-feasible-rate-v1`: it probes every initial rate,
then uses bounded doubling and directional bisection to select only the highest
observed feasible rate. `max_search_steps` covers automatically added probes;
it does not truncate the declared initial list. Use command-line `--set` to
override recipe-specific SLO values without changing the stored definition.

## Serving Bench request sources

Every serving Bench selects one closed request source. A random source declares
the desired complete profiling prompt shape:

```toml
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
```

Either random length may instead use an inclusive uniform selector:

```toml
request_source = { kind = "random", input_tokens = { kind = "inclusive_uniform", min = 7000, max = 9000 }, output_tokens = { kind = "inclusive_uniform", min = 900, max = 1100 } }
```

InferLab draws ISL and OSL independently from the closed integer intervals
under the Bench seed, then freezes the realized population before requests
start. Extending a case preserves the existing sequence prefix. An OSL interval
cannot span both one token and two-or-more tokens, and uniform ISL is not
combined with prefix sharing.

For synthetic sources, `input_tokens` targets the complete prompt length after
chat-template application. Population preparation evaluates the full local
template projection with the resolved model tokenizer and generation marker,
then adjusts only generated message content until that projection has exactly
the selected length. This projection sizes content only: the request remains
structured messages and the server still applies its template. If the default
or request-body template cannot be projected, or no exact generated length can
be constructed, InferLab keeps the unadjusted selected content length and
records the entry as fallback with its reason.

A fixed random shape may reserve one system-message content prefix shared by
every request and an independently generated user suffix:

```toml
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024, prefix_sharing = { shared_prefix_ratio = 0.75 } }
```

InferLab floors `input_tokens * shared_prefix_ratio` to obtain the shared
system-content budget. Exact targeting keeps that prefix unchanged and adjusts
only the user suffix. The ratio is therefore relative to the desired complete
prompt target, not the final pre-template content length. It controls a planned
cacheable-content budget; it does not guarantee the observed
`prompt_cache_read_ratio`, which still depends on backend cache policy, block
alignment, concurrency, and the first uncached request.

Use `random_mixture` when one Bench should sample several exact ISL/OSL pairs:

```toml
request_source = { kind = "random_mixture", shapes = [
  { input_tokens = 1024, output_tokens = 128, weight = 7 },
  { input_tokens = 8192, output_tokens = 1024, weight = 3 },
] }
```

Each request independently selects one shape with probability
`weight / sum(weight)` under the Bench seed. Weights are sampling proportions,
not exact request quotas. Shapes must be distinct, and their output lengths
must either all equal one or all be at least two so TPOT applicability remains
unambiguous. Every case starts from the same deterministic source sequence;
its warmup and profiling phases consume consecutive, non-overlapping entries.

Serving Bench always keeps inputs as structured messages and uses the resolved
chat-completions route. The model server applies its effective chat template;
InferLab does not expose a sibling workspace `chat_template` field, render a
flat prompt, or fall back to completions. Backend-specific server controls such
as `chat_template` and `chat_template_kwargs` may remain ordinary
non-structural members of `request_body` and are forwarded under their real
JSON names. Their support is backend-owned; InferLab records that they were
sent without claiming that an unsupported backend applied them.

Synthetic population evidence records the selected complete-prompt target,
realized pre-template content length, local prediction, exact or fallback
outcome, and fallback reason. When template resolution succeeds, the record
also preserves whether the concrete projection template came from
`request_body` or the tokenizer default, together with its exact content and
SHA-256 digest. Native request identities reconcile warmup and profiling
requests to their frozen population entries. Dataset admission bounds remain
pre-template message-content limits and do not rewrite source content. Neither
local value nor the locally resolved template replaces the model input observed
by the server. A successful Bench separately reports `mean_prompt_tokens`,
`min_prompt_tokens`, `max_prompt_tokens`, `stddev_prompt_tokens`, and the
`p50`/`p90`/`p95`/`p99` prompt-token metrics from AIPerf's backend-observed
`input_sequence_length`. Those values include the server-side template and are
also the token authority behind total-token throughput.

The release catalog currently exposes ShareGPT as a bounded conversational
source. InferLab pins the Apache-2.0
[ShareGPT Vicuna snapshot](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/bcd32a724d8460ebe14e1d05b0195e30e9a46cb1):

```toml
request_source = { kind = "dataset", dataset = "sharegpt", max_input_tokens = 8192 }
```

InferLab downloads the release-pinned snapshot on first execution, verifies
its digest, and reuses it from
`$XDG_CACHE_HOME/inferlab/datasets/sha256/<digest>` (normally
`~/.cache/inferlab/datasets/sha256/<digest>`). Dry-run reports the catalog and
cache state but does not download missing data.

Each selected conversation becomes one independent chat-completions request.
The final assistant message is held out to derive the output limit. If the
pre-template message-content length exceeds `max_input_tokens`, InferLab rolls
back complete trailing user/assistant exchanges until an earlier target fits;
it never truncates a message or discards the leading history. Set
`output_tokens` inside the table to replace target-derived output lengths,
including `output_tokens = 1` for a prefill-dominant run. The Bench-level `seed`
controls deterministic sampling without replacement. Command-line overrides
may change fields within the selected source, but cannot change
`request_source.kind`.

The same release catalog exposes NVIDIA SPEED-Bench profiles. Profile names
are catalog data rather than Rust or Python enum variants; the catalog maps
each identifier to its immutable snapshot, category filter, and AIPerf format.
Qualitative profiles use names such as `qualitative_coding`; throughput
profiles combine an input bucket and entropy tier, such as
`throughput_8k_mixed`:

```toml
[benches.speed-coding]
kind = "serving"
request_source = { kind = "dataset", dataset = "speed_bench", profile = "qualitative_coding", max_input_tokens = 8192, output_tokens = 4096 }
server_metrics = true
concurrency = [16]
prompts_per_concurrency = 5
warmup_prompts_per_concurrency = 0
timeout_seconds = 900
```

InferLab verifies the selected parquet snapshot, filters and samples without
replacement, and freezes only each row's first user turn as an independent
request. Later turns are recorded as omitted rather than silently becoming
sessions.

## Linear serving sessions

A linear session Bench selects a qualified conversational source instead of an
independent `request_source`:

```toml
[benches.linear-chat]
kind = "serving"
session_source = { dataset = "sharegpt", max_input_tokens = 8192, inter_turn_delay_scale = 1.0, max_inter_turn_delay_seconds = 3.0 }
concurrency = [8]
sessions_per_concurrency = 4
warmup_sessions_per_concurrency = 1
timeout_seconds = 900
```

Every later user turn waits for the preceding live assistant response and its
effective inter-turn delay. Concurrency counts active conversations, including
their delay intervals, rather than individual HTTP requests. Positive warmup
uses complete AIPerf-native sessions and drains before profiling. AIPerf adds
its native `warmup` system marker to warmup transport messages; InferLab keeps
that phase-tagged request and its observed prompt-token count, while excluding
all warmup traffic from profiling metrics. Native `session_num` values restart
per phase, so their durable request identity is the pair of phase and
`session_num`.

## Validate before launch

Use the commands in increasing order of machine dependence:

```sh
inferlab workspace show
inferlab stack status
inferlab recipe run smoke --dry-run
inferlab recipe run smoke
```

`workspace show` validates the public catalog. `stack status` separately
reports each selected Pixi environment's manifest-and-lock confirmation,
executes that stack's declared checks against a confirmed local realization,
and reports overall readiness. A failed check reports its captured output and
declared repair hint but does not repair or otherwise mutate the realization.
Dry-run then resolves local placement, effective settings, endpoints, device
assignments, commands, environment, and override provenance without launching
or writing an execution record.
