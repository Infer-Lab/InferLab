# Workspace definitions, placement, upgrades, and validation

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
canonical role. Integrations validate their typed fields. `extra_args` is the
explicit backend escape hatch, composed per flag group across the server base,
the selected case, and invocation overrides:

- A `--`-prefixed token opens a group; repeated flags coalesce into one group
  per layer, and `--flag=value` counts as the same group.
- A later layer's group replaces a same-named earlier group in place; new
  groups append; unmentioned groups are inherited. Cases are additive over the
  base — there is no removal path.
- A bare `--` starts one verbatim passthrough block that a later layer
  replaces as a whole. The sentinel is an InferLab-side composition marker:
  it never reaches the engine argv, which carries only the post-sentinel
  tokens appended after the managed tail so engine last-wins parsing applies
  the deliberate override.
- Naming an InferLab-owned option fails with a typed `invalid_settings` error
  rather than being silently dropped; place it after `--` for a deliberate
  verbatim override.
- A post-`--` token that names a flag from a group the integration documents
  as verbatim-replaceable suppresses the managed rendering of the whole group,
  and the verbatim block owns the spelling. This exists because store-true
  flags cannot be retracted by last-wins parsing.

## Synthetic acceptance

For speculative-decoding benchmarks that must use a controlled acceptance
length (for example InferenceX AgentX golden-AL policy), declare
`synthetic_acceptance` on the server or a case. The speculative method, draft
model, and method-specific flags stay in framework settings as usual —
InferLab only overlays the resolved acceptance length onto that operator-owned
configuration, and planning fails with a typed error when no speculative
configuration exists to overlay. A case-level declaration replaces the
server-level declaration wholesale.

```toml
# Explicit acceptance length:
[servers.example.synthetic_acceptance]
acceptance_length = 2.49

# Or a digest-pinned golden acceptance-length curve (InferenceX shape):
[servers.example.synthetic_acceptance.curve]
path = "curves/deepseek_mtp.yaml"
expected_sha256 = "<64 lowercase hex of the file bytes>"
model_key = "deepseek-v4-pro"
thinking_mode = "thinking_on"     # optional; defaults to thinking_on for matrix curves
# No draft count is declared: the integration reads it from the operator's
# speculative configuration and resolves the effective acceptance length.
```

A curve file maps each model key to either a flat list of
`{draft_length: acceptance_length}` entries or a `{thinking_mode: {draft_length:
acceptance_length}}` matrix. The plan-returned acceptance length, the
determined draft count, and the curve provenance are preserved in dry-run and
the serve record. An Eval bound to a
server with synthetic acceptance fails at planning, because synthetic
acceptance bypasses real draft-model verification; run correctness evals
against a variant without the declaration. Per-backend overlay support is
listed in the
[backend support matrix](../../../../../docs/backend-support.md).

A stack selects one integration and Pixi environment. Its `source_paths` name
workspace-relative framework sources; declared checks verify the realized
environment, and optional image postprocessing belongs to the stack rather
than an image definition. Use [Workspaces and stacks](workspaces-and-stacks.md)
for installation, confirmation, and lock operations.

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

## Context parallelism

Declare attention context parallelism on the server or a canonical role under
`parallelism.attention`:

```toml
[servers.example.roles.serve.parallelism.attention]
context_parallel_size = 2
```

The integration lowers the declared value; it is not an engine flag spelling.
vLLM lowers `single` and `decode` roles to decode context parallelism and
`prefill_decode` prefill roles to device-multiplying prefill context
parallelism. SGLang lowers `single` and `prefill` roles to prefill context
parallelism (`--enable-prefill-cp` with a default `zigzag` strategy) and
`decode` roles to decode context parallelism (`--dcp-size`). Context
parallelism on a `single` server never adds devices; only a `prefill_decode`
prefill role may grow its device count.

Model- and hardware-dependent applicability (MLA-only prefill CP, attention
head divisibility) remains the engine's launch-time verdict; planning cannot
prove it. Consult the bundled backend support matrix before claiming a CP
shape.

For DeepSeek-family models on SGLang, declare CP and select the DSA prefill-CP
spelling verbatim. The integration documents its whole prefill-CP flag group
as verbatim-replaceable, so one post-`--` mention suppresses the managed group
and the engine derives the remaining CP facts:

```toml
[servers.deepseek-mla.parallelism.attention]
context_parallel_size = 2

[servers.deepseek-mla.settings]
extra_args = ["--", "--enable-dsa-prefill-context-parallel"]
```

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

Machine `cache_root` provides the machine-local root from which InferLab
allocates runtime JIT caches using resolved stack and source identity. Cache
contents remain convenience state rather than portable evidence or stack
confirmation.

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

## Workload suites and recipes

A workload suite lists named Evals and Benches and may select one Eval as its
gate. A recipe selects one server and one workload suite; it does not duplicate
their definitions. Recipe execution preserves the declared workload order.

Invocation patches may address only the selected server and measurements in
that suite. Read [Invocation patches](execution-authoring.md#invocation-patches)
for their typed paths and identity restrictions.

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

## Upgrading to the protocol-v9 line

The next release after 0.13.1 hard-cuts the adapter protocol to version 9:
protocol version 8 input or output is rejected rather than partially
interpreted, and server records written before this line fail through the
schema-version gate (server record schema version 8). Existing workspaces must
update their exact package pins to `inferlab-adapter-sdk==0.9.0` and version
`0.8.0` of the selected vLLM, SGLang, TensorRT-LLM, TokenSpeed, or Specialized
Engine integration. Update the SDK and selected integration together, then run
`inferlab workspace lock` so the committed Pixi lock becomes the new workspace
authority.

## Upgrading to 0.12

InferLab 0.12 hard-cuts the adapter protocol to version 8: protocol version 7
input or output is rejected rather than partially interpreted, and server and
recipe records written before protocol version 8 fail through the
schema-version gate instead of loading. Existing workspaces must update their
exact package pins to `inferlab-adapter-sdk==0.8.0` and version `0.7.0` of the
selected vLLM, SGLang, TensorRT-LLM, or TokenSpeed integration. A Specialized
Engine workspace uses `inferlab-integration-specialized-engine==0.4.0`.
Update the SDK and selected integration together, then run
`inferlab workspace lock` so the committed Pixi lock becomes the new workspace
authority. The product-owned `inferlab-measurement-sdk` remains internal to
the installed measurement toolchain and must not be added to a serving
workspace.

The release-owned measurement toolchain (lm-eval runner and AIPerf Bench
runner `0.12.0`) is pinned by the product, not by the workspace. Re-run the
installer after upgrading so the installed runtime matches the new release:

```sh
inferlab toolchain install
```

Workspaces upgrading from 0.5 or earlier must replace the former
`routing_backend` field because the current control plane does not interpret
protocol-version-6 requests or responses.
A direct `single` server declares neither frontend backend; a routed `single`
declares `gateway_backend`; and a `prefill_decode` server declares both
`gateway_backend` and `pd_router_backend`. The control plane rejects the old
combined field rather than guessing how to divide its ownership.
