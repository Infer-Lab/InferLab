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

[servers.example.cases.tp2]
devices = 2

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

`readiness_timeout_seconds` owns the complete ordinary server-readiness wait;
each blocking process-status or HTTP attempt within that wait is capped by
`readiness_attempt_timeout_seconds`, patchable for one invocation as
`--set server.readiness_attempt_timeout_seconds=45`. Profiled servers use
separate capture budgets owned by
[Profiling authoring](execution-authoring.md#workload-profiling). Cleanup grace
and polling cadence are product policy rather than workspace settings; SSH
connection and keepalive policy remain in the selected OpenSSH target
configuration.

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

`extra_env` is the matching escape hatch for process environment variables:
a table of names to string values that the integration adds to each rendered
process. Like other settings tables, it composes per variable across the
server base, the selected case, and invocation overrides, and a later layer's
value for one name replaces the earlier one. On a containerized launch these
variables reach the container verbatim and apply only inside it.

## Synthetic acceptance

For speculative-decoding benchmarks that must use a controlled acceptance
length (for example InferenceX AgentX golden-AL policy), declare
`synthetic_acceptance` on the server or a case. The speculative method and
method-specific flags stay in framework settings as usual, and a separate
draft model is declared through `auxiliary_models` (see below) rather than
spelled into those settings — InferLab only overlays the resolved acceptance
length onto that operator-owned configuration, and planning fails with a
typed error when no speculative configuration exists to overlay. A case-level
declaration replaces the server-level declaration wholesale.

Declare exactly one form. An explicit acceptance length:

```toml
[servers.example.synthetic_acceptance]
acceptance_length = 2.49
```

Or a digest-pinned golden acceptance-length curve (InferenceX shape):

```toml
[servers.example.synthetic_acceptance.curve]
path = "curves/deepseek_mtp.yaml"
expected_sha256 = "<64 lowercase hex of the file bytes>"
model_key = "deepseek-v4-pro"
thinking_mode = "thinking_on"     # optional for matrix curves
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
[backend support matrix](backend-support.md).

## Auxiliary model weights

A launch that consumes additional weight artifacts beyond the served model —
today only a separate speculative-decoding draft model — declares them on the
server under `auxiliary_models`, keyed by the governed artifact kind and
referencing an ordinary workspace model:

```toml
[models.example-draft]
served_name = "example-draft"

[servers.example.auxiliary_models]
draft-model = "example-draft"
```

The kind vocabulary names the artifact's role, not a speculative method; the
method and its flags stay in the operator's framework settings. The
referenced model resolves its weights through the same `model_weights`
bindings as the served model, including per-machine `machine_locators`, and
the resolved locator rides each model-rank allocation into the rendered
command and the serve record. The declaration is the single authority for
the artifact: planning fails with a typed error when the operator's
speculative configuration already names the draft weights or offers no
splice target. The declaration is server-level only — cases must not declare
or override it. Per-backend consumption support is listed in the
[backend support matrix](backend-support.md).

## Stacks

A stack names its integration, the Pixi environment that realizes it, and the
workspace source paths it builds from. It may also declare realization checks
and image postprocess steps; both are workspace-relative Python scripts run
with the stack environment's interpreter:

```toml
[[stacks.vllm.checks]]
id = "serving-stack"
script = "tools/check_vllm_stack.py"
repair_hint = "pixi install --locked -e vllm"

[[stacks.vllm.image_postprocess]]
id = "trim-caches"
script = "tools/postprocess_image.py"
```

A check passes only on exit status zero, must not mutate the realization, and
reports facts rather than remedies; `repair_hint` carries the operator remedy.
One check set runs against every realization: the local environment, each
remote machine's environment before launch, and the image during assembly.
Postprocess steps run only during image assembly, after installation and
before the checks. Use [Workspaces and stacks](workspaces-and-stacks.md) for
installation, confirmation, and lock operations.

A stack may also declare `settings`: facts of the framework or Engine
implementation it realizes, shared by every server and role that uses the
stack. Its integration defines and validates them, fills its own defaults, and
returns the effective values, which dry-run output and the server record keep.
An integration that defines no stack setting rejects any declared one; today
only the [Specialized Engine](https://github.com/Infer-Lab/InferLab/blob/main/docs/specialized-engine.md) defines one,
`engine_command`:

```toml
[stacks.engine]
integration = "specialized-engine"
pixi_environment = "serving"

[stacks.engine.settings]
engine_command = ["example-token-engine", "serve"]
```

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

A routed `single` server may declare `replicas` greater than 1 on its `serve`
role when its integration supports it, and a server with a Gateway may declare
`gateway_settings`, a table its integration types and validates; a declared
entry the integration does not recognize fails planning. A server with a P/D
Router may likewise declare `pd_router_settings`, where SMG-backed
integrations take `prefill_policy` and `decode_policy`; on a `prefill_decode`
server, routing policies belong there rather than in `gateway_settings`.
Cases and `--set` patch both tables like other settings.

Selecting `dynamo` — `gateway_backend = "dynamo"` on a vLLM routed `single`
server, or both backend fields on a vLLM `prefill_decode` server with
`kv_transfer = "nixl"` — runs the Dynamo frontend and `dynamo.vllm` workers,
and InferLab also starts one etcd discovery process beside the Gateway for that
server. The stack environment must provide the `ai-dynamo` distribution and an
`etcd` executable. Dynamo and etcd listen only on IP addresses and advertise
them to processes on other machines, so a machine bound by hostname renders as
the first address that hostname resolves to other than loopback; a hostname
that does not resolve, or resolves only to loopback, is rejected before launch.
Dynamo parses a worker's control port
as a signed 16-bit value, so those machines must offer ports no higher than
32767. The vLLM `reasoning_parser` setting renders as Dynamo's reasoning
parser. The vLLM API-server settings `tool_call_parser`,
`enable_auto_tool_choice`, and `enable_prompt_tokens_details` are rejected —
the Dynamo frontend answers requests and reports cache-read usage itself — and
so are their options, and Dynamo's own discovery and listener options and
variables, anywhere in `extra_args` or `extra_env`.

## Devices and parallelism

`devices` is the whole-replica device count of each model-serving role. It
may be declared on the server, a canonical role, a case, a case role, or
with `--set`, and a role value at any layer wins over a common value. When
`devices` is declared it is the only placement authority:

- with no parallelism declared, the integration renders pure tensor
  parallelism of that width;
- with a declared layout, the integration fills omitted outer tensor
  parallelism to match `devices` and rejects a layout that its framework
  would place on a different number of devices.

```toml
[servers.example]
devices = 8

[servers.example.parallelism.attention]
data_parallel_size = 8
```

Without `devices`, the integration derives the count from `parallelism`.

The `parallelism` components map onto framework options, and each integration
reads them its own way. For example, vLLM multiplies outer tensor parallelism
by attention data parallelism, while SGLang treats outer tensor parallelism as
the whole world. The [backend support matrix](backend-support.md) states each
reading. The components are deprecated in favor of `devices` plus framework
layout settings and remain accepted. Keep attention data parallelism in
`parallelism.attention.data_parallel_size` rather than in `extra_args`: it
is the only source InferLab uses to prime and route every data-parallel rank.

## External KV store

A `single` server may run one external KV store that its model-serving
processes share. Declare it on the server base; a case or `--set` may patch
its settings but cannot add it or change its backend:

```toml
[servers.example.kv_store]
backend = "mooncake"

[servers.example.kv_store.settings]
eviction_high_watermark_ratio = 0.9
default_kv_lease_ttl_ms = 10000
```

InferLab runs the store as its own process beside the Gateway, or beside the
first replica on a direct server, with no devices. It starts the store first,
waits for its readiness, gives every Engine its address, stops it last, and
fails the server if it exits. The `mooncake` backend runs `mooncake_master`
from the stack environment and exchanges transfer metadata peer to peer, so no
metadata service runs. Other master flags pass through `settings.extra_args`.

Controlled cache starts stay honest with a store: a cold reset must also clear
the store, so vLLM resets with `reset_external=true`, which wipes the store,
while SGLang declares no reset with a store, because `/flush_cache` leaves its
HiCache storage backend intact, so its cold and primed starts are rejected at
planning. Each integration declares how each role uses the store: `read_write`
when its processes read matching prefixes and write the ones they compute
before the request completes, or `write` when they only write. When every
cache-owning role — `serve`, or `prefill` on a prefill/decode server — reads
and writes, the server shares one prefix cache, and a primed start sends one
conditioning request for the whole server. The Specialized Engine declares
`serve` or `prefill` as `read_write` and `decode` as `write`; vLLM and SGLang
declare none and keep their per-rank conditioning. A store applies to `single`
and `prefill_decode` servers; only the Specialized Engine consumes it on
`prefill_decode` so far, and the others reject it there.

How much memory each Engine contributes stays an Engine setting, because the
frameworks size it differently: vLLM's `kv_store_global_segment_size` is per
worker rank, SGLang's is per instance and split across its ranks, and a
Specialized Engine sizes its own tier. The
[backend support matrix](backend-support.md) lists which integrations consume
a store.

## Context parallelism

Declare attention context parallelism on the server or a canonical role under
`parallelism.attention`:

```toml
[servers.example.roles.serve.parallelism.attention]
context_parallel_size = 2
```

The integration lowers the declared value; it is not an engine flag spelling.
Context parallelism on a `single` server never adds devices; only a
`prefill_decode` prefill role without `devices` may grow its device count.
With `devices` declared, context parallelism must fit within it. Each backend's
lowering and qualification live in the [backend support matrix](backend-support.md).

Model- and hardware-dependent applicability (MLA-only prefill CP, attention
head divisibility) remains the engine's launch-time verdict; planning cannot
prove it.

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

A workflow selects its placement from an explicit `--placement`, then, for a
recipe run, the recipe's entry in the local `recipe_placements`, then the local
`default_placement`, then the sole declared placement; more than one remaining
candidate is ambiguous and fails naming them. `recipe_placements` keeps a
recipe's placement a machine-local fact while letting the recipe reproduce its
run without flags. Each entry must name a declared recipe and placement;
otherwise loading the workspace fails naming the entry:

```toml
[recipe_placements]
qualify-two-replicas = "two-replicas"
```

A `model_weights` binding may declare a fallback `locator`, per-machine
`machine_locators`, or both; each model-serving rank uses its machine's
locator when present and the fallback otherwise. Measurements run on the
controller and load the tokenizer from a controller-local rank's locator, or
else from the fallback `locator`, which must therefore be readable on the
controller. A placement that offers neither fails resolution.

Adapter invocation deadlines are machine-local because process startup and
container startup costs vary by site. The two paths remain independent:

```toml
[adapter]
timeout_seconds = 30        # process-backed plan/render
image_timeout_seconds = 120 # image-backed plan/render
```

Both values must be positive when declared; when omitted, the control plane's
built-in defaults apply. `image_device` is a separate,
optional workaround for container runtimes that cannot create a device-less
adapter container; it does not affect process-backed lowering.

Machine `cache_root` provides the machine-local root from which InferLab
allocates runtime JIT caches using resolved stack and source identity. Cache
contents remain convenience state rather than portable evidence or stack
confirmation.

On a shared multi-socket host, measurements can vary with load that other
sessions place on the CPUs and memory a server happens to use. Machine
`numa_nodes` binds every server process InferLab launches on that machine to
the listed NUMA nodes, usually the node that hosts the machine's devices
(`nvidia-smi topo -m` shows each GPU's NUMA affinity):

```toml
[machines.local]
host = "127.0.0.1"
devices = [0, 1]
ports = [8000, 8001]
numa_nodes = [0]
```

Host processes then run under `numactl --cpunodebind --membind`, which must be
installed on the launch machine; containerized processes receive the
container runtime's cpuset for those nodes. The binding appears in the
resolved commands in dry-run and records. Resolution, dry-run included, reads
the nodes' CPU lists on the launch machine and fails naming the machine when a
declared node is absent there or has no CPUs, or when a host launch's machine
lacks `numactl`; the list must be nonempty and must not repeat a node. Leave
it unset for a machine whose devices span NUMA nodes.

Servers on one host can share one machine binding, its devices, and its port
pool. Resolution, dry-run included, observes candidate machines in placement
order until one fits: on each launch path it lists the TCP sockets listening
there with `ss -Hltnp` (iproute2, which must be on the launch machine's
`PATH`), reads which devices report compute applications with the same
`nvidia-smi` probe the launch-time device check uses, and counts what the
workspace's records hold there. A starting server record holds its ports and
devices; a running one holds its ports, its devices being visible as compute
applications; a running recipe whose server record does not yet exist holds
both. Finished records, and records written by an earlier record version, hold
nothing. Pool allocation then takes the first remaining devices and ports in
declared order, and the resolved placement lists each skipped one under
`unavailable_devices` or `unavailable_ports` with its holder: a compute
application, a listening socket with its process when visible, or the record
that holds it together with the `serve stop` or `recipe stop` command that
releases it — see
[Stopping a run](serving-and-recipes.md#stopping-a-run). Too few free devices
fails resolution naming the holders; it does not wait. A launch path that
cannot be read fails resolution naming the machine. Launches in one workspace
take turns observing and allocating; a launch that waits for its turn names
the launch it waits on and ends on Ctrl+C.

One pool placement per host is then enough for concurrent servers of any
size:

```toml
[machines.shared]
host = "shared.example"
devices = [0, 1, 2, 3, 4, 5, 6, 7]
ports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]

[placements.shared]
machines = ["shared"]
```

A second TP2 server started while the first holds devices 0 and 1 receives 2
and 3, and its own ports. To pin a server to particular devices — for NUMA or
interconnect topology — bind them explicitly; explicit devices are never
moved and face only the launch-time device check, an explicit `endpoint_port`
that is held fails resolution, and `--wait-for-devices` can wait for explicit
devices to free:

```toml
[placements.shared-high.roles.serve]
machine = "shared"
devices = [4, 5, 6, 7]
```

Launches from one workspace take their devices and ports one at a time; a
launch from another workspace or user that has not yet started its processes
is not seen, and a collision with it fails at the launch-time device check or
at bind.

A machine with `launch = { kind = "ssh", target = "..." }` runs its processes
through noninteractive SSH (`BatchMode`), so authentication must not prompt.
Remote commands run in a login shell, whose initialization must put `pixi`, and
`docker` or `numactl` when used, on `PATH`. A host launch requires the machine's
`workspace` path to be a checkout at the controller's revision, dirty state,
source digest, and Pixi manifest and lock, with the stack environment already
installed there; a mismatch or a missing environment fails before launch. An
external-image launch uses the remote `workspace` only as a launch directory.

Use explicit rank placement when replicas span machines, roles use different
device counts, or the same model has different locators on each machine. This
example places two TP4 prefill replicas across pairs of machines, two TP2
decode replicas on individual machines, and a zero-device fused frontend on
the controller. The frontend is placed as `gateway` even though the same process
also realizes P/D Router, and it has no model locator, replica index, or rank:

```toml
default_placement = "cluster"

[model_weights.example]
locator = "/models/example"   # readable on the controller for measurements

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

## Local definitions

Experiment-only definitions — a Bench over a private corpus, a capture recipe,
a server variant with a debug hook — belong in git-ignored fragments under
`.inferlab/workspace.local.d/*.toml` rather than untracked `workspace.d`
files, which dirty the workspace and block `image build`. Add the directory to
`.gitignore`; InferLab rejects it otherwise:

```gitignore
.inferlab/local.toml
.inferlab/workspace.local.d/
```

A local fragment uses the `workspace.d` fragment format but may declare only
`models`, `servers`, `evals`, `benches`, `workload_suites`, and `recipes`. It
only adds: redeclaring an identifier the committed workspace declares fails
load naming both files, and committed definitions validate without local ones,
so they can never depend on them. Local definitions compose into `serve
start`, `recipe run`, `bench`, and their dry runs, stay outside the source
digest and dirty marker, and need no copy on remote launch machines, because
definitions resolve on the controller; a file a local definition makes a remote
process read, such as a `PYTHONPATH` entry, must still exist there. Each
record names the local files that composed under
`workspace.local_definitions`, with their SHA-256 digests, and sets
`workspace.revision_reproducible` to false, since the commit alone no longer
reproduces the run. `image build` and `run` never load local definitions, so
an image validation recipe must be committed. `workspace show`, the TUI, and the
web console show committed definitions only, and the web console's launch forms
offer only those; launch a local definition from the CLI, with `--detach` when
the run must outlive the session. The directory and its files must
stay git-ignored and untracked.

## Workload suites and recipes

A workload suite lists named Evals and Benches and may select one Eval as its
gate. A recipe selects one server and one workload suite; it does not duplicate
their definitions. Recipe execution preserves the declared workload order.

A recipe may also pin one of its server's cases, so the run does not depend on
`--case` or the server's `default_case`; an explicit `--case` still takes
precedence, and the record names which one selected the case. Its placement is
chosen in local bindings through `recipe_placements`
([Local bindings](#local-bindings)).

```toml
[recipes.qualify-two-replicas]
server = "engine"
workload_suite = "qualify"
case = "two-replicas"
```

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

`workspace show` validates the public catalog. `stack status` reports
realization confirmation and declared checks as described in
[Workspaces and stacks](workspaces-and-stacks.md#stack-lifecycle). Dry-run then
resolves local placement, effective settings, endpoints, device
assignments, commands, environment, and override provenance without launching
or writing an execution record.

## Upgrading across an adapter protocol hard cut

A release may hard-cut the adapter protocol: input or output from the previous
protocol line is rejected rather than partially interpreted, and records
written before the cut fail through the record schema-version gate instead of
loading. A protocol mismatch at runtime names both versions and the remedy.

To move a workspace across a hard cut, first learn the current correspondence
from the authority surfaces: `inferlab --version` states the adapter protocol
line the installed binary speaks, and the version correspondence table in the
[backend support matrix](backend-support.md) maps the product line to the exact adapter SDK and
framework integration package versions. Then update the adapter SDK and the selected
framework integration pins together to those versions — never one without the
other — and run `inferlab workspace lock` so the committed Pixi lock becomes
the new workspace authority.

The release-owned measurement toolchain is pinned by the product, not by the
workspace; reinstall it after upgrading as
[Prepare The Measurement Runtime](measurements.md#prepare-the-measurement-runtime)
describes. Migration
history for each release lives in the changelog.

Record schemas also change between releases without an adapter protocol cut.
A binary reads server records only at its own schema version, so after an
upgrade that changes it, `serve status`, `logs`, `stop`, and `bench --serve`
reject a server started by the previous binary. Stop managed servers before
upgrading, or stop them with the binary that started them.
