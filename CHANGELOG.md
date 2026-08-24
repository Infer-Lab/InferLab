# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.1] - 2026-08-23

### Fixed

- SSH subprocesses spawned by the control plane and the profiler transport no
  longer inherit the serving stack's dynamic-linker environment: when InferLab
  is invoked through a workspace Pixi activation, `LD_LIBRARY_PATH` and
  `LD_PRELOAD` are stripped at every SSH spawn boundary (launch, status, log
  sync, cleanup, and container-over-SSH), so the host SSH client cannot be
  corrupted by stack libraries. Operators who deployed a PATH wrapper to work
  around the leak can remove it.

## [0.13.0] - 2026-08-23

### Added

- The `random` serving-Bench request source accepts an optional `corpus`
  declaration — a workspace-relative `path` to an operator-supplied text
  corpus with an optional `expected_sha256` — so content-sensitive
  measurements such as speculative-decoding acceptance rates run on
  natural-language prompts instead of synthetic hash words. Each entry is cut
  as one exact token-length slice of the corpus token stream at an offset
  determined by the Bench seed and the entry's population index alone,
  preserving seed determinism and the invariant that a larger generated
  population keeps the same first entries. Declared digests are verified
  before preparation completes, a corpus shorter than the largest selected
  input target fails preparation, and `prefix_sharing` keeps its controlled
  semantics with one fixed corpus slice as the shared prefix. Independently
  drawn slices may overlap; that incidental sharing is natural reuse and is
  not measured or presented as controlled prefix geometry. Corpus slicing
  requires the default `flat` prompt kind, and invocation overrides cannot
  change the corpus `path` or `expected_sha256`.

- Serving Benches accept a `replay` request source that replays one
  workspace-local frozen population file (for example a previous record's
  `cases/request-source/artifacts/population.jsonl`) byte for byte. The
  source declares a workspace-relative `path`, an explicit `prompt` kind
  (`flat`/`rendered_chat` entries carry `text_input`, `server_chat` entries
  carry structured `messages`), an optional `expected_sha256` verified during
  preparation, and optional `prefix_sharing` whose geometry is resolved from
  the file entries. The file is the sole population authority — no selection,
  filtering, or transformation — and entry output targets stay entry-owned:
  files mixing output-one and larger outputs are rejected, and an
  insufficient population fails preparation instead of repeating entries.
  Dry-run reports the declared path, declared and observed digests, prompt
  kind, and entry count without fabricating unobserved facts, and invocation
  overrides cannot change the replay `path` or `expected_sha256`.

## [0.12.1] - 2026-08-22

### Added

- The Specialized Engine integration 0.5.0 declares the prefix-cache
  conditioning fan-out capability on its SMG Gateway frontend endpoint
  (`POST /prime_prefix_cache`), so a primed Bench against a multi-target
  SMG-fronted shape plans through the declared capability instead of being
  rejected; single-target shapes continue to condition through the ordinary
  serving flow.

### Fixed

- Gateway-fronted `cache.start = "primed"` no longer requires the frontend
  conditioning fan-out capability when the serving shape resolves to exactly
  one cache-owning target (one prefill replica with attention data-parallel
  size one): ordinary frontend routing cannot miss the sole target, so
  conditioning issues one untagged request through the ordinary serving flow
  exactly as on a direct Engine endpoint. The 0.12.0 release rejected such
  single-target shapes at planning, which broke established single-topology
  Gateway-fronted primed comparators; multi-target shapes (multiple prefill
  replicas or attention DP above one) still require the declared capability.

## [0.12.0] - 2026-08-22

### Added

- Verbatim-replaceable managed flag groups in `extra_args`: when a
  post-`--` token names a flag from a group the integration documents as
  verbatim-replaceable, the managed rendering of the whole group is
  suppressed and the verbatim block owns the spelling (store-true flags
  cannot be retracted by last-wins parsing). The SGLang integration applies
  this to its prefill-CP group (`--attention-context-parallel-size`,
  `--enable-prefill-cp`, `--cp-strategy`, and the DSA/NSA-family spellings),
  so DeepSeek-family models can be served with declarative
  `attention.context_parallel_size` plus a verbatim
  `--enable-dsa-prefill-context-parallel` while the engine derives the
  remaining CP facts.

- Gateway-fronted `cache.start = "primed"` conditioning fan-out: the built-in
  vLLM Mooncake, vLLM NIXL, and SGLang prefill/decode proxies now serve
  `POST /prime_prefix_cache`, routing one conditioning request through the
  ordinary pairing flow for every prefill replica and every attention
  data-parallel rank of that replica (rank-pinned via
  `X-Data-Parallel-Rank`; the NIXL and SGLang proxies learn each replica's
  data-parallel size from a control-plane-issued `--prefill-dp` launch
  argument, while the Mooncake proxy enumerates its discovered
  data-parallel engines). The vLLM and SGLang integrations declare the
  fan-out capability on their built-in frontend endpoint; a Gateway-fronted
  primed start whose backend does not declare it (for example the
  `vllm-router` and `sglang-router` pairs) is rejected at planning, and the
  record preserves per-(replica, rank) status, timing, and failure evidence.

- Engine-trace profiling capture over adapter protocol version 8: servers,
  cases, and invocation overrides may select `profiler.mechanism =
  "engine_trace"` (omission resolves to `managed_collection`). For local,
  non-containerized vLLM and SGLang servers the control plane assigns each
  engine-trace replica a persistent record-owned trace directory, the
  integrations render it into the framework-native profiler launch
  configuration (`--profiler-config` with the torch profiler directory for
  vLLM, `SGLANG_TORCH_PROFILER_DIR` for SGLang), and capture coverage is
  verified by the trace-storage delta over the capture: the dedicated
  per-replica trace directory must gain at least one new trace artifact per
  model-serving rank. The TensorRT-LLM, TokenSpeed, and
  Specialized Engine integrations reject the mechanism with a typed error.

- Serving integrations lower declared `attention.context_parallel_size` instead
  of rejecting it. The SGLang integration activates prefill context
  parallelism on `single` and `prefill` roles (`--enable-prefill-cp` with a
  default `zigzag` strategy, overridable through the `--` verbatim passthrough)
  and lowers `decode` roles to `--dcp-size`; the vLLM integration lowers
  `single` and `decode` roles to `--decode-context-parallel-size` and
  `prefill_decode` prefill roles to device-multiplying
  `--prefill-context-parallel-size`. Context parallelism on `single` servers
  never changes device counts; only a `prefill_decode` prefill role may grow
  its device count. Model-, hardware-, and backend-dependent applicability
  remains the framework's launch-time verdict.

### Changed

- Engine-trace capture window closing no longer draws the per-action
  `capture_control_deadline_seconds` budget. The close request is dispatched
  when the measured phase ends, and its response consumption, the artifact
  flush wait, and coverage verification consume the one global
  `capture_finalization_deadline_seconds` budget without restarting it
  (RFC-0004 0.30.1, ADR-0039). A delivery failure of the close — connection
  refusal, a dead engine process, or a prompt error status — remains
  window-closing control failure evidence adjudicated by coverage, while a
  slow or absent stop response is now neutral `flush_pending` evidence on the
  closing action record rather than a deadline failure, and the capture
  succeeds whenever coverage verifies. The flush-adjudication record renames
  `flush_confirmed` to `close_confirmed` because the control response no
  longer attests flush completion; workload record schema version is now 19.
  The undeclared finalization default follows the resolved capture mechanism
  (RFC-0003 0.14.1): 300 seconds for managed collection, 3600 seconds for
  engine trace. Managed-collection captures keep the per-action control
  budget and 300-second default unchanged.

- The built-in SGLang prefill/decode proxy now shares the common decode
  response stream used by the other paired-role proxies instead of a private
  near-copy. The stream takes an explicit client-drop policy: the SGLang
  proxy selects detach (a dropped client response leaves the prefill request
  draining to completion in the background, so the bootstrap-room-paired
  decode-side engine request is never stranded waiting for KV from a
  cancelled prefill — the previous behavior, now documented and pinned by
  tests), while the vLLM Mooncake proxy keeps the abort policy. Streaming
  behavior is unchanged for both proxies.

### Fixed

- The TokenSpeed integration now rejects a non-positive role `replica_count`
  at planning with a typed invalid-settings error naming the role, matching
  the vLLM, SGLang, and TensorRT-LLM adapters; `replica_count = 0` previously
  planned a role with zero replicas silently. Its `render_serve` also drops a
  duplicated multi-node rejection that repeated the single-topology
  allocation check.

- The built-in vLLM NIXL prefill/decode proxy now gates on backend readiness
  like its siblings: it polls every prefill and decode backend's
  `GET /v1/models` until all answer, reports `ready: false` with HTTP 503 from
  `/healthcheck` until then, and rejects `/v1/completions`,
  `/v1/chat/completions`, `/v1/models`, `/prime_prefix_cache`, and
  `/reset_prefix_cache` with 503 before readiness instead of forwarding to
  backends that are not yet serving.

- Profiler declarations can no longer silently drop: declaring
  `profiler.mechanism` or nsys escape inputs on a server whose profiling
  resolves off (no `profiling = true`, no requested capture) previously
  produced a server with no capture preparation at all. Resolution now fails
  with a typed error naming the declaration and the enable-profiling
  remediation.

- Built-in vLLM and SGLang prefill/decode frontend endpoints now declare
  backend prompt cache-read usage when both serving roles enable the
  reporting setting (`enable_prompt_tokens_details` / `enable_cache_report`).
  The built-in proxies forward engine responses verbatim, so the capability
  is real; without the declaration, primed and prefix-sharing Benches against
  a Gateway-fronted P/D pair failed planning with a remediation (enable the
  reporting setting and rebuild) that could never succeed for that topology.

- A serving Bench with `cache.start = "primed"` or declared prefix geometry
  (`prefix_sharing` / `shared_system_content`) against a server whose endpoint
  exposes no prompt cache-read capability previously ran to completion and
  then failed every request's normalization for missing backend cache-read
  usage. Planning now rejects the combination with a typed error naming the
  bench, the missing capability, and the remediation (enable the integration's
  cache-read reporting setting and rebuild the server); the runtime
  normalization and Bench client messages carry the same remediation as a
  backstop.

- A serving Bench with `cache.start = "primed"` under attention data
  parallelism previously sent one conditioning request that the frontend
  load balancer routed to a single data-parallel rank, leaving the remaining
  ranks cold. Conditioning now issues one recorded request per attention
  data-parallel rank of the public serving role, pinned through the
  `X-Data-Parallel-Rank` request header (honored by current vLLM main and
  SGLang), preserves per-rank status, token usage, and timing evidence, and
  fails the case when any rank's conditioning fails.

- The built-in vLLM Mooncake and NIXL prefill/decode proxies now serve
  `POST /reset_prefix_cache` by fanning out to every prefill and decode engine
  and reporting partial failures, so a serving Bench with a controlled cache
  start passes planning on built-in vLLM P/D pairs instead of forcing an
  uncontrolled (hot-prefix) start. The `vllm-router` pairing remains without
  reset control.

- Quality pass over the recent merge, proxy, and profiling changes. Tokens
  after a bare `--` in `extra_args` again form one verbatim passthrough block
  that an overriding layer replaces as a whole, server-common and role-level
  `extra_args` compose by the same per-flag-group merge as every other layer,
  and every declared `extra_args` array is validated at workspace load.
  Proxy cache fan-out shares one aggregation and response implementation,
  rejects an empty target set instead of recording a successful prime, and
  the control plane reconciles the returned replica/rank coverage against the
  planned shape. Engine-trace finalization only follows successful
  window-control actions and propagates trace-directory snapshot errors
  instead of exhausting the budget. Server and recipe records written before
  adapter protocol version 8 fail through the friendly schema-version gate.

- The `extra_args` passthrough sentinel no longer reaches the engine. The
  bare `--` is an InferLab-side composition marker, but the adapters
  previously spliced it into the rendered engine command, where
  argparse-based engine launchers reject it (`unrecognized arguments`) and
  the designed verbatim override never took effect. Rendered engine argv now
  carries only the post-sentinel tokens, appended after the managed tail so
  engine last-wins parsing applies the deliberate override.

- Engine-trace capture coverage is now counted against the replica's model
  device count instead of the process-rank count of the trace-directory
  snapshot. vLLM and SGLang write one trace artifact per model-serving rank
  (one per GPU) into the replica's shared trace directory, so a replica with
  `tensor_parallel_size = 2` must gain at least two new artifacts before the
  capture verifies; the previous per-process baseline verified as soon as the
  frontend's own trace landed, minutes before the worker ranks flushed theirs.
  The capture record carries the baseline as `expected_artifacts`.

### Changed

- The adapter protocol hard-cuts to version 8: `plan_serve` and
  `render_serve` carry the effective capture mechanism instead of a profiling
  flag, capture targets declare their mechanism, and engine-trace model-rank
  allocations carry the control-plane-assigned trace directory. Protocol
  version 7 input or output is rejected rather than partially interpreted;
  update workspace adapter pins and relock. Adapter SDK 0.8.0; vLLM, SGLang,
  TensorRT-LLM, and TokenSpeed integrations 0.7.0; Specialized Engine
  integration 0.4.0.

- `extra_args` now composes per flag group across the server base, the
  selected case, and invocation overrides instead of being replaced wholesale
  by each later layer. A later layer's group replaces a same-named earlier
  group in place, new groups append, unmentioned groups are inherited, and the
  `--` passthrough block is replaced as a whole. Cases are additive over the
  base; there is no removal path — dropping a base flag means restructuring
  the server definition.
- The bundled authoring guidance splits `measurement-authoring.md` into
  `eval-authoring.md` (Eval tasks, datasets, and inference requests) and
  `bench-authoring.md` (serving-Bench load, sources, sessions, metrics, and
  SLOs), and no longer cites specification documents that are not shipped
  with the plugin.

## [0.11.0] - 2026-08-19

### Added

- Serving Bench definitions accept `artifact_level = "diagnostic" | "performance"`.
  The `diagnostic` default retains the complete raw capture, including
  per-request raw records and the raw-derived AgentX evidence dimensions.
  The `performance` level trims raw capture for measurement runs where
  client-side instrumentation must not perturb the result, and records the
  trimmed dimensions as explicitly unavailable rather than fabricating them.

### Fixed

- Serving integrations no longer silently drop `extra_args` entries that name
  an InferLab-owned option (for example `--block-size`). Plan and render now
  fail with a typed `invalid_settings` error naming the offending flag and its
  remedy — use the owning typed setting, or place the flag after a `--`
  sentinel for a deliberate verbatim override — so the returned and recorded
  escape-hatch contents match the executed command. Adapter SDK 0.7.1; vLLM,
  SGLang, TensorRT-LLM, and TokenSpeed integrations 0.6.1; Specialized Engine
  0.3.1.

## [0.10.0] - 2026-08-08

### Added

- Recipe-owned and standalone measurements now prepare non-synthetic data
  assets before serving or inference begins. Release-catalog and AgentX inputs
  retain verified immutable closures, enumerable local lm-eval sources are
  rebound to read-only snapshots, and sources whose complete closure is owned
  elsewhere remain explicit opaque evidence. Dry-run reports only locally
  observable state and planned effects, while records preserve preparation,
  reuse, and reproducibility outcomes.
- Serving Bench accepts `cache = { start = "uncontrolled" | "cold" | "primed" }`.
  Native warmup drains before controlled reset, a primed case conditions the
  exact canonical shared prefix once, and profiling starts only after cache
  preparation succeeds. Prefix-sensitive cases retain per-request prompt and
  cache-read token evidence and report weighted observed reuse; direct vLLM
  and SGLang cache-start paths are qualified with their integration-owned
  reporting controls.
- Generative lm-eval definitions can select `prompt = { kind = "flat" }` or
  `prompt = { kind = "server_chat" }`, with omission resolving to `flat`.
  InferLab derives the native client and route from that authority, records it
  with every metric, and determines prompt-logprob probes from the request
  types a resolved task actually emits rather than from its definition
  language.
- The TUI now presents source-aware request, linear-session, and AgentX Bench
  definitions and records, together with domain labels, families, and units
  for every known Bench metric.
- The Specialized Engine integration exposes typed GPU-memory, workspace,
  prefix-cache GPU, and per-rank host-cache settings. Its public endpoint also
  declares explicit prompt-cache-read reporting, while vLLM and SGLang expose
  opt-in settings that enable and describe their backend reporting behavior.

### Changed

- Human progress on stderr now uses conventional timestamped, scoped log lines
  with concise item positions and elapsed durations while preserving the
  existing machine-readable stdout and operation-observation facts.
- The 0.10.0 workspace package set uses `inferlab-adapter-sdk==0.7.0`, version
  `0.6.0` of the vLLM, SGLang, TensorRT-LLM, and TokenSpeed integrations, and
  `inferlab-integration-specialized-engine==0.3.0`. Existing workspaces must
  update the exact SDK and selected integration pins together and relock.

### Fixed

- Local lm-eval sources are classified as closed only when the release-pinned
  loader can enumerate and rebind every consumed file; unsupported selectors,
  external paths, and function references remain runnable as explicit opaque
  sources instead of overstating reproducibility.
- Documentation light and dark modes now apply one coherent semantic palette
  across navigation, sidebars, controls, and primary content while retaining
  the InferLab accent and font choices.

## [0.9.1] - 2026-08-06

### Added

- Serving Bench supports the release-qualified SemiAnalysis AgentX trace-replay
  workflow through a closed `agentic_source = { dataset, profile }` boundary.
  InferLab verifies immutable 062126 corpus revisions and digests, while the
  pinned AIPerf 0.12 runtime owns trajectory warmup, root/subagent scheduling,
  source-response replay, native scenario validity, and branch evidence. The
  256k profile is qualified on direct vLLM with its 600-second cache-pressure
  warmup and a 900-second profiling window; results remain transport evidence,
  not agent-task quality scores.

### Changed

- Ordinary measurement definitions now expand concise authoring forms into
  explicit canonical values: static Bench defaults to `serving`, synthetic
  random sources default to exact flat completions, untagged `{ min, max }`
  token ranges mean inclusive-uniform selection, and OpenAI smoke defaults to
  prompt `Hello`, 16 output tokens, and a 60-second timeout.
- The release-owned Bench runtime now pins AIPerf 0.12.0. Native request-rate,
  linear-session, server-metric, and warmup behavior remains qualified, and
  profile capture retains one fail-closed acknowledged barrier around both
  request-rate and AgentX profiling phases.

### Fixed

- The built-in vLLM NIXL P/D Gateway now forwards streaming completion and
  chat-completion events as they arrive instead of buffering the decode body
  to completion, preserving client-observed TTFT and inter-token timing.

## [0.9.0] - 2026-08-05

### Added

- Serving Bench can materialize tokenizer-exact scalar prompts directly or
  freeze locally rendered chat from the tokenizer default or a
  definition-supplied template and kwargs. Both modes reconcile every
  completed request with AIPerf's backend-observed input length and fail a
  mismatch instead of reporting an inaccurate ISL.
- Exact flat and locally rendered synthetic prompts accept either shared token
  counts or per-entry ratios over one deterministic canonical prefix stream.
  Fixed and distributed ISL populations support the complete range from zero
  through full-prompt sharing while keeping declared geometry separate from
  observed cache-read evidence.

### Changed

- Synthetic random and weighted-mixture Bench sources now require an explicit
  prompt authority: `flat` and `rendered_chat` derive the completions route,
  while `server_chat` retains structured messages and chat completions.
  Dataset-backed requests and linear sessions remain server-rendered chat, and
  the request endpoint is not an independently selectable Bench option.
- Server-chat `shared_system_content` remains a pre-template compatibility
  shape and cannot be combined with exact final-prompt prefix geometry. A
  configured prefix ratio or full shared prompt does not by itself claim a
  cache-hit percentage or decode-only execution.
- The bundled agent skill now routes operators through an offline capability
  map and focused references covering the complete CLI, workspace-authoring,
  measurement, profiling, image, evidence, and plugin surfaces. CLI `--help`
  also states command-specific dry-run, record, override, image,
  profiling, and mutation boundaries at the point of use.

## [0.8.5] - 2026-08-05

### Changed

- InferLab 0.8.5 supersedes the halted aggregate 0.8.3 and 0.8.4 product
  releases. Their already-published crates remain immutable, but neither
  version received a product tag or GitHub Release; the governed time-control
  corrections below ship together under the 0.8.5 product identity.
- Product tags now stage qualified repository assets in a draft GitHub Release.
  Manual finalization adds and downloads the exact closed workspace-wheel
  inventory, publishes the verified aggregate Release, and only then unlocks
  manual crates.io and Python package-index publication commands.

### Fixed

- Readiness status, HTTP, and target-registry attempts now use the resolved
  `readiness_attempt_timeout_seconds` value (30 seconds by default), capped by
  the readiness operation's remaining budget. This removes the hidden 250 ms
  HTTP cap while keeping capture-armed readiness interruptible between bounded
  attempts.
- Prefix-cache reset now enforces one measurement-case deadline across client
  initialization, connection, response headers, and complete response-body
  consumption. A transport failure observed only after that deadline can no
  longer replace the case timeout. Prompt-logprob tokenizer and HTTP probes
  likewise consume their measurement case's remaining budget instead of
  acquiring a hidden 30-second cap.
- Profiler arming and finalization now consume one resolved budget across all
  targets and commands; framework window control consumes complete HTTP
  responses without a shorter connection cap, and report verification waits
  for asynchronous publication within the shared finalization budget.
- SSH invocations retain noninteractive login initialization while leaving
  connection and keepalive policy to the selected OpenSSH target configuration.
  Remote preflight and launch delivery remain unbounded by arbitrary terminal
  timeouts but now respond to operator interruption and reap their process group;
  bounded lifecycle and cleanup calls continue consuming their owning budgets.
- Machine-local adapter bindings can set `adapter.timeout_seconds` for
  process-backed plan/render invocations independently of
  `adapter.image_timeout_seconds`; effective defaults remain 30 and 120 seconds
  respectively, and adapter timing evidence records the selected budget.

## [0.8.3] - 2026-08-04

### Fixed

- `inferlab stack status` now runs each confirmed stack's declared local
  realization checks and reports structured check evidence plus overall
  readiness. Its existing `status` field remains the manifest-and-lock Pixi
  environment confirmation state; failed checks and check-launch errors leave
  the realization unchanged and make the command unsuccessful.
- Workspace source digests now encode each initialized recursive submodule's
  path and effective commit directly, so local branch, tag, and `git describe`
  presentation state cannot change source identity. Workspaces containing
  submodules receive a one-time source-digest re-key.
- SSH lifecycle commands now retain their own exit status after interactive
  login environment initialization, so a failing Bash logout hook cannot make
  a live remote server appear to have exited.

## [0.8.2] - 2026-08-02

### Fixed

- Workload-attached profiling now drains a positive AIPerf-native Bench warmup
  before opening the Nsight capture window, while retaining one native case
  run, one sequential request population, and profiling-only metrics.
- Nsight collection finalization now recognizes a completed repeat-range
  collection without issuing a redundant failing `nsys stop`. The effective
  profiler environment is applied consistently to launch, collection start,
  session inspection, and fallback collection stop.

## [0.8.1] - 2026-08-01

### Changed

- Product Releases now collect the exact independently versioned workspace-side
  wheels selected by the tagged source snapshot from the package index and
  publish them with verified checksum sidecars. Package-only publications no
  longer create package-scoped GitHub tags or Releases.

### Fixed

- Frozen synthetic Bench populations now derive each undeclared-prefix prompt
  from its seeded request identity instead of slicing a repeated short corpus,
  preventing accidental full-prompt prefix-cache hits between independent
  warmup and profiling requests. Records identify the corrected generator as
  `inferlab-synthetic-prompt-target-v3`.

## [0.8.0] - 2026-08-01

### Added

- Serving Bench random sources accept deterministic inclusive-uniform ISL and
  OSL selectors, while retaining fixed and weighted-shape AIPerf paths.
- The release dataset catalog includes immutable, first-turn NVIDIA
  SPEED-Bench qualitative and throughput profiles. Direct vLLM and SGLang
  endpoints, together with the Specialized Engine through TokenSpeed SMG's
  integration-declared Prometheus listener, can expose AIPerf server metrics,
  including validated `acceptance_length` and `acceptance_rate` SPEED report
  results.
- Serving Bench supports AIPerf-native linear ShareGPT sessions whose later
  turns depend on each live assistant response. Definitions control source
  think-time scaling and capping, while records preserve phase-qualified
  session and turn evidence.
- SGLang serving supports framework-controlled Nsight Systems capture for
  direct and prefill/decode topologies through integration-declared
  `/start_profile` and `/stop_profile` actions.

### Changed

- Serving Bench always sends structured messages through the chat-completions
  route and leaves the effective chat template with the model server. For
  synthetic sources, InferLab first attempts to size generated content against
  the complete local template projection so the selected ISL is exact; when
  the projection or an exact construction is unavailable, it preserves the
  unadjusted content-length request and records the fallback reason. AIPerf's
  backend-observed prompt-token metrics remain the authority for measured ISL
  and token throughput.
- Runtime lifecycle and profiling now have independent reusable Rust crate
  boundaries, while resolver, image-context, and serving-Bench coordination
  are divided by their owning domain without changing the operator commands.
- The 0.8.0 workspace package set uses `inferlab-adapter-sdk==0.6.1`, version
  `0.5.2` of the vLLM, SGLang, TensorRT-LLM, and TokenSpeed integrations, and
  `inferlab-integration-specialized-engine==0.2.2`. Existing workspaces must
  update the exact SDK and selected integration pins together and relock.

### Fixed

- Linear AIPerf session warmup accounts for the pinned load generator's
  terminal prefetch, keeping warmup and profiling template slices disjoint and
  preventing replay or loss at the phase boundary.
- An operator-authored `request_body.chat_template` remains a literal backend
  request value instead of being evaluated against AIPerf's own Jinja context.

## [0.7.1] - 2026-07-30

### Fixed

- `inferlab agent update` now replaces an older versioned local InferLab
  marketplace with the plugin package embedded in the current binary before
  refreshing the Codex or Claude installation. After installing the 0.7.1
  executable, 0.7.0 operators can run
  `inferlab agent update --agent all`; manual plugin uninstall and reinstall
  are no longer required.

## [0.7.0] - 2026-07-30

### Added

- Serving Bench random sources can declare one shared system-prefix ratio.
  InferLab resolves and records exact shared-prefix and unique-suffix token
  counts while keeping the configured ratio distinct from the backend's
  observed prompt-cache hit ratio.
- Serving Bench can draw a deterministic request population from an ordered,
  weighted mixture of exact input/output token shapes. Shape selection,
  warmup, and profiling remain on the existing release-pinned AIPerf path.

### Changed

- The public `inferlab-adapter-sdk` now contains only framework-integration
  protocol models and helpers. Eval and Bench use a new internal
  `inferlab-measurement-sdk` that is versioned and delivered with the InferLab
  product, so future measurement-only changes do not force adapter releases.
  `inferlab-measurement-sdk`, `inferlab-eval-runner`, and
  `inferlab-bench-runner` all carry product version `0.7.0` and are not
  workspace-side publications.
- The 0.7.0 workspace package set uses `inferlab-adapter-sdk==0.6.0`, version
  `0.5.1` of the vLLM, SGLang, TensorRT-LLM, and TokenSpeed integrations, and
  `inferlab-integration-specialized-engine==0.2.1`. These integration patches
  only adopt the narrower SDK dependency; serving behavior and adapter
  protocol version 7 are unchanged. Existing workspaces must update the exact
  adapter SDK and selected integration pins together and relock. The internal
  measurement SDK is not a workspace dependency.

## [0.6.1] - 2026-07-28

### Fixed

- Default `inferlab agent install` now materializes the embedded plugin under a
  durable, versioned InferLab data directory instead of registering a temporary
  marketplace source that disappears when the command exits. Re-running
  `inferlab agent install --agent all` with 0.6.1 repairs affected 0.6.0 Codex
  and Claude registrations, and `inferlab agent doctor` now reports a
  configured local InferLab marketplace whose directory is missing.
- Public CI provides the Protocol Buffers compiler required to compile and test
  the optional Specialized Engine SMG transport on a clean runner.

## [0.6.0] - 2026-07-27

### Added

- A reusable Specialized Engine integration runs one token-only Engine process
  behind a TokenSpeed SMG Gateway. The Engine owns an arbitrary nonzero
  single-process pure-TP device set and request-shaped batched-token capacity,
  while SMG retains HTTP, tokenizer, chat-template, detokenization, and response
  formatting responsibilities.

### Changed

- Adapter protocol version 7 replaces the combined routing-ownership result
  with separate Engine, Gateway, and P/D Router plans. Frontend allocations
  use stable named schema bindings for `[gateway]` and
  `[gateway, pd_router]`, carry no model or rank coordinates, and keep
  `render_source` limited to command lowering while InferLab retains runtime
  authority.
- Protocol-v7 profiling targets now bind typed capture-window actions to either
  their Engine replica entry or the separately planned Gateway. This lets a
  Specialized Engine keep a token-only process surface while InferLab captures
  its process tree and TokenSpeed SMG controls the profiling window. Managed
  capture defaults to CUDA and NVTX tracing; OS runtime tracing remains
  available through the typed trace override.
- Workspace serving configuration replaces `routing_backend` with independent
  `gateway_backend` and `pd_router_backend` facts. A direct `single` has no
  frontend, a routed `single` derives a Gateway-only process when a supported
  backend is selected, and prefill/decode derives one fused frontend process.
  Resolved plans and server-record schema 4 now close component facts and
  concrete allocations under `frontend`, bind each component by `process_id`,
  and own frontend allocations as a process collection so a future split
  Gateway/P/D Router implementation does not require another hierarchy change.
- The shared adapter SDK now owns protocol-v7 frontend-plan construction,
  allocation dispatch, and rendered-identity checks reused by all four
  maintained integrations.
- The protocol-v7 release set uses `inferlab-adapter-sdk==0.5.0`, version
  `0.5.0` of the vLLM, SGLang, TensorRT-LLM, and TokenSpeed integrations, and
  `inferlab-integration-specialized-engine==0.2.0`. Workspaces using
  protocol-v6 packages must update their exact package pins, replace
  `routing_backend` with the topology-appropriate Gateway and P/D Router
  fields, and relock before running InferLab 0.6.0.

### Fixed

- The GitHub Pages site follows the repository's canonical `InferLab` path
  without changing local preview routing.

## [0.5.0] - 2026-07-18

### Added

- `inferlab tui` provides a persistent, strictly view-only console for one
  discovered or explicitly selected workspace. Its Overview, Operations,
  Records, and Workspace views combine declared definitions, concurrent CLI
  observations, records, referenced logs, and scratchpad context without
  starting or changing an experiment.
- A static product and documentation website publishes the selected public
  guides together with the current RFC and ADR corpus through GitHub Pages,
  with search, locked local preview and production-build tasks, and
  revision-matched CI deployment.
- Public SGLang reference workflows and support documentation cover
  disaggregated prefill/decode serving with Model Gateway, Mooncake, and NIXL,
  while distinguishing execution-qualified pairings from supported but
  unqualified cross-pairings.

### Changed

- The TUI uses a responsive infrastructure-console hierarchy, typed Global Find
  and contextual log search, stable object navigation, and a record-local
  Metrics surface that compares one selected metric across authoritative case
  loads with horizontal bars and explicit missing or failed states.
- The complete Records catalog and Global Find scale to at least 1,000 records
  through source-aware disposable projections, tiered observation cadence, one
  fair refresh-wide active-server probe budget, and redraws driven by observable
  presentation changes rather than the input-poll loop.
- The final capless IL mark and InferLab Blue brand color are applied
  consistently across the website, favicon, plugin identity, and TUI loading
  and accent surfaces; constrained terminals retain a compact text fallback.
- The adapter SDK and each framework integration own independent package
  versions and package-scoped releases. InferLab product releases continue to
  version the Cargo workspace, embedded plugin, and internal measurement
  runners; exact workspace pins preserve artifacts and the adapter protocol
  version remains the runtime compatibility authority.
- Published framework workspace baselines are clean and reproducible, with
  generic local-binding examples and without machine-local state, credentials,
  model locators, or cross-framework package drift.

### Fixed

- TUI rows and details keep recorded lifecycle, observed process liveness, and
  refresh health separate: stopped servers no longer appear live, a dead process
  behind a recorded-running server becomes explicit attention, and a failed
  observation retains its prior value only as stale.
- Healthy automatic refresh shows a stable cadence instead of oscillating
  between `now` and elapsed ages. The indicator reports waiting before its
  first generation, becomes overdue only after two missed intervals, measures
  receipt age monotonically, and recovers after the next completed generation.
- Global Find no longer combines unrelated typed fields into false matches,
  keyboard selection opens the chosen result, referenced-log search retains its
  owning object and selected log, and responsive layouts preserve navigation
  and visible overflow down to the supported minimum terminal size.
- Human-facing data-age labels remain `now` for their complete first second;
  elapsed-duration fields retain subsecond precision.
- Website routes accept either local trailing-slash form while publishing one
  canonical form, human-facing brand text consistently uses InferLab, and
  projected Markdown renders one semantic page title instead of duplicating its
  leading heading.

## [0.4.0] - 2026-07-16

### Added

- Long-running commands now report phases, bounded item progress, lock contention, readiness failures, heartbeats, elapsed time, record directories, and durable log paths on stderr while keeping machine-readable stdout clean.
- lm-eval definitions can select built-in tasks or workspace-local task YAML, including task-owned datasets, splits, prompting, output type, and scoring. The resolved model locator supplies the default tokenizer, and likelihood tasks receive a bounded prompt-logprob and tokenizer-alignment probe before evaluation.
- Eval results now normalize task and metric identity deterministically, preserve raw native output and failure artifacts, and expose explicit transport, endpoint, response-shape, metric-selection, and tokenizer-alignment failures instead of silently changing task semantics.
- The release-owned Eval runtime includes an offline long-context, single-sample generation task with strict terminal-answer scoring. Eval definitions can repeat an eligible task with deterministic per-trial seeds, existing request concurrency, incremental per-trial evidence, and pass rate over issued trials.
- Eval and Bench definitions can carry task-specific OpenAI request parameters, including sampling, logprobs, reasoning effort, and chat-template arguments, with invocation-time nested overrides. Generate and serving workloads use named chat-completions routes, likelihood and smoke workloads use named completions routes, and Inferlab's built-in vLLM, SGLang, and TensorRT-LLM proxies support the corresponding chat path.
- Concurrency Benches can run AIPerf-native warmup before profiling. Normalized results include request latency, TTFT, and TPOT mean/min/max/stddev/p50/p90/p95/p99 plus prompt-cache read ratio when reported.
- Static and adaptive Benches support aggregate SLOs, per-request latency SLOs, minimum good-request ratio, goodput, and an automatically expanding and bisecting highest-feasible-rate search.
- Serving Bench can materialize a release-pinned ShareGPT snapshot into deterministic, tokenizer-bounded single-request populations with content-verified caching and complete acquisition, truncation, population, and native-request identity evidence.

### Changed

- The release-owned Bench environment now uses AIPerf 0.11.0; lm-eval remains pinned at 0.4.12.
- Adapter protocol version 6 makes completions and chat-completions routes explicit and carries the revised Eval and Bench request shapes. The 0.4.0 integration wheels are the tested lockstep set; workspaces pinned to 0.3.0 integrations must bump and relock with the binary.
- Eval and Bench cases now consume one end-to-end case budget beginning at their first case-owned action. Readiness, adapter invocation, container operations, and profiler control likewise have explicit owning deadlines, while cleanup and finalization use separate grace periods and never consume or rewrite the preceding business-operation budget.
- Serving Bench definitions now use the closed `request_source` union. The former flat random-token fields and adaptive `target_metric`, `target_threshold`, and `max_refinement_steps` fields have no compatibility aliases; use `aggregate_slos` and `max_search_steps`.
- Agent commands again delegate orchestration to the agent-plugin-installer batch API and remain outside the long-running-command progress contract.
- Workload plans, typed records, local process groups, measurement-runner operations, Python integration helpers, and server launch mechanisms now have explicit single owners. These internal boundary refactors are intended to preserve existing serving, placement, readiness, interruption, cleanup, and record behavior.

### Fixed

- Official static release binaries now detect the runtime Linux architecture and glibc compatibility from host facts, so measurement toolchain installation works on supported x86_64 and aarch64 glibc hosts instead of inheriting the musl build target.
- TPOT applicability, stable Bench metric names, lm-eval metric selection, operation terminal causes, and cleanup elapsed time now come from one typed authority at each boundary instead of duplicated string or boolean inference.
- A failed or timed-out measurement, adapter, profiler, container, or cleanup operation preserves the established business result, terminal cause, native command evidence, partial artifacts, and verified cleanup outcome without granting nested attempts fresh timeout budgets.

### Security

- Backend qualification remains scoped to the exact demonstrated topology, route, integration revision, and hardware baseline; public support documentation excludes credentials, private hosts, model locators, local paths, record identifiers, and private downstream revisions.

## [0.3.0] - 2026-07-14

### Changed

- Workspace schema version 2 replaces source sets, serving environments, serve profiles, and recipe-owned cases with stacks, independently launchable servers, server-owned cases, and recipes that compose one server with one workload suite. Local placements can bind explicit role/replica/rank allocations, including zero-device proxy ranks and machine-specific model locators.
- The matching CLI uses `workspace lock`, `stack status [STACK]`, `run --stack`, and `serve start <SERVER>`; serve, recipe, and image workflows accept explicit local placement selection.
- Adapter protocol version 5 plans canonical role/replica/rank hierarchies. Each role is the sole authority for effective settings and parallelism; concrete process allocations carry only placement and launch facts. Dry-run, execution, and records now share concrete resolved types, and lifecycle commands reload their complete authority from records without consulting current workspace configuration.
- Ad-hoc `run` resolves only committed stack and image declarations and no longer loads machine-local bindings. Server overrides mirror the typed workspace shape, keep backend values under explicit common or role `settings` paths, and accept quoted TOML key segments including setting names containing literal dots.
- Record identifiers include the workflow and selected server, recipe, Bench, or image name, plus the selected case where applicable and the creating process ID.

### Fixed

- Device hardware evidence now invokes NVIDIA's native `nvidia-smi --query-gpu` spelling while keeping Inferlab's public resource terminology consistently device-based.

## [0.2.0] - 2026-07-13

### Added

- TokenSpeed is now a supported serving integration through the new `inferlab-integration-tokenspeed` package, covering aggregated `ts serve` launches and SMG-routed prefill/decode serving over Mooncake with explicit attention, dense, and MoE parallelism.
- SGLang prefill/decode serving can use either the Inferlab built-in proxy or SGLang Router, independently of Mooncake or NIXL KV transfer.
- TensorRT-LLM prefill/decode serving can use its native disaggregated frontend or the Inferlab built-in proxy over NIXL.
- Framework integrations can render content-addressed launch files; the control plane validates, records, and atomically materializes them for local, SSH, and container launches.
- TensorRT-LLM is now a supported serving framework: the new `inferlab-integration-tensorrt-llm` package plans and renders `trtllm-serve` launches (declare `integration = "tensorrt-llm"` in a serve profile). It maps the shared parallelism vocabulary onto TensorRT-LLM's semantics — attention data parallelism is all-or-nothing (`--enable_attention_dp`), expert parallelism divides the tensor-parallel world — and rejects shapes TensorRT-LLM cannot serve (context parallel, MoE data parallel, dense tensor parallel) at planning time. Framework knobs only reachable through TensorRT-LLM's extra-LLM-API-options YAML (MoE backends, attention-DP balancing, KV block-reuse control) pass through the `extra_llm_api_options` path setting. Note: TensorRT-LLM exposes no prefix-cache flush endpoint, so benches against it cannot request `reset_prefix_cache`; disable KV block reuse at launch instead when a case needs cache isolation. The adapter boundary was smoke-validated against the official release image; the maintained DeepSeek-V4 SM120 baseline is source-built.
- `inferlab run [--environment ID] [--image RECORD | --external-image ID] [--mount PATH[:rw]]... [--gpus SPEC] -- CMD...` runs one ad-hoc command inside a serving environment — a local Pixi install, a built image, or an external image — attached to your terminal and exiting with the command's own status. There are no default mounts; `--mount` binds an absolute host path read-only unless suffixed `:rw`, and `--gpus` exposes an explicit GPU selection to a container.
- `inferlab env status [--environment ID]` reports whether each declared serving environment is `confirmed`, `never-installed`, or `not-usable`, as JSON, without needing local machine bindings or installing anything — useful right after a fresh checkout or a `git pull` to check before you launch anything. Exits non-zero if any environment isn't confirmed.
- A successful environment check is now remembered against the exact Pixi manifest and lock content that produced it, so a launch that finds nothing changed skips re-probing Pixi entirely; any edit to the manifest or lock invalidates the memory and forces a fresh check. `inferlab run` deliberately does not participate in this — it neither trusts nor produces this evidence, so an ad-hoc command can never make a real launch skip a check it should have made, or vice versa.
- `inferlab agent install` no longer needs a repository checkout: the Claude Code / Codex plugin package now ships embedded in the binary itself, so `inferlab agent install --agent all` works immediately after installing the CLI, offline. `--from-checkout <DIR>` is still available for testing local edits to the plugin before a release.
- Interrupting a recipe now reliably cleans up the eval/bench measurement processes it started, including a background sweep that catches any survivor left behind by an unclean exit.
- A toolchain removal that fails because something still has the install path open now names the exact holding process(es) in the error.

### Changed

- Recipe record IDs now include the selected recipe and case, omit process IDs, and use collision suffixes when needed.
- `inferlab agent install` defaults to the binary-embedded plugin package described above; `--from-checkout` remains a fully-supported explicit override.
- `scripts/install.sh` no longer downloads or unpacks the plugin package separately — it's already inside the binary it just installed.
- The operator skill now routes ad-hoc environment commands through `inferlab run` and calls out that invoking an interpreter or tool binary directly from inside a materialized environment prefix is unsupported.

### Fixed

- Bench requests to `/v1/completions` now preserve each synthetic prompt as a scalar string, allowing OpenAI-compatible servers without batched-prompt support to run the shared AIPerf workload.
- vLLM Router readiness can no longer preempt Inferlab's own readiness endpoint during prefill/decode startup.
- A serving environment that was never installed at all used to silently fall through to whatever happened to be on the ambient `PATH` instead of failing; environment checks (local and over SSH) now correctly catch this before anything launches, and separately catch an installed environment that's gone stale relative to a since-regenerated lock.
- The `inferlab` binary crate now packages and compiles cleanly from its published crate form (`cargo package`/`cargo publish`), with a test pinning the in-crate toolchain payload copies byte-identical to their Python sources so this can't silently regress.
- Serving from an external image could not report which adapter version did the lowering: the in-container adapter invocation saw the packages' code but not their distribution metadata. Each package's metadata now travels with its code, so records carry the exact pinned adapter version even for external-image launches.
- Stopping a server container launched with auto-remove could race the Docker daemon's own removal and get recorded as unverified cleanup even though the container was gone moments later; the stop now confirms by watching the container actually disappear (bounded), and only reports unverified if it never does.

## [0.1.0] - 2026-07-05

Initial release.

### Added

**Workspace model** — Workspaces declare recipes, serve profiles, serving environments, models, benchmarks, and correctness cases as typed, validated configuration, composed from `.inferlab/workspace.d/*.toml` fragments. Workspace state is tied to the git revision and content digest of the source tree, including submodules; a dirty working tree is recorded honestly instead of silently ignored. Local, machine-specific facts (model weight paths, machine bindings) live separately in `.inferlab/local.toml`, kept out of the versioned workspace definition. Workspace source integrity is enforced structurally — no path in the workspace configuration or a declared source set may be, or resolve through, a symlink that escapes the workspace root.

**Pixi environment lifecycle** — `inferlab env lock` produces the authoritative full workspace Pixi lock from a clean local prefix, with no manual manifest edits. Every workflow that touches a serving environment activates it read-only and fails before doing anything else if it isn't installed or the lock doesn't match the manifest — Inferlab itself never installs packages or updates the lock.

**Serving and recipes** — `serve start`/`status`/`logs`/`stop` and `recipe run` share one server lifecycle covering single- and multi-node deployments, local and over SSH, with full dry-run validation before anything launches. `recipe run` executes the closed loop end to end — start the server, wait for readiness, run the recipe's Eval gate and any eligible Benches, then tear the server down — recording every case, server, and measurement outcome in one aggregate record. Multi-node network setup is automatic: Inferlab probes every machine for a routable interface common to the whole placement and wires it into NCCL. Each server process gets its own deterministic runtime cache directory, derived from the workspace, environment, machine, and process identity.

**Measurement (Eval and Bench)** — `inferlab toolchain install` installs Inferlab's own release-pinned Eval (lm-eval-based) and Bench (AIPerf-based) runtimes, kept separate from your serving environment, on both x86_64 and aarch64 Linux. Bench runs go through a typed runner that translates recipe-declared cases into AIPerf configuration and returns normalized results and cleanup evidence even on failure or interruption. A standalone OpenAI-compatible smoke check needs no measurement toolchain at all. Workload profiling captures a Nsight Systems trace of a serving run, keyed to the actual assigned GPUs, with configurable capture windows and escape hatches for advanced `nsys` options.

**Framework integrations** — vLLM, including single-role and disaggregated prefill/decode (Mooncake and NIXL) topologies with a built-in reverse proxy or an external vLLM Router; and SGLang, using the shared tensor/data/expert/pipeline parallelism vocabulary.

**Runtime images** — `inferlab image build` assembles an OCI runtime image from a workspace's serving closure, deduplicating identical closures across requested model-validation targets and validating each output by actually running the eval/bench suite against it. Serving from a pre-built image reuses the same recipe and measurement machinery as a live install, including on multi-node and SSH placements. Workspaces can also declare a digest-pinned *external* image that Inferlab didn't build — Inferlab verifies it's present with the right digest on every machine that needs it and never pulls it for you. Declared environment checks and image postprocessing hooks run at defined points (image build, inside the built image, and host preflight) so per-site fixups are explicit and recorded.

**Distribution** — `inferlab agent install|update|uninstall|doctor` installs Inferlab's operator skill into Claude Code and/or Codex. Tagged releases publish Linux binaries (x86_64/aarch64), a reproducible plugin tarball, Python wheels for the workspace-side adapter packages, and the Rust crates, all stamped with one version, with the MIT license retained in every distributed form. A stable, append-only error-code registry means every runtime failure exits with exactly one `error[<code>]` diagnostic, and published codes never change meaning.

**Scratchpad** — `inferlab scratchpad note`/`show`, an append-only, file-first operator journal, with entries optionally tied to a specific record.

### Changed

- Multi-node placements express machine facts (hosts, devices, launch access, endpoints, execution-visible paths) in local, machine-specific bindings, keeping recipes themselves machine-independent.
- Runtime cache storage roots are configurable per machine, with a workspace-local default when none is bound.

### Removed

- The old adapter-mediated Eval and Bench request/response path is gone in favor of direct typed runner requests; framework integrations no longer lower ordinary Bench operations, only server-specific control.
- The binary no longer embeds or materializes workspace-side Python packages — those now ship as ordinary wheels from the package index, so a binary upgrade can no longer silently change adapter behavior under an unchanged workspace commit.

### Fixed

- Numerous image-build correctness and caching fixes: cache keys now cover every input that actually affects the built content (source-set paths, Pixi manifest digest, target platform, build-procedure identity), export archives are named so concurrent or repeated builds can't clobber each other's evidence, cache publication is safe under concurrent builds, and a workspace mutated during a build now fails the build loudly instead of shipping silently-wrong output.
- Container launches no longer leak: adapter containers are terminated and removed through an owned handle rather than left to the docker client, removal is bounded by a deadline instead of hanging, and an unconfirmed removal is reported honestly rather than assumed clean.
- Activation values that could break shell quoting or leak credentials are rejected at render time; container pass-through environment variables are validated and passed by name only, never by value.
- A disaggregated-serving streaming correctness fix: prefill no longer sends more tokens than the forced single-token prefill maximum allows.

### Security

- Model weights, weight locations, credentials, and any other undeclared private workspace content never enter a built image, its OCI output, or a shareable manifest.
