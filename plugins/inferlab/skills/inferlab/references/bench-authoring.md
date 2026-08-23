# Serving Bench load, sources, sessions, metrics, and SLOs

Start with the smallest definition that expresses the workload. A static synthetic Bench defaults to `serving`. A `random` or
`random_mixture` source defaults to an exact flat completion prompt, so fixed
ISL and OSL need no prompt table:

```toml
[benches.fixed-8k1k]
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

Use an untagged `{ min, max }` table for an inclusive-uniform integer range:

```toml
[benches.range-8k1k]
request_source = { kind = "random", input_tokens = { min = 6553, max = 8192 }, output_tokens = { min = 819, max = 1024 } }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

These are authoring defaults, not hidden execution state. The
`inferlab workspace show --json` command renders them explicitly as `serving`,
`flat`, and tagged `inclusive_uniform`. Existing
explicit forms remain valid. Add the advanced controls below only when the
workload needs their distinct semantics.

Eval definitions are covered by [eval-authoring.md](eval-authoring.md).

## Static and adaptive serving load

A static `serving` Bench uses either independent requests or dependent
sessions. Concurrency cases use `prompts_per_concurrency` or
`sessions_per_concurrency`. Request-rate cases instead use `request_count` or
`duration_seconds`; rates accept positive numbers or `"inf"`, and `burstiness`
controls supported stochastic arrival shaping. One definition cannot mix
incompatible load authorities.

An `adaptive-serving` Bench declares positive initial request rates, one or
more aggregate or request SLO constraints, a bounded search-step count, and an
optional minimum rate resolution. It measures every initial rate before its
bounded expansion and bisection policy.

## Serving Bench warmup and metrics

A concurrency Bench may run a native warmup phase before its profiled phase:

```toml
[benches.random-8k1k]
kind = "serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = 8192, output_tokens = 1024 }
concurrency = [1, 8]
prompts_per_concurrency = 4
warmup_prompts_per_concurrency = 2
request_body = { temperature = 1.0 }
timeout_seconds = 900
```

- For concurrency `c`, the resolved warmup count is
  `c * warmup_prompts_per_concurrency`, drawn from a disjoint prefix of the
  frozen request population.
- Warmup shares the profiled route, request source, request body, and
  concurrency, and is excluded from normalized metrics and profiling request
  counts.
- Cache start defaults to uncontrolled; declare a cold or primed start only
  when the experiment requires it:

```toml
cache = { start = "cold" }   # warmup drains, then reset, then profiling
```

- `cold` requires the selected integration/topology to expose a prefix-cache
  reset. `primed` additionally requires an exact flat or rendered-chat
  `prefix_sharing` source; after reset, InferLab sends the maximum canonical
  shared prefix once with one output token, then releases profiling. A rendered
  prefix that cannot tokenize exactly as an independent conditioning prompt
  fails preparation.
- Under attention data parallelism, conditioning fans out one recorded request
  per prefill replica and per attention data-parallel rank of that replica,
  pinned through the `X-Data-Parallel-Rank` request header, so no rank stays
  cold. The built-in vLLM Mooncake, vLLM NIXL, and SGLang prefill/decode
  proxies serve `POST /prime_prefix_cache` and route that fan-out through the
  ordinary pairing flow. The record preserves per-(replica, rank) status,
  token usage, and timing evidence, and any rank's conditioning failure fails
  the case. The fan-out capability is required only when more than one
  prefill-side cache-owning target (replica × attention DP rank) sits behind
  the frontend; a single-target Gateway-fronted shape conditions through the
  ordinary serving flow without it. The `vllm-router` and `sglang-router`
  pairs declare no primed fan-out capability and reject a multi-target primed
  start at planning.
- The built-in vLLM Mooncake and NIXL pairs also serve
  `POST /reset_prefix_cache`, fanning out to every prefill and decode engine,
  so a cold start passes planning on those pairs. The `vllm-router` pairing
  remains without reset control.
- Reset and conditioning occur after warmup, are excluded from profiling
  metrics, and do not consume population entries.
- The one case timeout covers warmup, reset, conditioning, profiling, and
  result handling; cleanup retains its separate grace. A default capture
  window opens after cache preparation and before profiling release. Any
  warmup or cache-preparation failure leaves the window unopened.

Every successful Bench reports `request_throughput`, `output_throughput`, and
`total_token_throughput`, plus `mean`, `min`, `max`, `stddev`, `p50`, `p90`,
`p95`, and `p99` for each of `request_latency_ms`, `ttft_ms`, and `tpot_ms`
(names such as `p95_tpot_ms`). TPOT is not applicable to an
`output_tokens = 1` prefill-dominant workload and its TPOT metrics are then
omitted.

A Bench with `prefix_sharing`, `shared_system_content`, or a primed start
requires backend prompt-token and cache-read-token usage on every completed
profiling request, and reports:

- `prompt_cache_read_tokens` and `uncached_prompt_tokens` with the same
  distribution statistics;
- `prompt_cache_read_ratio = sum(cache_read_tokens) / sum(prompt_tokens)`.

For direct vLLM, enable `enable_prompt_tokens_details = true` in the server
settings. For direct SGLang, enable `enable_cache_report = true`; SGLang's
OpenAI protocol omits the cache detail when the reported read is exactly zero,
so its integration records that endpoint representation and InferLab preserves
the request observation as zero. An undeclared missing value still fails
normalization. Built-in vLLM and SGLang prefill/decode frontend endpoints
declare the backend cache-read capability only when both roles enable the
reporting setting, because the built-in proxies forward engine responses
verbatim. A primed or prefix-geometry Bench against an endpoint without the
declared capability fails at planning with a typed error naming the bench, the
missing capability, and the remediation — enable the reporting setting on both
roles and rebuild the server — instead of running to completion and failing
every request's normalization. Other cases preserve AIPerf's optional aggregate
cache ratio when available. `good_request_ratio` and `goodput` are derived only
when a request SLO is configured.

Set `server_metrics = true` to ask AIPerf to collect the server's declared JSON
metrics export. This is accepted only when the selected integration/topology
declares a metrics endpoint. Native warmup and cache preparation remain outside
the exported profiling window. The integration may bind that endpoint to the
public serving port or to one named port it already
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

Set `artifact_level = "performance"` to skip AIPerf's full raw
request/response export while keeping the normalized per-request records and
summary; omission resolves to `"diagnostic"`, which retains the raw export.
Warmup validation and population-identity reconciliation read the normalized
records under either level. Session- and agentic-source cases remain valid at
the performance level; their raw-derived evidence dimensions are recorded as
unavailable due to the artifact level rather than failing the case.

## Serving Bench SLOs

A static or adaptive serving Bench may constrain normalized aggregate metrics,
individual request latency, or both. Aggregate constraints are inclusive and
AND-composed. Request SLOs count a request as good only when every configured
latency bound passes, then gate the case with `minimum_good_request_ratio`.

```toml
[benches.saturation]
kind = "adaptive-serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = 8192, output_tokens = 1024 }
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

Every serving Bench selects one closed request source. A synthetic random
source declares its desired complete profiling prompt shape. Omission selects
`flat`; declare it explicitly when emphasizing tokenizer-exact scalar prompts
sent to completions:

```toml
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = 8192, output_tokens = 1024 }
```

Either random length may instead use an inclusive-uniform selector. The
ordinary untagged range and the explicit tagged spelling resolve identically;
canonical JSON and records use the tagged spelling:

```toml
request_source = { kind = "random", input_tokens = { min = 7000, max = 9000 }, output_tokens = { min = 900, max = 1100 } }
```

InferLab draws ISL and OSL independently from the closed integer intervals
under the Bench seed, then freezes the realized population before requests
start. Extending a case preserves the existing sequence prefix. An OSL interval
cannot span both one token and two-or-more tokens.

Use `rendered_chat` to freeze either the model tokenizer's default chat template
or one definition-supplied template and kwargs. InferLab renders the complete
prompt once during population preparation, verifies its exact final token
length, and sends the resulting scalar prompt to completions. Template controls
for this mode belong only to `prompt`; duplicating them in `request_body` is a
validation error. Failure to resolve or satisfy the template is fatal rather
than a fallback.

```toml
request_source = { kind = "random", prompt = { kind = "rendered_chat", chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}", chat_template_kwargs = { enable_thinking = false } }, input_tokens = 8192, output_tokens = 1024 }
```

Flat and rendered-chat sources may declare exact final-prompt prefix geometry
as fixed tokens or a per-entry ratio:

```toml
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = { kind = "inclusive_uniform", min = 7000, max = 9000 }, output_tokens = 1024, prefix_sharing = { shared_prefix_ratio = 0.75 } }
```

For selected input length `I`, a ratio resolves to
`floor(I * shared_prefix_ratio)` shared tokens and an `I - shared` independently
generated suffix. Fixed tokens and equivalent ratios produce the same seeded
population. All entries use nested prefixes of one canonical stream, including
distributed ISL. Both `0.0` (independent flat prompts) and `1.0` (full-prompt
sharing) are valid. This is prompt geometry, not a requested or observed cache
hit percentage; backend cache-read metrics remain separate evidence.

Use `random_mixture` when one Bench should sample several exact ISL/OSL pairs:

```toml
request_source = { kind = "random_mixture", prompt = { kind = "flat" }, shapes = [
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

Use `server_chat` when the server must retain chat-template authority. Requests
stay as structured messages and use chat completions. Template controls may be
sent under their real backend-owned names in `request_body`; InferLab locally
projects the effective template only to target and explain ISL. If projection
is unavailable or exact targeting is unsatisfiable, the unadjusted content is
sent and the fallback is recorded.

```toml
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 8192, output_tokens = 1024, shared_system_content = { ratio = 0.75 } }
request_body = { chat_template_kwargs = { enable_thinking = false } }
```

`shared_system_content` is a server-chat compatibility shape: it reserves
pre-template system-message content and an independent user suffix. Its ratio
must be strictly between zero and one. It is not exact final-prompt prefix
geometry, cannot be combined with `prefix_sharing`, and cannot be declared on a
weighted mixture.

Synthetic population evidence records the selected complete-prompt target,
request representation, prompt kind, realized pre-template content length,
local prediction, exact or fallback outcome, and template content and digest
when one is frozen or projected. Prefix evidence preserves the declaration,
resolved prefix and suffix lengths, canonical-stream digest, and exact frozen
population. Native request identities reconcile warmup and profiling requests
to their frozen population entries. For flat and rendered-chat prompts, every
completed profiling request must also reconcile the selected ISL with AIPerf's
backend-observed `input_sequence_length`; a mismatch fails the case. Dataset
admission bounds remain pre-template message-content limits and do not rewrite
source content. A successful Bench separately reports `mean_prompt_tokens`,
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

## Replaying a recorded population

Any earlier Bench record freezes its request population at
`.inferlab/records/<record-id>/cases/request-source/artifacts/population.jsonl`.
Copy that file into the workspace and pin its digest to replay the exact same
requests against another server or configuration:

```toml
[benches.replay-c8k1k]
kind = "serving"
request_source = { kind = "replay", path = "populations/c8k1k.jsonl", expected_sha256 = "<64-hex digest of the file>", prompt = { kind = "flat" } }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

`path` is workspace-relative (absolute paths and `..` escapes are rejected),
and `prompt` must be declared explicitly: `flat` (or `rendered_chat`) entries
carry a `text_input` string, `server_chat` entries carry structured
`messages`. `expected_sha256` is optional; when declared, preparation verifies
the file bytes before any request runs. The file is the sole population
authority: InferLab offers no selection, filtering, or transformation — edit,
subset, or reorder the file itself instead. Each entry keeps its own recorded
`output_length`; mixing `1` and larger values in one file is rejected, and a
file with fewer entries than the largest case requires fails preparation
rather than repeating entries. `prefix_sharing` follows the generated-source
semantics with the shared geometry resolved from the file entries. Dry-run
reports the path, declared and observed digests, and entry count without
running anything, and command-line overrides cannot change the replay `path`
or `expected_sha256`.

## Drawing random content from a text corpus

A `random` source can replace its synthetic hash-word prompts with slices of
an operator-supplied text corpus, for content-sensitive measurements such as
speculative-decoding acceptance rates:

```toml
[benches.corpus-c8k1k]
kind = "serving"
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024, corpus = { path = "corpus/shakespeare.txt", expected_sha256 = "<64-hex digest of the file>" } }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

`corpus.path` is workspace-relative (absolute paths and `..` escapes are
rejected, exactly like a replay population file), and `expected_sha256` is
optional; when declared, preparation verifies the corpus bytes before any
request runs. Pin the digest whenever the corpus identity matters — an edited
corpus silently changes the workload otherwise. Corpus content cannot be
inlined or fetched remotely, and command-line overrides cannot change the
corpus `path` or `expected_sha256`. Corpus slicing requires
`prompt = { kind = "flat" }` (the default), because entry content must be one
exact token-length slice of the corpus stream.

InferLab tokenizes the corpus once with the resolved model tokenizer and cuts
each entry as one slice of exactly its selected `input_tokens` length at an
offset determined by the Bench seed and the entry's population index, so the
same seed reproduces the same prompts and generating a larger population keeps
the same first entries. A corpus whose token stream is shorter than the
largest selected input target fails preparation. `prefix_sharing` keeps its
controlled semantics: the shared prefix is one fixed corpus slice and each
entry's unique suffix is drawn independently, with primed-cache conditioning
unchanged. Slices are drawn independently and may overlap; such incidental
sharing is natural reuse — it is not measured, promised, or presented as
controlled prefix geometry. The record preserves the corpus path, declared
and observed digests, and each entry's slice offset and length.

## Source preparation and cold-to-warm verification

Non-synthetic measurement sources are prepared before a recipe launches its
server. Release-catalog and AgentX sources must close and verify their immutable
content; a task-owned lm-eval source that cannot expose its complete closure is
recorded explicitly as opaque and non-reproducible. Source preparation is
separate from lm-eval, AIPerf, or InferLab population materialization, so its
time and cache outcome are not charged to an arbitrary measurement case.

For a workspace lm-eval YAML using a file-backed `json`, `csv`, `parquet`,
`text`, or `arrow` loader, InferLab snapshots the YAML include closure and
workspace-local `data_files` before serving starts. Exact paths, lists, split
mappings, and file globs are expanded into the recorded ordered closure, and
the Eval client receives a generated task YAML bound only to the read-only
snapshot. Remote selectors, paths outside the workspace, and task function
references remain explicit opaque sources because preparation cannot bind
their complete file closure.

Use an isolated cache to exercise a real cold preparation followed by verified
reuse without changing the recipe or measurement definitions:

```sh
INFERLAB_ASSET_E2E_CACHE=$(mktemp -d)
XDG_CACHE_HOME="$INFERLAB_ASSET_E2E_CACHE" inferlab recipe run <RECIPE> > cold-recipe.json
XDG_CACHE_HOME="$INFERLAB_ASSET_E2E_CACHE" inferlab recipe run <RECIPE> > warm-recipe.json
```

Choose a recipe whose suite contains a release-qualified dataset or AgentX
source. Both records must report `source_preparation_completed = true` and
`serving_launch_attempted = true`. In `data_assets`, the same selection must
have the same `selection_key_sha256`; the cold attempt's terminal preparation
phase reports downloaded source bytes and a cache miss, while the warm attempt
reports reused source bytes and a verified full hit. The terminal `ready`
attempt must retain a closed content list and its expected-versus-observed
verification. Compare the two records' data-asset `selection`, `consumers`, and
the selected workload definitions under `resolved.measurements` to confirm
that only the cache outcome changed. A dry-run may inspect an existing local
path, but never downloads, verifies, or claims source readiness.

For a manual Bench, the same preparation occurs before its first inference
request. A preparation failure leaves the selected managed server running and
records `target_server_unchanged = true` in the Bench record's standalone
`data_assets` evidence.

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

## SemiAnalysis AgentX trace replay

AgentX is a third, deliberately closed static serving source. It is neither a
population of independent prompts nor a linear live-response session:

```toml
[benches.agentx]
agentic_source = { dataset = "semianalysis_agentx_062126_256k", profile = "inferencex" }
concurrency = [1]
timeout_seconds = 7200
```

`concurrency` counts root session-tree lanes. Spawned subagents can therefore
produce more simultaneous HTTP requests than the declared value. The
`inferencex` release profile fixes source-response replay, first-turn-prefix
cache busting, trajectory sampling, a 600-second cache-pressure warmup,
streaming chat requests, native failure thresholds, and a 900-second minimum
profiling duration. Omitting `duration_seconds` selects 1800 seconds. Live
server responses are measured but do not become the context for later source
turns, so this workflow measures replay transport behavior rather than agent
task quality.

The 256k corpus is approximately 569 MB. The full-context
`semianalysis_agentx_062126` corpus is approximately 1.85 GB. InferLab verifies
the immutable Hugging Face revision and complete `traces.jsonl` digest before
AIPerf materializes the trace trees. The release profile, not workspace fields,
owns loader, scenario, timing, warmup, cache-bust, and failure-policy details.
AgentX rejects request counts and rates, prompt and request-body controls,
linear-session counts and delays, cache-start controls, SLOs, and adaptive
serving. Use `inferlab workspace show --json` and dry-run to inspect the closed
effective policy before downloading or sending traffic.
