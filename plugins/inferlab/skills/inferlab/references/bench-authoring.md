# Serving Bench load, sources, sessions, metrics, and SLOs

- [Minimal definitions](#minimal-definitions)
- [Load: concurrency, rate, and adaptive](#load-concurrency-rate-and-adaptive)
- [Warmup](#warmup)
- [Prefix-cache start](#prefix-cache-start)
- [Metrics](#metrics)
- [SLOs](#slos)
- [Request sources](#request-sources): synthetic prompts, prefix geometry,
  mixtures, server chat, datasets, [replay](#replaying-a-recorded-population),
  [corpus](#drawing-random-content-from-a-text-corpus), and
  [images](#attaching-images-to-random-requests)
- [Source preparation and cold-to-warm verification](#source-preparation-and-cold-to-warm-verification)
- [Linear serving sessions](#linear-serving-sessions)
- [SemiAnalysis AgentX trace replay](#semianalysis-agentx-trace-replay)

## Minimal definitions

Start with the smallest definition that expresses the workload. `kind`
defaults to `serving`, and a `random` or `random_mixture` source defaults to
an exact `flat` completion prompt, so fixed ISL and OSL need no prompt table:

```toml
[benches.fixed-8k1k]
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

An untagged `{ min, max }` table selects an inclusive-uniform integer range:

```toml
[benches.range-8k1k]
request_source = { kind = "random", input_tokens = { min = 6553, max = 8192 }, output_tokens = { min = 819, max = 1024 } }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

These defaults are not hidden state: `inferlab workspace show --json` renders
them as `serving`, `flat`, and the tagged
`{ kind = "inclusive_uniform", min, max }` form, which is also accepted as
input. Add the controls below only when the workload needs them.

`request_body` follows the shared Eval/Bench rule in
[eval-authoring.md](eval-authoring.md#request-body). A Bench fragment
additionally may not declare `min_tokens`, `min_new_tokens`, or `ignore_eos`,
because InferLab derives output-length enforcement from `output_stop`.

### Exact length or end of sequence

`output_stop` on a `random`, `random_mixture`, `dataset`, or `session_source`
table chooses how the output-token limit is enforced:

- `"length"` (default): the limit is an exact length. Each request sets a
  matching minimum and ignores end of sequence.
- `"eos"`: the limit is a cap. The model stops at its natural end of sequence
  or at the limit, whichever comes first.

```toml
request_source = { kind = "dataset", dataset = "speed_bench", profile = "qualitative_qa", max_input_tokens = 8192, output_tokens = 1024, output_stop = "eos" }
```

Choose `"eos"` when the measurement depends on natural response length, such
as speculative-decoding acceptance; SPEED-Bench's own methodology does not
suppress end of sequence. Under `"eos"` a response can end after one token, so
per-token timing covers fewer tokens, and results are not comparable with
`"length"` runs. A `replay` source has no `output_stop`; see
[Replaying a recorded population](#replaying-a-recorded-population).

## Load: concurrency, rate, and adaptive

A static `serving` Bench sends either independent requests or dependent
sessions. Its case matrix holds concurrency cases, request-rate cases, or
both:

- Concurrency cases: `concurrency` with `prompts_per_concurrency` (or
  `sessions_per_concurrency` for sessions).
- Request-rate cases (independent requests only): `request_rates` (positive
  numbers or `"inf"`) with exactly one of `request_count` or
  `duration_seconds`. An `"inf"` rate cannot be combined with
  `duration_seconds`. `burstiness` shapes stochastic arrival.

```toml
[benches.rate-sweep]
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024 }
request_rates = [2.0, "inf"]
request_count = 64
timeout_seconds = 900
```

An `adaptive-serving` Bench declares positive finite `initial_request_rates`,
at least one aggregate or request SLO, `max_search_steps`, exactly one of
`request_count` or `duration_seconds`, and optionally `min_rate_resolution`;
see [SLOs](#slos) for an example and the search policy.

## Warmup

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

- For concurrency `c`, warmup sends `c * warmup_prompts_per_concurrency`
  requests drawn from a disjoint prefix of the frozen request population.
- Warmup shares the profiled route, request source, request body, and
  concurrency, drains before profiling, and is excluded from normalized
  metrics and profiling request counts.

## Prefix-cache start

`cache.start` selects the prefix-cache state profiling begins from. It
defaults to `uncontrolled` (no reset or conditioning, and the record claims no
known cache state); declare another start only when the experiment needs it:

```toml
cache = { start = "cold" }
```

How a `cold` or `primed` start keeps earlier content from being reused depends
on who owns the content:

- **Generated populations** (`random`, `random_mixture`) in a static matrix
  are **namespaced**; nothing is reset. Every execution draws a nonce, and each
  case reads its own population: warmup entries from the case's warmup
  namespace, profiling entries from its profiling namespace. No case, phase,
  earlier run, or other Bench of the run shares content with another, while
  the seed keeps the geometry (lengths, prefix lengths, groups) identical
  across runs. `cold` therefore means nothing the window could reuse exists
  when it opens, not that the cache is empty: it may hold unrelated entries.
  The server needs no reset capability, so this works on servers whose
  integration declares none. The dry-run shows the nonce, and the record keeps
  it with each case's namespaces, so every prompt can be regenerated.
- **Replay, dataset, and linear-session sources, and adaptive searches** keep
  resetting: the server must expose a prefix-cache reset (the public endpoint's
  reset action or a per-target reset of every model-serving replica), and
  profiling starts right after a successful reset.

`primed` additionally requires a `flat` or `rendered_chat` source with a
positive `prefix_sharing` length. After the reset, or directly for a namespaced case,
InferLab sends the canonical shared prefix through the maximum resolved shared
length — the case's profiling-namespace prefix when namespaced — with one
output token and the profiling route, model, and `request_body`, then releases
profiling. With `groups`, every group's prefix is sent, the first group first.
A rendered prefix that does not tokenize exactly as an independent prompt fails
preparation.

The [backend support matrix](backend-support.md) owns which integrations,
proxies, Gateways, and routers declare reset, conditioning fan-out, and
cache-read reporting, and which builds honor rank pinning.

### Which conditioning requests a primed start sends

A primed start must reach every cache-owning target: each prefill replica and
each attention data-parallel rank of that replica. What InferLab sends depends
on the server shape:

| Server shape | Conditioning |
| --- | --- |
| Engines share one prefix cache through a KV store | One target. One unpinned request through the ordinary serving flow, then a recorded wait for the store's publication bound (after the last group, with `groups`). |
| Direct Engine endpoint | One request per attention data-parallel rank of the public role, each pinned with the `X-Data-Parallel-Rank` header (a single rank needs no header). A build that ignores the header leaves ranks cold. |
| Gateway, one cache-owning target | One unpinned request through the ordinary serving flow; no fan-out capability needed. |
| Gateway, more than one target | The frontend's conditioning fan-out sends one pinned request per replica and rank. A frontend without a declared fan-out is rejected. |

The record keeps each (replica, rank) request's status, token usage, and
timing; any rank's failure fails the case. Decode-side cache state produced by
conditioning is kept as evidence, not claimed as coverage.

When the target count follows from the declared server shape, planning rejects
an unconditionable primed start. When it depends on facts an Engine reports
only at readiness, such as a Specialized Engine's data-parallel rank count or
its KV store access, InferLab applies the reported facts after the server is
ready: an unconditionable primed start then fails before any reset or
conditioning request, naming the reported counts, and the publication wait
uses the reported bound.

### Reset retries, timeouts, and ordering

- A backend may decline a reset it cannot perform yet (for example while KV
  blocks are still held for an unfinished transfer) by answering 2xx with the
  opposite boolean of the reset action's success predicate; vLLM answers
  `success: false`. InferLab retries after a growing delay until the reset
  succeeds, an attempt fails another way, or the case timeout is spent. A
  per-target reset retries only its declined targets; a public reset,
  including one fanned out by a built-in prefill/decode proxy, retries the
  whole action. Any other predicate mismatch (empty or unparseable body,
  missing value, mismatched string) fails the case at once.
- Every attempt is kept under `cache_preparation.reset.attempts`. A long run
  of declines points at engine state left over from the previous case.
- A reset, flush, or conditioning request sent through the built-in
  prefill/decode proxy has no per-target cap of its own: the case's remaining
  `timeout_seconds` bounds the whole fan-out, and the proxy cancels in-flight
  target work when the caller gives up.
- Reset and conditioning run after warmup, are excluded from profiling
  metrics, and consume no population entries.
- The one case timeout covers warmup, reset, conditioning, profiling, and
  result handling; cleanup keeps its own grace. A capture window opens after
  cache preparation and before profiling release; a warmup or
  cache-preparation failure leaves it unopened.

## Metrics

Every successful Bench reports `request_throughput`, `output_throughput`, and
`total_token_throughput`, plus `mean`, `min`, `max`, `stddev`, `p50`, `p90`,
`p95`, and `p99` for each of `request_latency_ms`, `ttft_ms`, and `tpot_ms`
(names such as `p95_tpot_ms`). TPOT metrics are omitted for an
`output_tokens = 1` prefill-dominant workload.

The same statistics of AIPerf's backend-observed `input_sequence_length` are
reported as `mean_prompt_tokens`, `p99_prompt_tokens`, and so on. They include
the server-side template and are the token authority behind total-token
throughput.

### Cache-read metrics

A Bench with `prefix_sharing`, `shared_system_content`, or a primed start
requires backend prompt-token and cache-read-token usage on every completed
profiling request, and reports:

- `prompt_cache_read_tokens` and `uncached_prompt_tokens` with the same
  distribution statistics;
- `prompt_cache_read_ratio = sum(cache_read_tokens) / sum(prompt_tokens)`.

The endpoint must declare the backend cache-read capability; the
[backend support matrix](backend-support.md) names the server setting each
backend needs, and a prefill/decode frontend declares it only when both roles
enable that setting. A primed or `prefix_sharing` Bench against an endpoint
without it fails at planning with an error naming the Bench, the missing
capability, and the remediation (enable the reporting setting and rebuild the
server). An integration may declare that its endpoint omits a zero read, as
SGLang's does; InferLab then records the omission as zero, while an undeclared
missing value fails normalization. Other Benches report AIPerf's aggregate
cache ratio when available.

A `cold` or `primed` case over a generated `prefix_sharing` population also
records each measured request's reuse opportunity: how many of its
shared-prefix tokens an earlier request of its prefix group sent after the
reset (conditioning included), and whether that prefix was `first` (nobody had
sent it), `in_flight` (sent, not yet published), or `available` (an earlier
request's first output token plus the store's declared or reported
publication bound, zero without a store, had passed). From these come:

- `available_miss_ratio`: the share of `available` requests that read no
  cached tokens;
- `available_reuse_efficiency`: cached tokens read by `available` requests
  over their reusable tokens.

Both are omitted when no request is `available`. With one cache-owning
target, an `available` miss means the cache did not serve what it could; with
several, routing may have sent the request to a target that never saw the
prefix. Engine-specific ceilings such as block rounding are left to whoever
reads the per-request evidence.

### Goodput, server metrics, and speculative decoding

- `good_request_ratio` and `goodput` are derived only when a request SLO is
  configured.
- `server_metrics = true` asks AIPerf to collect the server's declared JSON
  metrics export. It is accepted only when the integration/topology declares a
  metrics endpoint, which may sit on the public serving port or on a named port
  the integration already requires; InferLab freezes the exact URL, keeping
  framework routes such as `/metrics` or `/v1/metrics`. Warmup and cache
  preparation stay outside the exported window. The
  [backend support matrix](backend-support.md) lists which endpoints declare a
  metrics export and what each needs.
- A successful `speed_bench` case publishes `acceptance_length` and
  `acceptance_rate` from AIPerf's SPEED-Bench computation, taking each from the
  first of AIPerf's per-request records, run summary, and scraped server
  metrics that yields one, and recording which. Other sources keep raw server
  metrics but do not publish these two scalars.
- When the backend reports accepted and rejected prediction tokens in
  `usage.completion_tokens_details` (SMG does for a speculatively decoding
  Specialized Engine), any serving Bench also publishes
  `draft_accepted_tokens`, `draft_tokens`, and `draft_acceptance_rate`, the
  token-weighted share of drafted tokens accepted across the profiling window.
  They are omitted when no request reports drafts. Usage carries no
  verify-step count, so no acceptance length is derived from it.

### Artifact level

`artifact_level = "performance"` skips AIPerf's full raw request/response
export while keeping normalized per-request records and the summary; the
default `"diagnostic"` keeps the raw export. Warmup validation and population
reconciliation work at either level. Session and agentic cases stay valid at
`performance`; their raw-derived evidence is recorded as unavailable due to
the artifact level.

## SLOs

A static or adaptive serving Bench may constrain normalized aggregate metrics,
individual request latency, or both. Aggregate constraints are inclusive and
AND-composed. A request counts as good only when every configured latency
bound passes, and `minimum_good_request_ratio` gates the case.

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

Adaptive Bench uses `highest-feasible-rate-v1`: it measures every initial
rate, then uses bounded doubling and directional bisection, and selects the
highest observed feasible rate; it does not claim an unmeasured optimum.
`max_search_steps` counts only automatically added probes, never the declared
initial list. Use `--set` to override SLO values for one run without changing
the stored definition.

## Request sources

Every serving Bench selects one closed request source.

### Synthetic prompts: flat and rendered chat

A `random` source declares its complete profiling prompt shape. `flat` (the
default) sends tokenizer-exact scalar prompts to completions:

```toml
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = 8192, output_tokens = 1024 }
```

InferLab draws ISL and OSL independently from their ranges under the Bench
seed and freezes the population before requests start. Extending a case
preserves the existing sequence prefix. An OSL range cannot span both one
token and two or more tokens.

`rendered_chat` freezes the model tokenizer's default chat template, or one
definition-supplied template and kwargs. InferLab renders each complete prompt
once during population preparation, verifies its exact token length, and sends
the scalar prompt to completions. Template controls belong only to `prompt`;
repeating them in `request_body` is a validation error, and failure to resolve
or satisfy the template is fatal rather than a fallback.

```toml
request_source = { kind = "random", prompt = { kind = "rendered_chat", chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}", chat_template_kwargs = { enable_thinking = false } }, input_tokens = 8192, output_tokens = 1024 }
```

### Prefix geometry

Flat and rendered-chat sources may declare exact final-prompt prefix geometry
as fixed tokens (`shared_prefix_tokens`) or a per-entry ratio
(`shared_prefix_ratio`):

```toml
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = { min = 7000, max = 9000 }, output_tokens = 1024, prefix_sharing = { shared_prefix_ratio = 0.75 } }
```

For selected input length `I`, a ratio resolves to `floor(I * ratio)` shared
tokens plus an independently generated `I - shared` suffix. Fixed tokens and
equivalent ratios produce the same seeded population. All entries use nested
prefixes of one canonical stream, including under distributed ISL. Both `0.0`
(independent prompts) and `1.0` (full-prompt sharing) are valid. This is
prompt geometry, not a requested or observed cache-hit percentage; backend
cache-read metrics are separate evidence.

Production reuse usually comes from many distinct prefixes (system prompts,
tool definitions, few-shot blocks), a few hot and most rarely used. `groups`
splits the population over that many independent canonical prefixes, and
`popularity` sets how often each is drawn:

```toml
prefix_sharing = { shared_prefix_tokens = 4096, groups = 32, popularity = { zipf = 1.1 } }
```

Each entry draws its group under the Bench seed, independently of its input
length; the group of rank `k` (from one) is drawn with probability
proportional to `1 / k^s`. `popularity` defaults to `uniform`; a Zipf exponent
must be positive and needs at least two groups. Groups share no controlled
prefix, and one group reproduces the ungrouped population. Population evidence
records each entry's `prefix_group`, and the geometry summary records
`group_entries` per group.

### Weighted shape mixtures

`random_mixture` samples several exact ISL/OSL pairs:

```toml
request_source = { kind = "random_mixture", prompt = { kind = "flat" }, shapes = [
  { input_tokens = 1024, output_tokens = 128, weight = 7 },
  { input_tokens = 8192, output_tokens = 1024, weight = 3 },
] }
```

Each request selects one shape with probability `weight / sum(weight)` under
the Bench seed; weights are proportions, not quotas. Shapes must be distinct,
and their output lengths must all equal one or all be at least two. Every case
starts from the same deterministic sequence, with warmup and profiling taking
consecutive, non-overlapping entries.

### Server-rendered chat

`server_chat` keeps chat-template authority on the server: requests stay
structured messages on chat completions, and template controls go in
`request_body` under their backend-owned names. InferLab projects the
effective template locally only to target and explain ISL; when projection is
unavailable or exact targeting is unsatisfiable, the unadjusted content is
sent and the fallback is recorded.

```toml
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 8192, output_tokens = 1024, shared_system_content = { ratio = 0.75 } }
request_body = { chat_template_kwargs = { enable_thinking = false } }
```

`shared_system_content` reserves shared pre-template system-message content
and an independent user suffix, declared as `{ ratio = R }` (strictly between
zero and one) or `{ tokens = N }`. It is not exact final-prompt prefix
geometry: it cannot be combined with `prefix_sharing`, used by a primed start,
or declared on a weighted mixture.

### Population evidence

Synthetic population evidence records the complete-prompt target, request
representation, prompt kind, realized pre-template content length, local
prediction, exact or fallback outcome, and template content and digest when
one is frozen or projected. Prefix evidence keeps the declaration, resolved
prefix and suffix lengths, canonical-stream digest, and the frozen population.
Native request identities reconcile warmup and profiling requests to their
population entries. For flat and rendered-chat prompts, every completed
profiling request's selected ISL must match AIPerf's backend-observed
`input_sequence_length`, or the case fails. Dataset admission bounds are
pre-template message-content limits and never rewrite source content.

### ShareGPT

The release catalog pins an immutable snapshot of the Apache-2.0
[ShareGPT Vicuna dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
as a bounded conversational source; dry-run reports the pinned revision:

```toml
request_source = { kind = "dataset", dataset = "sharegpt", max_input_tokens = 8192 }
```

InferLab downloads the snapshot on first execution, verifies its digest, and
reuses it from `$XDG_CACHE_HOME/inferlab/datasets/sha256/<digest>` (normally
`~/.cache/inferlab/datasets/sha256/<digest>`). Dry-run reports catalog and
cache state but never downloads.

Each selected conversation becomes one independent chat-completions request,
with the final assistant message held out to derive the output limit. If the
pre-template content exceeds `max_input_tokens`, InferLab drops complete
trailing user/assistant exchanges until an earlier target fits; it never
truncates a message or drops leading history. `output_tokens` inside the table
replaces target-derived lengths (`output_tokens = 1` gives a prefill-dominant
run). The Bench `seed` controls sampling without replacement. Overrides may
change fields within the source but not `request_source.kind`.

### SPEED-Bench

The catalog also exposes NVIDIA SPEED-Bench profiles, each mapped to an AIPerf
SPEED-Bench dataset, a pinned content digest, and a category filter.
Qualitative profiles have names such as `qualitative_coding`; throughput
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

SPEED-Bench publishes placeholders for prompts whose source datasets forbid
redistribution. Before the first case, AIPerf resolves them from those
sources, and InferLab holds the resolved rows to the catalog digest, failing
preparation on a mismatch. Resolution downloads the sources once into the
Hugging Face cache and needs an account that has accepted the terms of the
gated `cais/hle` dataset; without it, preparation fails with AIPerf's message.
InferLab then samples without replacement and keeps only each row's first
user turn as an independent request; later turns are recorded as omitted.

## Replaying a recorded population

An independent-request Bench record (`random`, `random_mixture`, `dataset`, or
`replay` source) freezes its population at
`.inferlab/records/<record-id>/cases/request-source/artifacts/population.jsonl`.
Copy that file into the workspace and pin its digest to replay the same
requests against another server or configuration. Linear-session records store
turn templates and AgentX records freeze no population, so neither can be
replayed this way.

```toml
[benches.replay-c8k1k]
kind = "serving"
request_source = { kind = "replay", path = "populations/c8k1k.jsonl", expected_sha256 = "<64-hex digest of the file>", prompt = { kind = "flat" } }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

- `path` is workspace-relative; absolute paths and `..` escapes are rejected.
  `expected_sha256` is optional; when declared, preparation verifies the file
  before any request runs.
- `prompt` is required: `flat` (or `rendered_chat`) entries carry a
  `text_input` string, `server_chat` entries carry structured `messages`.
- The file is the sole population authority. InferLab does not select,
  filter, or transform it; edit, subset, or reorder the file instead. A file
  with fewer entries than the largest case needs fails preparation rather than
  repeating entries.
- Each entry keeps its recorded `output_length`; mixing `1` and larger values
  in one file is rejected.
- `prefix_sharing` follows the generated-source semantics, with geometry
  resolved from the entries. `groups` and `popularity` cannot be declared: a
  replayed population brings its own prefixes.
- Dry-run reports the path, declared and observed digests, and entry count.
  Overrides cannot change `path` or `expected_sha256`.

Every replayed request starts from the exact-length default (end of sequence
suppressed), and each entry's `extra` members override it member by member.
An entry without a stop member generates exactly its `output_length`. To stop
at end of sequence with `output_length` as a cap, set
`"extra": {"ignore_eos": false}`; other members, such as a per-entry
`temperature`, pass through the same way. Populations InferLab writes under
`output_stop = "eos"` carry `"ignore_eos": false` and replay as recorded;
populations exported under `eos` by earlier releases lack it and replay at
full length until you add it or export them again. Population evidence
records each entry's resolved `output_stop`.

## Drawing random content from a text corpus

A `random` source can replace its synthetic prompts with slices of an
operator-supplied text corpus, for content-sensitive measurements such as
speculative-decoding acceptance:

```toml
[benches.corpus-c8k1k]
kind = "serving"
request_source = { kind = "random", input_tokens = 8192, output_tokens = 1024, corpus = { path = "corpus/example.txt", expected_sha256 = "<64-hex digest of the file>" } }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

- `corpus.path` follows the replay path rules, and `expected_sha256` is
  optional. Pin it whenever corpus identity matters: an edited corpus
  otherwise silently changes the workload. Corpus content cannot be inlined
  or fetched remotely, and overrides cannot change `path` or
  `expected_sha256`.
- Corpus slicing requires `prompt = { kind = "flat" }` (the default), because
  each entry is one exact token-length slice of the corpus stream.
- InferLab tokenizes the corpus once with the resolved model tokenizer and
  cuts each entry at an offset determined by the Bench seed and the entry's
  population index, so the same seed reproduces the same prompts and a larger
  population keeps the same first entries. A corpus shorter than the largest
  selected input target fails preparation.
- With `prefix_sharing`, the shared prefix is one fixed corpus slice and each
  suffix is drawn independently; primed conditioning is unchanged. Slices may
  overlap; that incidental sharing is not measured or presented as controlled
  prefix geometry.
- The record keeps the corpus path, declared and observed digests, and each
  entry's slice offset and length.

## Attaching images to random requests

A `random` source can attach images to every request for vision-language
servers:

```toml
[benches.vlm-random]
kind = "serving"
request_source = { kind = "random", input_tokens = 512, output_tokens = 128, images = { width = 512, height = 384, count = 2 } }
concurrency = [1, 4]
prompts_per_concurrency = 4
timeout_seconds = 900
```

- `width` and `height` are required fixed pixel values; `count` defaults to
  one image per request. Distribution spellings are rejected.
- `input_tokens` keeps its text meaning; image tokens ride on top and show up
  in backend-reported prompt tokens.
- Images force the `server_chat` prompt (an omitted `prompt` resolves to it;
  `flat` or `rendered_chat` is rejected), because image content parts exist
  only on chat completions. Other source kinds reject `images`.

The measurement runtime synthesizes noise images by default. To load real
images, declare one workspace-relative directory:

```toml
request_source = { kind = "random", input_tokens = 512, output_tokens = 128, images = { width = 512, height = 384, source = { path = "corpora/vlm-smoke-images", expected_sha256 = "<64-hex directory digest>", sampling = "shuffle-cycle" } } }
```

The path follows the corpus rules; `expected_sha256` binds the directory
through InferLab's recursive enumeration digest (regular files ordered by
relative path). A missing or empty directory, or a digest mismatch, fails
preparation. `sampling` is `random-with-replacement`, `shuffle-cycle`
(default), or `sequential-cycle`. Overrides cannot change the source `path`,
digest, or `sampling`; dimensions and `count` remain overridable. Images are
attached at run time in a sequence the Bench seed determines; frozen
populations stay text-only, and the record keeps the effective policy and the
directory's observed digest, not per-request image identities.

## Source preparation and cold-to-warm verification

Non-synthetic sources are prepared before a recipe launches its server, or
before a manual Bench's first request. Release-catalog and AgentX sources must
close and verify their immutable content; a task-owned lm-eval source that
cannot expose its complete closure is recorded as opaque and
non-reproducible. Preparation is separate from lm-eval, AIPerf, or population
materialization, so its time and cache outcome are not charged to a
measurement case. [eval-authoring.md](eval-authoring.md#source-preparation)
describes how a workspace lm-eval YAML's file closure is snapshotted. A failed
manual-Bench preparation leaves the managed server running and records
`target_server_unchanged = true` in the Bench record's `data_assets`.

To exercise a real cold preparation followed by verified reuse, point every
cache binding preparation honors into a scratch directory. Hugging Face
downloads, including AgentX corpora, follow `HF_HOME` and `HF_HUB_CACHE`
before `XDG_CACHE_HOME`:

```sh
C=$(mktemp -d)
export XDG_CACHE_HOME="$C" HF_HOME="$C/hf" HF_HUB_CACHE="$C/hf/hub" HF_DATASETS_CACHE="$C/hf/datasets"
inferlab recipe run <RECIPE> > cold-recipe.json
inferlab recipe run <RECIPE> > warm-recipe.json
```

Use a recipe whose suite contains a release-qualified dataset or AgentX
source. Then check:

- Both records report `source_preparation_completed = true` and
  `serving_launch_attempted = true`.
- In `data_assets`, the selection has the same `source_key_sha256` in both.
  The cold attempt's terminal phase reports downloaded bytes and a cache miss;
  the warm attempt reports reused bytes and a verified full hit.
- The terminal `ready` attempt keeps a closed content list and its
  expected-versus-observed verification.
- The data-asset `source`, `effective_selection`, and `consumers`, and the
  workload definitions under `resolved.measurements`, are identical, so only
  the cache outcome changed.

A dry-run may inspect an existing local path but never downloads, verifies,
or claims source readiness.

## Linear serving sessions

A linear session Bench selects a qualified conversational source instead of
an independent `request_source`:

```toml
[benches.linear-chat]
kind = "serving"
session_source = { dataset = "sharegpt", max_input_tokens = 8192, inter_turn_delay_scale = 1.0, max_inter_turn_delay_seconds = 3.0 }
concurrency = [8]
sessions_per_concurrency = 4
warmup_sessions_per_concurrency = 1
timeout_seconds = 900
```

Each later user turn waits for the preceding live assistant response and its
effective inter-turn delay. Concurrency counts active conversations, delay
intervals included, not HTTP requests. Positive warmup runs complete
AIPerf-native sessions and drains before profiling. AIPerf adds its native
`warmup` system marker to warmup transport messages; InferLab keeps that
phase-tagged request and its observed prompt-token count, while excluding all
warmup traffic from profiling metrics. Native `session_num` values restart per
phase, so a request's durable identity is the pair of phase and
`session_num`.

## SemiAnalysis AgentX trace replay

AgentX is a third, closed static serving source: neither a population of
independent prompts nor a linear live-response session.

```toml
[benches.agentx]
agentic_source = { dataset = "semianalysis_agentx_062126_256k", profile = "inferencex" }
concurrency = [1]
timeout_seconds = 7200
```

`concurrency` counts root session-tree lanes; spawned subagents can produce
more simultaneous HTTP requests than that. The `inferencex` release profile
fixes source-response replay, first-turn-prefix cache busting, trajectory
sampling, per-trace and global caps on recorded idle gaps, a per-lane
cache-pressure warmup, streaming chat requests, native failure thresholds, and
minimum and default profiling durations. Idle periods longer than the caps are
shortened, so replay timing differs from the raw trace. Omitting
`duration_seconds` or `seed` selects the profile's default, not the ordinary
Bench seed default of 0. Live responses are measured but do not become context
for later turns, so this measures replay transport behavior, not agent task
quality.

Choose the corpus the served context length can hold:
`semianalysis_agentx_062126_256k` for a context limit of at least 256k tokens,
or the full-context `semianalysis_agentx_062126` for servers with a larger
native context. InferLab does not compare the corpus with the server's context
length: AIPerf records over-length requests as context overflows, excludes
them from latency distributions, and judges the overflow rate against its
native scenario threshold; an invalid native submission fails the case.

AIPerf decides whether a warmup failure aborts the case, as InferenceX runs
do: any failed root-stream warmup request, including a single cache-pressure
request, aborts before profiling, while a failed subagent warmup request does
not by itself. Warmup failures are recorded either way.

Both corpora are sizable downloads. Source preparation verifies the immutable
Hugging Face revision and the complete `traces.jsonl` digest before the server
launches (or before a manual Bench's first request). AIPerf then validates and
materializes the trace trees when each case's client starts, inside that
case's `timeout_seconds` together with warmup, profiling, and result handling.
Budget the timeout for all of them.

The release profile, not workspace fields, owns loader, scenario, timing,
warmup, cache-bust, and failure policy. Besides `agentic_source`, an AgentX
Bench accepts only `concurrency`, `duration_seconds` (at least the profile
minimum), `timeout_seconds`, `seed`, `server_metrics`, and `artifact_level`.
It rejects request counts and rates, prompt and request-body controls,
linear-session counts and delays, cache-start controls, SLOs, and adaptive
serving. `inferlab workspace show --json` shows only the dataset and profile
names; use dry-run (`recipe run --dry-run`, or `bench --dry-run` against a
server record) to inspect the closed effective policy before downloading or
sending traffic.
