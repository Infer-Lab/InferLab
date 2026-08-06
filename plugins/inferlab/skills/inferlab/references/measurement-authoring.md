# Eval, Bench, datasets, sessions, metrics, and SLOs

## Ordinary measurement path

Start with the smallest definition that expresses the workload. The built-in
OpenAI smoke defaults to prompt `Hello`, 16 maximum output tokens, and a
60-second timeout:

```toml
[evals.smoke]
kind = "openai-smoke"
```

A static synthetic Bench defaults to `serving`. A `random` or
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
`flat`, tagged `inclusive_uniform`, and the effective smoke values. Existing
explicit forms remain valid. Add the advanced controls below only when the
workload needs their distinct semantics.

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

`openai-smoke` is the smallest completion-path correctness Eval. An lm-eval
definition controls its request fragment, sample limit, few-shot count, seed,
trials, output bound, concurrency, selected metric and optional filter,
threshold, and timeout while the task retains dataset and scoring authority.

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
