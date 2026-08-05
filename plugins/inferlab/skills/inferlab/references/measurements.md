# Eval And Bench Measurements

Install the release-owned measurement runtimes before the first lm-eval or
Bench execution:

```sh
inferlab toolchain install
```

The installed runtime is fixed by the InferLab product release. Serving
workspaces do not declare or install its internal measurement SDK.

## Eval

`openai-smoke` is the smallest completion-path correctness gate. An `lm-eval`
definition selects exactly one task as a pinned lm-eval name, a release-bundled
task, or a tracked workspace YAML. The task owns dataset, split, prompt, output
type, filters, and scorer. The definition controls `request_body`, `limit`,
`few_shot`, `seed`, `trials`, `max_tokens`, `concurrency`, selected `metric`,
optional `metric_filter`, `threshold`, and `timeout_seconds`.

Generative tasks use chat completions. Prompt-logprob task types use completions
only after a tokenizer/logprob alignment probe. The resolved model-weight
locator is the sole tokenizer source. The smoke path uses completions and no
request-body fragment.

`request_body` carries backend inference parameters such as temperature,
reasoning effort, or chat-template kwargs. It cannot override structural facts
owned by InferLab or the task: model, prompt/messages, route, streaming,
one-completion policy, output bound, stop, or the repeated-trial seed schedule.

```sh
inferlab recipe run qualify \
  --set evals.reasoning.trials=5 \
  --set evals.reasoning.request_body.temperature=1.0 \
  --set 'evals.reasoning.request_body.reasoning_effort="high"' \
  --set evals.reasoning.request_body.chat_template_kwargs.enable_thinking=true \
  --dry-run
```

## Static And Adaptive Serving Bench

A `serving` Bench uses either independent `request_source` entries or dependent
`session_source` conversations. It may define concurrency cases with
`prompts_per_concurrency` or request-rate cases with `request_count` or
`duration_seconds`; request rates accept positive numbers or `"inf"`.
`burstiness` controls supported stochastic arrival shaping. One definition does
not mix incompatible load authorities.

Warmup fields are phase-specific: `warmup_prompts_per_concurrency` for
independent requests and `warmup_sessions_per_concurrency` for sessions. Warmup
uses the same route, source, body, and load, consumes a disjoint prefix of the
frozen population, and is excluded from normalized profiling counts and
metrics. `reset_prefix_cache = true` requests one integration-declared reset
before warmup. The case `timeout_seconds` covers reset, warmup, profiling, and
result handling.

An `adaptive-serving` Bench declares positive `initial_request_rates`, one or
more aggregate/request SLO constraints, `max_search_steps`, and optional
`min_rate_resolution`. Its `highest-feasible-rate-v1` policy measures every
initial rate, then performs bounded doubling and directional bisection. It
selects only the highest measured feasible rate and does not claim an
unobserved optimum.

Manual and recipe execution consume the same resolved Bench:

```sh
inferlab bench <BENCH> --serve <SERVER_RECORD_ID> [--set PATH=VALUE] [--dry-run]
inferlab recipe run <RECIPE> --set 'benches.<BENCH>.concurrency=[1,8]'
```

## Prompt Authority And Synthetic Shape

Synthetic `random` and `random_mixture` sources must select one prompt kind:

- `flat`: create an exact tokenizer-length scalar prompt and use completions;
- `rendered_chat`: freeze the tokenizer's default chat template or one
  definition-supplied `chat_template` plus `chat_template_kwargs`, verify the
  exact final length, then use completions; or
- `server_chat`: send structured messages through chat completions and leave
  final rendering with the server. Local projection targets and explains the
  requested length; an unavailable or unsatisfiable projection records a
  fallback rather than changing route.

For `rendered_chat`, template controls belong only under `prompt`; duplicating
them in `request_body` is invalid. Template resolution or exact construction
failure is fatal. For `server_chat`, backend template controls retain their
real names under `request_body`; a server with no usable template fails rather
than falling back to completions.

`input_tokens` and `output_tokens` accept a fixed integer or
`{ kind = "inclusive_uniform", min = ..., max = ... }`. ISL and OSL are drawn
independently under the Bench seed and frozen before traffic. An OSL range may
not span both one token and two-or-more tokens because TPOT applicability would
be ambiguous.

`random_mixture` uses distinct exact `{ input_tokens, output_tokens, weight }`
shapes. Weights are per-request sampling proportions, not quotas. Every case
starts from the same deterministic source sequence; warmup and profiling take
consecutive non-overlapping entries.

Exact `flat` and `rendered_chat` sources may declare one final-prompt prefix:

```toml
prefix_sharing = { shared_prefix_tokens = 4096 }
# or
prefix_sharing = { shared_prefix_ratio = 0.75 }
```

For input length `I`, the ratio resolves to
`floor(I * shared_prefix_ratio)`. Values from zero through one are valid,
including no sharing and full-prompt sharing. Distributed ISL uses nested
prefixes of one canonical stream. Equivalent resolved token schedules produce
the same seeded population. This controls prompt geometry only; it neither
controls initial cache state nor promises an observed cache-hit percentage.

`server_chat` instead may use pre-template `shared_system_content` with fixed
tokens or a ratio strictly inside `(0, 1)`. It cannot combine with exact
`prefix_sharing`, and a weighted mixture cannot use it.

## Dataset Sources And Sessions

The release catalog, not an arbitrary URL or engine-native loader, owns dataset
identity and immutable snapshots. Missing data is downloaded on first real
execution, digest-verified, and cached by content; dry-run reports cache state
without downloading.

- ShareGPT independent requests retain structured messages, hold out the final
  assistant target for the output limit, and roll back complete trailing
  exchanges until the pre-template message-content length fits
  `max_input_tokens`. `output_tokens` may replace the target-derived limit.
- SPEED-Bench selects a catalog profile, filters and samples the pinned parquet
  snapshot, and freezes only each row's first user turn as one independent
  request. Later turns are recorded as omitted. Profile identifiers are catalog
  data, not a fixed source-code enum.
- A linear ShareGPT `session_source` sends later user turns only after the live
  assistant response and the scaled/capped inter-turn delay. Concurrency counts
  active conversations including delay intervals. A turn failure is terminal
  for that session. Warmup uses complete native sessions.

Dataset admission measures message content before the server's chat template.
Backend-observed prompt-token evidence remains the measured post-template ISL.

## Metrics, SLOs, And Server Exports

Every successful non-empty Bench reports request, output-token, and total-token
throughput plus mean/min/max/stddev and p50/p90/p95/p99 prompt tokens, request
latency, and TTFT. TPOT reports the same distribution when output length makes
it applicable. `output_tokens = 1` is prefill-dominant and omits TPOT.

`prompt_cache_read_ratio` appears only when valid AIPerf cache-usage evidence
exists. It is observed evidence, not the configured prefix ratio.
`good_request_ratio` and `goodput` appear only with a request SLO.

Aggregate SLO constraints use one inclusive `at_least` or `at_most` bound and
are AND-composed. A request SLO applies latency/TTFT/TPOT upper bounds per
request and gates the case with `minimum_good_request_ratio`. A complete set of
inference errors can remain service-quality evidence with request SLOs; broken
transport, timeout, protocol, or native artifacts remain measurement failures.

`server_metrics = true` asks AIPerf to collect the integration-declared JSON
metrics endpoint and requires zero warmup. The integration owns the exact path,
including `/metrics` or `/v1/metrics`, and any separately allocated metrics
port. Direct SGLang requires its effective `enable_metrics = true`. A successful
SPEED-Bench case additionally normalizes acceptance length and rate; other
sources retain raw server metrics without claiming those SPEED scalars.

## Evidence Checks

The record freezes the resolved route, prompt authority, tokenizer identity,
template provenance, request population, prefix schedule, warmup/profiling
phase identity, request-body fragment, native commands, artifacts, and
normalized metrics. Completed exact `flat` and `rendered_chat` requests must
reconcile their selected ISL with AIPerf's backend-observed
`input_sequence_length`; a mismatch invalidates the case.

Read [Evidence and diagnosis](evidence-and-diagnosis.md) before comparing runs.
