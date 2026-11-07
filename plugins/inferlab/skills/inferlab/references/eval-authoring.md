# Eval tasks, datasets, and inference requests

Serving Benches are covered by [bench-authoring.md](bench-authoring.md).

## OpenAI smoke

Start with the smallest definition that expresses the workload. The built-in
OpenAI smoke, the smallest completion-path correctness check, needs only its
kind:

```toml
[evals.smoke]
kind = "openai-smoke"
```

`prompt`, `max_tokens`, and `timeout_seconds` may be declared;
`inferlab workspace show --json` renders the effective values explicitly.

For a vision-language server, `vision = true` sends one chat-completions user
message whose content is the effective prompt as a text part followed by a
release-owned fixed test image as an `image_url` data-URI part:

```toml
[evals.smoke-vision]
kind = "openai-smoke"
vision = true
```

Success and failure mirror the completions smoke, judging the first choice's
`message.content` string in place of `text`; the image content itself is never
judged, and there is nothing else to configure.

## lm-eval tasks

An lm-eval definition selects exactly one task: a pinned lm-eval task name, a
release-bundled InferLab task, or a workspace-owned task YAML:

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

The task owns `dataset_path`, `dataset_name`, split selection, prompting,
output type, filters, and scoring; InferLab adds no second dataset layer. The
definition controls `request_body`, `limit` (samples), `few_shot`, `seed`,
`trials`, `max_tokens`, request `concurrency`, `metric` and optional
`metric_filter`, `threshold`, and `timeout_seconds`.

Workspace YAML paths must be workspace-relative tracked `.yaml` or `.yml`
files. InferLab resolves their include closure, records the effective task
configuration and dataset selection, and includes that closure in source
identity. Release-bundled tasks are addressed only by catalog name and carry a
release-owned closure digest.

`timeout_seconds` is one budget for the whole Eval case: task and tokenizer
loading, the prompt-logprob probe, and every repeated trial consume it; a
trial never gets a fresh deadline. Cleanup keeps its own grace.

### Per-model thresholds

When the expected score depends on the model, key thresholds by workspace
model instead of copying the Eval; `threshold` stays the default:

```toml
[evals.builtin.model_thresholds]
qwen3-8b = 0.96
```

Every key must name a declared model. A recipe or image validation that runs
the Eval against a server for that model uses its entry, and the plan records
the effective threshold and the model that selected it. `--set` patches either
level: `evals.builtin.threshold=…` changes the default, and
`evals.builtin.model_thresholds.qwen3-8b=…` changes that model's entry.

### Tokenizer and server requirements

InferLab uses the resolved model-weight locator as the Hugging Face tokenizer
locator, so it must contain a usable tokenizer and be readable on the
controller, where measurements run;
[workspace-definition.md](workspace-definition.md) owns which locator is
selected. Any Eval, including the smoke, fails planning against a server that
declares synthetic acceptance.

## Prompt authority

A `generate_until` task may declare how prompts are rendered; omission
resolves to `flat`:

```toml
[evals.gsm8k]
kind = "lm-eval"
task = "gsm8k"
prompt = { kind = "flat" }      # omit for the same result
metric = "exact_match"
metric_filter = "strict-match"
threshold = 0.90
timeout_seconds = 900
```

- `flat` sends the task's own few-shot context as plain text on the
  completions path, keeping the continuation format the task's scoring
  filters expect. Use it unless the chat template is part of what you measure.
- `server_chat` sends structured messages on the chat-completions path and
  lets the server apply its template. Choose it when the evaluated behavior
  depends on that template or on server-side controls such as
  `chat_template_kwargs`, which `request_body` accepts only under
  `server_chat`; a `flat` definition declaring one fails resolution naming the
  member.

The two are different measurements of the same task: records carry the
resolved authority and whether it was declared or defaulted, and scores must
not be compared across them.

Tasks whose output type is `loglikelihood`, `loglikelihood_rolling`, or
`multiple_choice`, and dynamic Python tasks, must not declare `prompt`. They
use completions and first run a prompt-logprob/tokenizer alignment probe,
because the pinned lm-eval chat client does not implement loglikelihood
scoring. The probe concludes `supported`, `unsupported` (the endpoint's prompt
echo does not align with the tokenizer, or scores no prompt positions), or
`inconclusive` (transport, timeout, non-2xx, or malformed response). Anything
but `supported` fails the Eval before lm-eval inference; InferLab never falls
back to generation scoring.

## Request body

`request_body` carries task-specific inference parameters such as sampling,
reasoning effort, logprobs, or, under `server_chat`, chat-template arguments:

```toml
[evals.reasoning]
kind = "lm-eval"
task = { yaml = "evals/reasoning.yaml" }
prompt = { kind = "server_chat" }
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

Nested values may be patched for one run, for example
`--set evals.reasoning.request_body.temperature=0.6`.

`request_body` is a JSON request fragment, not a replacement request. This is
the shared rule for Eval and Bench: the measurement runtime owns `model`,
`prompt`, `messages`, `stream`, `n`, `max_tokens`, `max_completion_tokens`,
and `stop`, and a fragment declaring any of them fails validation naming the
member. An lm-eval fragment also may not declare `seed` (Eval owns the
repeated-trial seed schedule), or `chat_template`/`chat_template_kwargs` under
`flat`. [bench-authoring.md](bench-authoring.md#minimal-definitions) lists the
members a Bench additionally reserves. Dry-run and the record keep the
complete effective fragment.

## Per-sample evidence

Every lm-eval run keeps lm-eval's per-sample log as `samples_*.jsonl` raw
artifacts (kind `lm-eval-samples-projected`): each question's responses,
filtered answer, target, and score. InferLab omits each question's document
and rendered prompt, which the task and `doc_id` reproduce and which can
dominate the log on long-context tasks; the prompt is `null` at its position
in `arguments`, and lm-eval's `doc_hash` and `prompt_hash` stay to verify a
reproduction. lm-eval writes this log only when evaluation completes.

A single-trial Eval also writes `generation-outcomes.jsonl` (kind
`lm-eval-generation-outcomes`) as it runs, so it survives a timeout or
interruption:

- Before generation, one `instance` entry per request with its `task_name`,
  `doc_id`, and `context_sha256`.
- Then one `attempt` entry per HTTP attempt, retries included, with the
  `context_sha256` it carried, send and completion times, HTTP status or
  transport error, the returned `choices` (text, reasoning content, finish
  reason) without log-probabilities, and the endpoint's `usage`.

Join an attempt to its question through `context_sha256`, and to its score
through `doc_id`. Loglikelihood requests are not listed there; their results
are in the sample log.

## Repeated Trials

`trials` greater than one repeats one lm-eval definition as fixed outer trials
inside the same Eval case and budget, and gates on the observed pass rate. It
is accepted only for a task that resolves to `generate_until`, selects exactly
one sample, and produces exactly one scored generation per trial; other tasks
fail resolution. `threshold` must then lie in `[0, 1]`, because it is compared
with the pass rate.

Trial `i`, counted from one, uses request seed `base_seed + i - 1`, where
`base_seed` is the definition `seed` or the release default. The record keeps
every trial's identity, effective seed, and outcome, and compares the pass
rate over issued trials with the threshold even when fewer than the requested
trials completed. Patch the count for one run with
`--set evals.<ID>.trials=<N>`.

## Source preparation

Preparation timing and the cold-to-warm verification procedure are shared
with serving sources; see
[bench-authoring.md](bench-authoring.md#source-preparation-and-cold-to-warm-verification).

For a workspace lm-eval YAML using a file-backed `json`, `csv`, `parquet`,
`text`, or `arrow` loader, InferLab snapshots the YAML include closure and
workspace-local `data_files` before serving starts. Exact paths, lists, split
mappings, and file globs expand into the recorded ordered closure, and the
Eval client receives a generated task YAML bound only to the read-only
snapshot. Remote selectors, paths outside the workspace, and task function
references remain explicit opaque sources, because preparation cannot bind
their complete file closure.
