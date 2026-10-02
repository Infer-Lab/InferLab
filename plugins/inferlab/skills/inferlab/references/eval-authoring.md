# Eval tasks, datasets, and inference requests

Start with the smallest definition that expresses the workload. The built-in
OpenAI smoke needs only its kind:

```toml
[evals.smoke]
kind = "openai-smoke"
```

The smoke carries authoring defaults, not hidden execution state: its
`prompt`, `max_tokens`, and `timeout_seconds` may be declared, and
`inferlab workspace show --json` renders the effective values explicitly.
Existing explicit forms remain valid. Serving Benches are
covered by [bench-authoring.md](bench-authoring.md).

## Vision smoke

For a vision-language server, `vision = true` turns the smoke toward the
vision path:

```toml
[evals.smoke-vision]
kind = "openai-smoke"
vision = true
```

A vision smoke routes to chat completions and sends one user message whose
content carries the effective prompt as a text part followed by the
release-owned fixed test image as an `image_url` data-URI part, so the check
exercises the vision path instead of text-only completions. Success and
failure mirror the completions smoke with the first choice's `message.content`
string in place of `text`; the image content itself is never judged. The image
bytes are fixed by the release, so there is nothing else to configure.

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
definition controls its request fragment (`request_body`), sample limit
(`limit`), few-shot count (`few_shot`), `seed`, `trials`, output bound
(`max_tokens`), request `concurrency`, selected `metric` and optional
`metric_filter`, `threshold`, and `timeout_seconds`, while the task retains
dataset and scoring authority.

`timeout_seconds` is one budget for the whole Eval case: task and tokenizer
loading, the prompt-logprob probe, and every repeated trial consume it, and a
trial never receives a fresh deadline. Cleanup keeps its own grace.

InferLab uses a resolved model-weight locator as the Hugging Face tokenizer
locator, so it must contain a usable tokenizer. Measurements run on the
controller, so the locator must be readable there;
[workspace-definition.md](workspace-definition.md) owns which locator is
selected. Any Eval, including the smoke, fails planning against a server that
declares synthetic acceptance.
A `generate_until` task may declare a prompt rendering authority, and omitting
it resolves to `flat`:

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

`flat` sends the task's own few-shot context as ordinary text on the completions
path, so the task keeps the continuation format its scoring filters expect. Use
it unless the model chat template is itself part of what you are measuring.
`server_chat` sends structured messages on the chat-completions path and lets
the model server apply its own template; choose it when the evaluated behavior
depends on that template or on server-side controls such as
`chat_template_kwargs`. The two are different measurements of the same task, so
records carry the resolved authority and whether it was declared or defaulted;
do not compare scores across them.

Server-side template controls such as `chat_template` and `chat_template_kwargs`
are accepted in `request_body` only under `server_chat`; a `flat` definition that
declares one fails resolution and names the conflicting member.

Tasks whose resolved output type is `loglikelihood`, `loglikelihood_rolling`, or
`multiple_choice` must not declare `prompt`. They use completions and first run a
prompt-logprob/tokenizer alignment probe, because the pinned lm-eval chat client
does not implement loglikelihood scoring. Dynamic Python tasks are treated the
same way. The probe concludes `supported`, `unsupported` (the endpoint's
prompt echo does not align with the tokenizer, or it scores no prompt
positions), or `inconclusive` (transport, timeout, non-2xx, or malformed
response). Any conclusion other than `supported` fails the Eval before lm-eval
inference; InferLab never falls back to generation scoring.

Use `request_body` for task-specific inference parameters such as sampling,
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

The same nested values may be patched for one run, for example
`--set evals.reasoning.request_body.temperature=0.6`. `request_body` is a JSON
request fragment, not a replacement request. This page owns the shared rule for
Eval and Bench: the measurement runtime owns `model`, `prompt`, `messages`,
`stream`, `n`, `max_tokens`, `max_completion_tokens`, and `stop`, and a
fragment that declares any of them fails validation naming the member. An
lm-eval fragment additionally may not declare `seed`, because Eval owns the
repeated-trial seed schedule, and under `flat` it may not declare
`chat_template` or `chat_template_kwargs`.
[bench-authoring.md](bench-authoring.md) lists the members a Bench additionally
reserves. The complete effective fragment is preserved in dry-run and record
evidence.

## Repeated Trials

`trials` greater than one repeats one lm-eval definition as fixed outer trials
inside the same Eval case and budget, and gates on the observed pass rate. It
is accepted only for a task that resolves to `generate_until`, selects exactly
one sample, and produces exactly one scored generation per trial; other tasks
fail resolution. With `trials` greater than one, `threshold` must lie in
`[0, 1]` because it is compared with the pass rate.

Trial `i`, counted from one, uses request seed `base_seed + i - 1`, where
`base_seed` is the definition `seed` or the release default when it is omitted.
The record preserves every trial's identity, effective seed, and outcome, and
compares the pass rate over issued trials with the threshold even when fewer
than the requested trials completed. Patch the count for one run with
`--set evals.<ID>.trials=<N>`.

## Source preparation

Source preparation semantics and the cold-to-warm verification procedure are
shared with serving sources; see
[bench-authoring.md](bench-authoring.md#source-preparation-and-cold-to-warm-verification).

For a workspace lm-eval YAML using a file-backed `json`, `csv`, `parquet`,
`text`, or `arrow` loader, InferLab snapshots the YAML include closure and
workspace-local `data_files` before serving starts. Exact paths, lists, split
mappings, and file globs are expanded into the recorded ordered closure, and
the Eval client receives a generated task YAML bound only to the read-only
snapshot. Remote selectors, paths outside the workspace, and task function
references remain explicit opaque sources because preparation cannot bind
their complete file closure.
