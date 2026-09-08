# Eval tasks, datasets, and inference requests

Start with the smallest definition that expresses the workload. The built-in
OpenAI smoke needs only its kind:

```toml
[evals.smoke]
kind = "openai-smoke"
```

The smoke carries authoring defaults, not hidden execution state.
`inferlab workspace show --json` renders the effective smoke values
explicitly, and existing explicit forms remain valid. Serving Benches are
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
definition controls its request fragment, sample limit, few-shot count, seed,
trials, output bound, concurrency, selected metric and optional filter,
threshold, and timeout while the task retains dataset and scoring authority.

InferLab uses the resolved model-weight locator as the Hugging Face tokenizer
locator. This follows the normal model-directory convention and avoids a
second tokenizer setting; the locator must contain a usable tokenizer.
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
same way. A probe failure makes support inconclusive rather than silently
removing the task.

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

## Source preparation

Source preparation semantics and the cold-to-warm verification procedure are
shared with serving sources; see
[bench-authoring.md](bench-authoring.md#source-preparation-and-cold-to-warm-verification).
