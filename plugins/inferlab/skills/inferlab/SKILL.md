---
name: inferlab
description: "Use when operating InferLab: run reproducible LLM inference experiments through a versioned workspace — serve lifecycles, closed-loop eval/bench recipes, standalone benches, workload-attached profiling (managed Nsight Systems collection or engine-native traces), runtime images, and the scratchpad journal — always reading results from file-first execution records."
---

# InferLab Operator Workflow

InferLab runs reproducible LLM inference experiments. A committed workspace
fixes the shareable baseline; `.inferlab/local.toml` supplies machine-private
bindings; git-ignored `.inferlab/workspace.local.d/` holds experiment-only
definitions; managed workflows write durable records under
`.inferlab/records/<ID>/`. Select declared objects and run them.

## Operating Rules

- Records are the interface. Read `record.json`, case results, logs, and raw
  artifacts instead of treating terminal text as experiment evidence.
- Use `--dry-run` before a stateful workflow when resolution is uncertain.
- Use repeatable `--set PATH=VALUE` for typed, invocation-scoped TOML patches;
  do not edit a committed definition merely to perform one variation.
- Keep workspace facts, local bindings, and execution evidence in their owning
  locations. Never copy private bindings into tracked workspace TOML.
- Do not hand-compose framework launch commands. Measure through a declared
  InferLab Bench, the AIPerf-backed measurement authority across serving
  engines, not an engine-native benchmark.
- Run probes through `inferlab run`. Never invoke a binary directly from a
  `.pixi/envs/<env>/bin/` path because that bypasses manifest activation.

## Read The Relevant Reference

The [capability map](references/capability-map.md) lists every command and
authoring surface with its owning reference; read it first for broad inventory
or an unfamiliar workspace. For a focused task, read the smallest matching
reference completely:

| Task | Reference |
| --- | --- |
| Author or change workspace, placement, profiling, image, Eval, or Bench definitions | [Workspace authoring](references/workspace-authoring.md) |
| Inspect a workspace or check, lock, and diagnose stack realization | [Workspaces and stacks](references/workspaces-and-stacks.md) |
| Start, inspect, stop, detach, or run a recipe around a managed server | [Serving and recipes](references/serving-and-recipes.md) |
| Execute or inspect Eval, Bench, dataset, session, metric, SLO, or prompt evidence | [Measurements](references/measurements.md) |
| Run or diagnose an Eval/Bench capture (managed Nsight Systems or engine-native trace) | [Profiling](references/profiling.md) |
| Build, select, validate, or probe a runtime image; record a workload InferLab does not model | [Images and ad-hoc execution](references/images-and-run.md) |
| Read records, compare results, use the TUI, web console, or scratchpad, or diagnose a failure | [Evidence and diagnosis](references/evidence-and-diagnosis.md) |
| Check whether a backend, topology, parallelism mode, or profiling path is qualified | [Backend support matrix](references/backend-support.md) |
| Install, update, diagnose, or remove the agent plugin | [Agent plugin](references/agent-plugin.md) |

The [backend support matrix](references/backend-support.md) ships with this skill at the installed InferLab
version; its version correspondence table maps each release line to the
Adapter SDK and integration packages it speaks.

## First Run

From the workspace root:

```sh
inferlab workspace show
# write .inferlab/local.toml (copy .inferlab/local.example.toml when provided)
pixi install --locked --all
inferlab stack status
inferlab toolchain install                            # for lm-eval Evals, serving Benches, or image package builds
```

`workspace show` needs no local bindings. `stack status` checks the selected
Pixi realization without model or placement bindings. Resolving a server,
recipe, or image then needs the applicable local facts; a standalone `bench`
resolves from its target server record instead.

Each command's `--help` is authoritative for its flags. Final reports go to
stdout and diagnostics to stderr; which report each workflow prints, including
on failure and under `--detach`, is described in
[Record-first reading](references/evidence-and-diagnosis.md#record-first-reading).

## Privacy

Never put credentials, private model paths, hostnames, ports, device UUIDs,
usernames, or local scratch paths into tracked files or anything published.
Local records are intentionally unredacted and must be access-controlled by the
operator. Portable image contexts and artifacts exclude machine-private facts
by construction.
