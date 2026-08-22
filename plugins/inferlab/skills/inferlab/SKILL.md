---
name: inferlab
description: "Use when operating InferLab: run reproducible LLM inference experiments through a versioned workspace — serve lifecycles, closed-loop eval/bench recipes, standalone benches, workload-attached profiling (managed Nsight Systems collection or engine-native traces), runtime images, and the scratchpad journal — always reading results from file-first execution records."
---

# InferLab Operator Workflow

InferLab runs reproducible LLM inference experiments. A committed workspace
fixes the shareable baseline; `.inferlab/local.toml` supplies machine-private
bindings; managed workflows write durable records under
`.inferlab/records/<ID>/`. Select declared objects and run them. Do not
hand-compose framework launch commands or substitute an engine-native benchmark
for a declared InferLab Bench.

## Operating Rules

- Records are the interface. Read `record.json`, case results, logs, and raw
  artifacts instead of treating terminal text as experiment evidence.
- Use `--dry-run` before a stateful workflow when resolution is uncertain.
- Use repeatable `--set PATH=VALUE` for typed, invocation-scoped TOML patches;
  do not edit a committed definition merely to perform one variation.
- Keep workspace facts, local bindings, and execution evidence in their owning
  locations. Never copy private bindings into tracked workspace TOML.
- Run probes through `inferlab run`. Never invoke a binary directly from a
  `.pixi/envs/<env>/bin/` path because that bypasses manifest activation.
- Keep InferLab's AIPerf-backed Bench definition as the measurement authority
  across serving engines.

## Read The Relevant Reference

For broad inventory or an unfamiliar workspace, read the
[capability map](references/capability-map.md) first. For a focused task, read
the smallest matching reference completely:

| Task | Reference |
| --- | --- |
| Author or change workspace, placement, profiling, image, Eval, or Bench definitions | [Workspace authoring](references/workspace-authoring.md) |
| Inspect a workspace or check, lock, and diagnose stack realization | [Workspaces and stacks](references/workspaces-and-stacks.md) |
| Start, inspect, stop, or run a recipe around a managed server | [Serving and recipes](references/serving-and-recipes.md) |
| Execute or inspect Eval, Bench, dataset, session, metric, SLO, or prompt evidence | [Measurements](references/measurements.md) |
| Run or diagnose an Eval/Bench capture (managed Nsight Systems or engine-native trace) | [Profiling](references/profiling.md) |
| Build, select, validate, or probe a runtime image | [Images and ad-hoc execution](references/images-and-run.md) |
| Inspect records, compare results, use the TUI or scratchpad, or diagnose a failure | [Evidence and diagnosis](references/evidence-and-diagnosis.md) |
| Install, update, diagnose, or remove the agent plugin | [Agent plugin](references/agent-plugin.md) |

Backend-specific qualification boundaries remain in the bundled
[backend support matrix](../../../../docs/backend-support.md). It comes from the
same source snapshot as the installed plugin; do not substitute the latest
website projection when reproducing an older InferLab release.

## First Run

From the workspace root:

```sh
inferlab workspace show
cp .inferlab/local.example.toml .inferlab/local.toml  # when provided
pixi install --locked --all
inferlab stack status
inferlab toolchain install                            # only for Eval/Bench
```

`workspace show` needs no local bindings. `stack status` checks the selected
Pixi realization without model or placement bindings. Resolving a server,
recipe, Bench, or image then needs the applicable local facts.

## Command Surface

```text
inferlab tui
inferlab workspace show|lock
inferlab stack status [STACK]
inferlab toolchain install
inferlab serve start <SERVER> [--case C] [--placement P]
inferlab serve status|logs|stop <RECORD_ID>
inferlab recipe run <RECIPE> [--case C] [--placement P]
inferlab bench <BENCH> --serve <SERVER_RECORD_ID>
inferlab run [--stack S] -- <CMD>...
inferlab image build <IMAGE>
inferlab scratchpad note|show
inferlab agent install|update|uninstall|doctor
inferlab license
```

Every non-dry-run managed `serve start`, `recipe run`, `bench`, and
`image build` prints one final JSON report containing a record `id`. A failed
managed workflow still finalizes evidence and attempts cleanup.

## Privacy

Never put credentials, private model paths, hostnames, ports, device UUIDs,
usernames, or local scratch paths into tracked files or anything published.
Local records are intentionally unredacted and must be access-controlled by the
operator. Portable image contexts and artifacts exclude machine-private facts
by construction.
