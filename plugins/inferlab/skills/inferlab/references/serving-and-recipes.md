# Serving And Recipes

For server, case, topology, placement, readiness, suite, and recipe authoring,
read [Workspace definitions and placement](workspace-definition.md). For typed
invocation patches, read
[Invocation patches](execution-authoring.md#invocation-patches). This reference
covers resolution and managed lifecycle execution.

## Resolution

Use `--dry-run` to resolve placement, ranks, devices, endpoints, commands,
environment, effective settings, integration identity, and override provenance
without launching or writing a record:

```sh
inferlab serve start example --case tp2 --placement local --dry-run
inferlab recipe run qualify --case tp2 --placement local --dry-run
```

A sole server case is selected automatically. When several cases exist, the
stored default applies unless the invocation supplies `--case`. The selected
integration validates the complete resolved topology and backend pairing before
launch.

## Manual Server Lifecycle

```sh
inferlab serve start <SERVER> [--case C] [--placement P]
inferlab serve status <RECORD_ID>
inferlab serve logs <RECORD_ID>
inferlab bench <BENCH> --serve <RECORD_ID>
inferlab serve stop <RECORD_ID>
```

`start` emits the running server record id. `status`, `logs`, and `stop`
use only that record; they do not reload workspace or local bindings. `stop`
is idempotent and finalizes cleanup evidence. Always stop a manual server after
the last measurement.

Ordinary readiness, capture preparation, framework control, and report
finalization use separate resolved budgets. Read
[Profiling](profiling.md) for capture lifecycle interpretation.

## Closed Loop

A recipe run starts the selected server, executes the ordered workload suite,
applies its gate, stops all processes, and aggregates child records:

```sh
inferlab recipe run <RECIPE> [--case C] [--placement P]
```

Failure is still evidence: the recipe record preserves the failing phase,
measurement and server child references, per-process cleanup, and logs. A prior
SLO failure does not suppress later static Bench cases; execution failure,
timeout, or interruption follows the closed-loop failure path.

`recipe run --capture <WORKLOAD_ID>` is repeatable and captures only selected
Evals or Benches. `bench --capture` attaches capture to a manual Bench. Capture
does not mutate the stored measurement definition.

## Image Selection

`serve start` and `recipe run` accept either
`--image <IMAGE_BUILD_RECORD>` or `--external-image <ID>`, never both. A built
image must contain a successful host-platform assembly compatible with the
server stack and local placement. InferLab probes a declared external image on
every launch machine and does not pull it automatically. Read
[Images and ad-hoc execution](images-and-run.md) for build and probe workflows.
