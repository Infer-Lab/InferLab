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

Case selection follows the server's declared cases and default; see
[Workspace definitions](workspace-definition.md#minimal-workspace). The selected
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
is idempotent and finalizes cleanup evidence, including the device-residual
check described in
[Failure and cleanup](evidence-and-diagnosis.md#failure-and-cleanup). Always
stop a manual server after the last measurement.

Ordinary readiness, capture preparation, framework control, and report
finalization use separate resolved budgets. Read
[Profiling](profiling.md) for capture lifecycle interpretation.

## Closed Loop

A recipe run starts the selected server, executes the ordered workload suite,
applies its gate, stops all processes, and aggregates child records:

```sh
inferlab recipe run <RECIPE> [--case C] [--placement P]
```

A failed recipe still finalizes its record; see
[Failure and cleanup](evidence-and-diagnosis.md#failure-and-cleanup). A prior
SLO failure does not suppress later static Bench cases; execution failure,
timeout, or interruption follows the closed-loop failure path.

`recipe run --capture <WORKLOAD_ID>` is repeatable and captures only selected
Evals or Benches. `bench --capture` attaches capture to a manual Bench. Capture
does not mutate the stored measurement definition.

## Occupied Devices

A launch fails before starting any process when an assigned device already
holds a compute application, naming each occupant. Add `--wait-for-devices` to
`serve start` or `recipe run` to wait instead: the launch rechecks the occupied
devices until they free, shows the occupants while it waits, and then
continues; the record's `device_wait` keeps how long it waited and on whom.
Interrupt the wait as you would the launch. The wait is not a queue: runs
waiting on the same devices can start together once they free, so start the
next run in a sequence only after the previous one ends. A pool placement
skips devices already occupied at resolution and fails resolution without
waiting when too few are free; the wait applies only to the launch-time check
of the devices already assigned.

## Detached Runs

`serve start`, `recipe run`, and `bench` accept `--detach` for runs that must
outlive the invoking session:

```sh
inferlab recipe run <RECIPE> --detach
inferlab job interrupt <JOB_ID>
```

A detached invocation launches the same command as a job — the same jobs the
web console launches, kept in `.inferlab/runtime/jobs/<job-id>/` with separate
stdout and stderr logs and the exit status — and returns once the command has
created its first record. Its JSON names `job_id`, `job_dir`, the `stdout` and
`stderr` log paths, and `record_id` and `record_dir`; follow the run through
that record (for a recipe, the recipe record). If the command ends before
creating a record, the invocation fails with the command's stderr. The
invocation names the job on stderr as soon as it starts, and interrupting the
waiting invocation leaves the job running. `job interrupt` delivers the
interrupt a terminal Ctrl+C delivers, only to the recorded live process, and
never escalates to a kill, so the run finalizes its record and cleans up as it
would after Ctrl+C. `--detach` cannot combine with `--dry-run`.

## Stopping a Run

```sh
inferlab recipe stop <RECIPE_RECORD_ID> [--force]
inferlab serve stop <SERVER_RECORD_ID> [--force]
```

The process that runs a recipe controls its record until the record is
finalized; the process that starts a server controls the server record until
the server runs or its start fails. Each controller holds a lock beside its
record and records its host, process ID, start time, and command there, so any
host sharing the workspace can stop it. `recipe stop` of a running recipe and
`serve stop` of a starting server act on that controller:

- While the controller lives, the command names it, asks it to stop, and waits
  until it has finalized the record — exactly as Ctrl+C in the controller's
  terminal would — then prints that record. Interrupting the wait leaves the
  request standing.
- When the controller is gone, for example killed, the command stops what the
  record started (a recipe's server through `serve stop`) and finalizes the
  record as failed with `operator_stop: {"forced": false}`.
- `--force` proceeds as though the controller were gone even though its lock
  is held, for a controller whose host died while the file system still holds
  its lock; the record shows `operator_stop: {"forced": true}`. Use it only
  after confirming the named controller no longer runs.

`recipe stop` of a finished recipe changes nothing and prints it. `serve stop`
of a running server stops it as before.

## Image Selection

See
[Built and external image selection](images-and-run.md#built-and-external-image-selection).
