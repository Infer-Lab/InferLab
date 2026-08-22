# Workload-Attached Profiling

For profiling enablement, deadline fields, mechanism selection, Nsight Systems
settings, escapes, and environment syntax, read
[Profiling authoring](execution-authoring.md#workload-profiling). This reference
covers capture execution, lifecycle interpretation, and evidence.

## Run A Capture

Capture remains attached to a selected Eval or Bench:

```sh
inferlab recipe run <RECIPE> --capture <WORKLOAD_ID>
inferlab bench <BENCH> --serve <SERVER_RECORD_ID> --capture
```

Recipe `--capture` is repeatable. The server must have launched with capture
targets prepared; a manual Bench cannot retrofit targets onto an ordinary
running server. Image-backed server launches reject either capture mechanism
because InferLab does not claim host profiling of an in-container server
process. Engine trace additionally requires a local, non-containerized
placement.

## Capture Lifecycle

Under managed collection, each model-rank target owns one Nsight Systems
session. Under engine trace there is no per-rank Nsight Systems session:
InferLab assigns each engine-trace replica a persistent record-owned trace
directory, the framework profiler writes one trace artifact per model-serving
rank into it, and coverage verifies a storage delta of at least one new
artifact per model device of the replica.

InferLab enumerates the expected semantic windows, arms every selected target,
opens the framework range before the bound measurement phase, closes it after
client completion or failure, finalizes collection, and verifies required
report coverage.

The configured arm, framework-control, and finalization budgets cover distinct
parts of that lifecycle. They do not replace the measurement timeout. Capture-
armed readiness has no overall deadline, but each readiness attempt remains
bounded so process exit and operator interruption stay observable. Under
engine trace, closing the window draws only the one global finalization
budget; the per-action control budget applies to managed collection.

A positive AIPerf Bench warmup drains before InferLab opens the capture window.
Warmup remains in native measurement artifacts but outside the trace window
and normalized profiling metrics. A warmup failure leaves the window unopened.

A delivery failure of the close — connection refusal, a dead engine process,
or a prompt error status — remains window-closing control failure evidence
adjudicated by coverage. A slow or absent stop response is instead neutral
flush-pending evidence on the closing action record, not a deadline failure.
Final report coverage decides capture success; missing coverage fails the
profile and retains both control and coverage evidence.

## Evidence

Server and workload records preserve raw and effective profiler settings,
target role, replica, process and rank, session, range-to-window mapping,
effective commands and environment, framework control requests and responses,
finalization, report paths, and verification. A profiling failure still follows
the measurement and server cleanup path.
