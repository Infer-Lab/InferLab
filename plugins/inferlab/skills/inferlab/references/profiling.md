# Workload-Attached Profiling

Profiling is an observation mode on a named Eval or Bench, not another workload
kind. A profiling-capable server prepares every integration-declared model-rank
target at launch:

```toml
[servers.example]
profiling = true
capture_arm_deadline_seconds = 60
capture_control_deadline_seconds = 60
capture_finalization_deadline_seconds = 300
```

A server case or invocation may patch profiling and its deadlines. Capture the
selected workload with:

```sh
inferlab recipe run <RECIPE> --capture <WORKLOAD_ID>
inferlab bench <BENCH> --serve <SERVER_RECORD_ID> --capture
```

Recipe `--capture` is repeatable. The server must have been launched with
capture targets prepared; a manual Bench cannot retrofit them onto an ordinary
running server. Image-backed server launches reject capture because InferLab
does not claim host profiling of an in-container server process.

## Window And Deadline Semantics

Each process rank owns one Nsight Systems session. InferLab enumerates the
expected semantic windows, arms all selected targets, opens the framework range
before its bound measurement phase, closes it after client completion or
failure, finalizes collection, and verifies every required report.

- `capture_arm_deadline_seconds` is one shared budget across preparation and
  arming of all selected targets.
- `capture_control_deadline_seconds` applies to the complete HTTP response for
  each framework start or stop action.
- `capture_finalization_deadline_seconds` is one shared budget across session
  inspection, any required collection stop, asynchronous report completion,
  and coverage verification for all targets.

These budgets do not replace ordinary measurement timeouts. Capture-armed
server readiness has no overall readiness timeout because instrumentation can
multiply framework startup cost, but every readiness attempt still uses
`readiness_attempt_timeout_seconds` and process exit or operator interruption
remains terminal.

A positive AIPerf Bench warmup drains before InferLab opens the capture window.
Warmup remains in the native artifacts but not the trace window or normalized
profiling metrics. A warmup failure leaves the window unopened. The capture
closes at the existing client-completion boundary.

A failed close action is preserved but does not by itself discard a complete
trace: final report coverage decides success. Missing report coverage fails the
profile and retains both control and coverage evidence.

## Nsight Systems Escapes

Defaults use the `nsys` executable and `cuda,nvtx` trace set. A server may
declare common escapes and a model-serving role may refine them:

```toml
[servers.example.profiler.nsys]
executable = "/opt/nsight-systems/bin/nsys"
trace = ["cuda", "nvtx", "osrt"]
launch_options = []
start_options = []
sampling = "none"
context_switch = "none"

[servers.example.profiler.nsys.env]
PATH = "/opt/nsight-systems/bin:/usr/bin"
```

`executable`, `trace`, `sampling`, and `context_switch` are dedicated
replacement fields. `launch_options` and `start_options` are separate lists;
role lists append after common lists. Environment maps merge per key with the
role winning and apply to every managed Nsight command. On the launch command,
the environment is inherited by the wrapped server; control and finalization
commands do not mutate the already-running server.

InferLab rejects escape options that attempt to replace managed session,
report, range, overwrite, export, or launch-wait facts. Managed options are
appended after escapes and remain authoritative. Environment keys must be POSIX
identifiers.

Use OS runtime tracing when the experiment needs it; disabling `osrt` is not a
general cure for startup or finalization timing. Select it explicitly in
`trace`, keep the lifecycle deadlines sized for the additional capture cost,
and inspect the per-target evidence.

## Evidence

Server and workload records preserve raw/effective escapes, target role,
replica, process and rank, session, range-to-window mapping, effective commands
and environment, framework control requests and responses, finalization,
report paths, and verification. A profiling failure still follows the
measurement and server cleanup path.
