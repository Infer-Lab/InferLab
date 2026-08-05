# Serving And Recipes

## Definitions And Resolution

A server names a stack, model, topology, readiness budget, framework settings,
parallelism, roles, cases, and optional frontend/KV-transfer/profiling facts. A
direct `single` server has no Gateway; a routed `single` names
`gateway_backend`; `prefill_decode` names both `gateway_backend` and
`pd_router_backend` and uses canonical `prefill` and `decode` Engine roles.
Backend support is pairing-specific, so check the release backend matrix.

Common settings apply to every model-serving role; role settings apply after
them. Cases may patch settings, parallelism, replica counts, frontend backend
identity, KV transfer, profiling, and readiness without changing the server's
component shape. A sole case is automatic; multiple cases require
`default_case` unless the invocation supplies `--case`.

Use `--dry-run` to resolve placement, ranks, devices, endpoints, commands,
environment, effective settings, integration identity, and override provenance
without launching or writing a record:

```sh
inferlab serve start example --case tp2 --placement local --dry-run
inferlab recipe run qualify --case tp2 --placement local --dry-run
```

Repeat `--set PATH=VALUE` for typed TOML patches. Server paths begin with
`server.`, for example:

```sh
inferlab serve start example \
  --set server.readiness_timeout_seconds=1800 \
  --set server.settings.max_model_len=32768 \
  --set server.roles.serve.parallelism.outer.tensor_parallel_size=4 \
  --dry-run
```

## Manual Server Lifecycle

```sh
inferlab serve start <SERVER> [--case C] [--placement P]
inferlab serve status <RECORD_ID>
inferlab serve logs <RECORD_ID>
inferlab bench <BENCH> --serve <RECORD_ID>
inferlab serve stop <RECORD_ID>
```

`start` emits the running server record id. `status`, `logs`, and `stop` use
only that record; they do not reload workspace or local bindings. `stop` is
idempotent and finalizes cleanup evidence. Always stop a manual server after
the last measurement.

`readiness_timeout_seconds` owns ordinary overall readiness.
`readiness_attempt_timeout_seconds` caps each blocking process-status or HTTP
attempt and defaults to 30 seconds. Capture-armed readiness is unbounded overall
but retains the bounded attempt timeout so process exit and interruption remain
observable. Profiling has separate arm, control, and finalization deadlines;
read [Profiling](profiling.md).

## Closed Loop

A workload suite lists named Evals and Benches and may identify one Eval gate.
A recipe selects one server and one suite. Running it starts the server, runs
the suite, stops all processes, and aggregates child records:

```sh
inferlab recipe run <RECIPE> [--case C] [--placement P]
```

Failure is still evidence: the recipe record preserves the failing phase,
measurement and server child references, per-process cleanup, and logs. A prior
SLO failure does not suppress later static Bench cases; execution failure,
timeout, or interruption follows the closed-loop failure path.

Recipe overrides may address only the selected server and measurements in the
suite. Measurement paths are `evals.<ID>.*` and `benches.<ID>.*`. They cannot
change definition identity or kind, suite membership, gate, or recipe server.

`recipe run --capture <WORKLOAD_ID>` is repeatable and captures only selected
Evals or Benches. `bench --capture` attaches capture to a manual Bench. The
stored measurement definition itself does not acquire a profiling mode.

## Image Selection

`serve start` and `recipe run` accept either `--image <IMAGE_BUILD_RECORD>` or
`--external-image <ID>`, never both. A built image must contain a successful
host-platform assembly compatible with the server stack and local placement.
An external image is a workspace-declared digest-pinned artifact whose claimed
integration matches the stack; InferLab probes every launch machine and does
not pull it automatically. Image-backed launches reject profiling until an
in-container profiler contract exists. Read
[Images and ad-hoc execution](images-and-run.md) for declarations and probes.
