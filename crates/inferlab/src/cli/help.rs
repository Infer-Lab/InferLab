pub(super) const ROOT: &str = "Run reproducible LLM inference experiments from a versioned workspace.

Committed workspace definitions own shareable stacks, servers, measurements, recipes, and images. Machine-private bindings own model locators, machines, devices, ports, and placement. Managed workflows write file-first evidence under .inferlab/records; diagnostics and progress go to stderr while final machine-readable reports stay on stdout.";

pub(super) const ROOT_EXAMPLES: &str = "FIRST RUN:
  inferlab workspace show
  pixi install --locked --all
  inferlab stack status
  inferlab toolchain install        # only when running Eval or Bench

SAFE DISCOVERY:
  Add --dry-run to serve start, recipe run, bench, or image build before a stateful execution.";

pub(super) const TUI: &str = "Observe operations, records, workspace definitions, metrics, referenced logs, and scratchpad entries in a persistent view-only terminal interface.

The TUI does not launch workflows, mutate records, or write a UI session. Refreshes label facts by authority and retain the last successful observation as stale when a later read fails.";

pub(super) const TUI_KEYS: &str = "KEYS:
  1-4       Select Overview, Operations, Records, or Workspace
  arrows    Navigate; Enter opens details; Esc returns
  Ctrl+K    Find typed objects
  /         Filter the current list or search a referenced log
  m         Compare one recorded metric across cases in one workload
  r         Request a refresh
  q         Exit";

pub(super) const STACK_STATUS: &str = "Report Pixi confirmation, declared realization-check evidence, and overall readiness for one stack or every declared stack.

This command does not require machine-local bindings. A confirmed environment is checked on every invocation; a failed check reports its captured output and declared repair hint without repairing or changing the retained Pixi confirmation.";

pub(super) const TOOLCHAIN_INSTALL: &str = "Install and verify the release-owned lm-eval and AIPerf measurement runtimes.

This toolchain is needed only for lm-eval and serving Bench measurements. It is separate from each serving stack's Pixi environment, and its internal measurement packages must not be added to a serving workspace.";

pub(super) const SERVE_START: &str = "Resolve and start one named server as a managed long-running lifecycle.

Resolution freezes the selected case, placement, ranks, devices, endpoints, integration lowering, effective settings, environment, image selection, and override provenance. Overrides address server.<PATH> fields only and cannot change server identity or topology. A real start creates a server record before launch; status, logs, and stop later use that record without reloading workspace or local bindings. Dry-run performs resolution without starting a server or writing a record, but integration planning still runs.";

pub(super) const SERVE_START_EXAMPLES: &str = "EXAMPLES:
  inferlab serve start qwen --case tp2 --placement local --dry-run
  inferlab serve start qwen --set server.settings.max_model_len=32768

IMAGE SELECTION:
  --image selects a successful InferLab image-build record.
  --external-image selects a declared digest-pinned image InferLab did not build.
  The two selections are mutually exclusive.";

pub(super) const SELECTION_OVERRIDE: &str = "Apply a typed invocation patch with a TOML value. May be repeated; later assignments win. The command details define which paths it accepts.";

pub(super) const RECIPE_RUN: &str = "Resolve and run one recipe as a recorded closed loop: start its named server, execute the selected Eval and Bench suite, stop every process, and aggregate child evidence.

Overrides accept server.<PATH> and selected evals.<ID>.<PATH> or benches.<ID>.<PATH> fields. They cannot change definition identity or kind, recipe server, workload-suite membership, or gate. --image and --external-image are mutually exclusive. Failure still finalizes the recipe and child records and attempts cleanup. Dry-run freezes the effective server and measurement plan without launching or writing a record. Repeat --capture for selected workload ids; each capture is an observation mode on that named Eval or Bench.";

pub(super) const RECIPE_RUN_EXAMPLES: &str = "EXAMPLES:
  inferlab recipe run qualify --case tp1 --placement local --dry-run
  inferlab recipe run qualify --set evals.smoke.limit=10
  inferlab recipe run qualify --capture latency --capture throughput";

pub(super) const BENCH: &str = "Run one named serving Bench against an explicit running managed-server record.

The stored Bench definition remains the authority for request source, prompt representation, load, warmup, metrics, SLOs, timeout, and cache controls. Dry-run still requires the target server because endpoint, model, tokenizer, and integration capabilities come from its record, but it sends no measurement traffic and writes no Bench record. --capture requires profiler targets prepared when that server was started.";

pub(super) const BENCH_OVERRIDE: &str = "Override one typed field inside the selected Bench definition with a TOML value, for example concurrency=[1,8] or request_body.temperature=1.0. Later assignments win. The Bench identity and kind cannot be changed.";

pub(super) const RUN: &str = "Execute one unrecorded diagnostic command with the same stack activation or container substitution used by InferLab launches.

Do not invoke .pixi/envs/<env>/bin tools directly: that bypasses manifest activation. A container receives no host mount or device implicitly. Local --stack and the two container-image selectors are mutually exclusive; --mount and --devices require a container image.";

pub(super) const RUN_EXAMPLES: &str = "EXAMPLES:
  inferlab run -- python -c 'import vllm; print(vllm.__version__)'
  inferlab run --stack vllm -- pytest tests/ -k smoke
  inferlab run --image <RECORD_ID> --devices 0 -- nvidia-smi -L
  inferlab run --external-image official --mount /data -- python3 /data/probe.py

This command writes no execution record; use a managed workflow for evidence.";

pub(super) const IMAGE_BUILD: &str = "Build one named runtime image as a recorded closed loop: resolve source and base-image identities, assemble every producible platform, inspect immutable image identity, optionally export unique OCI archives, and run eligible recipe validations.

Image build requires a clean workspace. One platform or validation failure does not suppress the remaining batch. Built images remain in local builder storage unless exported; this workflow never pushes to a registry. Dry-run performs no package build, assembly, export, or validation.";

pub(super) const IMAGE_BUILD_EXAMPLES: &str = "EXAMPLES:
  inferlab image build vllm-runtime --dry-run
  inferlab image build vllm-runtime --builder local --placement local
  inferlab image build vllm-runtime --export /absolute/output/directory";

pub(super) const SCRATCHPAD_NOTE: &str = "Append one entry to the workspace-local operator narrative.

Entries may link existing records but never alter workspace resolution, execution, records, or source identity. --record is repeatable, and the value last resolves to the newest local record.";

pub(super) const SCRATCHPAD_SHOW: &str = "Render the workspace-local append-only operator narrative in chronological order.

The default view shows the recent tail; --all shows the complete journal. --topic selects one topic stream. Reading the journal never alters workspace resolution, execution, records, or source identity.";

pub(super) const AGENT_INSTALL: &str = "Install the versioned InferLab agent plugin independently of experiment workspaces and records.

Installation defaults to the package embedded in this binary and needs no checkout or network access. --from-checkout selects an explicit repository checkout or unpacked release archive instead. The package is validated before any native agent CLI runs. The command emits one JSON report with a row per selected runtime and every native command attempted in order.";

pub(super) const AGENT_UPDATE: &str = "Update the installed InferLab plugin through each selected agent runtime's registered marketplace.

An older InferLab marketplace source is repaired from the package embedded in this binary before refresh when needed. The operation does not read experiment workspace facts or records and emits one JSON report with every native command attempted in order.";

pub(super) const AGENT_UNINSTALL: &str = "Uninstall the InferLab plugin through each selected native agent runtime.

The operation does not read experiment workspace facts or records. It emits one JSON report with a row per selected runtime and every native command attempted in order; a partial failure retains completed actions.";

pub(super) const AGENT_DOCTOR: &str = "Read-only diagnosis of selected native agent CLIs and registered InferLab marketplace sources.

Doctor does not install, update, or remove a plugin and does not read experiment workspace facts or records. It emits one JSON report containing readiness and source health plus every native probe attempted in order.";
