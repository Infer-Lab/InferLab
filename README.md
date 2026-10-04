# InferLab

<p align="center">
  <img src="https://raw.githubusercontent.com/Infer-Lab/InferLab/main/website/src/assets/inferlab-mark.svg" width="112" alt="InferLab logo">
</p>

<p align="center"><strong>Declare once. Run anywhere. Keep the evidence.</strong></p>

<p align="center">
  <a href="https://github.com/Infer-Lab/InferLab/actions/workflows/ci.yml"><img src="https://github.com/Infer-Lab/InferLab/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://github.com/Infer-Lab/InferLab/releases/latest"><img src="https://img.shields.io/github/v/release/Infer-Lab/InferLab" alt="Latest GitHub release"></a>
  <a href="https://crates.io/crates/inferlab"><img src="https://img.shields.io/crates/v/inferlab" alt="inferlab on crates.io"></a>
  <a href="https://github.com/Infer-Lab/InferLab/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Infer-Lab/InferLab" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://infer-lab.github.io/InferLab/">Website</a> ·
  <a href="https://infer-lab.github.io/InferLab/docs/">Documentation</a> ·
  <a href="https://github.com/Infer-Lab/InferLab/releases">Releases</a> ·
  <a href="https://infer-lab.github.io/InferLab/docs/architecture/rfc/">Specification</a>
</p>

Reproducible LLM inference experiments. A committed **workspace** fixes the
shareable baseline — stacks, named servers and cases, recipes, and eval/bench
definitions. A git-ignored **local bindings** file supplies machine-private
facts (model weights, machines, devices, ports, and placement). Managed serving,
measurement, and image-build workflows write durable, file-first **records**
you can inspect, compare, and reproduce.

- **Serve lifecycles** — long-running framework servers, single-role or
  prefill/decode disaggregated across machines, with readiness, logs, and
  verified cleanup in the record.
- **Five framework integrations** — vLLM, SGLang, TensorRT-LLM, TokenSpeed,
  and the reusable Specialized Engine contract behind one typed adapter
  protocol.
- **Closed-loop recipes** — serve + eval (lm-eval) + bench (AIPerf) suites in
  one command, with per-case metrics and raw artifacts preserved.
- **Measurements** — native and lm-eval Evals including vision smoke; serving
  Benches over random, corpus, replay, and image-decorated request sources,
  linear sessions, and SemiAnalysis AgentX agentic trace replay; standalone
  `inferlab bench` against a running server.
- **Runtime images** — build and validate OCI images from the workspace, or
  select a digest-pinned external image, then launch servers, recipes, and
  ad-hoc `inferlab run` commands from them.
- **Profiling** — attach managed Nsight Systems collection or engine-native
  traces to selected workloads.
- **Source identity** — records carry the workspace revision and a source
  digest; a clean-workspace run is reproducible from a fresh checkout.
- **Operator journal** — an append-only scratchpad on the same time axis as
  the records.
- **View-only TUI** — one responsive workspace console for declared definitions,
  concurrent CLI work, records, referenced logs, and journal context.

<p align="center">
  <img src="https://raw.githubusercontent.com/Infer-Lab/InferLab/main/docs/assets/inferlab-tui-demo.png" width="976" alt="InferLab view-only TUI: a sidebar with views and status counts, an Overview of issues, current work, and recent records by day, and a benchmark record with its per-case metrics">
</p>
<p align="center"><sub>Rendered by the real InferLab TUI with synthetic demo data; no local workspace or machine identifiers are shown.</sub></p>

- **Web console** — the same views in a browser over several workspaces,
  Bench comparison across records, and serve, recipe, bench, and journal
  workflows launched as detached CLI jobs behind a per-start access token.

<p align="center">
  <img src="https://raw.githubusercontent.com/Infer-Lab/InferLab/main/docs/assets/inferlab-web-compare.png" width="976" alt="InferLab web console comparing output-token throughput across three Bench records from two workspaces, one line per record over concurrency on a log scale, with a table of the same values">
</p>
<p align="center"><sub>Rendered by the real InferLab web console with synthetic demo data; no local workspace or machine identifiers are shown.</sub></p>

## Install

Download a release binary (x86_64 / aarch64 Linux):

```sh
curl -fsSL https://github.com/Infer-Lab/InferLab/releases/latest/download/install.sh | sh
```

Or install from the crates registry (Rust 1.89+):

```sh
cargo install inferlab
```

Or build from a checkout with `cargo install --path crates/inferlab`. The
published library crates (`inferlab-runtime`, `inferlab-profiler`,
`inferlab-protocol`, `inferlab-proxy`, `inferlab-serve-domain`) exist to
build the binary; their APIs are experimental and carry no stability promise
yet.

`inferlab --version` prints the adapter protocol line the binary speaks; the
[backend support](plugins/inferlab/skills/inferlab/references/backend-support.md) correspondence table names the
Adapter SDK and integration packages a workspace pins for it.

### Agent skill

InferLab ships an operator skill for Claude Code and Codex, embedded in the
binary at the same version — no checkout or network access needed:

```sh
inferlab agent install --agent all
```

`--from-checkout <DIR>` overrides the source with a local checkout or
unpacked release plugin tarball, for testing an unreleased change.

## Quick start

In a workspace (see [`docs/rfc/`](docs/rfc/) for the full contract, starting at RFC-0001):

```sh
inferlab workspace show                     # validate and browse public definitions
# write .inferlab/local.toml (copy .inferlab/local.example.toml when provided)
pixi install --locked --all                 # realize every stack's selected Pixi environment
inferlab stack status                       # confirm environments and run declared stack checks
inferlab toolchain install                  # only for lm-eval Evals or serving Benches
inferlab tui                                # observe this workspace; never starts or changes work
inferlab web                                # browser console; prints a tokenized local URL

inferlab recipe run my-recipe --dry-run     # validate placement, devices, commands, environment
inferlab recipe run my-recipe --case tp2    # closed loop: serve + eval/bench + cleanup
inferlab serve start my-server --case tp2   # or drive the pieces manually
inferlab bench random-8k1k --serve <ID>
inferlab serve stop <ID>
```

`inferlab toolchain install` fetches the pinned measurement packages, so it
needs network access, and it must be rerun after upgrading InferLab.

Non-dry-run `recipe run`, `bench`, and `image build` print JSON naming a
record under `.inferlab/records/<ID>/`, even when they fail; `serve start`
prints it only on success, and a start that fails after writing its record
names it in the error on stderr. `--dry-run`
on those commands resolves and validates without launching or writing one.

## Documentation

- [Product and documentation website](https://infer-lab.github.io/InferLab/):
  the searchable public entry point for current operator guidance, RFCs, and
  ADRs.
- [Workspace authoring](plugins/inferlab/skills/inferlab/references/workspace-authoring.md):
  public definitions, local bindings, heterogeneous P/D placement, typed
  patches, and dry-run.
- [Backend support](plugins/inferlab/skills/inferlab/references/backend-support.md): maintained backend capabilities
  and integration package names.
- [View-only TUI](docs/tui.md): workspace observation, source labels, views,
  keys, search, refresh, and stale-state semantics.
- [Web console](docs/web.md): browser access, registered workspaces, Bench
  comparison, launched jobs, and notifications.
- [Specialized Engines](docs/specialized-engine.md): the reusable token-worker
  Engine contract for hardware-by-model specialized backends.
- [RFC-0001 — Specification Overview And Authority Map](docs/rfc/RFC-0001.md):
  the entry point of the normative external contract; topic RFCs under
  [docs/rfc/](docs/rfc/) own workspaces/stacks, servers/execution,
  measurements/toolchains, evidence, the integration protocol, runtime
  images, agent plugin distribution, runtime time bounds, the view-only TUI,
  the public website, and the web console.
- [Architecture decisions](docs/adr/): accepted ADRs plus superseded and
  rejected historical decisions.
- [`plugins/inferlab/skills/inferlab/SKILL.md`](plugins/inferlab/skills/inferlab/SKILL.md):
  the operator workflow, as taught to agents.

## License

MIT — see [LICENSE](LICENSE); `inferlab license` prints the full text.
