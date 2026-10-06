# Images And Ad-Hoc Execution

For runtime-image definitions, external-image declarations, builders, container
bindings, and their exact fields, read
[Runtime image authoring](execution-authoring.md#runtime-images-and-ad-hoc-execution).
This reference covers image workflows, selection, and probes.

## Build Workflow

Resolve first, then run the recorded build:

```sh
inferlab image build <IMAGE> --dry-run
inferlab image build <IMAGE> --builder local --placement local
inferlab image build <IMAGE> --export /absolute/output/directory
```

Dry-run resolves package builds, including the project directory each selected
source path builds at, content-closure reuse, platforms, inspection, export
paths, and validation eligibility without assembly. A real build creates
one record, assembles and inspects every producible platform, optionally exports
unique OCI archives, and runs eligible recipe validations. One platform or
validation failure does not suppress the remaining batch.

An image that selects packages needs `inferlab toolchain install` first, dry-run
included: its image-packaging runtime makes every built package path-neutral.
Builds receive compiler path maps, so source, Pixi environment, Cargo home,
and workspace paths appear as fixed `/opt/inferlab-*` paths; library search
paths are rewritten to the image environment; and each package, including its
device code, is verified to contain no workspace, environment, home, Cargo
home, or build-tree path before it is cached or assembled. The maps reach C,
C++, and CUDA through `CFLAGS`, `CXXFLAGS`, and `NVCC_APPEND_FLAGS`, and rustc
through the rustflags variable Cargo already honors: `CARGO_ENCODED_RUSTFLAGS`
or `RUSTFLAGS` when the stack activation sets one, otherwise
`CARGO_BUILD_RUSTFLAGS`, which keeps a package's configured `build.rustflags`.
A failure names the member and what it embeds — usually a build backend that
ignores those variables, a Rust package whose Cargo configuration sets a
matching `target` rustflags table, a crate that embeds
`env!("CARGO_MANIFEST_DIR")`, or device code compiled with `-lineinfo`. Built wheels
reach the image only as installed packages. Debug information follows the
stack's build configuration: set the framework's build type (for vLLM,
`CMAKE_BUILD_TYPE=Release` in stack activation) to ship smaller images without
it.

Built images remain in local builder storage; this workflow does not push to a
registry.

## Built And External Image Selection

`serve start`, `recipe run`, and `run` may select a successful
`--image <IMAGE_BUILD_RECORD>` assembly. Server and recipe selection requires
the image record's stack to equal the server's stack, a successful assembly for
the host platform, and a placement on the single local machine whose builder
storage holds the image; InferLab does not distribute a built image between
machines.

`--external-image <ID>` selects a workspace-declared digest-pinned artifact
whose integration claim matches the server stack's integration; its commands
are lowered with the workspace's `adapter` Pixi environment (see
[External images](execution-authoring.md#runtime-images-and-ad-hoc-execution)).
InferLab verifies the image in builder storage on the controller and every
launch machine and never pulls it automatically. An external-image placement
may span SSH machines, and a remote machine then needs no workspace
realization: its `workspace` path is only a launch directory. Resolution
probes the framework version inside the image, and the record marks the launch
as not qualified by the workspace, carrying that observed version. Built and
external selections are mutually exclusive. Image-backed launches reject
profiling until an in-container profiler contract exists.

## Ad-Hoc Probes

`inferlab run` executes one command with the same activation used by product
launches. Without `--record` it writes no execution record:

```sh
inferlab run -- python -c "import vllm; print(vllm.__version__)"
inferlab run --stack vllm -- pytest tests/ -k smoke
inferlab run --image <RECORD_ID> --devices 0 -- nvidia-smi -L
inferlab run --external-image official --mount /data -- python3 /data/probe.py
```

With multiple stacks, `--stack` is required. Container probes receive no host
mount, device, or machine container grant implicitly, and an external image
declaring `entrypoint = "image"` runs the command through its own entrypoint. `--mount /absolute/path` is same-path read-only;
append `:rw` for write access. `--devices 0,1` exposes only those host device
indexes. Local probes execute with `CUDA_VISIBLE_DEVICES` set from the
workspace's default placement when every machine it references launches
locally — an explicit `CUDA_VISIBLE_DEVICES` in the environment wins, so
ad-hoc GPU work lands on the workspace's own devices without extra flags.
Use `run` for diagnostics and for workloads InferLab does not model; a run
record says what ran, never a qualification, which requires a managed
recorded workflow.

## Recorded Runs And The InferLab Context

For workloads InferLab does not model — agentic evaluation harnesses, model
quantization, one-off conversions — local `run` is a thin wrapper: the command
runs unchanged and InferLab offers it a context through environment variables
the command may read or ignore.

| Variable | Provided |
| --- | --- |
| `INFERLAB_CONTEXT` (context contract version) and `INFERLAB_WORKSPACE_ROOT` | always |
| `INFERLAB_RECORD_ID`, `INFERLAB_RECORD_ARTIFACTS` (a record-owned directory the command may write) | with `--record` |
| `INFERLAB_SERVE_RECORD`, `INFERLAB_SERVE_BASE_URL` (scheme, host, port; no path), `INFERLAB_SERVE_MODEL` (served model name) | with `--serve <SERVER_RECORD_ID>` of a running server |
| `INFERLAB_MODEL_ID`, `INFERLAB_MODEL_PATH` (the `model_weights` locator for this machine) | with `--model <MODEL>` |

`--record` writes a `run` record under `.inferlab/records/<UTC>-run-<stack>-<pid>/`:
argv, stack and lock identity, workspace snapshot, the provided context and
device projection, the linked server's state at start and exit, the parent
record when nested, stdout and stderr logs, exit status, and timing. The
record says what was provided, not what the command used, and claims no
outputs: the command's own arguments stay the authority for where it wrote.
These options apply to local stack execution only.

Read the context from a script, not from the `inferlab run` command line —
`"$INFERLAB_SERVE_BASE_URL"` written there expands in your shell before the
context exists. A workspace-owned `scripts/swebench.sh`:

```sh
#!/usr/bin/env bash
set -euo pipefail
export OPENAI_BASE_URL="$INFERLAB_SERVE_BASE_URL/v1/"
mini-extra swebench --model "hosted_vllm/$INFERLAB_SERVE_MODEL" \
  --subset verified --slice 0:50 -o out/swebench
cp out/swebench/preds.json "$INFERLAB_RECORD_ARTIFACTS/"
```

```sh
inferlab serve start qwen3-8b-nvfp4
inferlab run --stack swe --record --serve <SERVER_RECORD_ID> -- ./scripts/swebench.sh
```

A script may itself call `inferlab run --record` for each step; a nested run
replaces the inherited context and records the outer run as its parent.
Setting `INFERLAB_` context variables by hand fails the run.

A recorded run behaves differently from an unrecorded one in three ways:

- **Streams.** The command writes to pipes, not a terminal: its output is
  copied to your terminal and the record logs, so progress bars, colors, and
  Python's output buffering follow their non-terminal behavior. It does not
  read the terminal; piped or redirected input still reaches it.
- **Interrupts.** Ctrl-C reaches InferLab, not the command. InferLab sends
  SIGTERM to every process group of the command's process tree, then SIGKILL
  to the processes still alive after the shared grace, and finalizes the
  record as `interrupted`. A harness that saves partial results on SIGINT
  must also handle SIGTERM. A nested run is spared SIGKILL until it finalizes
  its own record. Processes that left the tree and containers started
  through a daemon, such as harness containers started through Docker, are
  not cleaned up by InferLab.
- **Closed readers.** Closing the reader of `inferlab run --record` (for
  example piping into `head`) does not stop the command: the record keeps
  its complete output.

Never execute a binary directly through `.pixi/envs/<env>/bin/`; the
[skill entry point](../SKILL.md) explains why `run` replaces it.
