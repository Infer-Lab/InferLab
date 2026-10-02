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
launches and writes no execution record:

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
Use `run` for diagnostics, not evidence; qualification requires a
managed recorded workflow.

Never execute a binary directly through `.pixi/envs/<env>/bin/`; the
[skill entry point](../SKILL.md) explains why `run` replaces it.
