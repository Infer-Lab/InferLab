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

Dry-run resolves package builds, content-closure reuse, platforms, inspection,
export paths, and validation eligibility without assembly. A real build creates
one record, assembles and inspects every producible platform, optionally exports
unique OCI archives, and runs eligible recipe validations. One platform or
validation failure does not suppress the remaining batch.

Built images remain in local builder storage; this workflow does not push to a
registry. Portable image contexts and metadata exclude model weights, model
locators, workspace paths, builder hosts, placements, and other machine-private
facts.

## Built And External Image Selection

`serve start`, `recipe run`, and `run` may select a successful
`--image <IMAGE_BUILD_RECORD>` assembly. Server and recipe selection requires
the host platform, stack, placement, and builder storage to remain compatible;
InferLab does not distribute an image between machines.

`--external-image <ID>` selects a workspace-declared digest-pinned artifact.
InferLab verifies it in builder storage on every launch machine and never pulls
it automatically. Built and external selections are mutually exclusive.
Image-backed launches reject profiling until an in-container profiler contract
exists.

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
mount or device implicitly. `--mount /absolute/path` is same-path read-only;
append `:rw` for write access. `--devices 0,1` exposes only those host device
indexes. Use `run` for diagnostics, not evidence; qualification requires a
managed recorded workflow.

Never execute a binary directly through `.pixi/envs/<env>/bin/`. That bypasses
manifest activation variables and package activation scripts, so it observes a
different environment from InferLab.
