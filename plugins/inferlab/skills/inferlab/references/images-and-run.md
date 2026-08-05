# Images And Ad-Hoc Execution

## Runtime Image Definitions

A named image selects one stack, a base image that InferLab resolves to an
immutable per-platform digest, one or more target platforms, an optional subset
of the stack's source packages, and recipe-based validations:

```toml
[images.vllm-runtime]
stack = "vllm"
base_image = "registry.example/cuda:13.0"
platforms = ["linux/amd64"]
packages = ["upstream/vllm"]
validations = [
  { recipe = "smoke", server_case = "tp1" },
]
```

Omitting `packages` selects all stack `source_paths`. A validation references a
recipe and optional server case; it does not repeat model, server, placement, or
measurement configuration. Image build requires a clean workspace.

Local bindings currently support only a local Docker builder:

```toml
[builders.local]
kind = "local-docker"
```

Run the closed loop with:

```sh
inferlab image build <IMAGE> --dry-run
inferlab image build <IMAGE> --builder local --placement local
inferlab image build <IMAGE> --export /absolute/output/directory
```

Dry-run resolves package builds, content-closure reuse, platforms, inspection,
export paths, and validation eligibility without assembly. A real build creates
one record, builds release inputs, assembles and inspects each producible
platform, optionally exports unique OCI archives, and runs every eligible
recipe validation. One platform or validation failure does not suppress the
remaining batch. Built images remain in local builder storage; this workflow
does not push to a registry.

Generated contexts and portable image metadata exclude model weights, model
locators, workspace paths, builder hosts, placements, and other machine-private
facts. Model weights remain runtime inputs. Equal content closure and platform
reuse one assembly within the invocation.

## Container Bindings

Machine-local `container` facts apply to image validation, built-image launch,
and external-image launch:

```toml
[machines.local.container]
pass_env = ["HF_TOKEN"]
devices = ["/dev/infiniband"]
memlock_unlimited = true
capabilities = ["IPC_LOCK", "SYS_NICE"]
```

Environment values are passed by name and never enter image content or the
recorded command. Device paths must be absolute. Supported capabilities are
`IPC_LOCK`, `SYS_NICE`, and `SYS_PTRACE`; privileged mode is never used.

Adapter image plan/render uses local `[adapter].image_timeout_seconds` and the
optional `image_device` workaround. These are distinct from builder or
validation timeouts.

## Built And External Images

`serve start`, `recipe run`, and `run` may select a successful
`--image <IMAGE_BUILD_RECORD>` assembly. Server/recipe selection requires the
host platform, stack, placement, and builder storage to be compatible and does
not distribute the image between machines.

A workspace may instead declare an image InferLab did not build:

```toml
[external_images.official]
reference = "registry.example/server@sha256:<64-hex-digest>"
integration = "vllm"
```

Select it with `--external-image official`. InferLab verifies the digest exists
in builder storage on every launch machine; it never pulls automatically and a
failure names the exact pull command. Built and external selections are
mutually exclusive. External images are declared, not qualified by InferLab.

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
indexes. Use `run` for diagnostics, not evidence; a qualification must use a
managed recorded workflow.

Never execute `.pixi/envs/<env>/bin/python` or another binary through its
materialized prefix. That bypasses manifest activation variables and package
activation scripts, so it observes a different environment from InferLab.
