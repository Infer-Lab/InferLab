# Workspaces And Stacks

## Authority And Discovery

The public authority is `.inferlab/workspace.toml` plus identifier-disjoint
fragments under `.inferlab/workspace.d/*.toml`. The root owns
`schema_version`; fragments may own complete named definitions. Run:

```sh
inferlab workspace show
inferlab workspace show --json
```

The first form validates and browses the merged catalog. The JSON form emits
the canonical merged public definition for another tool. Neither reads local
bindings. The global `--workspace <DIR>` selects a root explicitly; otherwise
InferLab searches the current directory and its parents.

Machine-private authority belongs only in ignored `.inferlab/local.toml`, or an
alternate file selected with `--local` on `serve start`, `recipe run`, and
`image build`. It holds model locators, machines, devices, ports, launch
targets, placements, builders, and adapter settings. Keep a tracked
`.inferlab/local.example.toml` generic.

## Stack Lifecycle

Each stack names one integration, one Pixi environment, zero or more
workspace-relative `source_paths`, declared `checks`, and optional
`image_postprocess` scripts. Prepare a fresh checkout with:

```sh
pixi install --locked --all
inferlab stack status
```

Plain `pixi install --locked` realizes only Pixi's implicit default
environment. Use `--all` or select the named environment explicitly.

`stack status [STACK]` needs no local bindings. It distinguishes the retained
Pixi confirmation (`confirmed`, `never-installed`, or `not-usable`) from the
current declared-check outcomes and overall readiness. Checks run on every
status call for a confirmed realization and before launches. Failure reports
captured output and the declared `repair_hint`; InferLab does not run the
repair. `workspace lock` produces the committed lock from a clean local prefix.

Machine `cache_root` provides a machine-local root for runtime JIT caches.
InferLab allocates cache locations from the resolved stack and source identity;
the cache remains convenience state rather than portable evidence.

## Local Placement

A minimal local binding is:

```toml
default_placement = "local"

[model_weights.example]
locator = "/models/example"

[machines.local]
host = "127.0.0.1"
devices = [0, 1]
ports = [8000]

[placements.local]
machines = ["local"]
```

Machines may launch locally or through a declared OpenSSH target and may name a
remote workspace. OpenSSH configuration owns connection and keepalive policy.
Do not add SSH timing knobs to workspace TOML.

Role placement may use a machine pool, one direct rank, one multi-rank replica,
or an ordered replica list. Replica and rank indexes derive from list order.
Use `machine_locators` when model paths differ by machine. A prefill/decode
frontend is placed as the zero-device `gateway` role even when one process
realizes both Gateway and P/D Router.

Machine-local adapter operations have independent optional deadlines:

```toml
[adapter]
timeout_seconds = 30
image_timeout_seconds = 120
```

`timeout_seconds` covers process-backed integration plan/render; the image
deadline covers image-backed plan/render. `image_device` is a separate optional
workaround for container runtimes that cannot create a device-less adapter
container.

## Product Upgrade

Use the bundled workspace-authoring guide linked from `SKILL.md` as the
authority for the current adapter protocol and exact SDK/integration package
pins. It comes from the same source snapshot as the installed plugin. Update
the adapter SDK and selected integration together, then run
`inferlab workspace lock`. The product-owned measurement SDK is internal to
`toolchain install` and does not belong in a serving workspace.

Existing synthetic Bench definitions must add explicit prompt authority. Read
[Measurements](measurements.md) before migrating them.
