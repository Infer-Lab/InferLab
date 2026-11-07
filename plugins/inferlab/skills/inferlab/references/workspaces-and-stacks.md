# Workspaces And Stacks

For workspace-definition syntax, machine-local bindings, placement shapes, and
upgrade pins, read [Workspace definitions and placement](workspace-definition.md).
This reference covers discovery, stack realization, and lock operations.

## Authority And Discovery

The committed authority is `.inferlab/workspace.toml` plus
identifier-disjoint fragments under `.inferlab/workspace.d/*.toml`.
Machine-private bindings live in the ignored `.inferlab/local.toml` or the file
selected with `--local`; see
[Workspace authoring](workspace-authoring.md) for all three sources. Run:

```sh
inferlab workspace show
inferlab workspace show --json
```

The first form validates and browses the merged catalog; the JSON form emits
the canonical merged public definition for another tool. Neither reads local
bindings or local definitions: `serve start`, `recipe run`, `bench`, and their
dry runs validate those. The global `--workspace <DIR>` selects a root
explicitly; otherwise InferLab searches the current directory and its
parents.

## Stack Lifecycle

Prepare every declared Pixi environment in a fresh checkout, then inspect the
selected stack realization:

```sh
pixi install --locked --all
inferlab stack status
```

Plain `pixi install --locked` realizes only Pixi's implicit default environment.
Use `--all` or select the named environment explicitly.

`stack status [STACK]` needs no local bindings. It reports the retained Pixi
confirmation (`confirmed`, `never-installed`, or `not-usable`) separately from
the current declared-check outcomes and overall readiness. Declared checks
(see [Stacks](workspace-definition.md#stacks)) run on every status call for a
confirmed realization and before launches. A failing check reports its
captured output and the declared repair hint; InferLab does not run the
repair.

## Lock And Upgrade Workflow

Read [Upgrading](workspace-definition.md#upgrading-across-an-adapter-protocol-hard-cut)
before changing workspace package pins. Update the committed manifest, then
produce and inspect the lock through InferLab:

```sh
inferlab workspace lock
inferlab workspace show
inferlab stack status
```

`workspace lock` writes the committed Pixi lock. When packages named under
`pypi-options.no-build-isolation` are also local `path` dependencies, it first
locks and installs every environment without them so those packages can build
against it, then locks the full manifest. A failed or interrupted lock
restores the manifest and the previous lock. A successful lock does not rebuild
an environment that was already realized; `stack status` shows whether it is
still confirmed.
