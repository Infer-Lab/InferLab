# Workspaces And Stacks

For workspace-definition syntax, machine-local bindings, placement shapes, and
upgrade pins, read [Workspace definitions and placement](workspace-definition.md).
This reference covers discovery, stack realization, and lock operations.

## Authority And Discovery

The public authority is `.inferlab/workspace.toml` plus identifier-disjoint
fragments under `.inferlab/workspace.d/*.toml`. Machine-private authority belongs
in ignored `.inferlab/local.toml` or the explicit `--local` selection. Run:

```sh
inferlab workspace show
inferlab workspace show --json
```

The first form validates and browses the merged catalog. The JSON form emits
the canonical merged public definition for another tool. Neither reads local
bindings. The global `--workspace <DIR>` selects a root explicitly; otherwise
InferLab searches the current directory and its parents.

## Stack Lifecycle

Prepare every declared Pixi environment in a fresh checkout, then inspect the
selected stack realization:

```sh
pixi install --locked --all
inferlab stack status
```

Plain `pixi install --locked` realizes only Pixi's implicit default environment.
Use `--all` or select the named environment explicitly.

`stack status [STACK]` needs no local bindings. It distinguishes the retained
Pixi confirmation (`confirmed`, `never-installed`, or `not-usable`) from the
current declared-check outcomes and overall readiness. Checks run on every
status call for a confirmed realization and before launches. Failure reports
captured output and the declared repair hint; InferLab does not run the repair.

Machine-local runtime caches remain convenience state. Their presence is not
stack confirmation or portable experiment evidence.

## Lock And Upgrade Workflow

Read [Upgrading](workspace-definition.md#upgrading-to-012) before changing
workspace package pins. Update the committed manifest, then produce and inspect
the lock through the workspace command:

```sh
inferlab workspace lock
inferlab workspace show
inferlab stack status
```

`workspace lock` requires a clean local prefix and writes the committed Pixi
lock. A successful lock does not establish that a previously realized
environment has been rebuilt; `stack status` reports confirmation and current
declared checks separately.
