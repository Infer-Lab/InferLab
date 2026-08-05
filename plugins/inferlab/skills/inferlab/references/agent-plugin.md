# Agent Plugin Operations

The InferLab binary carries a plugin package at the same product version. Agent
operations are independent of workspace definitions, local bindings, and
execution records and do not write experiment records. Use each subcommand's
`--help` for its current source selection, runtime selection, report contract,
and mutation boundary.

```sh
inferlab agent doctor [--agent codex|claude|all]
inferlab agent install [--agent codex|claude|all]
inferlab agent update [--agent codex|claude|all]
inferlab agent uninstall [--agent codex|claude|all]
```

Use `install` for a fresh runtime or to qualify an unreleased checkout, `update`
after changing the InferLab binary, `doctor` for read-only source and CLI
diagnosis, and `uninstall` only when removing the plugin. For a checkout:

```sh
inferlab agent install --agent all --from-checkout /path/to/inferlab
```

Read the final machine-readable report rather than scraping native agent CLI
output. A partial failure still retains the attempted commands and per-runtime
outcomes.

After installing a new InferLab binary, run:

```sh
inferlab agent update --agent all
inferlab agent doctor --agent all
```

`inferlab license` prints the complete retained MIT notice. Plugin archives and
the embedded package carry the same notice.
