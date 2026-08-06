# Workspace authoring references

An InferLab workspace has two authorities:

- committed `.inferlab/workspace.toml` and `.inferlab/workspace.d/*.toml` files
  describe shareable models, stacks, servers, cases, measurements, and recipes;
- git-ignored `.inferlab/local.toml` binds those definitions to model weights,
  machines, devices, ports, and placement for one operator.

Run `inferlab workspace show` to validate and browse the committed authority.
It does not read local bindings or inspect a stack realization. Use
`inferlab workspace show --json` when another tool needs the canonical merged
definition.

Read only the reference matching the authoring task:

| Task | Reference |
| --- | --- |
| Workspace definitions, placement, upgrades, and validation | [workspace-definition.md](workspace-definition.md) |
| Profiling, runtime images, and invocation patches | [execution-authoring.md](execution-authoring.md) |
| Eval, Bench, datasets, sessions, metrics, and SLOs | [measurement-authoring.md](measurement-authoring.md) |
