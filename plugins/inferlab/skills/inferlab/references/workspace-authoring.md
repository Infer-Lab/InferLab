# Workspace authoring references

An InferLab workspace has three sources of definitions and bindings:

- committed `.inferlab/workspace.toml` and `.inferlab/workspace.d/*.toml` files
  describe shareable models, stacks, servers, cases, measurements, and recipes;
- git-ignored `.inferlab/local.toml` binds those definitions to model weights,
  machines, devices, ports, and placement for one operator;
- git-ignored `.inferlab/workspace.local.d/*.toml` files optionally add one
  operator's experiment-only models, servers, Evals, Benches, workload suites,
  and recipes (see
  [Local definitions](workspace-definition.md#local-definitions)).

Read only the reference matching the authoring task:

| Task | Reference |
| --- | --- |
| Workspace definitions, placement, upgrades, and validation | [workspace-definition.md](workspace-definition.md) |
| Profiling, runtime images, and invocation patches | [execution-authoring.md](execution-authoring.md) |
| Eval tasks, datasets, and inference requests | [eval-authoring.md](eval-authoring.md) |
| Serving Bench load, sources, sessions, metrics, and SLOs | [bench-authoring.md](bench-authoring.md) |
