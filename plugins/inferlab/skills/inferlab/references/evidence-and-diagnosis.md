# Evidence And Diagnosis

## Record-First Reading

Managed workflows write `.inferlab/records/<ID>/record.json` plus referenced
logs and artifacts. `recipe run` and `bench` print the record as one JSON
report carrying its `id`, and `image build` prints a report keyed `record_id`;
all three print it even when the workflow fails and then exit with `E4001`. A
failed `serve start` prints no report and names its record in the stderr
error. Do not scrape diagnostic progress for results.

- Eval records reference `cases/eval/result.json` and `cases/eval/artifacts/`.
  They preserve normalized metrics, gate and repeated-trial evidence, task
  resolution, probes, request-body identity, native command, and artifacts.
- Bench records reference `cases/<case>/result.json` and each case's artifacts.
  They preserve the resolved route, effective load, warmup/profiling slices,
  request population, prompt authority, tokenizer identity, template
  provenance, prefix evidence, the request-body fragment, normalized metrics,
  SLO conclusions, capture, native commands, and native AIPerf output.
- Server records preserve placement, device hardware identity, topology,
  frontend and role/rank bindings, effective settings, readiness, profiler
  targets, image selection, process state, and cleanup, including the
  device-residual evidence described under
  [Failure And Cleanup](#failure-and-cleanup).
- Recipe and image records preserve child references and partial outcomes rather
  than flattening or discarding a failed branch.

Compare runs on typed record facts: declared/effective definitions, revision,
`source_digest`, package and integration identity, route, population, case load,
metrics, and cleanup. `revision_reproducible: true` means the run used a clean
source revision; it does not compare that run with another record. Reproduce by
checking out the recorded revision, recreating local bindings, and comparing
both revision and source digest.

## TUI

`inferlab tui` is view-only and writes no session or workflow state:

```sh
inferlab tui
inferlab tui --refresh-interval 2s
inferlab --workspace /path/to/workspace tui
```

Views cover Overview, Operations, Records, and Workspace declarations. Details
label each fact as declared, recorded, ephemeral, or observed. Running server
records receive bounded live process observation; a failed refresh retains the
last successful value as stale rather than rewriting the record.

Run `inferlab tui --help` for the current binary's key summary. The complete
navigation, layout, search, comparison, and source-health contract is in the
[TUI guide](https://infer-lab.github.io/InferLab/docs/guides/tui/).

Concurrent CLI operations publish ephemeral observations under
`.inferlab/runtime/observations/`. Normal completion removes them. Abrupt
residue can appear stale in the TUI, but it is not execution evidence and must
not be reconstructed into a record.

## Scratchpad

The append-only narrative lives at
`.inferlab/scratchpads/journal.jsonl` and never changes execution resolution:

```sh
inferlab scratchpad note "tp1 OOMs at readiness" --record last --topic flash
inferlab scratchpad note "compare with baseline" --record <ID> --author <NAME>
inferlab scratchpad show --topic flash
inferlab scratchpad show --all
```

`--record` is repeatable; `last` resolves the newest local record. Record links
must exist. The default view shows the recent tail.

## Failure And Cleanup

Stable error codes, not messages, are the machine-facing failure contract. The
message names the failing fact and retains diagnostic context. Important
families include configuration/environment/toolchain (`E1xxx`),
integration/protocol (`E2xxx`), placement and network resolution (`E3xxx`),
closed-loop failure, lifecycle, profiling, image, and data-asset preparation
(`E4xxx`), record/operation observation (`E5xxx`), scratchpad (`E6xxx`), agent
plugin (`E7xxx`), ad-hoc execution (`E8xxx`), and command output (`E9xxx`).

A failed launch or closed-loop run still finalizes its record and attempts
cleanup. Inspect the record before retrying. For a suspected manual server leak:

```sh
inferlab serve status <SERVER_RECORD_ID>
inferlab serve logs <SERVER_RECORD_ID>
inferlab serve stop <SERVER_RECORD_ID>
```

`stop` is idempotent. Distinguish business-result failure, profiling/control
failure, process cleanup failure, and incomplete evidence instead of reducing
all of them to a nonzero shell status.

After a server's process cleanup verifies, InferLab probes every assigned
device on its launch machine and records the outcome at
`process_evidence.<process>.cleanup[].device_residuals`: `freed`,
`residual_held` with the observed bytes, or `probe_unavailable`. Only memory
held by processes that no longer exist on that machine, or exist only as
zombies, counts as residual; memory another live session holds on the same
device does not. A device still holding residual memory is probed again until
it frees or the settle window recorded as `device_residual_settle_window_ms`
closes; residual memory at the window's end makes the cleanup unverified, and
the error names the machine and device. `probe_unavailable`, for example on a
machine without the probe tool, does not fail cleanup.

A server record is readable only by a binary that writes the same server record
schema version; [workspace-definition.md](workspace-definition.md) owns the
upgrade procedure that follows from it.

## Privacy

Do not publish records without an operator privacy review. The full privacy
rule set lives in [SKILL.md](../SKILL.md#privacy).
