"""Prepare the source selected by one lm-eval definition without materializing it."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    ClientStatus,
    EvalDefinitionInputLmEval,
    EvalTaskSourceInputBuiltIn,
    EvalTaskSourceInputBundled,
    EvalTaskSourceInputWorkspaceYaml,
    MeasurementDataAssetAcquiredSource,
    MeasurementDataAssetCacheOutcome,
    MeasurementDataAssetCacheStore,
    MeasurementDataAssetContentEntry,
    MeasurementDataAssetEffectiveSelection,
    MeasurementDataAssetEffectiveSelectionEval,
    MeasurementDataAssetPreparationNextPhase,
    MeasurementDataAssetPreparationPhaseResolve,
    MeasurementDataAssetPreparationPhaseSnapshotLocal,
    MeasurementDataAssetPreparationRequest,
    MeasurementDataAssetPreparationResult,
    MeasurementDataAssetReadiness,
    MeasurementDataAssetReadinessClosed,
    MeasurementDataAssetReadinessOpaque,
    MeasurementDataAssetRemoteMetadataOutcome,
    MeasurementDataAssetSourceBytesOutcome,
    MeasurementDataAssetSourceInputEval,
    MeasurementDataAssetVerification,
    MeasurementDataFiles,
)

from inferlab_eval_runner.local_eval_source import (
    plan_local_eval_source,
    snapshot_local_eval_source,
)
from inferlab_eval_runner.task_resolution import resolve_lm_eval_data_source


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _normalized_data_files(value: object) -> dict[str, list[str]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"train": [value]}
    if isinstance(value, list) and all(isinstance(pattern, str) for pattern in value):
        return {"train": cast(list[str], value)}
    if isinstance(value, dict) and all(isinstance(split, str) for split in value):
        normalized: dict[str, list[str]] = {}
        for split, raw_patterns in value.items():
            if isinstance(raw_patterns, str):
                normalized[cast(str, split)] = [raw_patterns]
            elif isinstance(raw_patterns, list) and all(
                isinstance(pattern, str) for pattern in raw_patterns
            ):
                normalized[cast(str, split)] = cast(list[str], raw_patterns)
            else:
                raise ValueError("lm-eval returned an unsupported data_files value")
        return normalized
    raise ValueError("lm-eval returned an unsupported data_files shape")


def _reported_huggingface_cache_stores() -> list[MeasurementDataAssetCacheStore]:
    def local_outcome(path: Path) -> MeasurementDataAssetCacheOutcome:
        try:
            path.stat()
        except FileNotFoundError:
            return MeasurementDataAssetCacheOutcome.miss
        except OSError:
            return MeasurementDataAssetCacheOutcome.unavailable
        if not path.is_dir():
            return MeasurementDataAssetCacheOutcome.unavailable
        try:
            populated = next(path.iterdir(), None) is not None
        except OSError:
            return MeasurementDataAssetCacheOutcome.unavailable
        return (
            MeasurementDataAssetCacheOutcome.partial_reuse
            if populated
            else MeasurementDataAssetCacheOutcome.miss
        )

    stores: list[MeasurementDataAssetCacheStore] = []
    try:
        hub_constants = importlib.import_module("huggingface_hub.constants")
        hub_cache = hub_constants.HUGGINGFACE_HUB_CACHE
    except (AttributeError, ImportError):
        stores.append(
            MeasurementDataAssetCacheStore(
                authority="huggingface_hub",
                purpose="repository_files",
                path=None,
                outcome=MeasurementDataAssetCacheOutcome.unavailable,
            )
        )
    else:
        hub_cache_path = Path(hub_cache)
        stores.append(
            MeasurementDataAssetCacheStore(
                authority="huggingface_hub",
                purpose="repository_files",
                path=str(hub_cache_path),
                outcome=local_outcome(hub_cache_path),
            )
        )
    try:
        datasets_config = importlib.import_module("datasets.config")
        datasets_cache = datasets_config.HF_DATASETS_CACHE
    except (AttributeError, ImportError):
        stores.append(
            MeasurementDataAssetCacheStore(
                authority="huggingface_datasets",
                purpose="dataset_materialization",
                path=None,
                outcome=MeasurementDataAssetCacheOutcome.unavailable,
            )
        )
    else:
        datasets_cache_path = Path(datasets_cache)
        stores.append(
            MeasurementDataAssetCacheStore(
                authority="huggingface_datasets",
                purpose="dataset_materialization",
                path=str(datasets_cache_path),
                outcome=local_outcome(datasets_cache_path),
            )
        )
    return stores


def _bundled_readiness(
    source: EvalTaskSourceInputBundled,
) -> MeasurementDataAssetReadinessClosed:
    root = Path(source.path).resolve(strict=True).parent
    assets = [
        ("estonia/dataset.json", root / "dataset.json", source.dataset_asset_sha256),
        ("estonia/estonia.py", root / "estonia.py", source.scorer_sha256),
        (
            "estonia/estonia.yaml",
            Path(source.path).resolve(strict=True),
            source.task_definition_sha256,
        ),
        ("estonia/prompt.txt", root / "prompt.txt", source.prompt_asset_sha256),
    ]
    closure: list[MeasurementDataAssetContentEntry] = []
    verification: list[MeasurementDataAssetVerification] = []
    for relative_path, path, expected in assets:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        matched = observed == expected
        verification.append(
            MeasurementDataAssetVerification(
                subject=relative_path,
                expected=expected,
                observed=observed,
                matched=matched,
            )
        )
        if not matched:
            raise ValueError(
                f"bundled task asset {relative_path!r} does not match its release digest"
            )
        closure.append(
            MeasurementDataAssetContentEntry(
                relative_path=relative_path,
                sha256=observed,
            )
        )
    return MeasurementDataAssetReadinessClosed(
        kind="closed",
        acquired_source=MeasurementDataAssetAcquiredSource.model_validate(
            {
                "kind": "release_qualified",
                "identity": f"inferlab-bundled-task:{source.task_closure_sha256}",
                "closure": [entry.model_dump() for entry in closure],
            }
        ),
        verification=verification,
    )


def prepare_eval_data_asset(
    request: MeasurementDataAssetPreparationRequest,
) -> MeasurementDataAssetPreparationResult:
    source_input = request.source.root
    if not isinstance(source_input, MeasurementDataAssetSourceInputEval):
        raise TypeError("Eval source preparation requires an Eval source input")
    phase = request.phase.root
    if not isinstance(
        phase,
        (
            MeasurementDataAssetPreparationPhaseResolve,
            MeasurementDataAssetPreparationPhaseSnapshotLocal,
        ),
    ):
        raise ValueError("Eval source preparation supports resolve and snapshot-local phases")
    definition = source_input.definition.root
    if not isinstance(definition, EvalDefinitionInputLmEval):
        raise ValueError("OpenAI smoke Eval has no measurement data asset")

    task_source = definition.task.root
    resolution: dict[str, object] | None = None
    if isinstance(task_source, EvalTaskSourceInputBuiltIn):
        task_identity = task_source.name
        selection_value: dict[str, object] = {}
    else:
        resolution = resolve_lm_eval_data_source(
            definition,
            Path(source_input.workspace_root),
            [Path(path) for path in source_input.workspace_source_exclusions],
            None,
        )
        raw_selection = resolution.get("effective_dataset_selection")
        if not isinstance(raw_selection, dict):
            raise ValueError("resolved lm-eval task omitted its effective dataset selection")
        selection_value = raw_selection
        resolved_identity = resolution.get("task_identity")
        if not isinstance(resolved_identity, str):
            raise ValueError("resolved lm-eval task omitted its task identity")
        task_identity = resolved_identity
    normalized_data_files = _normalized_data_files(selection_value.get("data_files"))
    selection = MeasurementDataAssetEffectiveSelectionEval(
        kind="eval",
        task_identity=task_identity,
        dataset_path=_optional_string(selection_value.get("dataset_path")),
        dataset_name=_optional_string(selection_value.get("dataset_name")),
        evaluation_split=_optional_string(selection_value.get("evaluation_split")),
        fewshot_split=_optional_string(selection_value.get("fewshot_split")),
        data_files=(
            MeasurementDataFiles(root=normalized_data_files)
            if normalized_data_files is not None
            else None
        ),
    )

    workspace_root = Path(source_input.workspace_root).resolve(strict=True)
    local_plan, local_opaque_reason = (
        plan_local_eval_source(resolution, workspace_root)
        if isinstance(task_source, EvalTaskSourceInputWorkspaceYaml) and resolution is not None
        else (None, None)
    )
    if isinstance(phase, MeasurementDataAssetPreparationPhaseSnapshotLocal):
        if local_plan is None:
            raise ValueError("workspace Eval source did not resolve to a closable local file set")
        return MeasurementDataAssetPreparationResult(
            schema_version=1,
            status=ClientStatus.succeeded,
            effective_selection=MeasurementDataAssetEffectiveSelection(root=selection),
            readiness=MeasurementDataAssetReadiness(
                root=snapshot_local_eval_source(local_plan, Path(request.artifact_dir))
            ),
            cache_stores=[],
            remote_metadata=MeasurementDataAssetRemoteMetadataOutcome.not_accessed,
            source_bytes=MeasurementDataAssetSourceBytesOutcome.reused,
            error=None,
        )

    readiness: MeasurementDataAssetReadinessClosed | MeasurementDataAssetReadinessOpaque
    if isinstance(task_source, EvalTaskSourceInputBundled):
        readiness = _bundled_readiness(task_source)
    elif local_plan is not None:
        return MeasurementDataAssetPreparationResult(
            schema_version=1,
            status=ClientStatus.succeeded,
            effective_selection=MeasurementDataAssetEffectiveSelection(root=selection),
            readiness=None,
            next_phase=MeasurementDataAssetPreparationNextPhase(root="snapshot_local"),
            cache_stores=[],
            remote_metadata=MeasurementDataAssetRemoteMetadataOutcome.not_accessed,
            source_bytes=MeasurementDataAssetSourceBytesOutcome.reused,
            error=None,
        )
    else:
        unresolved_path = (
            task_source.path if isinstance(task_source, EvalTaskSourceInputWorkspaceYaml) else None
        )
        readiness = MeasurementDataAssetReadinessOpaque(
            kind="opaque",
            reason=local_opaque_reason
            or (
                "the pinned lm-eval task interface does not expose and bind the complete "
                "dataset content closure before task materialization"
            ),
            unresolved_path=unresolved_path,
            deferred_source_access=True,
        )
    return MeasurementDataAssetPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        effective_selection=MeasurementDataAssetEffectiveSelection(root=selection),
        readiness=MeasurementDataAssetReadiness(root=readiness),
        cache_stores=(
            []
            if isinstance(task_source, EvalTaskSourceInputBundled)
            else _reported_huggingface_cache_stores()
        ),
        remote_metadata=MeasurementDataAssetRemoteMetadataOutcome.not_accessed,
        source_bytes=MeasurementDataAssetSourceBytesOutcome.not_accessed,
        error=None,
    )
