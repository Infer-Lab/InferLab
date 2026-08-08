"""Prepare a release-qualified AgentX source before serving starts."""

from __future__ import annotations

from huggingface_hub import constants
from inferlab_measurement_sdk import (
    BenchAgenticAcquisitionOutcome,
    BenchDatasetCacheState,
    ClientStatus,
    MeasurementDataAssetAcquiredSource,
    MeasurementDataAssetCacheOutcome,
    MeasurementDataAssetCacheStore,
    MeasurementDataAssetEffectiveSelection,
    MeasurementDataAssetEffectiveSelectionAgentic,
    MeasurementDataAssetPreparationPhaseAcquire,
    MeasurementDataAssetPreparationPhaseResolve,
    MeasurementDataAssetPreparationRequest,
    MeasurementDataAssetPreparationResult,
    MeasurementDataAssetReadiness,
    MeasurementDataAssetReadinessClosed,
    MeasurementDataAssetRemoteMetadataOutcome,
    MeasurementDataAssetSourceBytesOutcome,
    MeasurementDataAssetSourceInputAgentic,
    MeasurementDataAssetVerification,
)

from inferlab_bench_runner.agentic_source import (
    acquire_resolved_agentic_source,
    resolve_agentic_source,
)


def _cache_store(
    outcome: MeasurementDataAssetCacheOutcome,
) -> list[MeasurementDataAssetCacheStore]:
    return [
        MeasurementDataAssetCacheStore(
            authority="huggingface_hub",
            purpose="dataset_repository_files",
            path=str(constants.HUGGINGFACE_HUB_CACHE),
            outcome=outcome,
        )
    ]


def _selection(
    repository: str,
    requested_revision: str,
    observed_revision: str | None,
    filename: str,
) -> MeasurementDataAssetEffectiveSelection:
    return MeasurementDataAssetEffectiveSelection(
        root=MeasurementDataAssetEffectiveSelectionAgentic(
            kind="agentic",
            repository=repository,
            requested_revision=requested_revision,
            observed_revision=observed_revision,
            filename=filename,
        )
    )


def prepare_agentic_data_asset(
    request: MeasurementDataAssetPreparationRequest,
) -> MeasurementDataAssetPreparationResult:
    source_input = request.source.root
    if not isinstance(source_input, MeasurementDataAssetSourceInputAgentic):
        raise TypeError("AgentX source preparation requires an agentic source input")
    source = source_input.source
    phase = request.phase.root
    if isinstance(phase, MeasurementDataAssetPreparationPhaseResolve):
        resolution = resolve_agentic_source(source)
        selection = _selection(
            source.catalog.repository,
            source.catalog.revision,
            resolution.observed_revision,
            source.catalog.filename,
        )
        cache_outcome = (
            MeasurementDataAssetCacheOutcome.partial_reuse
            if resolution.cache_state is BenchDatasetCacheState.present
            else MeasurementDataAssetCacheOutcome.miss
            if resolution.cache_state is BenchDatasetCacheState.missing
            else MeasurementDataAssetCacheOutcome.unavailable
        )
        return MeasurementDataAssetPreparationResult(
            schema_version=1,
            status=(ClientStatus.succeeded if resolution.error is None else ClientStatus.failed),
            effective_selection=selection,
            readiness=None,
            cache_stores=_cache_store(cache_outcome),
            remote_metadata=(
                MeasurementDataAssetRemoteMetadataOutcome.accessed
                if resolution.metadata_accessed
                else MeasurementDataAssetRemoteMetadataOutcome.not_accessed
            ),
            source_bytes=MeasurementDataAssetSourceBytesOutcome.not_accessed,
            error=resolution.error,
        )
    if not isinstance(phase, MeasurementDataAssetPreparationPhaseAcquire):
        raise TypeError(f"unsupported AgentX preparation phase {type(phase).__name__}")
    cache_state = (
        BenchDatasetCacheState.present
        if phase.cache_state_before
        in {
            MeasurementDataAssetCacheOutcome.full_hit,
            MeasurementDataAssetCacheOutcome.partial_reuse,
        }
        else BenchDatasetCacheState.missing
    )
    acquisition = acquire_resolved_agentic_source(source, phase.resolved_revision, cache_state)
    verification = acquisition.verification
    effective_selection = _selection(
        source.catalog.repository,
        source.catalog.revision,
        verification.observed_revision,
        source.catalog.filename,
    )
    cache_outcome = MeasurementDataAssetCacheOutcome.unavailable
    source_bytes = MeasurementDataAssetSourceBytesOutcome.unavailable
    if verification.acquisition_outcome is BenchAgenticAcquisitionOutcome.reused:
        cache_outcome = MeasurementDataAssetCacheOutcome.full_hit
        source_bytes = MeasurementDataAssetSourceBytesOutcome.reused
    elif verification.acquisition_outcome is BenchAgenticAcquisitionOutcome.downloaded:
        cache_outcome = MeasurementDataAssetCacheOutcome.miss
        source_bytes = MeasurementDataAssetSourceBytesOutcome.downloaded
    if acquisition.error is not None:
        return MeasurementDataAssetPreparationResult(
            schema_version=1,
            status=ClientStatus.failed,
            effective_selection=effective_selection,
            readiness=None,
            cache_stores=_cache_store(cache_outcome),
            remote_metadata=(
                MeasurementDataAssetRemoteMetadataOutcome.accessed
                if cache_state is not BenchDatasetCacheState.present
                else MeasurementDataAssetRemoteMetadataOutcome.not_accessed
            ),
            source_bytes=source_bytes,
            error=acquisition.error,
        )
    observed_revision = verification.observed_revision
    observed_sha256 = verification.observed_sha256
    if observed_revision is None or observed_sha256 is None:
        raise ValueError("successful AgentX acquisition omitted immutable source identity")
    readiness = MeasurementDataAssetReadiness(
        root=MeasurementDataAssetReadinessClosed(
            kind="closed",
            acquired_source=MeasurementDataAssetAcquiredSource.model_validate(
                {
                    "kind": "release_qualified",
                    "identity": (
                        f"hf-dataset:{source.catalog.repository}@{observed_revision}:"
                        f"{source.catalog.filename}#{observed_sha256}"
                    ),
                    "closure": [
                        {
                            "relative_path": source.catalog.filename,
                            "sha256": observed_sha256,
                        }
                    ],
                }
            ),
            verification=[
                MeasurementDataAssetVerification(
                    subject="repository_revision",
                    expected=source.catalog.revision,
                    observed=observed_revision,
                    matched=observed_revision == source.catalog.revision,
                ),
                MeasurementDataAssetVerification(
                    subject=source.catalog.filename,
                    expected=source.catalog.sha256,
                    observed=observed_sha256,
                    matched=observed_sha256 == source.catalog.sha256,
                ),
            ],
        )
    )
    return MeasurementDataAssetPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        effective_selection=effective_selection,
        readiness=readiness,
        cache_stores=_cache_store(cache_outcome),
        remote_metadata=(
            MeasurementDataAssetRemoteMetadataOutcome.accessed
            if cache_state is not BenchDatasetCacheState.present
            else MeasurementDataAssetRemoteMetadataOutcome.not_accessed
        ),
        source_bytes=source_bytes,
        error=None,
    )
