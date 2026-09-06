"""Shared measurement-client result plumbing for the runner entrypoints."""

from pathlib import Path

from ._generated import (
    BenchClientResult,
    BenchPopulationPreparationResult,
    ClientStatus,
    EvalClientResult,
    MeasurementDataAssetPreparationResult,
    MeasurementDataAssetRemoteMetadataOutcome,
    MeasurementDataAssetSourceBytesOutcome,
)

SCHEMA_VERSION: int = 1
HUGGINGFACE_HUB_CACHE_PURPOSE: str = "repository_files"

type ClientResult = (
    BenchClientResult
    | BenchPopulationPreparationResult
    | EvalClientResult
    | MeasurementDataAssetPreparationResult
)


def failed_data_asset_preparation_result(
    error: BaseException,
) -> MeasurementDataAssetPreparationResult:
    return MeasurementDataAssetPreparationResult(
        schema_version=SCHEMA_VERSION,
        status=ClientStatus.failed,
        effective_selection=None,
        readiness=None,
        cache_stores=[],
        remote_metadata=MeasurementDataAssetRemoteMetadataOutcome.unavailable,
        source_bytes=MeasurementDataAssetSourceBytesOutcome.unavailable,
        error=str(error),
    )


def write_result(path: Path, result: ClientResult) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
