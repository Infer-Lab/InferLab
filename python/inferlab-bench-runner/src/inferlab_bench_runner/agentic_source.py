"""Acquire and verify one release-qualified AgentX dataset snapshot."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
)
from inferlab_measurement_sdk import (
    BenchAgenticAcquisitionOutcome,
    BenchAgenticSourceInput,
    BenchAgenticSourceVerification,
    BenchDatasetCacheState,
)


@dataclass(frozen=True)
class AgenticSourceAcquisition:
    verification: BenchAgenticSourceVerification
    error: str | None


@dataclass(frozen=True)
class AgenticSourceResolution:
    observed_revision: str | None
    cache_state: BenchDatasetCacheState | None
    cache_path: Path | None
    metadata_accessed: bool
    error: str | None


def verify_downloaded_snapshot(
    source: BenchAgenticSourceInput,
    observed_revision: str,
    path: Path,
    cache_state_before: BenchDatasetCacheState | None = None,
) -> BenchAgenticSourceVerification:
    catalog = source.catalog
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    observed_sha256 = digest.hexdigest()
    return BenchAgenticSourceVerification(
        repository=catalog.repository,
        expected_revision=catalog.revision,
        observed_revision=observed_revision,
        filename=catalog.filename,
        expected_sha256=catalog.sha256,
        observed_sha256=observed_sha256,
        cache_path=str(path),
        cache_state_before=cache_state_before or catalog.cache_state,
        acquisition_outcome=(
            BenchAgenticAcquisitionOutcome.reused
            if (cache_state_before or catalog.cache_state) is BenchDatasetCacheState.present
            else BenchAgenticAcquisitionOutcome.downloaded
        ),
    )


def _partial_verification(
    source: BenchAgenticSourceInput,
    observed_revision: str | None,
    *,
    cache_path: Path | None = None,
) -> BenchAgenticSourceVerification:
    catalog = source.catalog
    return BenchAgenticSourceVerification(
        repository=catalog.repository,
        expected_revision=catalog.revision,
        observed_revision=observed_revision,
        filename=catalog.filename,
        expected_sha256=catalog.sha256,
        observed_sha256=None,
        cache_path=str(cache_path) if cache_path is not None else None,
        cache_state_before=catalog.cache_state,
        acquisition_outcome=(
            None
            if cache_path is None
            else BenchAgenticAcquisitionOutcome.reused
            if catalog.cache_state is BenchDatasetCacheState.present
            else BenchAgenticAcquisitionOutcome.downloaded
        ),
    )


def source_verification_error(
    source: BenchAgenticSourceInput,
    verification: BenchAgenticSourceVerification,
) -> str | None:
    catalog = source.catalog
    if verification.observed_revision != catalog.revision:
        return (
            "agentic dataset revision does not match the release catalog: "
            f"expected={catalog.revision}, observed={verification.observed_revision}"
        )
    if verification.observed_sha256 != catalog.sha256:
        return (
            "agentic dataset content digest does not match the release catalog: "
            f"expected={catalog.sha256}, observed={verification.observed_sha256}"
        )
    return None


def acquire_and_verify_agentic_source(
    source: BenchAgenticSourceInput,
) -> AgenticSourceAcquisition:
    resolution = resolve_agentic_source(source)
    if resolution.error is not None or resolution.observed_revision is None:
        return AgenticSourceAcquisition(
            verification=_partial_verification(source, resolution.observed_revision),
            error=resolution.error or "agentic dataset revision was not resolved",
        )
    return acquire_resolved_agentic_source(
        source,
        resolution.observed_revision,
        resolution.cache_state or BenchDatasetCacheState.missing,
    )


def resolve_agentic_source(source: BenchAgenticSourceInput) -> AgenticSourceResolution:
    observed_revision = source.catalog.revision
    cached = try_to_load_from_cache(
        repo_id=source.catalog.repository,
        filename=source.catalog.filename,
        repo_type="dataset",
        revision=observed_revision,
    )
    cache_path = Path(cached) if isinstance(cached, str) else None
    return AgenticSourceResolution(
        observed_revision=observed_revision,
        cache_state=(
            BenchDatasetCacheState.present
            if cache_path is not None
            else BenchDatasetCacheState.missing
        ),
        cache_path=cache_path,
        metadata_accessed=False,
        error=None,
    )


def acquire_resolved_agentic_source(
    source: BenchAgenticSourceInput,
    observed_revision: str,
    cache_state_before: BenchDatasetCacheState,
) -> AgenticSourceAcquisition:
    try:
        path = Path(
            hf_hub_download(
                repo_id=source.catalog.repository,
                filename=source.catalog.filename,
                repo_type="dataset",
                revision=observed_revision,
                local_files_only=cache_state_before is BenchDatasetCacheState.present,
            )
        )
    except (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        OSError,
    ) as error:
        return AgenticSourceAcquisition(
            verification=_partial_verification(source, observed_revision),
            error=f"failed to acquire agentic dataset {source.catalog.repository!r}: {error}",
        )
    try:
        verification = verify_downloaded_snapshot(
            source, observed_revision, path, cache_state_before
        )
    except OSError as error:
        return AgenticSourceAcquisition(
            verification=_partial_verification(
                source,
                observed_revision,
                cache_path=path,
            ),
            error=f"failed to hash agentic dataset {path}: {error}",
        )
    return AgenticSourceAcquisition(
        verification=verification,
        error=source_verification_error(source, verification),
    )
