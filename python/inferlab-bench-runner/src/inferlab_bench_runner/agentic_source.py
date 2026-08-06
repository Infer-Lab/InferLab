"""Acquire and verify one release-qualified AgentX dataset snapshot."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
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


def _cached_main_revision(repository: str) -> str:
    repositories = [
        repo
        for repo in scan_cache_dir().repos
        if repo.repo_type == "dataset" and repo.repo_id == repository
    ]
    revisions = [
        revision.commit_hash
        for repo in repositories
        for revision in repo.revisions
        if "main" in revision.refs
    ]
    if len(revisions) != 1:
        raise ValueError(
            f"Hugging Face cache has no unique main revision for dataset {repository!r}"
        )
    return revisions[0]


def _main_revision(repository: str) -> tuple[str, bool]:
    try:
        revision = HfApi().dataset_info(repository, revision="main").sha
    except (HfHubHTTPError, OfflineModeIsEnabled):
        return _cached_main_revision(repository), True
    if revision is None:
        raise ValueError(f"Hugging Face returned no main revision for dataset {repository!r}")
    return revision, False


def verify_downloaded_snapshot(
    source: BenchAgenticSourceInput,
    observed_revision: str,
    path: Path,
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
        cache_state_before=catalog.cache_state,
        acquisition_outcome=(
            BenchAgenticAcquisitionOutcome.reused
            if catalog.cache_state is BenchDatasetCacheState.present
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
    try:
        observed_revision, offline = _main_revision(source.catalog.repository)
    except (OSError, ValueError) as error:
        return AgenticSourceAcquisition(
            verification=_partial_verification(source, None),
            error=f"failed to resolve agentic dataset revision: {error}",
        )
    try:
        path = Path(
            hf_hub_download(
                repo_id=source.catalog.repository,
                filename=source.catalog.filename,
                repo_type="dataset",
                revision=observed_revision,
                local_files_only=offline,
            )
        )
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as error:
        return AgenticSourceAcquisition(
            verification=_partial_verification(source, observed_revision),
            error=f"failed to acquire agentic dataset {source.catalog.repository!r}: {error}",
        )
    try:
        verification = verify_downloaded_snapshot(source, observed_revision, path)
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
