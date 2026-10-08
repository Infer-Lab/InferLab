//! Rust-owned release-catalog acquisition and closure verification.

use super::super::model::ReleaseCatalogDataAssetSource;
use super::super::model::{
    DataAssetAcquiredSource, DataAssetCacheOutcome, DataAssetCacheStore, DataAssetContentEntry,
    DataAssetPreparationAttempt, DataAssetPreparationPhase, DataAssetPreparationPhaseEvidence,
    DataAssetReadiness, DataAssetRemoteMetadataOutcome, DataAssetSourceBytesOutcome,
    DataAssetVerification,
};
use super::python;
use crate::InferlabError;
use crate::workload::plan::ClientCommandPlan;
use crate::workload::record::{DatasetAcquisitionEvidence, DatasetAcquisitionOutcome};
use crate::workload::runtime::{
    acquire_dataset_snapshot, publish_materialized_snapshot, reuse_cached_snapshot,
};
use inferlab_protocol::{
    MeasurementDataAssetPreparationPhase, MeasurementDataAssetPreparationRequest,
    MeasurementDataAssetSourceInput, ProtocolVersion,
};
use std::fs;
use std::path::Path;

pub(super) fn prepare(
    root: &Path,
    owner_record_id: &str,
    source: &ReleaseCatalogDataAssetSource,
    attempt: &mut DataAssetPreparationAttempt,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    match (&source.url, &source.aiperf_dataset, &source.command) {
        (Some(url), None, _) => download(
            &source.cache_path,
            url,
            &source.expected_sha256,
            attempt,
            persist,
        ),
        (None, Some(dataset), Some(command)) => materialize(
            root,
            owner_record_id,
            &source.cache_path,
            dataset,
            command,
            &source.expected_sha256,
            attempt,
            persist,
        ),
        _ => Err(InferlabError::DatasetPreparation {
            message: format!(
                "release-catalog source {:?} must name either a byte snapshot or an AIPerf public dataset with its runner",
                source.dataset
            ),
        }),
    }
}

fn download(
    cache_path: &Path,
    url: &str,
    expected_sha256: &str,
    attempt: &mut DataAssetPreparationAttempt,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    attempt.begin_acquisition()?;
    let (cache_outcome, observed_bytes) = observe_cache(cache_path);
    attempt.commit_phase(DataAssetPreparationPhaseEvidence {
        phase: DataAssetPreparationPhase::CacheObservation,
        process: None,
        request: None,
        result: None,
        stdout: None,
        stderr: None,
        effective_selection: None,
        cache_stores: vec![DataAssetCacheStore {
            authority: "inferlab_http_cas".to_owned(),
            purpose: "release_catalog_source".to_owned(),
            path: Some(cache_path.to_path_buf()),
            outcome: cache_outcome,
        }],
        remote_metadata: DataAssetRemoteMetadataOutcome::NotAccessed,
        source_bytes: DataAssetSourceBytesOutcome::NotAccessed,
        observed_bytes,
        observed_sha256: None,
        error: None,
    });
    persist(std::slice::from_ref(attempt))?;
    let acquisition = match acquire_dataset_snapshot(cache_path, url, expected_sha256) {
        Ok(acquisition) => acquisition,
        Err(failure) => {
            let (evidence, error) = *failure;
            attempt.commit_phase(DataAssetPreparationPhaseEvidence {
                phase: DataAssetPreparationPhase::AcquireAndVerify,
                process: None,
                request: None,
                result: None,
                stdout: None,
                stderr: None,
                effective_selection: None,
                cache_stores: Vec::new(),
                remote_metadata: DataAssetRemoteMetadataOutcome::Unavailable,
                source_bytes: DataAssetSourceBytesOutcome::Unavailable,
                observed_bytes: evidence.observed_bytes,
                observed_sha256: evidence.observed_sha256,
                error: evidence.error,
            });
            persist(std::slice::from_ref(attempt))?;
            return Err(error);
        }
    };
    finish(attempt, cache_path, expected_sha256, acquisition, persist)
}

/// Materialize an AIPerf-backed source on a cache miss: the measurement runner
/// asks the release-pinned AIPerf for the rows, and the control plane holds
/// them to the catalog's pinned digest before publishing, as for a download
/// ([[RFC-0004:C-BENCH-REQUEST-SOURCES]]).
#[allow(clippy::too_many_arguments)]
fn materialize(
    root: &Path,
    owner_record_id: &str,
    cache_path: &Path,
    dataset: &str,
    command: &ClientCommandPlan,
    expected_sha256: &str,
    attempt: &mut DataAssetPreparationAttempt,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    attempt.begin_acquisition()?;
    let (cache_outcome, observed_bytes) = observe_cache(cache_path);
    attempt.commit_phase(DataAssetPreparationPhaseEvidence {
        phase: DataAssetPreparationPhase::CacheObservation,
        process: None,
        request: None,
        result: None,
        stdout: None,
        stderr: None,
        effective_selection: None,
        cache_stores: vec![DataAssetCacheStore {
            authority: "inferlab_http_cas".to_owned(),
            purpose: "release_catalog_source".to_owned(),
            path: Some(cache_path.to_path_buf()),
            outcome: cache_outcome,
        }],
        remote_metadata: DataAssetRemoteMetadataOutcome::NotAccessed,
        source_bytes: DataAssetSourceBytesOutcome::NotAccessed,
        observed_bytes,
        observed_sha256: None,
        error: None,
    });
    persist(std::slice::from_ref(attempt))?;
    let acquisition = if cache_path.is_file() {
        reuse_cached_snapshot(cache_path, expected_sha256)
    } else {
        let artifact_dir = python::asset_directory(root, owner_record_id, &attempt.attempt_id);
        let output_path = artifact_dir.join("aiperf-public-dataset.jsonl");
        let cache_root = aiperf_cache_root(cache_path)?;
        let request = MeasurementDataAssetPreparationRequest {
            protocol_version: ProtocolVersion::CURRENT,
            phase: MeasurementDataAssetPreparationPhase::Materialize,
            source: MeasurementDataAssetSourceInput::AiperfPublicDataset {
                dataset: dataset.to_owned(),
                output_path: output_path.clone(),
                cache_root: cache_root.clone(),
            },
            artifact_dir,
        };
        let result = python::run_phase(
            root,
            owner_record_id,
            &attempt.attempt_id,
            DataAssetPreparationPhase::Materialize,
            command,
            &request,
        )?;
        python::commit_materialize(attempt, result, &cache_root, persist)?;
        publish_materialized_snapshot(&output_path, cache_path, expected_sha256)
    };
    let acquisition = match acquisition {
        Ok(acquisition) => acquisition,
        Err(failure) => {
            let (evidence, error) = *failure;
            attempt.commit_phase(DataAssetPreparationPhaseEvidence {
                phase: DataAssetPreparationPhase::AcquireAndVerify,
                process: None,
                request: None,
                result: None,
                stdout: None,
                stderr: None,
                effective_selection: None,
                cache_stores: Vec::new(),
                remote_metadata: DataAssetRemoteMetadataOutcome::Unavailable,
                source_bytes: DataAssetSourceBytesOutcome::Unavailable,
                observed_bytes: evidence.observed_bytes,
                observed_sha256: evidence.observed_sha256,
                error: evidence.error,
            });
            persist(std::slice::from_ref(attempt))?;
            return Err(error);
        }
    };
    finish(attempt, cache_path, expected_sha256, acquisition, persist)
}

/// AIPerf's dataset cache is relative to the working directory, so the runner
/// works from `<cache home>/inferlab/aiperf`, a machine-local sibling of the
/// `<cache home>/inferlab/datasets/sha256/<digest>` source cache.
fn aiperf_cache_root(cache_path: &Path) -> Result<std::path::PathBuf, InferlabError> {
    cache_path
        .ancestors()
        .nth(3)
        .map(|inferlab| inferlab.join("aiperf"))
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: format!(
                "dataset cache path {} is not under an InferLab cache home",
                cache_path.display()
            ),
        })
}

fn observe_cache(cache_path: &Path) -> (DataAssetCacheOutcome, Option<u64>) {
    match fs::metadata(cache_path) {
        Ok(metadata) if metadata.is_file() => {
            (DataAssetCacheOutcome::PartialReuse, Some(metadata.len()))
        }
        Ok(_) => (DataAssetCacheOutcome::Unavailable, None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            (DataAssetCacheOutcome::Miss, None)
        }
        Err(_) => (DataAssetCacheOutcome::Unavailable, None),
    }
}

fn finish(
    attempt: &mut DataAssetPreparationAttempt,
    cache_path: &Path,
    expected_sha256: &str,
    acquisition: DatasetAcquisitionEvidence,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    let observed =
        acquisition
            .observed_sha256
            .clone()
            .ok_or_else(|| InferlabError::DatasetPreparation {
                message: "successful release-catalog acquisition omitted its digest".to_owned(),
            })?;
    let (source_bytes, cache_outcome) = match acquisition.outcome {
        DatasetAcquisitionOutcome::Reused => (
            DataAssetSourceBytesOutcome::Reused,
            DataAssetCacheOutcome::FullHit,
        ),
        DatasetAcquisitionOutcome::Downloaded => (
            DataAssetSourceBytesOutcome::Downloaded,
            DataAssetCacheOutcome::Miss,
        ),
        DatasetAcquisitionOutcome::Materialized => (
            DataAssetSourceBytesOutcome::Materialized,
            DataAssetCacheOutcome::Miss,
        ),
        DatasetAcquisitionOutcome::Failed => (
            DataAssetSourceBytesOutcome::Unavailable,
            DataAssetCacheOutcome::Unavailable,
        ),
    };
    attempt.commit_phase(DataAssetPreparationPhaseEvidence {
        phase: DataAssetPreparationPhase::AcquireAndVerify,
        process: None,
        request: None,
        result: None,
        stdout: None,
        stderr: None,
        effective_selection: None,
        cache_stores: vec![DataAssetCacheStore {
            authority: "inferlab_http_cas".to_owned(),
            purpose: "release_catalog_source".to_owned(),
            path: Some(cache_path.to_path_buf()),
            outcome: cache_outcome,
        }],
        remote_metadata: if matches!(acquisition.outcome, DatasetAcquisitionOutcome::Downloaded) {
            DataAssetRemoteMetadataOutcome::Accessed
        } else {
            DataAssetRemoteMetadataOutcome::NotAccessed
        },
        source_bytes,
        observed_bytes: acquisition.observed_bytes,
        observed_sha256: acquisition.observed_sha256.clone(),
        error: acquisition.error,
    });
    attempt.complete(
        DataAssetReadiness::Closed {
            acquired_source: Box::new(DataAssetAcquiredSource::ReleaseQualified {
                identity: format!("sha256:{observed}"),
                closure: vec![DataAssetContentEntry {
                    relative_path: cache_path.file_name().map_or_else(
                        || "source".to_owned(),
                        |name| name.to_string_lossy().into_owned(),
                    ),
                    sha256: observed.clone(),
                }],
            }),
            verification: vec![DataAssetVerification {
                subject: "release_catalog_source".to_owned(),
                expected: expected_sha256.to_owned(),
                observed: Some(observed),
                matched: true,
            }],
            eval_binding: None,
        },
        "release catalog digest matched the complete declared source closure",
    )?;
    persist(std::slice::from_ref(attempt))
}

#[cfg(test)]
mod tests {
    use super::observe_cache;
    use crate::workload::data_asset::model::DataAssetCacheOutcome;

    #[test]
    fn cache_observation_distinguishes_missing_and_unverified_local_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("source.json");
        assert_eq!(observe_cache(&path), (DataAssetCacheOutcome::Miss, None));

        std::fs::write(&path, b"local bytes")?;
        assert_eq!(
            observe_cache(&path),
            (DataAssetCacheOutcome::PartialReuse, Some(11))
        );
        Ok(())
    }
}
