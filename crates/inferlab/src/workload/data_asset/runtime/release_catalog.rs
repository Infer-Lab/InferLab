//! Rust-owned release-catalog acquisition and closure verification.

use super::super::model::{
    DataAssetAcquiredSource, DataAssetCacheOutcome, DataAssetCacheStore, DataAssetContentEntry,
    DataAssetPreparationAttempt, DataAssetPreparationPhase, DataAssetPreparationPhaseEvidence,
    DataAssetReadiness, DataAssetRemoteMetadataOutcome, DataAssetSourceBytesOutcome,
    DataAssetVerification,
};
use crate::InferlabError;
use crate::workload::record::{DatasetAcquisitionEvidence, DatasetAcquisitionOutcome};
use crate::workload::runtime::acquire_dataset_snapshot;
use std::fs;
use std::path::Path;

pub(super) fn prepare(
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
