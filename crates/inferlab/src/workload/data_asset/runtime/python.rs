//! Invocation and evidence translation for Python-owned source preparation.

use super::super::model::{
    DataAssetAcquiredSource, DataAssetCacheOutcome, DataAssetCacheStore, DataAssetContentEntry,
    DataAssetEffectiveSelection, DataAssetPreparationAttempt, DataAssetPreparationPhase,
    DataAssetPreparationPhaseEvidence, DataAssetReadiness, DataAssetRemoteMetadataOutcome,
    DataAssetSourceBytesOutcome, DataAssetVerification, PreparedEvalSource,
};
use crate::InferlabError;
use crate::workload::plan::ClientCommandPlan;
use crate::workload::runtime::{ClientProcessPaths, run_unbounded_client};
use inferlab_protocol::{
    ClientStatus, MeasurementDataAssetPreparationNextPhase, MeasurementDataAssetPreparationRequest,
    MeasurementDataAssetPreparationResult,
};
use std::fs;
use std::path::{Path, PathBuf};

pub(super) struct PythonPhaseResult {
    result: Option<MeasurementDataAssetPreparationResult>,
    process: Option<crate::workload::record::ClientProcessEvidence>,
    error: Option<String>,
    paths: ClientProcessPaths,
}

pub(super) struct SourceObservation {
    pub effective_selection: Option<DataAssetEffectiveSelection>,
    pub cache_stores: Vec<DataAssetCacheStore>,
    pub snapshot_local: bool,
}

pub(super) enum EvalResolveOutcome {
    Ready,
    SnapshotLocal,
}

pub(super) struct AgenticResolution {
    pub observed_revision: String,
    pub cache_state: DataAssetCacheOutcome,
}

pub(super) fn run_phase(
    root: &Path,
    owner_record_id: &str,
    attempt_id: &str,
    phase: DataAssetPreparationPhase,
    command: &ClientCommandPlan,
    request: &MeasurementDataAssetPreparationRequest,
) -> Result<PythonPhaseResult, InferlabError> {
    let directory = asset_directory(root, owner_record_id, attempt_id).join(phase.as_str());
    fs::create_dir_all(&directory).map_err(|source| InferlabError::RecordIo {
        path: directory.clone(),
        source,
    })?;
    let paths = ClientProcessPaths {
        request: directory.join("request.json"),
        result: directory.join("result.json"),
        stdout: directory.join("stdout.log"),
        stderr: directory.join("stderr.log"),
    };
    let outcome = run_unbounded_client::<MeasurementDataAssetPreparationResult>(
        command,
        request,
        &paths,
        &["--prepare-source"],
    )?;
    Ok(PythonPhaseResult {
        result: outcome.result,
        process: outcome.process,
        error: outcome.error,
        paths,
    })
}

enum ObservationKind {
    Eval,
    Agentic,
}

pub(super) fn observe_eval(
    command: &ClientCommandPlan,
    request: MeasurementDataAssetPreparationRequest,
) -> Result<SourceObservation, InferlabError> {
    observe(command, request, ObservationKind::Eval)
}

pub(super) fn observe_agentic(
    command: &ClientCommandPlan,
    request: MeasurementDataAssetPreparationRequest,
) -> Result<SourceObservation, InferlabError> {
    observe(command, request, ObservationKind::Agentic)
}

fn observe(
    command: &ClientCommandPlan,
    mut request: MeasurementDataAssetPreparationRequest,
    kind: ObservationKind,
) -> Result<SourceObservation, InferlabError> {
    let directory = tempfile::tempdir().map_err(|source| InferlabError::RecordIo {
        path: std::env::temp_dir(),
        source,
    })?;
    request.artifact_dir = directory.path().join("artifacts");
    let paths = ClientProcessPaths {
        request: directory.path().join("request.json"),
        result: directory.path().join("result.json"),
        stdout: directory.path().join("stdout.log"),
        stderr: directory.path().join("stderr.log"),
    };
    let outcome = run_unbounded_client::<MeasurementDataAssetPreparationResult>(
        command,
        &request,
        &paths,
        &["--prepare-source"],
    )?;
    if let Some(error) = outcome.error {
        let diagnostic = fs::read_to_string(&paths.stderr)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map_or_else(String::new, |value| format!("; stderr: {}", value.trim()));
        let process = outcome.process.as_ref().map_or_else(
            || "not started".to_owned(),
            |process| format!("exit={:?}", process.exit_code),
        );
        return Err(InferlabError::DatasetPreparation {
            message: format!("{error}; process {process}{diagnostic}"),
        });
    }
    let result = outcome
        .result
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: "source observation client returned no result".to_owned(),
        })?;
    if result.status != ClientStatus::Succeeded {
        return Err(InferlabError::DatasetPreparation {
            message: result
                .error
                .clone()
                .unwrap_or_else(|| "source observation client failed".to_owned()),
        });
    }
    if let Some(error) = result.error {
        return Err(InferlabError::DatasetPreparation { message: error });
    }
    let snapshot_local = match kind {
        ObservationKind::Eval => {
            if !matches!(
                result.effective_selection.as_ref(),
                Some(inferlab_protocol::MeasurementDataAssetEffectiveSelection::Eval { .. })
            ) {
                return Err(InferlabError::DatasetPreparation {
                    message: "Eval source observation omitted its Eval effective selection"
                        .to_owned(),
                });
            }
            match (result.readiness.as_ref(), result.next_phase) {
                (None, Some(MeasurementDataAssetPreparationNextPhase::SnapshotLocal)) => true,
                (Some(_), None) => false,
                (None, None) => {
                    return Err(InferlabError::DatasetPreparation {
                        message:
                            "Eval source observation returned neither readiness nor a continuation"
                                .to_owned(),
                    });
                }
                (Some(_), Some(_)) => {
                    return Err(InferlabError::DatasetPreparation {
                        message: "source observation returned both readiness and a continuation"
                            .to_owned(),
                    });
                }
            }
        }
        ObservationKind::Agentic => {
            if result.readiness.is_some()
                || result.next_phase.is_some()
                || !matches!(
                    result.effective_selection.as_ref(),
                    Some(inferlab_protocol::MeasurementDataAssetEffectiveSelection::Agentic { .. })
                )
            {
                return Err(InferlabError::DatasetPreparation {
                    message: "AgentX source observation returned an invalid resolution outcome"
                        .to_owned(),
                });
            }
            false
        }
    };
    Ok(SourceObservation {
        effective_selection: result.effective_selection.map(effective_selection),
        cache_stores: result.cache_stores.into_iter().map(cache_store).collect(),
        snapshot_local,
    })
}

fn commit_result(
    attempt: &mut DataAssetPreparationAttempt,
    phase: DataAssetPreparationPhase,
    result: PythonPhaseResult,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<MeasurementDataAssetPreparationResult, InferlabError> {
    let result_value = result.result.as_ref();
    let error = result
        .error
        .or_else(|| result_value.and_then(|value| value.error.clone()));
    attempt.commit_phase(DataAssetPreparationPhaseEvidence {
        phase,
        process: result.process,
        request: Some(result.paths.request),
        result: Some(result.paths.result),
        stdout: Some(result.paths.stdout),
        stderr: Some(result.paths.stderr),
        effective_selection: result_value
            .and_then(|value| value.effective_selection.clone())
            .map(effective_selection),
        cache_stores: result_value.map_or_else(Vec::new, |value| {
            value
                .cache_stores
                .clone()
                .into_iter()
                .map(cache_store)
                .collect()
        }),
        remote_metadata: result_value
            .map_or(DataAssetRemoteMetadataOutcome::Unavailable, |value| {
                remote_metadata(value.remote_metadata)
            }),
        source_bytes: result_value.map_or(DataAssetSourceBytesOutcome::Unavailable, |value| {
            source_bytes(value.source_bytes)
        }),
        observed_bytes: None,
        observed_sha256: None,
        error: error.clone(),
    });
    persist(std::slice::from_ref(attempt))?;
    if let Some(error) = error {
        return Err(InferlabError::DatasetPreparation { message: error });
    }
    let result_value = result
        .result
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: format!(
                "measurement data-asset {} phase returned no result",
                phase.as_str()
            ),
        })?;
    if result_value.status != ClientStatus::Succeeded {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "measurement data-asset {} phase did not succeed",
                phase.as_str()
            ),
        });
    }
    Ok(result_value)
}

pub(super) fn commit_eval_resolve(
    attempt: &mut DataAssetPreparationAttempt,
    result: PythonPhaseResult,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<EvalResolveOutcome, InferlabError> {
    let result = commit_result(attempt, DataAssetPreparationPhase::Resolve, result, persist)?;
    match (result.readiness, result.next_phase) {
        (Some(readiness), None) => {
            attempt.complete(
                readiness_from_wire(readiness),
                "owning runtime established and bound the complete immutable closure",
            )?;
            persist(std::slice::from_ref(attempt))?;
            Ok(EvalResolveOutcome::Ready)
        }
        (None, Some(MeasurementDataAssetPreparationNextPhase::SnapshotLocal)) => {
            Ok(EvalResolveOutcome::SnapshotLocal)
        }
        (None, None) => Err(InferlabError::DatasetPreparation {
            message: "Eval source resolution returned neither readiness nor a continuation"
                .to_owned(),
        }),
        (Some(_), Some(_)) => Err(InferlabError::DatasetPreparation {
            message: "Eval source resolution returned both readiness and a continuation".to_owned(),
        }),
    }
}

pub(super) fn commit_agentic_resolve(
    attempt: &mut DataAssetPreparationAttempt,
    result: PythonPhaseResult,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<AgenticResolution, InferlabError> {
    let result = commit_result(attempt, DataAssetPreparationPhase::Resolve, result, persist)?;
    if result.readiness.is_some() || result.next_phase.is_some() {
        return Err(InferlabError::DatasetPreparation {
            message:
                "AgentX source resolution returned an unexpected terminal or continuation outcome"
                    .to_owned(),
        });
    }
    let observed_revision = result
        .effective_selection
        .as_ref()
        .and_then(|selection| match selection {
            inferlab_protocol::MeasurementDataAssetEffectiveSelection::Agentic {
                observed_revision,
                ..
            } => observed_revision.clone(),
            inferlab_protocol::MeasurementDataAssetEffectiveSelection::Eval { .. } => None,
        })
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: "AgentX source resolution returned no immutable revision".to_owned(),
        })?;
    let cache_state = result
        .cache_stores
        .first()
        .map_or(DataAssetCacheOutcome::Unavailable, |store| {
            cache_outcome(store.outcome)
        });
    Ok(AgenticResolution {
        observed_revision,
        cache_state,
    })
}

pub(super) fn commit_terminal(
    attempt: &mut DataAssetPreparationAttempt,
    phase: DataAssetPreparationPhase,
    result: PythonPhaseResult,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    let result = commit_result(attempt, phase, result, persist)?;
    if result.next_phase.is_some() {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "measurement data-asset {} phase returned a continuation",
                phase.as_str()
            ),
        });
    }
    let readiness = result
        .readiness
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: format!(
                "measurement data-asset {} phase omitted source readiness",
                phase.as_str()
            ),
        })?;
    attempt.complete(
        readiness_from_wire(readiness),
        "owning runtime established and bound the complete immutable closure",
    )?;
    persist(std::slice::from_ref(attempt))
}

fn effective_selection(
    selection: inferlab_protocol::MeasurementDataAssetEffectiveSelection,
) -> DataAssetEffectiveSelection {
    match selection {
        inferlab_protocol::MeasurementDataAssetEffectiveSelection::Eval {
            task_identity,
            dataset_path,
            dataset_name,
            evaluation_split,
            fewshot_split,
            data_files,
        } => DataAssetEffectiveSelection::Eval {
            task_identity,
            dataset_path,
            dataset_name,
            evaluation_split,
            fewshot_split,
            data_files: data_files.map(|files| files.0),
        },
        inferlab_protocol::MeasurementDataAssetEffectiveSelection::Agentic {
            repository,
            requested_revision,
            observed_revision,
            filename,
        } => DataAssetEffectiveSelection::Agentic {
            repository,
            requested_revision,
            observed_revision,
            filename,
        },
    }
}

fn cache_outcome(
    outcome: inferlab_protocol::MeasurementDataAssetCacheOutcome,
) -> DataAssetCacheOutcome {
    match outcome {
        inferlab_protocol::MeasurementDataAssetCacheOutcome::FullHit => {
            DataAssetCacheOutcome::FullHit
        }
        inferlab_protocol::MeasurementDataAssetCacheOutcome::Miss => DataAssetCacheOutcome::Miss,
        inferlab_protocol::MeasurementDataAssetCacheOutcome::PartialReuse => {
            DataAssetCacheOutcome::PartialReuse
        }
        inferlab_protocol::MeasurementDataAssetCacheOutcome::Unavailable => {
            DataAssetCacheOutcome::Unavailable
        }
    }
}

fn cache_store(store: inferlab_protocol::MeasurementDataAssetCacheStore) -> DataAssetCacheStore {
    DataAssetCacheStore {
        authority: store.authority,
        purpose: store.purpose,
        path: store.path,
        outcome: cache_outcome(store.outcome),
    }
}

fn remote_metadata(
    outcome: inferlab_protocol::MeasurementDataAssetRemoteMetadataOutcome,
) -> DataAssetRemoteMetadataOutcome {
    match outcome {
        inferlab_protocol::MeasurementDataAssetRemoteMetadataOutcome::NotAccessed => {
            DataAssetRemoteMetadataOutcome::NotAccessed
        }
        inferlab_protocol::MeasurementDataAssetRemoteMetadataOutcome::Accessed => {
            DataAssetRemoteMetadataOutcome::Accessed
        }
        inferlab_protocol::MeasurementDataAssetRemoteMetadataOutcome::Unavailable => {
            DataAssetRemoteMetadataOutcome::Unavailable
        }
    }
}

fn source_bytes(
    outcome: inferlab_protocol::MeasurementDataAssetSourceBytesOutcome,
) -> DataAssetSourceBytesOutcome {
    match outcome {
        inferlab_protocol::MeasurementDataAssetSourceBytesOutcome::NotAccessed => {
            DataAssetSourceBytesOutcome::NotAccessed
        }
        inferlab_protocol::MeasurementDataAssetSourceBytesOutcome::Reused => {
            DataAssetSourceBytesOutcome::Reused
        }
        inferlab_protocol::MeasurementDataAssetSourceBytesOutcome::Downloaded => {
            DataAssetSourceBytesOutcome::Downloaded
        }
        inferlab_protocol::MeasurementDataAssetSourceBytesOutcome::Unavailable => {
            DataAssetSourceBytesOutcome::Unavailable
        }
    }
}

fn readiness_from_wire(
    readiness: inferlab_protocol::MeasurementDataAssetReadiness,
) -> DataAssetReadiness {
    match readiness {
        inferlab_protocol::MeasurementDataAssetReadiness::Closed {
            acquired_source,
            verification,
            eval_binding,
        } => DataAssetReadiness::Closed {
            acquired_source: Box::new(match *acquired_source {
                inferlab_protocol::MeasurementDataAssetAcquiredSource::ReleaseQualified {
                    identity,
                    closure,
                } => DataAssetAcquiredSource::ReleaseQualified {
                    identity,
                    closure: closure
                        .into_iter()
                        .map(|entry| DataAssetContentEntry {
                            relative_path: entry.relative_path,
                            sha256: entry.sha256,
                        })
                        .collect(),
                },
                inferlab_protocol::MeasurementDataAssetAcquiredSource::LocalFileClosure {
                    source_root,
                    files,
                } => DataAssetAcquiredSource::LocalFileClosure {
                    source_root,
                    files: files
                        .into_iter()
                        .map(|entry| DataAssetContentEntry {
                            relative_path: entry.relative_path,
                            sha256: entry.sha256,
                        })
                        .collect(),
                },
            }),
            verification: verification
                .into_iter()
                .map(|item| DataAssetVerification {
                    subject: item.subject,
                    expected: item.expected,
                    observed: item.observed,
                    matched: item.matched,
                })
                .collect(),
            eval_binding: eval_binding.map(|binding| {
                Box::new(PreparedEvalSource {
                    workspace_root: binding.workspace_root,
                    task_path: binding.task_path,
                })
            }),
        },
        inferlab_protocol::MeasurementDataAssetReadiness::Opaque {
            reason,
            unresolved_path,
            deferred_source_access,
        } => DataAssetReadiness::Opaque {
            reason,
            unresolved_path,
            deferred_source_access,
        },
    }
}

pub(super) fn asset_directory(root: &Path, owner_record_id: &str, attempt_id: &str) -> PathBuf {
    root.join(crate::record::RECORDS_DIR)
        .join(owner_record_id)
        .join("data-assets")
        .join(attempt_id)
}

#[cfg(test)]
mod tests {
    use super::{PythonPhaseResult, commit_eval_resolve};
    use crate::workload::data_asset::model::{
        DataAssetConsumer, DataAssetConsumerKind, DataAssetDryRunProjection, DataAssetPlan,
        DataAssetSource, EvalDataAssetSource,
    };
    use crate::workload::plan::ClientCommandPlan;
    use crate::workload::runtime::ClientProcessPaths;
    use crate::workspace::{EvalDefinition, EvalTaskSource};
    use inferlab_protocol::{
        ClientStatus, MeasurementDataAssetEffectiveSelection,
        MeasurementDataAssetPreparationNextPhase, MeasurementDataAssetPreparationResult,
        MeasurementDataAssetReadiness, MeasurementDataAssetRemoteMetadataOutcome,
        MeasurementDataAssetSourceBytesOutcome,
    };

    fn attempt() -> crate::workload::data_asset::DataAssetPreparationAttempt {
        crate::workload::data_asset::DataAssetPreparationAttempt::from(&DataAssetPlan {
            attempt_id: "data-asset-fixture".to_owned(),
            source_key_sha256: "digest".to_owned(),
            source: DataAssetSource::Eval {
                source: Box::new(EvalDataAssetSource {
                    workspace_root: ".".into(),
                    workspace_source_exclusions: Vec::new(),
                    definition: Box::new(EvalDefinition::LmEval {
                        task: EvalTaskSource::WorkspaceYaml {
                            yaml: "task.yaml".into(),
                        },
                        prompt: Default::default(),
                        request_body: Default::default(),
                        limit: None,
                        few_shot: None,
                        seed: None,
                        trials: 1,
                        max_tokens: None,
                        concurrency: None,
                        metric: "acc".to_owned(),
                        metric_filter: None,
                        threshold: 0.0,
                        timeout_seconds: 60,
                    }),
                    bundled_task: None,
                    command: ClientCommandPlan {
                        argv: Vec::new(),
                        env: Default::default(),
                        cwd: ".".into(),
                    },
                    acquisition_runtime_identity: "runner".to_owned(),
                }),
            },
            consumers: vec![DataAssetConsumer {
                kind: DataAssetConsumerKind::Eval,
                definition_id: "eval".to_owned(),
            }],
            dry_run: DataAssetDryRunProjection::Planned {
                external_work: Vec::new(),
                unavailable: Vec::new(),
            },
        })
    }

    #[test]
    fn eval_resolution_rejects_readiness_with_a_continuation_after_recording_the_phase()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let mut attempt = attempt();
        attempt.begin_resolution()?;
        let result = PythonPhaseResult {
            result: Some(MeasurementDataAssetPreparationResult {
                schema_version: 1,
                status: ClientStatus::Succeeded,
                effective_selection: Some(MeasurementDataAssetEffectiveSelection::Eval {
                    task_identity: "fixture".to_owned(),
                    dataset_path: Some("json".to_owned()),
                    dataset_name: None,
                    evaluation_split: Some("test".to_owned()),
                    fewshot_split: None,
                    data_files: Some(inferlab_protocol::MeasurementDataFiles(
                        std::collections::BTreeMap::from([(
                            "train".to_owned(),
                            vec!["data.json".to_owned()],
                        )]),
                    )),
                }),
                readiness: Some(MeasurementDataAssetReadiness::Opaque {
                    reason: "fixture".to_owned(),
                    unresolved_path: None,
                    deferred_source_access: true,
                }),
                next_phase: Some(MeasurementDataAssetPreparationNextPhase::SnapshotLocal),
                cache_stores: Vec::new(),
                remote_metadata: MeasurementDataAssetRemoteMetadataOutcome::NotAccessed,
                source_bytes: MeasurementDataAssetSourceBytesOutcome::NotAccessed,
                error: None,
            }),
            process: None,
            error: None,
            paths: ClientProcessPaths {
                request: directory.path().join("request.json"),
                result: directory.path().join("result.json"),
                stdout: directory.path().join("stdout.log"),
                stderr: directory.path().join("stderr.log"),
            },
        };

        let error = commit_eval_resolve(&mut attempt, result, &mut |_| Ok(()))
            .err()
            .ok_or("contradictory outcome unexpectedly succeeded")?;
        assert!(
            error
                .to_string()
                .contains("both readiness and a continuation")
        );
        let evidence = serde_json::to_value(&attempt)?;
        assert_eq!(evidence["phases"].as_array().map(Vec::len), Some(1));
        assert_eq!(evidence["state"], "resolving");
        Ok(())
    }
}
