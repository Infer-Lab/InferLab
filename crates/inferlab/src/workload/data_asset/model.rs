//! Planned data-source identity and durable preparation evidence.

use crate::InferlabError;
use crate::toolchain::BundledEvalTask;
use crate::workload::domain::{BenchDatasetFilter, ResolvedBenchAgenticSource};
use crate::workload::plan::ClientCommandPlan;
use crate::workspace::EvalDefinition;
use inferlab_runtime::operation_bound::OperationTimingEvidence;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DataAssetConsumerKind {
    Eval,
    Bench,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DataAssetConsumer {
    pub kind: DataAssetConsumerKind,
    pub definition_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum DataAssetSource {
    Eval {
        #[serde(flatten)]
        source: Box<EvalDataAssetSource>,
    },
    ReleaseCatalog {
        #[serde(flatten)]
        source: Box<ReleaseCatalogDataAssetSource>,
    },
    Agentic {
        #[serde(flatten)]
        source: Box<AgenticDataAssetSource>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EvalDataAssetSource {
    pub workspace_root: PathBuf,
    pub workspace_source_exclusions: Vec<PathBuf>,
    pub definition: Box<EvalDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundled_task: Option<Box<BundledEvalTask>>,
    pub command: ClientCommandPlan,
    pub acquisition_runtime_identity: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReleaseCatalogDataAssetSource {
    pub dataset: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    pub url: String,
    pub upstream_identity: String,
    pub expected_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<BenchDatasetFilter>,
    pub cache_path: PathBuf,
    pub acquisition_runtime_identity: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AgenticDataAssetSource {
    pub command: ClientCommandPlan,
    #[serde(flatten)]
    pub definition: Box<ResolvedBenchAgenticSource>,
    pub acquisition_runtime_identity: String,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DataAssetKeyInput<'a> {
    Eval {
        definition: &'a EvalDefinition,
        bundled_task: &'a Option<Box<BundledEvalTask>>,
        workspace_source_exclusions: &'a [PathBuf],
        acquisition_runtime_identity: &'a str,
    },
    ReleaseCatalog {
        dataset: &'a str,
        profile: &'a Option<String>,
        url: &'a str,
        upstream_identity: &'a str,
        expected_sha256: &'a str,
        configuration: &'a Option<String>,
        split: &'a Option<String>,
        filter: &'a Option<BenchDatasetFilter>,
        acquisition_runtime_identity: &'a str,
    },
    Agentic {
        dataset: &'a str,
        profile: &'a str,
        repository: &'a str,
        revision: &'a str,
        filename: &'a str,
        expected_sha256: &'a str,
        acquisition_runtime_identity: &'a str,
    },
}

impl DataAssetSource {
    pub(super) fn key_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        let input = match self {
            Self::Eval { source } => DataAssetKeyInput::Eval {
                definition: &source.definition,
                bundled_task: &source.bundled_task,
                workspace_source_exclusions: &source.workspace_source_exclusions,
                acquisition_runtime_identity: &source.acquisition_runtime_identity,
            },
            Self::ReleaseCatalog { source } => DataAssetKeyInput::ReleaseCatalog {
                dataset: &source.dataset,
                profile: &source.profile,
                url: &source.url,
                upstream_identity: &source.upstream_identity,
                expected_sha256: &source.expected_sha256,
                configuration: &source.configuration,
                split: &source.split,
                filter: &source.filter,
                acquisition_runtime_identity: &source.acquisition_runtime_identity,
            },
            Self::Agentic { source } => DataAssetKeyInput::Agentic {
                dataset: &source.definition.dataset,
                profile: &source.definition.profile,
                repository: &source.definition.catalog.repository,
                revision: &source.definition.catalog.revision,
                filename: &source.definition.catalog.filename,
                expected_sha256: &source.definition.catalog.sha256,
                acquisition_runtime_identity: &source.acquisition_runtime_identity,
            },
        };
        serde_json::to_vec(&input)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum DataAssetEffectiveSelection {
    Eval {
        task_identity: String,
        #[serde(default)]
        dataset_path: Option<String>,
        #[serde(default)]
        dataset_name: Option<String>,
        #[serde(default)]
        evaluation_split: Option<String>,
        #[serde(default)]
        fewshot_split: Option<String>,
        #[serde(default)]
        data_files: Option<std::collections::BTreeMap<String, Vec<String>>>,
    },
    Agentic {
        repository: String,
        requested_revision: String,
        #[serde(default)]
        observed_revision: Option<String>,
        filename: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DataAssetCacheOutcome {
    FullHit,
    Miss,
    PartialReuse,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DataAssetCacheStore {
    pub authority: String,
    pub purpose: String,
    #[serde(default)]
    pub path: Option<PathBuf>,
    pub outcome: DataAssetCacheOutcome,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DataAssetRemoteMetadataOutcome {
    NotAccessed,
    Accessed,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DataAssetSourceBytesOutcome {
    NotAccessed,
    Reused,
    Downloaded,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DataAssetContentEntry {
    pub relative_path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DataAssetVerification {
    pub subject: String,
    pub expected: String,
    #[serde(default)]
    pub observed: Option<String>,
    pub matched: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum DataAssetAcquiredSource {
    ReleaseQualified {
        identity: String,
        closure: Vec<DataAssetContentEntry>,
    },
    LocalFileClosure {
        source_root: PathBuf,
        files: Vec<DataAssetContentEntry>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreparedEvalSource {
    pub workspace_root: PathBuf,
    pub task_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum DataAssetReadiness {
    Closed {
        acquired_source: Box<DataAssetAcquiredSource>,
        #[serde(default)]
        verification: Vec<DataAssetVerification>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        eval_binding: Option<Box<PreparedEvalSource>>,
    },
    Opaque {
        reason: String,
        #[serde(default)]
        unresolved_path: Option<PathBuf>,
        deferred_source_access: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DataAssetPlan {
    pub attempt_id: String,
    pub source_key_sha256: String,
    pub(super) source: DataAssetSource,
    pub(super) consumers: Vec<DataAssetConsumer>,
    pub(super) dry_run: DataAssetDryRunProjection,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DataAssetObservation {
    SelectedLocalPathPresent,
    SelectedLocalPathMissing,
    CachePathPresent,
    CachePathMissing,
    ReleaseBundledClosureSelected,
    CompleteLocalClosureEnumerated,
    OwningRuntimeSourceObserved,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DataAssetPlannedEffect {
    OwningRuntimeSourceResolution,
    ReleaseSourceAcquisitionAndVerification,
    ReleaseAssetVerification,
    SourceResolutionOrAcquisition,
    ImmutableLocalSnapshot,
    SourceAcquisitionOrConsumerMaterialization,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DataAssetUnavailableFact {
    CompleteLocalClosure,
    PreparedSnapshotIdentity,
    ReproducibilityConclusion,
    DigestVerification,
    AcquiredSource,
    ExistingMaterializationIdentity,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", tag = "state", deny_unknown_fields)]
pub(super) enum DataAssetDryRunProjection {
    Planned {
        external_work: Vec<DataAssetPlannedEffect>,
        unavailable: Vec<DataAssetUnavailableFact>,
    },
    LocalObservation {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effective_selection: Option<DataAssetEffectiveSelection>,
        cache_stores: Vec<DataAssetCacheStore>,
        observations: Vec<DataAssetObservation>,
        planned_external_work: Vec<DataAssetPlannedEffect>,
        unavailable: Vec<DataAssetUnavailableFact>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DataAssetPreparationState {
    Planned,
    Resolving,
    Acquiring,
    Ready,
    Failed,
    Interrupted,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DataAssetPreparationPhase {
    Resolve,
    SnapshotLocal,
    Acquire,
    CacheObservation,
    AcquireAndVerify,
}

impl DataAssetPreparationPhase {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Resolve => "resolve",
            Self::SnapshotLocal => "snapshot-local",
            Self::Acquire => "acquire",
            Self::CacheObservation => "cache_observation",
            Self::AcquireAndVerify => "acquire_and_verify",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DataAssetPreparationPhaseEvidence {
    pub phase: DataAssetPreparationPhase,
    pub process: Option<super::super::record::ClientProcessEvidence>,
    pub request: Option<PathBuf>,
    pub result: Option<PathBuf>,
    pub stdout: Option<PathBuf>,
    pub stderr: Option<PathBuf>,
    pub effective_selection: Option<DataAssetEffectiveSelection>,
    pub cache_stores: Vec<DataAssetCacheStore>,
    pub remote_metadata: DataAssetRemoteMetadataOutcome,
    pub source_bytes: DataAssetSourceBytesOutcome,
    pub observed_bytes: Option<u64>,
    pub observed_sha256: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "conclusion", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum DataAssetReproducibility {
    Reproducible { basis: String },
    NonReproducible { reason: String },
    NotEstablished,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DataAssetPreparationAttempt {
    pub attempt_id: String,
    pub source_key_sha256: String,
    pub(super) source: DataAssetSource,
    pub(super) consumers: Vec<DataAssetConsumer>,
    state: DataAssetPreparationState,
    phases: Vec<DataAssetPreparationPhaseEvidence>,
    effective_selection: Option<DataAssetEffectiveSelection>,
    readiness: Option<DataAssetReadiness>,
    reproducibility: DataAssetReproducibility,
    error: Option<String>,
}

impl From<&DataAssetPlan> for DataAssetPreparationAttempt {
    fn from(plan: &DataAssetPlan) -> Self {
        Self {
            attempt_id: plan.attempt_id.clone(),
            source_key_sha256: plan.source_key_sha256.clone(),
            source: plan.source.clone(),
            consumers: plan.consumers.clone(),
            state: DataAssetPreparationState::Planned,
            phases: Vec::new(),
            effective_selection: None,
            readiness: None,
            reproducibility: DataAssetReproducibility::NotEstablished,
            error: None,
        }
    }
}

impl DataAssetPreparationAttempt {
    fn transition_to(&mut self, next: DataAssetPreparationState) -> Result<(), InferlabError> {
        let valid = matches!(
            (self.state, next),
            (
                DataAssetPreparationState::Planned,
                DataAssetPreparationState::Resolving
                    | DataAssetPreparationState::Acquiring
                    | DataAssetPreparationState::Failed
                    | DataAssetPreparationState::Interrupted
            ) | (
                DataAssetPreparationState::Resolving,
                DataAssetPreparationState::Acquiring
                    | DataAssetPreparationState::Ready
                    | DataAssetPreparationState::Failed
                    | DataAssetPreparationState::Interrupted
            ) | (
                DataAssetPreparationState::Acquiring,
                DataAssetPreparationState::Ready
                    | DataAssetPreparationState::Failed
                    | DataAssetPreparationState::Interrupted
            ) | (
                DataAssetPreparationState::Ready,
                DataAssetPreparationState::Failed | DataAssetPreparationState::Interrupted
            )
        );
        if !valid {
            return Err(InferlabError::DatasetPreparation {
                message: format!(
                    "invalid data-asset attempt transition from {:?} to {:?}",
                    self.state, next
                ),
            });
        }
        self.state = next;
        Ok(())
    }

    pub(super) fn begin_resolution(&mut self) -> Result<(), InferlabError> {
        self.transition_to(DataAssetPreparationState::Resolving)
    }

    pub(super) fn begin_acquisition(&mut self) -> Result<(), InferlabError> {
        self.transition_to(DataAssetPreparationState::Acquiring)
    }

    pub(super) fn commit_phase(&mut self, evidence: DataAssetPreparationPhaseEvidence) {
        if let Some(selection) = evidence.effective_selection.clone() {
            self.effective_selection = Some(selection);
        }
        self.phases.push(evidence);
    }

    pub(super) fn complete(
        &mut self,
        readiness: DataAssetReadiness,
        closed_basis: &str,
    ) -> Result<(), InferlabError> {
        self.transition_to(DataAssetPreparationState::Ready)?;
        self.reproducibility = match &readiness {
            DataAssetReadiness::Closed { .. } => DataAssetReproducibility::Reproducible {
                basis: closed_basis.to_owned(),
            },
            DataAssetReadiness::Opaque { reason, .. } => {
                DataAssetReproducibility::NonReproducible {
                    reason: reason.clone(),
                }
            }
        };
        self.readiness = Some(readiness);
        self.error = None;
        Ok(())
    }

    pub(super) fn terminate(
        &mut self,
        interrupted: bool,
        error: String,
    ) -> Result<(), InferlabError> {
        self.transition_to(if interrupted {
            DataAssetPreparationState::Interrupted
        } else {
            DataAssetPreparationState::Failed
        })?;
        self.readiness = None;
        self.reproducibility = DataAssetReproducibility::NotEstablished;
        self.error = Some(error);
        Ok(())
    }

    pub(crate) fn eval_binding(&self) -> Option<PreparedEvalSource> {
        self.readiness
            .as_ref()
            .and_then(|readiness| match readiness {
                DataAssetReadiness::Closed { eval_binding, .. } => eval_binding.as_deref().cloned(),
                DataAssetReadiness::Opaque { .. } => None,
            })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "owner", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum WorkloadDataAssetEvidence {
    Recipe {
        recipe_record_id: String,
        attempt_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prepared_source: Option<PreparedEvalSource>,
    },
    Standalone {
        attempts: Vec<DataAssetPreparationAttempt>,
        target_server_unchanged: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timing: Option<OperationTimingEvidence>,
    },
    None,
}

#[cfg(test)]
mod tests {
    use super::{
        DataAssetAcquiredSource, DataAssetConsumer, DataAssetConsumerKind,
        DataAssetPreparationAttempt, DataAssetPreparationState, DataAssetReadiness,
        DataAssetReproducibility, DataAssetSource, ReleaseCatalogDataAssetSource,
    };

    fn attempt() -> DataAssetPreparationAttempt {
        DataAssetPreparationAttempt {
            attempt_id: "data-asset-fixture".to_owned(),
            source_key_sha256: "digest".to_owned(),
            source: DataAssetSource::ReleaseCatalog {
                source: Box::new(ReleaseCatalogDataAssetSource {
                    dataset: "fixture".to_owned(),
                    profile: None,
                    url: "https://example.invalid/data.json".to_owned(),
                    upstream_identity: "revision".to_owned(),
                    expected_sha256: "digest".to_owned(),
                    configuration: None,
                    split: None,
                    filter: None,
                    cache_path: "fixture-cache.json".into(),
                    acquisition_runtime_identity: "inferlab-0.10.0".to_owned(),
                }),
            },
            consumers: vec![DataAssetConsumer {
                kind: DataAssetConsumerKind::Bench,
                definition_id: "bench".to_owned(),
            }],
            state: DataAssetPreparationState::Planned,
            phases: Vec::new(),
            effective_selection: None,
            readiness: None,
            reproducibility: DataAssetReproducibility::NotEstablished,
            error: None,
        }
    }

    #[test]
    fn readiness_is_committed_as_one_terminal_transition() -> Result<(), crate::InferlabError> {
        let mut attempt = attempt();
        attempt.begin_acquisition()?;
        attempt.complete(
            DataAssetReadiness::Closed {
                acquired_source: Box::new(DataAssetAcquiredSource::ReleaseQualified {
                    identity: "sha256:digest".to_owned(),
                    closure: Vec::new(),
                }),
                verification: Vec::new(),
                eval_binding: None,
            },
            "fixture closure",
        )?;

        assert_eq!(attempt.state, DataAssetPreparationState::Ready);
        assert!(attempt.readiness.is_some());
        assert!(matches!(
            attempt.reproducibility,
            DataAssetReproducibility::Reproducible { .. }
        ));
        assert!(attempt.error.is_none());
        assert!(attempt.begin_acquisition().is_err());
        Ok(())
    }

    #[test]
    fn termination_clears_nonterminal_reproducibility_claims() -> Result<(), crate::InferlabError> {
        let mut attempt = attempt();
        attempt.begin_resolution()?;
        attempt.terminate(false, "resolution failed".to_owned())?;

        assert_eq!(attempt.state, DataAssetPreparationState::Failed);
        assert!(attempt.readiness.is_none());
        assert!(matches!(
            attempt.reproducibility,
            DataAssetReproducibility::NotEstablished
        ));
        assert_eq!(attempt.error.as_deref(), Some("resolution failed"));
        Ok(())
    }
}
