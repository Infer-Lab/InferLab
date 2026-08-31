//! Runtime orchestration at the source-preparation boundary.

mod python;
mod release_catalog;

use super::model::{
    DataAssetCacheOutcome, DataAssetDryRunProjection, DataAssetObservation, DataAssetPlan,
    DataAssetPlannedEffect, DataAssetPreparationAttempt, DataAssetPreparationPhase,
    DataAssetSource, DataAssetUnavailableFact, EvalDataAssetSource,
};
use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::toolchain::BundledEvalTask;
use crate::workload::domain::ResolvedBenchAgenticSource;
use crate::workload::plan::ClientCommandPlan;
use crate::workload::wire;
use crate::workspace::EvalDefinition;
use inferlab_protocol::{
    MeasurementDataAssetPreparationPhase, MeasurementDataAssetPreparationRequest,
    MeasurementDataAssetSourceInput, ProtocolVersion,
};
use std::path::{Path, PathBuf};

pub(crate) fn observe_data_asset_dry_run(plans: &mut [DataAssetPlan]) -> Result<(), InferlabError> {
    for plan in plans {
        let observation = match &plan.source {
            DataAssetSource::Eval { source } => {
                let request = eval_request(
                    &source.workspace_root,
                    &source.workspace_source_exclusions,
                    &source.definition,
                    source.bundled_task.as_deref(),
                    MeasurementDataAssetPreparationPhase::Resolve,
                    PathBuf::new(),
                )?;
                python::observe_eval(&source.command, request)?
            }
            DataAssetSource::Agentic { source } => python::observe_agentic(
                &source.command,
                agentic_request(
                    &source.definition,
                    MeasurementDataAssetPreparationPhase::Resolve,
                    PathBuf::new(),
                ),
            )?,
            DataAssetSource::ReleaseCatalog { .. } => continue,
        };
        plan.dry_run = DataAssetDryRunProjection::LocalObservation {
            effective_selection: observation.effective_selection,
            cache_stores: observation.cache_stores,
            observations: vec![if observation.snapshot_local {
                DataAssetObservation::CompleteLocalClosureEnumerated
            } else {
                DataAssetObservation::OwningRuntimeSourceObserved
            }],
            planned_external_work: vec![if observation.snapshot_local {
                DataAssetPlannedEffect::ImmutableLocalSnapshot
            } else {
                DataAssetPlannedEffect::SourceAcquisitionOrConsumerMaterialization
            }],
            unavailable: vec![
                DataAssetUnavailableFact::AcquiredSource,
                DataAssetUnavailableFact::ExistingMaterializationIdentity,
                DataAssetUnavailableFact::ReproducibilityConclusion,
            ],
        };
    }
    Ok(())
}

pub(crate) fn prepare_data_assets(
    root: &Path,
    owner_record_id: &str,
    plans: &[DataAssetPlan],
    attempts: &mut [DataAssetPreparationAttempt],
    progress: &Progress,
    mut persist: impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    for index in 0..attempts.len() {
        let plan = plans
            .iter()
            .find(|plan| plan.attempt_id == attempts[index].attempt_id)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("missing data-asset plan {:?}", attempts[index].attempt_id),
            })?;
        let result = prepare_one(
            root,
            owner_record_id,
            plan,
            &mut attempts[index],
            progress,
            &mut persist,
        );
        if let Err(error) = result {
            let interrupted = inferlab_runtime::interrupt::received();
            let failed_attempt_id = attempts[index].attempt_id.clone();
            attempts[index].terminate(interrupted, error.to_string())?;
            for pending in &mut attempts[index + 1..] {
                let message = if interrupted {
                    format!(
                        "source preparation interrupted before attempt {:?} began",
                        pending.attempt_id
                    )
                } else {
                    format!("source preparation stopped after attempt {failed_attempt_id:?} failed")
                };
                pending.terminate(true, message)?;
            }
            persist(&attempts[index..])?;
            return Err(error);
        }
    }
    Ok(())
}

fn prepare_one(
    root: &Path,
    owner_record_id: &str,
    plan: &DataAssetPlan,
    attempt: &mut DataAssetPreparationAttempt,
    progress: &Progress,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    progress
        .phase(Phase::named("measurement source preparation").current_item(&plan.attempt_id))?;
    match &plan.source {
        DataAssetSource::Eval { source } => {
            prepare_eval(root, owner_record_id, source, attempt, persist)
        }
        DataAssetSource::ReleaseCatalog { source } => release_catalog::prepare(
            &source.cache_path,
            &source.url,
            &source.expected_sha256,
            attempt,
            persist,
        ),
        DataAssetSource::Agentic { source } => prepare_agentic(
            root,
            owner_record_id,
            &source.command,
            &source.definition,
            attempt,
            persist,
        ),
    }
}

fn prepare_eval(
    root: &Path,
    owner_record_id: &str,
    source: &EvalDataAssetSource,
    attempt: &mut DataAssetPreparationAttempt,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    attempt.begin_resolution()?;
    persist(std::slice::from_ref(attempt))?;
    let artifact_dir = python::asset_directory(root, owner_record_id, &attempt.attempt_id);
    let request = eval_request(
        &source.workspace_root,
        &source.workspace_source_exclusions,
        &source.definition,
        source.bundled_task.as_deref(),
        MeasurementDataAssetPreparationPhase::Resolve,
        artifact_dir.clone(),
    )?;
    let result = python::run_phase(
        root,
        owner_record_id,
        &attempt.attempt_id,
        DataAssetPreparationPhase::Resolve,
        &source.command,
        &request,
    )?;
    match python::commit_eval_resolve(attempt, result, persist)? {
        python::EvalResolveOutcome::Ready => Ok(()),
        python::EvalResolveOutcome::SnapshotLocal => {
            attempt.begin_acquisition()?;
            persist(std::slice::from_ref(attempt))?;
            let request = eval_request(
                &source.workspace_root,
                &source.workspace_source_exclusions,
                &source.definition,
                source.bundled_task.as_deref(),
                MeasurementDataAssetPreparationPhase::SnapshotLocal,
                artifact_dir,
            )?;
            let result = python::run_phase(
                root,
                owner_record_id,
                &attempt.attempt_id,
                DataAssetPreparationPhase::SnapshotLocal,
                &source.command,
                &request,
            )?;
            python::commit_terminal(
                attempt,
                DataAssetPreparationPhase::SnapshotLocal,
                result,
                persist,
            )
        }
    }
}

fn prepare_agentic(
    root: &Path,
    owner_record_id: &str,
    command: &ClientCommandPlan,
    source: &ResolvedBenchAgenticSource,
    attempt: &mut DataAssetPreparationAttempt,
    persist: &mut impl FnMut(&[DataAssetPreparationAttempt]) -> Result<(), InferlabError>,
) -> Result<(), InferlabError> {
    attempt.begin_resolution()?;
    persist(std::slice::from_ref(attempt))?;
    let artifact_dir = python::asset_directory(root, owner_record_id, &attempt.attempt_id);
    let request = agentic_request(
        source,
        MeasurementDataAssetPreparationPhase::Resolve,
        artifact_dir.clone(),
    );
    let result = python::run_phase(
        root,
        owner_record_id,
        &attempt.attempt_id,
        DataAssetPreparationPhase::Resolve,
        command,
        &request,
    )?;
    let resolution = python::commit_agentic_resolve(attempt, result, persist)?;
    attempt.begin_acquisition()?;
    persist(std::slice::from_ref(attempt))?;
    let request = agentic_request(
        source,
        MeasurementDataAssetPreparationPhase::Acquire {
            resolved_revision: resolution.observed_revision,
            cache_state_before: cache_outcome_to_wire(resolution.cache_state),
        },
        artifact_dir,
    );
    let result = python::run_phase(
        root,
        owner_record_id,
        &attempt.attempt_id,
        DataAssetPreparationPhase::Acquire,
        command,
        &request,
    )?;
    python::commit_terminal(attempt, DataAssetPreparationPhase::Acquire, result, persist)
}

fn eval_request(
    workspace_root: &Path,
    workspace_source_exclusions: &[PathBuf],
    definition: &EvalDefinition,
    bundled_task: Option<&BundledEvalTask>,
    phase: MeasurementDataAssetPreparationPhase,
    artifact_dir: PathBuf,
) -> Result<MeasurementDataAssetPreparationRequest, InferlabError> {
    Ok(MeasurementDataAssetPreparationRequest {
        protocol_version: ProtocolVersion::V9,
        phase,
        source: MeasurementDataAssetSourceInput::Eval {
            workspace_root: workspace_root.to_path_buf(),
            workspace_source_exclusions: workspace_source_exclusions.to_vec(),
            definition: Box::new(wire::eval_definition_input(definition, bundled_task)?),
        },
        artifact_dir,
    })
}

fn agentic_request(
    source: &ResolvedBenchAgenticSource,
    phase: MeasurementDataAssetPreparationPhase,
    artifact_dir: PathBuf,
) -> MeasurementDataAssetPreparationRequest {
    MeasurementDataAssetPreparationRequest {
        protocol_version: ProtocolVersion::V9,
        phase,
        source: MeasurementDataAssetSourceInput::Agentic {
            source: Box::new(wire::bench_agentic_source_input(source)),
        },
        artifact_dir,
    }
}

fn cache_outcome_to_wire(
    outcome: DataAssetCacheOutcome,
) -> inferlab_protocol::MeasurementDataAssetCacheOutcome {
    match outcome {
        DataAssetCacheOutcome::FullHit => {
            inferlab_protocol::MeasurementDataAssetCacheOutcome::FullHit
        }
        DataAssetCacheOutcome::Miss => inferlab_protocol::MeasurementDataAssetCacheOutcome::Miss,
        DataAssetCacheOutcome::PartialReuse => {
            inferlab_protocol::MeasurementDataAssetCacheOutcome::PartialReuse
        }
        DataAssetCacheOutcome::Unavailable => {
            inferlab_protocol::MeasurementDataAssetCacheOutcome::Unavailable
        }
    }
}
