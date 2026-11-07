//! Bench execution coordinator. Policy and evidence owners live in focused
//! child modules; AIPerf remains the native request-execution boundary.

pub(super) mod adaptive;
mod case;
mod matrix;
mod native;
mod phase_barrier;
mod prefix_cache;
pub(super) mod result;
mod reuse;
pub(super) mod session;
#[cfg(test)]
mod test_support;

use self::adaptive::run_adaptive;
use self::matrix::run_matrix_cases;
use super::client::sweep_stale_client_groups;
use super::preparation::prepare_bench_request_source;
use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::server;
use crate::workload::record::{
    WorkloadKind, WorkloadRecord, WorkloadRecordSession, WorkloadStatus,
};
use crate::workload::{
    BenchExecutionPlan, BenchPlan, ResolvedWorkloadPlan, WorkloadDataAssetEvidence,
    WorkloadServerAccess, prepare_data_assets,
};
use inferlab_runtime::operation_bound::{OperationBound, OperationTerminalCause};
use std::path::Path;

pub(crate) fn run_bench(
    root: &Path,
    record_id: &str,
    plan: &BenchPlan,
    server_access: WorkloadServerAccess<'_>,
    record_evidence: ResolvedWorkloadPlan,
    progress: &Progress,
    data_assets: WorkloadDataAssetEvidence,
) -> Result<WorkloadRecord, InferlabError> {
    // Earlier runs' unclean exits leave recorded client groups behind;
    // terminate identity-matching survivors before this run launches its
    // own clients ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    sweep_stale_client_groups(root);
    let mut standalone_attempts = match &data_assets {
        WorkloadDataAssetEvidence::Standalone { attempts, .. } => attempts.clone(),
        WorkloadDataAssetEvidence::Recipe { .. } | WorkloadDataAssetEvidence::None => Vec::new(),
    };
    let data_asset_plans = match &record_evidence {
        ResolvedWorkloadPlan::ManualBench(manual) => manual.data_assets.clone(),
        ResolvedWorkloadPlan::Bench(_) | ResolvedWorkloadPlan::Eval(_) => Vec::new(),
    };
    let mut session = WorkloadRecordSession::begin(
        root,
        record_id,
        WorkloadKind::Bench,
        &plan.id,
        record_evidence,
        data_assets,
    )?;
    progress.phase(Phase::named("record created").record(
        record_id,
        root.join(crate::record::RECORDS_DIR).join(record_id),
    ))?;
    let server_record_id = server_access.record_id().to_owned();
    match server_access {
        WorkloadServerAccess::RecipeOwned { .. } => {
            execute_bench(root, &server_record_id, plan, &mut session, progress)?
        }
        WorkloadServerAccess::ManagedServer { record_id } => {
            if !standalone_attempts.is_empty() {
                let preparation_bound = OperationBound::unbounded();
                let owner_record_id = session.record_mut().id.clone();
                let preparation = prepare_data_assets(
                    root,
                    &owner_record_id,
                    &data_asset_plans,
                    &mut standalone_attempts,
                    progress,
                    |updates| {
                        for update in updates {
                            session.update_standalone_data_asset_attempt(update)?;
                        }
                        Ok(())
                    },
                );
                if let Err(error) = preparation {
                    session.set_standalone_data_asset_timing(preparation_bound.timing(
                        "before_first_source_preparation_effect",
                        if inferlab_runtime::interrupt::received() {
                            OperationTerminalCause::Interrupted
                        } else {
                            OperationTerminalCause::Failed
                        },
                    ))?;
                    finish_failed_bench(&mut session, error.to_string())?;
                    return Ok(session.into_record());
                }
                session.set_standalone_data_asset_timing(preparation_bound.timing(
                    "before_first_source_preparation_effect",
                    OperationTerminalCause::Succeeded,
                ))?;
            }
            let operation = match server::acquire_operation(root, record_id) {
                Ok(operation) => operation,
                Err(error) => {
                    finish_failed_bench(&mut session, error.to_string())?;
                    return Ok(session.into_record());
                }
            };
            let admission = match server::lost_process_error(root, record_id) {
                Ok(lost) => lost,
                Err(error) => Some(error.to_string()),
            };
            if let Some(error) = admission {
                finish_failed_bench(&mut session, error)?;
                return Ok(session.into_record());
            }
            execute_bench(root, &server_record_id, plan, &mut session, progress)?;
            drop(operation);
        }
    }
    Ok(session.into_record())
}

/// Facts the server reports only at readiness — a cache-owning Engine's
/// data-parallel rank count, and whether its store is shared with what
/// publication bound — are applied to this run's plan before preparation; a
/// primed start whose reported targets the frontend cannot condition fails
/// before any reset or conditioning request ([[RFC-0004:C-BENCH-CACHE-STATE]]).
fn apply_observed_serving_facts(
    root: &Path,
    server_record_id: &str,
    plan: &mut BenchPlan,
) -> Result<(), InferlabError> {
    if !plan.client.serving_facts_observed {
        return Ok(());
    }
    let shape =
        crate::workload::conditioning_shape_of(&server::load_record(root, server_record_id)?);
    if shape.data_parallel_pending || shape.sharing_pending {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "bench {:?} needs the cache-owning serving facts server {server_record_id:?} reports at readiness, but its record holds none",
                plan.id
            ),
        });
    }
    plan.client.prefix_publication_ms = shape.publication_wait_ms;
    if let Some(conditioning) = plan.client.prefix_cache_conditioning.as_mut() {
        if shape.gateway_frontend && shape.conditioning_targets > 1 && !conditioning.frontend_fanout
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "bench {:?} selects cache.start = \"primed\", but server {server_record_id:?} reported {} cache-owning targets at readiness (attention data-parallel size {}): frontend routing cannot address an individual replica or data-parallel rank, so conditioning cannot cover every cache-owning rank",
                    plan.id, shape.conditioning_targets, shape.attention_data_parallel_size
                ),
            });
        }
        conditioning.attention_data_parallel_size = shape.attention_data_parallel_size;
        conditioning.publication_wait_ms = shape.publication_wait_ms;
    }
    Ok(())
}

fn execute_bench(
    root: &Path,
    server_record_id: &str,
    plan: &BenchPlan,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<(), InferlabError> {
    let mut plan = plan.clone();
    if let Err(error) = apply_observed_serving_facts(root, server_record_id, &mut plan) {
        session.record_mut().error = Some(error.to_string());
        session.record_mut().passed = Some(false);
        return session.finish(WorkloadStatus::Failed);
    }
    if let Err(error) = prepare_bench_request_source(&mut plan, session, progress) {
        session.record_mut().error = Some(error.to_string());
        session.record_mut().passed = Some(false);
        return session.finish(WorkloadStatus::Failed);
    }
    session.set_prepared_bench_plan(plan.clone())?;
    session.rewrite()?;
    let window_ids = match &plan.execution {
        BenchExecutionPlan::Matrix { cases } => {
            cases.iter().map(|case| case.id.clone()).collect::<Vec<_>>()
        }
        BenchExecutionPlan::Adaptive { .. } if plan.capture => {
            let message = "adaptive Bench does not have a static capture-window set".to_owned();
            session.record_mut().capture = Some(inferlab_profiler::record::CaptureRecord::failed(
                message.clone(),
            ));
            session.record_mut().error = Some(message);
            session.record_mut().passed = Some(false);
            return session.finish(WorkloadStatus::Failed);
        }
        BenchExecutionPlan::Adaptive { .. } => Vec::new(),
    };
    let mut capture = if plan.capture {
        let selection = match crate::server::running_profiler_selection(root, server_record_id) {
            Ok(selection) => selection,
            Err(error) => {
                let message = error.to_string();
                session.record_mut().capture = Some(
                    inferlab_profiler::record::CaptureRecord::failed(message.clone()),
                );
                session.record_mut().error = Some(message);
                session.record_mut().passed = Some(false);
                return session.finish(WorkloadStatus::Failed);
            }
        };
        match inferlab_profiler::session::CaptureSession::open(
            server_record_id,
            &session.record_mut().id,
            &window_ids,
            selection,
        ) {
            Ok(capture) => Some(capture),
            Err(record) => {
                let record = *record;
                let message = record
                    .error
                    .clone()
                    .unwrap_or_else(|| "failed to open Bench capture".to_owned());
                session.record_mut().capture = Some(record);
                session.record_mut().error = Some(message);
                session.record_mut().passed = Some(false);
                return session.finish(WorkloadStatus::Failed);
            }
        }
    } else {
        None
    };
    let outcome = match &plan.execution {
        BenchExecutionPlan::Matrix { cases } => run_matrix_cases(
            server_record_id,
            &plan,
            cases,
            session,
            capture.as_mut(),
            progress,
        ),
        BenchExecutionPlan::Adaptive {
            policy,
            preparation_order,
            initial_request_rates,
            max_search_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
        } => run_adaptive(
            server_record_id,
            &plan,
            policy,
            preparation_order,
            initial_request_rates,
            *max_search_steps,
            *min_rate_resolution,
            *request_count,
            *duration_seconds,
            session,
            progress,
        ),
    };
    if capture.is_some() {
        progress.phase(Phase::named("profiler finalization").current_item(&plan.id))?;
    }
    let capture_record = capture.map(inferlab_profiler::session::CaptureSession::finish);
    let capture_succeeded = capture_record
        .as_ref()
        .is_none_or(inferlab_profiler::record::CaptureRecord::succeeded);
    if let Some(message) = capture_record
        .as_ref()
        .filter(|record| !record.succeeded())
        .and_then(|record| record.error.clone())
    {
        session.record_mut().error = Some(message);
    }
    session.record_mut().capture = capture_record;
    let outcome = match outcome {
        Ok(outcome) => BenchRunOutcome {
            measurement_succeeded: outcome.measurement_succeeded && capture_succeeded,
            passed: outcome.passed && capture_succeeded,
        },
        Err(error) => {
            session.record_mut().error = Some(error.to_string());
            BenchRunOutcome {
                measurement_succeeded: false,
                passed: false,
            }
        }
    };
    session.record_mut().passed = Some(outcome.passed);
    session.finish(if outcome.measurement_succeeded {
        WorkloadStatus::Succeeded
    } else {
        WorkloadStatus::Failed
    })
}

pub(super) struct BenchRunOutcome {
    measurement_succeeded: bool,
    passed: bool,
}

fn finish_failed_bench(
    session: &mut WorkloadRecordSession,
    error: String,
) -> Result<(), InferlabError> {
    session.record_mut().passed = Some(false);
    session.record_mut().error = Some(error);
    session.finish(WorkloadStatus::Failed)
}

pub(crate) fn skip(
    root: &Path,
    record_id: &str,
    resolved: ResolvedWorkloadPlan,
    reason: &str,
    progress: &Progress,
    data_assets: WorkloadDataAssetEvidence,
) -> Result<WorkloadRecord, InferlabError> {
    let kind = resolved.kind();
    let definition_id = resolved.definition_id().to_owned();
    let mut session =
        WorkloadRecordSession::begin(root, record_id, kind, &definition_id, resolved, data_assets)?;
    progress.phase(Phase::named("record created").record(
        record_id,
        root.join(crate::record::RECORDS_DIR).join(record_id),
    ))?;
    session.record_mut().skip_reason = Some(reason.to_owned());
    session.finish(WorkloadStatus::Skipped)?;
    Ok(session.into_record())
}
