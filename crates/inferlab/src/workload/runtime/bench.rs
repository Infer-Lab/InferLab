//! Bench execution coordinator. Policy and evidence owners live in focused
//! child modules; AIPerf remains the native request-execution boundary.

pub mod adaptive;
mod case;
mod matrix;
mod native;
mod phase_barrier;
mod prefix_cache;
pub mod result;
pub mod session;

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
use crate::workload::{BenchExecutionPlan, BenchPlan, ResolvedWorkloadPlan, WorkloadServerAccess};
use std::path::Path;

pub fn run_bench(
    root: &Path,
    record_id: &str,
    plan: &BenchPlan,
    server_access: WorkloadServerAccess<'_>,
    record_evidence: ResolvedWorkloadPlan,
    progress: &Progress,
) -> Result<WorkloadRecord, InferlabError> {
    // Earlier runs' unclean exits leave recorded client groups behind;
    // terminate identity-matching survivors before this run launches its
    // own clients ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    sweep_stale_client_groups(root);
    let mut session = WorkloadRecordSession::begin(
        root,
        record_id,
        WorkloadKind::Bench,
        &plan.id,
        record_evidence,
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
            let operation = match server::acquire_operation(root, record_id) {
                Ok(operation) => operation,
                Err(error) => {
                    finish_failed_bench(&mut session, error.to_string())?;
                    return Ok(session.into_record());
                }
            };
            let admission =
                server::status(root, record_id).and_then(|report| server::require_running(&report));
            if let Err(error) = admission {
                finish_failed_bench(&mut session, error.to_string())?;
                return Ok(session.into_record());
            }
            execute_bench(root, &server_record_id, plan, &mut session, progress)?;
            drop(operation);
        }
    }
    Ok(session.into_record())
}

fn execute_bench(
    root: &Path,
    server_record_id: &str,
    plan: &BenchPlan,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<(), InferlabError> {
    let mut plan = plan.clone();
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
        BenchExecutionPlan::Matrix { cases } => {
            run_matrix_cases(&plan, cases, session, capture.as_mut(), progress)
        }
        BenchExecutionPlan::Adaptive {
            policy,
            initial_request_rates,
            max_search_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
        } => run_adaptive(
            &plan,
            policy,
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

pub struct BenchRunOutcome {
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

pub fn skip<T>(
    root: &Path,
    record_id: &str,
    kind: WorkloadKind,
    definition_id: &str,
    plan: &T,
    reason: &str,
    progress: &Progress,
) -> Result<WorkloadRecord, InferlabError>
where
    T: Clone + Into<ResolvedWorkloadPlan>,
{
    let resolved = plan.clone().into();
    let mut session = WorkloadRecordSession::begin(root, record_id, kind, definition_id, resolved)?;
    progress.phase(Phase::named("record created").record(
        record_id,
        root.join(crate::record::RECORDS_DIR).join(record_id),
    ))?;
    session.record_mut().skip_reason = Some(reason.to_owned());
    session.finish(WorkloadStatus::Skipped)?;
    Ok(session.into_record())
}
