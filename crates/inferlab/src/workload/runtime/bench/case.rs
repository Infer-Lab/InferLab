//! One Bench case budget: prefix-cache action, AIPerf release, capture window,
//! result adjudication, and record assembly.

use super::super::{AcceptedClient, AdjudicatedClient};
use super::native::{adjudicate_bench_client, run_bench_client};
use super::phase_barrier::{PROFILE_BARRIER_ENV, ProfileBarrier};
use super::prefix_cache::reset_prefix_cache;
use super::result::evaluate_case_slos;
use crate::InferlabError;
use crate::workload::domain::ResolvedBenchSource;
use crate::workload::plan::session_population_layout;
use crate::workload::record::{
    BenchCaseEvidence, BenchCaseRecord, BenchPopulationSliceEvidence, ClientCasePaths,
    PrefixCacheResetEvidence, WorkloadRecordSession, WorkloadStatus,
};
use crate::workload::{BenchCasePlan, BenchPlan};
use inferlab_protocol::BenchClientResult;
use inferlab_runtime::operation_bound::{
    OperationBound, OperationTerminalCause, OperationTimingEvidence,
};
use std::thread::{self, ScopedJoinHandle};
use std::time::Duration;

pub fn run_bench_case(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    capture: Option<&mut inferlab_profiler::session::CaptureSession>,
) -> Result<BenchCaseRecord, InferlabError> {
    let paths = session.case_paths(&case.id)?;
    let budget = Duration::from_secs(plan.client.effective_definition.timeout_seconds);
    let reset_bound = plan
        .client
        .prefix_cache_reset
        .as_ref()
        .map(|_| OperationBound::finite(budget));
    let reset = plan
        .client
        .prefix_cache_reset
        .as_ref()
        .zip(reset_bound.as_ref())
        .map(|(action, bound)| reset_prefix_cache(&plan.client.endpoint, action, bound));
    if reset_bound.as_ref().is_some_and(OperationBound::is_expired) {
        let timing = reset_bound.as_ref().map_or_else(
            || {
                OperationBound::finite(Duration::ZERO).timing(
                    "before_prefix_cache_reset",
                    OperationTerminalCause::TimedOut,
                )
            },
            |bound| {
                bound.timing(
                    "before_prefix_cache_reset",
                    OperationTerminalCause::TimedOut,
                )
            },
        );
        return Ok(failed_case(
            plan,
            case,
            paths,
            reset,
            timing,
            "measurement-case budget expired during prefix-cache reset",
        ));
    }
    if reset.as_ref().is_some_and(|evidence| !evidence.succeeded) {
        let timing = reset_bound.as_ref().map_or_else(
            || {
                OperationBound::finite(Duration::ZERO)
                    .timing("before_prefix_cache_reset", OperationTerminalCause::Failed)
            },
            |bound| bound.timing("before_prefix_cache_reset", OperationTerminalCause::Failed),
        );
        return Ok(failed_case(
            plan,
            case,
            paths,
            reset,
            timing,
            "prefix-cache reset failed",
        ));
    }
    let run_and_adjudicate = || -> Result<_, InferlabError> {
        let adjudicated = match reset_bound.as_ref() {
            Some(bound) => {
                let accepted = run_bench_client(plan, case, session, &paths, bound, &[])?;
                adjudicate_bench_client(accepted, bound, plan, case)
            }
            None => {
                let bound = OperationBound::finite(budget);
                let accepted = run_bench_client(plan, case, session, &paths, &bound, &[])?;
                adjudicate_bench_client(accepted, &bound, plan, case)
            }
        };
        Ok(adjudicated)
    };
    let adjudicated = match capture {
        Some(capture)
            if case.warmup_request_count > 0
                || case.warmup_session_count.is_some_and(|count| count > 0) =>
        {
            run_captured_after_warmup(
                plan,
                case,
                session,
                &paths,
                capture,
                reset_bound.as_ref(),
                budget,
            )
        }
        Some(capture) => capture.run_window(&case.id, run_and_adjudicate),
        None => run_and_adjudicate(),
    }?;
    let AdjudicatedClient {
        mut accepted,
        mut succeeded,
        mut error,
    } = adjudicated;
    if reset.is_some() {
        accepted.timing.start_boundary = "before_prefix_cache_reset".to_owned();
    } else {
        accepted.timing.start_boundary = "before_external_client_release".to_owned();
    }
    let result = accepted.result;
    let slo = if succeeded {
        match result
            .as_ref()
            .map(|result| evaluate_case_slos(&plan.client.slo, result))
        {
            Some(Ok(evaluation)) => evaluation,
            Some(Err(slo_error)) => {
                succeeded = false;
                error = Some(slo_error);
                None
            }
            None => None,
        }
    } else {
        None
    };
    Ok(BenchCaseRecord {
        id: case.id.clone(),
        status: if succeeded {
            WorkloadStatus::Succeeded
        } else {
            WorkloadStatus::Failed
        },
        request: paths.request,
        result: paths.result,
        stdout: Some(paths.stdout),
        stderr: Some(paths.stderr),
        process: accepted.run.process,
        timing: accepted.timing,
        evidence: BenchCaseEvidence {
            prefix_cache_reset: reset,
            metrics: result.as_ref().map(|result| result.metrics.clone()),
            slo,
            population_slice: bench_population_slice(plan, case),
            completed_requests: result.as_ref().map(|result| result.completed_requests),
            failed_requests: result.as_ref().map(|result| result.failed_requests),
            normalization_schema: result
                .as_ref()
                .map(|result| result.normalization_schema.clone()),
            session: result
                .as_ref()
                .and_then(|result| result.session_evidence.clone()),
            prompt_token_reconciliation: result.as_ref().map_or_else(Vec::new, |result| {
                result.prompt_token_reconciliation.clone()
            }),
            report_invocations: result
                .as_ref()
                .map_or_else(Vec::new, |result| result.report_invocations.clone()),
        },
        native_command: result.as_ref().map(|result| result.native_command.clone()),
        native_exit_code: result.as_ref().and_then(|result| result.native_exit_code),
        raw_artifacts: result.as_ref().map(|result| result.raw_artifacts.clone()),
        error,
    })
}

fn run_captured_after_warmup(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    capture: &mut inferlab_profiler::session::CaptureSession,
    reset_bound: Option<&OperationBound>,
    budget: Duration,
) -> Result<AdjudicatedClient<BenchClientResult>, InferlabError> {
    let case_bound = OperationBound::finite(budget);
    let bound = reset_bound.unwrap_or(&case_bound);
    let barrier = ProfileBarrier::bind()?;
    let barrier_address = barrier.address().to_owned();
    let runtime_environment = [(PROFILE_BARRIER_ENV, barrier_address.as_str())];

    thread::scope(|scope| {
        let client = scope
            .spawn(|| run_bench_client(plan, case, session, paths, bound, &runtime_environment));
        let release = match barrier.wait_for_ready(&client) {
            Ok(Some(release)) => release,
            Ok(None) => {
                return finish_before_profile_release(client, plan, case, bound, capture);
            }
            Err(error) => {
                finish_client_cleanup(client, plan, case, bound);
                return Err(error);
            }
        };
        let mut client = Some(client);
        let mut release = Some(release);
        let result = capture.run_window(&case.id, || {
            let release = release
                .take()
                .ok_or_else(|| InferlabError::ProfileBarrierProtocol {
                    message: "capture window attempted to release profiling twice".to_owned(),
                })?;
            release.acknowledge()?;
            let client = client
                .take()
                .ok_or_else(|| InferlabError::ProfileBarrierProtocol {
                    message: "capture window attempted to join the Bench client twice".to_owned(),
                })?;
            let accepted = join_bench_client(client)?;
            Ok(adjudicate_bench_client(accepted, bound, plan, case))
        });
        if result.is_err() {
            drop(release.take());
            if let Some(client) = client.take() {
                finish_client_cleanup(client, plan, case, bound);
            }
        }
        result
    })
}

fn finish_before_profile_release(
    client: ScopedJoinHandle<'_, Result<AcceptedClient<BenchClientResult>, InferlabError>>,
    plan: &BenchPlan,
    case: &BenchCasePlan,
    bound: &OperationBound,
    capture: &mut inferlab_profiler::session::CaptureSession,
) -> Result<AdjudicatedClient<BenchClientResult>, InferlabError> {
    let accepted = match join_bench_client(client) {
        Ok(accepted) => accepted,
        Err(error) => {
            capture.record_unopened_window(&case.id, false, error.to_string())?;
            return Err(error);
        }
    };
    let mut adjudicated = adjudicate_bench_client(accepted, bound, plan, case);
    let message =
        "Bench client exited before AIPerf reported profiling readiness; capture remained unopened"
            .to_owned();
    capture.record_unopened_window(&case.id, true, message.clone())?;
    adjudicated.succeeded = false;
    adjudicated.error = Some(
        adjudicated
            .error
            .take()
            .map_or(message.clone(), |error| format!("{error}; {message}")),
    );
    Ok(adjudicated)
}

fn finish_client_cleanup(
    client: ScopedJoinHandle<'_, Result<AcceptedClient<BenchClientResult>, InferlabError>>,
    plan: &BenchPlan,
    case: &BenchCasePlan,
    bound: &OperationBound,
) {
    if let Ok(accepted) = join_bench_client(client) {
        adjudicate_bench_client(accepted, bound, plan, case);
    }
}

fn join_bench_client(
    client: ScopedJoinHandle<'_, Result<AcceptedClient<BenchClientResult>, InferlabError>>,
) -> Result<AcceptedClient<BenchClientResult>, InferlabError> {
    client
        .join()
        .map_err(|_| InferlabError::ProfileBarrierProtocol {
            message: "Bench client supervision thread panicked".to_owned(),
        })?
}

fn failed_case(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    paths: ClientCasePaths,
    reset: Option<PrefixCacheResetEvidence>,
    timing: OperationTimingEvidence,
    error: &str,
) -> BenchCaseRecord {
    BenchCaseRecord {
        id: case.id.clone(),
        status: WorkloadStatus::Failed,
        request: paths.request,
        result: paths.result,
        stdout: Some(paths.stdout),
        stderr: Some(paths.stderr),
        process: None,
        timing,
        evidence: BenchCaseEvidence {
            prefix_cache_reset: reset,
            metrics: None,
            slo: None,
            population_slice: bench_population_slice(plan, case),
            completed_requests: None,
            failed_requests: None,
            normalization_schema: None,
            session: None,
            prompt_token_reconciliation: Vec::new(),
            report_invocations: Vec::new(),
        },
        native_command: None,
        native_exit_code: None,
        raw_artifacts: None,
        error: Some(error.to_owned()),
    }
}

fn bench_population_slice(
    plan: &BenchPlan,
    case: &BenchCasePlan,
) -> Option<BenchPopulationSliceEvidence> {
    let population = plan.client.population.as_ref()?;
    match &plan.client.effective_definition.source {
        ResolvedBenchSource::Requests { .. } => Some(BenchPopulationSliceEvidence::Requests {
            population_sha256: population.sha256.clone(),
            warmup_start: 0,
            warmup_count: case.warmup_request_count,
            profiling_start: case.warmup_request_count,
            profiling_count: case.request_count,
        }),
        ResolvedBenchSource::Sessions { .. } => {
            let warmup_session_count = case.warmup_session_count.unwrap_or(0);
            let profiling_session_count = case.session_count.unwrap_or(0);
            let profiling_start =
                session_population_layout(warmup_session_count, profiling_session_count)?
                    .profiling_start;
            Some(BenchPopulationSliceEvidence::Sessions {
                population_sha256: population.sha256.clone(),
                warmup_start: 0,
                warmup_session_count,
                warmup_request_count: case.warmup_request_count,
                warmup_template_identities: population
                    .session_templates
                    .iter()
                    .take(warmup_session_count as usize)
                    .map(|template| template.template_identity.clone())
                    .collect(),
                profiling_start,
                profiling_session_count,
                profiling_request_count: case.request_count,
                profiling_template_identities: population
                    .session_templates
                    .iter()
                    .skip(profiling_start as usize)
                    .take(profiling_session_count as usize)
                    .map(|template| template.template_identity.clone())
                    .collect(),
            })
        }
    }
}
