//! One Bench case budget: prefix-cache action, AIPerf release, capture window,
//! result adjudication, and record assembly.

use super::super::{AcceptedClient, AdjudicatedClient};
use super::native::{adjudicate_bench_client, run_bench_client};
use super::phase_barrier::{
    PROFILE_BARRIER_ENV, PROFILE_BARRIER_REQUIRES_WARMUP_ENV, ProfileBarrier,
};
use super::prefix_cache::{CachePreparationInput, prepare_prefix_cache};
use super::result::evaluate_case_slos;
use crate::InferlabError;
use crate::workload::domain::ResolvedBenchSource;
use crate::workload::plan::session_population_layout;
use crate::workload::record::{
    BenchCachePreparationEvidence, BenchCachePreparationPhase, BenchCachePreparationTransition,
    BenchCaseEvidence, BenchCaseRecord, BenchPopulationSliceEvidence, ClientCasePaths,
    DataAssetMaterializationEvidence, WorkloadRecordSession, WorkloadStatus,
};
use crate::workload::{BenchCasePlan, BenchPlan, BenchPreparationStep};
use inferlab_protocol::BenchClientResult;
use inferlab_runtime::operation_bound::{
    OperationBound, OperationTerminalCause, OperationTimingEvidence,
};
use std::thread::{self, ScopedJoinHandle};
use std::time::Duration;

struct CaseRun {
    adjudicated: AdjudicatedClient<BenchClientResult>,
    cache_preparation: Option<BenchCachePreparationEvidence>,
}

struct CaseRunFailure {
    message: String,
    cache_preparation: Option<BenchCachePreparationEvidence>,
}

pub(super) fn run_bench_case(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    capture: Option<&mut inferlab_profiler::session::CaptureSession>,
) -> Result<BenchCaseRecord, InferlabError> {
    let paths = session.case_paths(&case.id)?;
    let budget = Duration::from_secs(plan.client.effective_definition.timeout_seconds);
    let bound = OperationBound::finite(budget);
    let requires_warmup_barrier = case
        .preparation_order
        .contains(&BenchPreparationStep::WarmupDrain);
    let controlled_cache = case
        .preparation_order
        .contains(&BenchPreparationStep::CacheReset);
    let pre_client_preparation = if controlled_cache && !requires_warmup_barrier {
        plan.client.prefix_cache_reset.as_ref().map(|action| {
            prepare_prefix_cache(
                CachePreparationInput {
                    endpoint: &plan.client.endpoint,
                    action,
                    start: plan.client.effective_definition.cache_start,
                    conditioning: plan.client.prefix_cache_conditioning.as_ref(),
                    population: plan.client.population.as_ref(),
                    warmup_drained: false,
                },
                &bound,
            )
        })
    } else {
        None
    };
    if let Some(preparation) = pre_client_preparation.as_ref()
        && let Some(message) = cache_preparation_error(preparation)
    {
        return Ok(failed_preparation_case(
            plan,
            case,
            session,
            paths,
            pre_client_preparation,
            &bound,
            message,
        ));
    }
    let run_and_adjudicate = |bound: &OperationBound| -> Result<_, InferlabError> {
        let accepted = run_bench_client(plan, case, session, &paths, bound, &[])?;
        Ok(adjudicate_bench_client(accepted, bound, plan, case))
    };
    let case_run = if let Some(mut preparation) = pre_client_preparation {
        let mut release_and_run = || {
            preparation
                .transitions
                .push(BenchCachePreparationTransition {
                    phase: BenchCachePreparationPhase::ProfilingReleased,
                    elapsed_ms: bound.elapsed_ms(),
                });
            run_and_adjudicate(&bound)
        };
        let adjudicated = match capture {
            Some(capture) => capture.run_window(&case.id, release_and_run),
            None => release_and_run(),
        }?;
        CaseRun {
            adjudicated,
            cache_preparation: Some(preparation),
        }
    } else {
        match capture {
            Some(capture) if requires_warmup_barrier => match run_after_setup_barrier(
                plan,
                case,
                session,
                &paths,
                Some(capture),
                &bound,
                requires_warmup_barrier,
            ) {
                Ok(case_run) => Ok(case_run),
                Err(failure) => {
                    return Ok(failed_barrier_case(
                        plan, case, session, paths, failure, &bound,
                    ));
                }
            },
            Some(capture) => capture
                .run_window(&case.id, || {
                    let bound = OperationBound::finite(budget);
                    run_and_adjudicate(&bound)
                })
                .map(|adjudicated| CaseRun {
                    adjudicated,
                    cache_preparation: None,
                }),
            None if requires_warmup_barrier => match run_after_setup_barrier(
                plan,
                case,
                session,
                &paths,
                None,
                &bound,
                requires_warmup_barrier,
            ) {
                Ok(case_run) => Ok(case_run),
                Err(failure) => {
                    return Ok(failed_barrier_case(
                        plan, case, session, paths, failure, &bound,
                    ));
                }
            },
            None => {
                let bound = OperationBound::finite(budget);
                run_and_adjudicate(&bound).map(|adjudicated| CaseRun {
                    adjudicated,
                    cache_preparation: None,
                })
            }
        }?
    };
    let CaseRun {
        adjudicated,
        cache_preparation,
    } = case_run;
    let AdjudicatedClient {
        mut accepted,
        mut succeeded,
        mut error,
    } = adjudicated;
    accepted.timing.start_boundary = if controlled_cache && !requires_warmup_barrier {
        "before_prefix_cache_reset"
    } else {
        "before_external_client_release"
    }
    .to_owned();
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
            preparation_attempt_id: session.data_asset_attempt_id().map(str::to_owned),
            data_asset_materialization: agentic_data_asset_materialization(
                plan,
                session,
                result.as_ref(),
            ),
            data_asset_materialization_identity: population_materialization_identity(plan),
            cache_preparation,
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
            agentic: result
                .as_ref()
                .and_then(|result| result.agentic_evidence.clone()),
            prompt_token_reconciliation: result.as_ref().map_or_else(Vec::new, |result| {
                result.prompt_token_reconciliation.clone()
            }),
            prompt_cache_observations: result
                .as_ref()
                .map_or_else(Vec::new, |result| result.prompt_cache_observations.clone()),
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

fn run_after_setup_barrier(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    mut capture: Option<&mut inferlab_profiler::session::CaptureSession>,
    bound: &OperationBound,
    requires_warmup: bool,
) -> Result<CaseRun, Box<CaseRunFailure>> {
    let barrier = ProfileBarrier::bind().map_err(|error| {
        Box::new(CaseRunFailure {
            message: error.to_string(),
            cache_preparation: None,
        })
    })?;
    let barrier_address = barrier.address().to_owned();
    let requires_warmup_value = if requires_warmup { "1" } else { "0" };
    let runtime_environment = [
        (PROFILE_BARRIER_ENV, barrier_address.as_str()),
        (PROFILE_BARRIER_REQUIRES_WARMUP_ENV, requires_warmup_value),
    ];

    thread::scope(|scope| {
        let client = scope
            .spawn(|| run_bench_client(plan, case, session, paths, bound, &runtime_environment));
        let release = match barrier.wait_for_ready(&client) {
            Ok(Some(release)) => release,
            Ok(None) => {
                return finish_before_profile_release(
                    client,
                    plan,
                    case,
                    bound,
                    capture.as_deref_mut(),
                )
                .map_err(|error| {
                    Box::new(CaseRunFailure {
                        message: error.to_string(),
                        cache_preparation: None,
                    })
                });
            }
            Err(error) => {
                let message = error.to_string();
                if let Some(capture) = capture.as_deref_mut()
                    && let Err(error) =
                        capture.record_unopened_window(&case.id, true, message.clone())
                {
                    return Err(Box::new(CaseRunFailure {
                        message: format!("{message}; {error}"),
                        cache_preparation: None,
                    }));
                }
                return finish_after_barrier_failure(client, plan, case, bound, None, message);
            }
        };
        let mut client = Some(client);
        let mut release = Some(release);
        let mut cache_preparation = plan.client.prefix_cache_reset.as_ref().map(|action| {
            prepare_prefix_cache(
                CachePreparationInput {
                    endpoint: &plan.client.endpoint,
                    action,
                    start: plan.client.effective_definition.cache_start,
                    conditioning: plan.client.prefix_cache_conditioning.as_ref(),
                    population: plan.client.population.as_ref(),
                    warmup_drained: requires_warmup,
                },
                bound,
            )
        });
        let preparation_error = cache_preparation.as_ref().and_then(cache_preparation_error);
        if let Some(message) = preparation_error {
            drop(release.take());
            let Some(client) = client.take() else {
                return Err(Box::new(CaseRunFailure {
                    message: "cache preparation lost Bench client ownership".to_owned(),
                    cache_preparation,
                }));
            };
            let accepted = match join_bench_client(client) {
                Ok(accepted) => accepted,
                Err(error) => {
                    return Err(Box::new(CaseRunFailure {
                        message: error.to_string(),
                        cache_preparation,
                    }));
                }
            };
            let mut adjudicated = adjudicate_bench_client(accepted, bound, plan, case);
            adjudicated.succeeded = false;
            adjudicated.error = Some(message.to_owned());
            if let Some(capture) = capture.as_deref_mut()
                && let Err(error) =
                    capture.record_unopened_window(&case.id, true, message.to_owned())
            {
                return Err(Box::new(CaseRunFailure {
                    message: error.to_string(),
                    cache_preparation,
                }));
            }
            return Ok(CaseRun {
                adjudicated,
                cache_preparation,
            });
        }
        let mut release_and_join = || {
            let release = release
                .take()
                .ok_or_else(|| InferlabError::ProfileBarrierProtocol {
                    message: "Bench case attempted to release profiling twice".to_owned(),
                })?;
            release.acknowledge()?;
            if let Some(preparation) = cache_preparation.as_mut() {
                preparation
                    .transitions
                    .push(BenchCachePreparationTransition {
                        phase: BenchCachePreparationPhase::ProfilingReleased,
                        elapsed_ms: bound.elapsed_ms(),
                    });
            }
            let client = client
                .take()
                .ok_or_else(|| InferlabError::ProfileBarrierProtocol {
                    message: "Bench case attempted to join its client twice".to_owned(),
                })?;
            let accepted = join_bench_client(client)?;
            Ok(CaseRun {
                adjudicated: adjudicate_bench_client(accepted, bound, plan, case),
                cache_preparation: cache_preparation.take(),
            })
        };
        let result: Result<CaseRun, InferlabError> = match capture {
            Some(capture) => capture.run_window(&case.id, release_and_join),
            None => release_and_join(),
        };
        match result {
            Ok(case_run) => Ok(case_run),
            Err(error) => {
                drop(release.take());
                let Some(client) = client.take() else {
                    return Err(Box::new(CaseRunFailure {
                        message: error.to_string(),
                        cache_preparation,
                    }));
                };
                finish_after_barrier_failure(
                    client,
                    plan,
                    case,
                    bound,
                    cache_preparation,
                    error.to_string(),
                )
            }
        }
    })
}

fn failed_barrier_case(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    paths: ClientCasePaths,
    failure: Box<CaseRunFailure>,
    bound: &OperationBound,
) -> BenchCaseRecord {
    let terminal_cause = if bound.is_expired() {
        OperationTerminalCause::TimedOut
    } else {
        OperationTerminalCause::Failed
    };
    let CaseRunFailure {
        message,
        cache_preparation,
    } = *failure;
    failed_case_record(
        plan,
        case,
        session,
        paths,
        cache_preparation,
        bound.timing("before_external_client_release", terminal_cause),
        &message,
    )
}

fn finish_after_barrier_failure(
    client: ScopedJoinHandle<'_, Result<AcceptedClient<BenchClientResult>, InferlabError>>,
    plan: &BenchPlan,
    case: &BenchCasePlan,
    bound: &OperationBound,
    cache_preparation: Option<BenchCachePreparationEvidence>,
    message: String,
) -> Result<CaseRun, Box<CaseRunFailure>> {
    let accepted = match join_bench_client(client) {
        Ok(accepted) => accepted,
        Err(error) => {
            return Err(Box::new(CaseRunFailure {
                message: format!("{message}; {error}"),
                cache_preparation,
            }));
        }
    };
    let mut adjudicated = adjudicate_bench_client(accepted, bound, plan, case);
    adjudicated.succeeded = false;
    adjudicated.error = Some(
        adjudicated
            .error
            .take()
            .map_or(message.clone(), |error| format!("{error}; {message}")),
    );
    Ok(CaseRun {
        adjudicated,
        cache_preparation,
    })
}

fn cache_preparation_error(preparation: &BenchCachePreparationEvidence) -> Option<&'static str> {
    if !preparation.reset.succeeded {
        Some("prefix-cache reset failed")
    } else if preparation.start == crate::workspace::BenchCacheStart::Primed
        && preparation
            .conditioning
            .as_ref()
            .is_none_or(|conditioning| !conditioning.succeeded)
    {
        Some("prefix-cache conditioning failed")
    } else {
        None
    }
}

fn failed_preparation_case(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    paths: ClientCasePaths,
    cache_preparation: Option<BenchCachePreparationEvidence>,
    bound: &OperationBound,
    error: &str,
) -> BenchCaseRecord {
    let terminal_cause = if bound.is_expired() {
        OperationTerminalCause::TimedOut
    } else {
        OperationTerminalCause::Failed
    };
    failed_case_record(
        plan,
        case,
        session,
        paths,
        cache_preparation,
        bound.timing("before_prefix_cache_reset", terminal_cause),
        error,
    )
}

fn failed_case_record(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    paths: ClientCasePaths,
    cache_preparation: Option<BenchCachePreparationEvidence>,
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
            preparation_attempt_id: session.data_asset_attempt_id().map(str::to_owned),
            data_asset_materialization: None,
            data_asset_materialization_identity: population_materialization_identity(plan),
            cache_preparation,
            metrics: None,
            slo: None,
            population_slice: bench_population_slice(plan, case),
            completed_requests: None,
            failed_requests: None,
            normalization_schema: None,
            session: None,
            agentic: None,
            prompt_token_reconciliation: Vec::new(),
            prompt_cache_observations: Vec::new(),
            report_invocations: Vec::new(),
        },
        native_command: None,
        native_exit_code: None,
        raw_artifacts: None,
        error: Some(error.to_owned()),
    }
}

fn finish_before_profile_release(
    client: ScopedJoinHandle<'_, Result<AcceptedClient<BenchClientResult>, InferlabError>>,
    plan: &BenchPlan,
    case: &BenchCasePlan,
    bound: &OperationBound,
    mut capture: Option<&mut inferlab_profiler::session::CaptureSession>,
) -> Result<CaseRun, InferlabError> {
    let accepted = match join_bench_client(client) {
        Ok(accepted) => accepted,
        Err(error) => {
            if let Some(capture) = capture.as_deref_mut() {
                capture.record_unopened_window(&case.id, false, error.to_string())?;
            }
            return Err(error);
        }
    };
    let mut adjudicated = adjudicate_bench_client(accepted, bound, plan, case);
    let message =
        "Bench client exited before AIPerf reported profiling readiness; capture remained unopened"
            .to_owned();
    if let Some(capture) = capture {
        capture.record_unopened_window(&case.id, true, message.clone())?;
    }
    adjudicated.succeeded = false;
    adjudicated.error = Some(
        adjudicated
            .error
            .take()
            .map_or(message.clone(), |error| format!("{error}; {message}")),
    );
    Ok(CaseRun {
        adjudicated,
        cache_preparation: None,
    })
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

fn agentic_data_asset_materialization(
    plan: &BenchPlan,
    session: &WorkloadRecordSession,
    result: Option<&BenchClientResult>,
) -> Option<DataAssetMaterializationEvidence> {
    let ResolvedBenchSource::Agentic { agentic_source } = &plan.client.effective_definition.source
    else {
        return None;
    };
    result
        .and_then(|result| result.agentic_evidence.as_deref())
        .and_then(|evidence| evidence.run.as_ref())?;
    Some(DataAssetMaterializationEvidence {
        preparation_attempt_id: session.data_asset_attempt_id()?.to_owned(),
        authority: "aiperf_case_startup".to_owned(),
        materialization_identity: format!(
            "{}:{}:{}",
            agentic_source.catalog.aiperf_loader,
            agentic_source.catalog.source_format,
            agentic_source.catalog.materialization_identity
        ),
        dataset_fingerprint: None,
        unavailable_reason: Some(
            "AIPerf did not report a distinct derived-cache fingerprint".to_owned(),
        ),
    })
}

fn population_materialization_identity(plan: &BenchPlan) -> Option<String> {
    match &plan.client.effective_definition.source {
        ResolvedBenchSource::Requests {
            request_source: crate::workload::domain::ResolvedBenchRequestSource::Dataset { .. },
        } => {}
        ResolvedBenchSource::Sessions { .. } => {}
        ResolvedBenchSource::Agentic { .. } | ResolvedBenchSource::Requests { .. } => return None,
    }
    plan.client
        .population
        .as_ref()
        .map(|population| format!("sha256:{}", population.sha256))
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
        ResolvedBenchSource::Agentic { .. } => None,
    }
}
