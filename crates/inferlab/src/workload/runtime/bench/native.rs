//! AIPerf request construction, release, result acceptance, and adjudication.

use super::super::client::{
    accept_client_result, client_terminal_cause, freeze_adjudicated_timing,
    reject_late_adjudication, remaining_seconds, run_client_with_environment,
};
use super::super::{AcceptedClient, AdjudicatedClient};
use super::result::bench_result_error;
use crate::InferlabError;
use crate::workload::domain::{ResolvedBenchRequestSource, ResolvedBenchSource};
use crate::workload::record::{ClientCasePaths, WorkloadRecordSession};
use crate::workload::wire;
use crate::workload::{BenchCasePlan, BenchPlan, LoadShape};
use crate::workspace::RequestRate;
use inferlab_protocol::{
    BenchCaseInput, BenchClientRequest, BenchClientResult, BenchLoadInput, ProtocolVersion,
};
use inferlab_runtime::operation_bound::OperationBound;

pub(super) fn run_bench_client(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    bound: &OperationBound,
    runtime_environment: &[(&str, &str)],
) -> Result<AcceptedClient<BenchClientResult>, InferlabError> {
    let request = BenchClientRequest {
        protocol_version: ProtocolVersion::V7,
        endpoint: wire::endpoint_input(&plan.client.endpoint),
        model: wire::model_input(&plan.client.model),
        definition: wire::bench_definition_input(&plan.client.effective_definition)?,
        population: plan.client.population.as_ref().map(wire::population_input),
        case: BenchCaseInput {
            load_shape: bench_load_input(&case.load_shape),
            request_count: case.request_count,
            warmup_request_count: case.warmup_request_count,
            duration_seconds: case.duration_seconds,
            session_count: case.session_count,
            warmup_session_count: case.warmup_session_count,
        },
        case_budget_seconds: remaining_seconds(bound),
        artifact_dir: paths.artifact_dir.clone(),
    };
    let run = run_client_with_environment(
        &plan.client.command,
        &request,
        session,
        paths,
        bound,
        runtime_environment,
    )?;
    Ok(accept_client_result::<BenchClientResult>(
        &session.absolute(&paths.result),
        "Bench client",
        run,
        bound,
    ))
}

pub(super) fn adjudicate_bench_client(
    mut accepted: AcceptedClient<BenchClientResult>,
    bound: &OperationBound,
    plan: &BenchPlan,
    case: &BenchCasePlan,
) -> AdjudicatedClient<BenchClientResult> {
    reject_late_adjudication(&mut accepted, bound);
    let domain_error = accepted.result.as_ref().and_then(|result| {
        bench_result_error(
            result,
            plan.client.tpot_applicability.is_applicable(),
            plan.client.effective_definition.server_metrics
                && matches!(
                    plan.client.effective_definition.source.request_source(),
                    Some(ResolvedBenchRequestSource::Dataset { dataset, .. })
                        if dataset == "speed_bench"
                ),
            case.session_count
                .map(|profiling| (case.warmup_session_count.unwrap_or_default(), profiling)),
            match &plan.client.effective_definition.source {
                ResolvedBenchSource::Agentic { agentic_source } => Some(agentic_source),
                ResolvedBenchSource::Requests { .. } | ResolvedBenchSource::Sessions { .. } => None,
            },
            case.request_count,
            plan.client.slo.request.as_ref(),
        )
    });
    reject_late_adjudication(&mut accepted, bound);
    let error = accepted.decode_error.take().or(domain_error);
    let succeeded = accepted.result.is_some() && error.is_none();
    let terminal_cause = client_terminal_cause(&accepted, succeeded);
    freeze_adjudicated_timing(&mut accepted, bound, terminal_cause);
    accepted.run.finish_cleanup();
    AdjudicatedClient {
        accepted,
        succeeded,
        error,
    }
}

fn bench_load_input(load: &LoadShape) -> BenchLoadInput {
    match load {
        LoadShape::ConcurrencyLimited { concurrency } => BenchLoadInput::ConcurrencyLimited {
            concurrency: *concurrency,
        },
        LoadShape::RequestRateLimited {
            request_rate: RequestRate::Finite(request_rate),
            burstiness,
        } => BenchLoadInput::RequestRateLimited {
            request_rate: *request_rate,
            burstiness: *burstiness,
        },
        LoadShape::RequestRateLimited {
            request_rate: RequestRate::Unbounded,
            ..
        } => BenchLoadInput::UnboundedRequestRate,
    }
}
