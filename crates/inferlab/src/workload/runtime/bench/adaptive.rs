//! Adaptive request-rate search policy and SLO probe classification.

use super::BenchRunOutcome;
use super::case::run_bench_case;
use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::workload::adaptive::{AdaptiveRatePlanner, Observation, ProbeClassification};
use crate::workload::record::{
    AdaptiveBenchSummary, CaseSloEvaluation, SloBoundDirection, SloEvaluationOutcome,
    WorkloadRecordSession, WorkloadStatus,
};
use crate::workload::{BenchCasePlan, BenchPlan, LoadShape, resolved_request_count};
use crate::workspace::{BenchDefinition, RequestRate};
use inferlab_runtime::interrupt;

#[allow(clippy::too_many_arguments)]
pub fn run_adaptive(
    plan: &BenchPlan,
    policy: &str,
    initial_rates: &[f64],
    max_search_steps: u32,
    min_rate_resolution: Option<f64>,
    request_count: Option<u32>,
    duration_seconds: Option<u64>,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<BenchRunOutcome, InferlabError> {
    let planner = AdaptiveRatePlanner::new(
        initial_rates.to_vec(),
        max_search_steps,
        min_rate_resolution,
    );
    let mut distinct_initial_rates = initial_rates.to_vec();
    distinct_initial_rates.sort_by(f64::total_cmp);
    distinct_initial_rates.dedup();
    let maximum_probe_count = distinct_initial_rates
        .len()
        .saturating_add(max_search_steps as usize);
    let mut observations = Vec::new();
    let mut measurement_failed = false;
    while let Some(rate) = planner.next_rate(&observations) {
        if interrupt::received() {
            session.record_mut().skip_reason =
                Some("remaining Bench probes skipped because recipe was interrupted".to_owned());
            break;
        }
        let case = BenchCasePlan {
            id: format!("probe-{:03}", observations.len()),
            load_shape: LoadShape::RequestRateLimited {
                request_rate: RequestRate::Finite(rate),
                burstiness: adaptive_burstiness(plan),
            },
            request_count: resolved_request_count(
                &plan.id,
                &RequestRate::Finite(rate),
                request_count,
                duration_seconds,
            )?,
            warmup_request_count: 0,
            session_count: None,
            warmup_session_count: None,
        };
        let paths = session.case_paths(&case.id)?;
        progress.phase(
            Phase::named("adaptive probe")
                .item(&case.id, observations.len() + 1, maximum_probe_count)
                .log(session.absolute(&paths.stderr)),
        )?;
        let mut record = run_bench_case(plan, &case, session, None)?;
        let case_succeeded = record.status == WorkloadStatus::Succeeded;
        let classification = record.evidence.slo.as_ref().map(classify_slo_evaluation);
        if case_succeeded && classification.is_none() {
            record.status = WorkloadStatus::Failed;
            record.error = Some("adaptive Bench probe has no case-level SLO evaluation".to_owned());
        }
        session.push_bench_case(record)?;
        session.rewrite()?;
        if !case_succeeded {
            measurement_failed = true;
            break;
        }
        let Some(classification) = classification else {
            measurement_failed = true;
            break;
        };
        observations.push(Observation {
            rate,
            classification,
        });
    }
    let normally_completed = !measurement_failed && !interrupt::received();
    let selected_rate = normally_completed
        .then(|| planner.selected_rate(&observations))
        .flatten();
    session.set_adaptive_bench_summary(AdaptiveBenchSummary {
        policy: policy.to_owned(),
        selected_rate,
        boundary_bracketed: selected_rate.is_some() && planner.boundary_bracketed(&observations),
        normal_termination_reason: normally_completed
            .then(|| planner.termination_reason(&observations)),
        case_ids: session
            .bench_cases()?
            .iter()
            .map(|case| case.id.clone())
            .collect(),
    })?;
    Ok(BenchRunOutcome {
        measurement_succeeded: normally_completed,
        passed: normally_completed && selected_rate.is_some(),
    })
}

pub fn classify_slo_evaluation(evaluation: &CaseSloEvaluation) -> ProbeClassification {
    if evaluation.passed {
        return ProbeClassification::Feasible;
    }
    if evaluation.aggregate_slos.iter().any(|constraint| {
        constraint.direction == SloBoundDirection::AtMost
            && constraint.outcome == SloEvaluationOutcome::Failed
    }) || evaluation
        .request_slo
        .as_ref()
        .is_some_and(|request| request.ratio_outcome == SloEvaluationOutcome::Failed)
    {
        return ProbeClassification::Above;
    }
    if evaluation.aggregate_slos.iter().any(|constraint| {
        constraint.direction == SloBoundDirection::AtLeast
            && constraint.outcome == SloEvaluationOutcome::Failed
    }) {
        return ProbeClassification::Below;
    }
    if evaluation
        .aggregate_slos
        .iter()
        .any(|constraint| constraint.outcome == SloEvaluationOutcome::Unavailable)
    {
        return ProbeClassification::Indeterminate;
    }
    ProbeClassification::Indeterminate
}

fn adaptive_burstiness(plan: &BenchPlan) -> Option<f64> {
    match &plan.definition {
        BenchDefinition::AdaptiveServing { burstiness, .. } => *burstiness,
        BenchDefinition::Serving { .. } => None,
    }
}
