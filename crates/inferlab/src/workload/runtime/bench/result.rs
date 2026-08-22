//! AIPerf result-envelope validation and InferLab SLO adjudication.

#[cfg(test)]
mod tests;

use super::session::linear_session_result_error;
use crate::bench_metric::BenchMetric;
use crate::workload::domain::{
    AggregateSloBound, ResolvedBenchAgenticSource, ResolvedBenchRequestSource,
    ResolvedBenchSloPolicy, ResolvedBenchSource,
};
use crate::workload::record::{
    AggregateSloEvaluation, CaseSloEvaluation, RequestSloEvaluation, SloBoundDirection,
    SloEvaluationOutcome,
};
use crate::workload::{BenchCasePlan, BenchPlan};
use crate::workspace::{BenchArtifactLevel, RequestSlo};
use inferlab_protocol::{BenchClientResult, ClientStatus};

/// Raw-artifact-derived agentic evidence dimensions recorded as unavailable at
/// the `performance` artifact level ([[RFC-0005:C-BENCH-AGENTIC-TRACE-EVIDENCE]]).
pub(super) const PERFORMANCE_AGENTIC_UNAVAILABLE_DIMENSIONS: [&str; 3] = [
    "source_coordinate_mapping",
    "cache_bust_observations",
    "warmup_source_coordinate_records",
];

pub(super) struct BenchResultExpectations<'a> {
    pub(super) tpot_applicable: bool,
    pub(super) speed_bench_server_metrics: bool,
    pub(super) sessions: Option<(u32, u32)>,
    pub(super) agentic_source: Option<&'a ResolvedBenchAgenticSource>,
    pub(super) artifact_level: BenchArtifactLevel,
    pub(super) request_count: u32,
    pub(super) request_slo: Option<&'a RequestSlo>,
    pub(super) prompt_cache_evidence: bool,
}

impl<'a> BenchResultExpectations<'a> {
    pub(super) fn for_case(plan: &'a BenchPlan, case: &BenchCasePlan) -> Self {
        let source = &plan.client.effective_definition.source;
        Self {
            tpot_applicable: plan.client.tpot_applicability.is_applicable(),
            speed_bench_server_metrics: plan.client.effective_definition.server_metrics
                && matches!(
                    source.request_source(),
                    Some(ResolvedBenchRequestSource::Dataset { dataset, .. })
                        if dataset == "speed_bench"
                ),
            sessions: case
                .session_count
                .map(|profiling| (case.warmup_session_count.unwrap_or_default(), profiling)),
            agentic_source: match source {
                ResolvedBenchSource::Agentic { agentic_source } => Some(agentic_source),
                ResolvedBenchSource::Requests { .. } | ResolvedBenchSource::Sessions { .. } => None,
            },
            artifact_level: plan.client.effective_definition.artifact_level,
            request_count: case.request_count,
            request_slo: plan.client.slo.request.as_ref(),
            prompt_cache_evidence: plan
                .client
                .effective_definition
                .requires_prompt_cache_evidence(),
        }
    }
}

pub(super) fn bench_result_error(
    result: &BenchClientResult,
    expectations: BenchResultExpectations<'_>,
) -> Option<String> {
    let BenchResultExpectations {
        tpot_applicable,
        speed_bench_server_metrics,
        sessions: expected_sessions,
        agentic_source: expected_agentic_source,
        artifact_level,
        request_count,
        request_slo,
        prompt_cache_evidence: requires_prompt_cache_evidence,
    } = expectations;
    if result.status == ClientStatus::Failed {
        return Some(
            result
                .error
                .clone()
                .unwrap_or_else(|| "Bench client reported failure".to_owned()),
        );
    }
    if result.normalization_schema != "aiperf-summary-v1" {
        return Some(format!(
            "Bench client returned unsupported normalization schema {:?}",
            result.normalization_schema
        ));
    }
    if expected_sessions.is_some() && expected_agentic_source.is_some() {
        return Some(
            "Bench case cannot expect both linear-session and agentic evidence".to_owned(),
        );
    }
    if let Some((warmup_sessions, profiling_sessions)) = expected_sessions {
        let Some(session) = result.session_evidence.as_ref() else {
            return Some("Bench client omitted linear-session evidence".to_owned());
        };
        if let Some(error) = linear_session_result_error(
            result,
            session,
            warmup_sessions,
            profiling_sessions,
            request_count,
            artifact_level,
        ) {
            return Some(format!(
                "Bench client returned invalid linear-session evidence: {error}"
            ));
        }
    } else if result.session_evidence.is_some() {
        return Some(
            "Bench client returned linear-session evidence for a non-session case".to_owned(),
        );
    }
    if let Some(source) = expected_agentic_source {
        let Some(evidence) = result.agentic_evidence.as_deref() else {
            return Some("Bench client omitted agentic evidence".to_owned());
        };
        if let Some(error) = agentic_result_error(result, evidence, source, artifact_level) {
            return Some(format!(
                "Bench client returned invalid agentic evidence: {error}"
            ));
        }
    } else if result.agentic_evidence.is_some() {
        return Some("Bench client returned agentic evidence for a non-agentic case".to_owned());
    }
    if speed_bench_server_metrics {
        let expected = [
            ("acceptance_length", "accept_length"),
            ("acceptance_rate", "accept_rate"),
        ];
        if result.report_invocations.len() != expected.len() {
            return Some(format!(
                "Bench client returned {} SPEED-Bench report invocations, expected {}",
                result.report_invocations.len(),
                expected.len()
            ));
        }
        for (invocation, (purpose, native_metric)) in result.report_invocations.iter().zip(expected)
        {
            let has_metric = invocation
                .command
                .windows(2)
                .any(|pair| pair == ["--metric", native_metric]);
            if invocation.purpose != purpose
                || invocation.exit_code != Some(0)
                || invocation.interrupted
                || invocation.timed_out
                || !invocation
                    .command
                    .iter()
                    .any(|arg| arg == "speed-bench-report")
                || !has_metric
            {
                return Some(format!(
                    "Bench client returned invalid {purpose} report process evidence"
                ));
            }
        }
    } else if !result.report_invocations.is_empty()
        || result.metrics.contains_key("acceptance_length")
        || result.metrics.contains_key("acceptance_rate")
    {
        return Some(
            "Bench client returned SPEED-Bench acceptance evidence for an inapplicable case"
                .to_owned(),
        );
    }
    if let Some(request_slo) = request_slo {
        if result
            .completed_requests
            .checked_add(result.failed_requests)
            != Some(u64::from(request_count))
        {
            return Some(format!(
                "Bench client request counts do not match resolved request_count {request_count}"
            ));
        }
        let Some(evidence) = result.request_slo.as_ref() else {
            return Some("Bench client omitted request-SLO evidence".to_owned());
        };
        if !evidence.request_count_reconciled {
            return Some("Bench client did not reconcile native request identities".to_owned());
        }
        if evidence.profiling_duration_source != "native-profiling-request-window" {
            return Some(format!(
                "Bench client returned unsupported profiling duration source {:?}",
                evidence.profiling_duration_source
            ));
        }
        if !evidence.profiling_duration_seconds.is_finite()
            || evidence.profiling_duration_seconds <= 0.0
            || !evidence.good_request_ratio.is_finite()
            || !(0.0..=1.0).contains(&evidence.good_request_ratio)
            || !evidence.goodput.is_finite()
            || evidence.goodput < 0.0
            || evidence.good_requests > result.completed_requests
        {
            return Some("Bench client returned invalid request-SLO evidence".to_owned());
        }
        let attempted = result.completed_requests + result.failed_requests;
        let expected_ratio = evidence.good_requests as f64 / attempted as f64;
        let expected_goodput = evidence.good_requests as f64 / evidence.profiling_duration_seconds;
        if !same_finite_value(evidence.good_request_ratio, expected_ratio)
            || !same_finite_value(evidence.goodput, expected_goodput)
            || !result
                .metrics
                .get("good_request_ratio")
                .is_some_and(|value| same_finite_value(*value, evidence.good_request_ratio))
            || !result
                .metrics
                .get("goodput")
                .is_some_and(|value| same_finite_value(*value, evidence.goodput))
        {
            return Some(
                "Bench client request-SLO metrics disagree with file-bound evidence".to_owned(),
            );
        }
        if evidence.native_aggregate_good_request_count_consistent == Some(false)
            || evidence
                .native_aggregate_good_request_count
                .is_some_and(|value| value != evidence.good_requests)
        {
            return Some(
                "Bench client native aggregate good-request count is inconsistent".to_owned(),
            );
        }
        if !request_slo.minimum_good_request_ratio.is_finite() {
            return Some("resolved request-SLO ratio is not finite".to_owned());
        }
    } else if expected_agentic_source.is_some() {
        if result.completed_requests == 0 {
            return Some("Bench client reported no completed agentic requests".to_owned());
        }
    } else {
        if result.completed_requests == 0 {
            return Some("Bench client reported no completed requests".to_owned());
        }
        if result.failed_requests != 0 {
            return Some(format!(
                "Bench client reported {} failed requests",
                result.failed_requests
            ));
        }
    }
    let metric_error = |metric: &str| {
        result.metrics.get(metric).map_or_else(
            || {
                Some(format!(
                    "Bench client result is missing required metric {metric:?}"
                ))
            },
            |value| {
                (!value.is_finite())
                    .then(|| format!("Bench client result metric {metric:?} is not finite"))
            },
        )
    };
    if result.completed_requests > 0 {
        for metric in BenchMetric::required_result_metrics(tpot_applicable) {
            if let Some(error) = metric_error(&metric.name()) {
                return Some(error);
            }
        }
        if speed_bench_server_metrics {
            for metric in BenchMetric::required_speed_bench_metrics() {
                if let Some(error) = metric_error(&metric.name()) {
                    return Some(error);
                }
            }
        }
    }
    if let Some(value) = result.metrics.get("prompt_cache_read_ratio")
        && (!value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Some(format!(
            "Bench client result metric \"prompt_cache_read_ratio\" is outside [0, 1]: {value}"
        ));
    }
    if let Some(error) = prompt_cache_result_error(result, requires_prompt_cache_evidence) {
        return Some(error);
    }
    if let Some(value) = result.metrics.get("acceptance_length")
        && (!value.is_finite() || *value < 1.0)
    {
        return Some(format!(
            "Bench client result metric \"acceptance_length\" is below one: {value}"
        ));
    }
    if let Some(value) = result.metrics.get("acceptance_rate")
        && (!value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Some(format!(
            "Bench client result metric \"acceptance_rate\" is outside [0, 1]: {value}"
        ));
    }
    None
}

fn prompt_cache_result_error(result: &BenchClientResult, required: bool) -> Option<String> {
    if !required {
        return (!result.prompt_cache_observations.is_empty()).then(|| {
            "Bench client returned per-request prompt-cache evidence for an inapplicable case"
                .to_owned()
        });
    }
    if result.prompt_cache_observations.len() != result.completed_requests as usize {
        return Some(format!(
            "Bench client returned {} prompt-cache observations for {} completed requests; the server may not expose backend prompt cache-read usage — enable the serving integration's cache-read reporting setting and rebuild the server",
            result.prompt_cache_observations.len(),
            result.completed_requests
        ));
    }
    let mut request_ids = std::collections::BTreeSet::new();
    let mut total_prompt = 0_u64;
    let mut total_cache = 0_u64;
    for observation in &result.prompt_cache_observations {
        if !request_ids.insert(observation.request_id)
            || observation.prompt_tokens == 0
            || observation.cache_read_tokens > observation.prompt_tokens
            || observation.uncached_prompt_tokens
                != observation.prompt_tokens - observation.cache_read_tokens
            || !same_finite_value(
                observation.cache_read_ratio,
                observation.cache_read_tokens as f64 / observation.prompt_tokens as f64,
            )
        {
            return Some(
                "Bench client returned invalid per-request prompt-cache evidence".to_owned(),
            );
        }
        total_prompt = match total_prompt.checked_add(observation.prompt_tokens) {
            Some(value) => value,
            None => return Some("Bench client prompt-token observations overflow".to_owned()),
        };
        total_cache = match total_cache.checked_add(observation.cache_read_tokens) {
            Some(value) => value,
            None => return Some("Bench client cache-read observations overflow".to_owned()),
        };
    }
    for family in ["prompt_cache_read_tokens", "uncached_prompt_tokens"] {
        for statistic in ["mean", "min", "max", "stddev", "p50", "p90", "p95", "p99"] {
            let name = format!("{statistic}_{family}");
            if !result
                .metrics
                .get(&name)
                .is_some_and(|value| value.is_finite())
            {
                return Some(format!(
                    "Bench client result is missing finite required metric {name:?}"
                ));
            }
        }
    }
    let expected_ratio = total_cache as f64 / total_prompt as f64;
    if !result
        .metrics
        .get("prompt_cache_read_ratio")
        .is_some_and(|value| same_finite_value(*value, expected_ratio))
    {
        return Some(
            "Bench client prompt_cache_read_ratio disagrees with per-request usage".to_owned(),
        );
    }
    None
}

fn agentic_result_error(
    result: &BenchClientResult,
    evidence: &inferlab_protocol::BenchAgenticResultEvidence,
    source: &ResolvedBenchAgenticSource,
    artifact_level: BenchArtifactLevel,
) -> Option<String> {
    let catalog = &source.catalog;
    if evidence.source.repository != catalog.repository
        || evidence.source.expected_revision != catalog.revision
        || evidence.source.observed_revision.as_deref() != Some(catalog.revision.as_str())
        || evidence.source.filename != catalog.filename
        || evidence.source.expected_sha256 != catalog.sha256
        || evidence.source.observed_sha256.as_deref() != Some(catalog.sha256.as_str())
    {
        return Some("source verification does not match the resolved release catalog".to_owned());
    }
    let Some(run) = evidence.run.as_deref() else {
        return Some("successful agentic result omitted native run evidence".to_owned());
    };
    if run.scenario != catalog.scenario {
        return Some("native scenario does not match the resolved release profile".to_owned());
    }
    if run.native_run_id.is_empty() {
        return Some("native run identity is empty".to_owned());
    }
    let raw_expected = artifact_level == BenchArtifactLevel::Diagnostic;
    let mut expected_unavailable = catalog.unavailable_dimensions.clone();
    if !raw_expected {
        expected_unavailable.extend(
            PERFORMANCE_AGENTIC_UNAVAILABLE_DIMENSIONS
                .iter()
                .map(|dimension| (*dimension).to_owned()),
        );
    }
    if raw_expected
        != (run.warmup_source_coordinate_records.is_some()
            && run.source_coordinate_records.is_some()
            && run.distinct_source_traces.is_some()
            && run.cache_bust_records.is_some()
            && run.raw_records_artifact.is_some())
    {
        return Some(
            "raw-derived agentic evidence dimensions disagree with the artifact level".to_owned(),
        );
    }
    if !run.warmup_succeeded
        || run.warmup_error_records != 0
        || !run.profiling_began_after_warmup_and_drain
        || run
            .warmup_source_coordinate_records
            .is_some_and(|coordinates| coordinates != run.warmup_records)
    {
        return Some(
            "native warmup outcome does not establish a clean profiling handoff".to_owned(),
        );
    }
    if !run.submission_valid {
        let reasons = if run.submission_invalid_reasons.is_empty() {
            "unspecified".to_owned()
        } else {
            run.submission_invalid_reasons.join(", ")
        };
        return Some(format!("native scenario submission is invalid: {reasons}"));
    }
    if !run.submission_invalid_reasons.is_empty() {
        return Some("valid native scenario includes invalidity reasons".to_owned());
    }
    if run.profiling_records == 0
        || run
            .source_coordinate_records
            .is_some_and(|coordinates| coordinates != run.profiling_records)
        || run.distinct_source_traces.is_some_and(|traces| traces == 0)
        || run.distinct_runtime_conversations == 0
        || run.distinct_transport_requests != run.profiling_records
    {
        return Some(
            "profiling records and source/runtime coordinates do not reconcile".to_owned(),
        );
    }
    if run.ordinary_failure_count > result.failed_requests
        || run.context_overflow_count > run.profiling_records
    {
        return Some("failure classifications exceed native profiling counts".to_owned());
    }
    if run.unavailable_dimensions != expected_unavailable {
        return Some(
            "unavailable dimensions do not match the resolved release profile and artifact level"
                .to_owned(),
        );
    }
    let has_path = |path: &std::path::Path| {
        result
            .raw_artifacts
            .iter()
            .any(|artifact| artifact.path == path)
    };
    for required in &catalog.required_artifacts {
        let present = match required.as_str() {
            "aggregate" => has_path(&run.aggregate_artifact),
            "records" => result
                .raw_artifacts
                .iter()
                .any(|artifact| artifact.name == "aiperf_records"),
            "raw_records" if raw_expected => run
                .raw_records_artifact
                .as_ref()
                .is_some_and(|path| has_path(path)),
            "raw_records" => true,
            _ => false,
        };
        if !present {
            return Some(format!("required native artifact {required:?} is absent"));
        }
    }
    None
}

fn same_finite_value(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= f64::EPSILON * left.abs().max(right.abs()).max(1.0) * 8.0
}

pub(crate) fn evaluate_case_slos(
    policy: &ResolvedBenchSloPolicy,
    result: &BenchClientResult,
) -> Result<Option<CaseSloEvaluation>, String> {
    if policy.aggregate.is_empty() && policy.request.is_none() {
        return Ok(None);
    }
    let mut aggregate_evaluations = Vec::with_capacity(policy.aggregate.len());
    for constraint in &policy.aggregate {
        let (direction, bound) = match constraint.bound {
            AggregateSloBound::AtMost(bound) => (SloBoundDirection::AtMost, bound),
            AggregateSloBound::AtLeast(bound) => (SloBoundDirection::AtLeast, bound),
        };
        let metric_name = constraint.metric.name();
        let observed = result.metrics.get(&metric_name).copied();
        let outcome = match observed {
            Some(value) if !value.is_finite() => {
                return Err(format!(
                    "Bench aggregate SLO metric {:?} is not finite",
                    metric_name
                ));
            }
            Some(value) => match direction {
                SloBoundDirection::AtMost if value <= bound => SloEvaluationOutcome::Passed,
                SloBoundDirection::AtLeast if value >= bound => SloEvaluationOutcome::Passed,
                _ => SloEvaluationOutcome::Failed,
            },
            None if constraint
                .metric
                .missing_is_unavailable(result.completed_requests) =>
            {
                SloEvaluationOutcome::Unavailable
            }
            None => {
                return Err(format!(
                    "Bench result is missing configured aggregate SLO metric {:?}",
                    metric_name
                ));
            }
        };
        aggregate_evaluations.push(AggregateSloEvaluation {
            metric: constraint.metric,
            direction,
            bound,
            observed,
            outcome,
        });
    }
    let request_evaluation = match policy.request.as_ref() {
        Some(slo) => {
            let evidence = result.request_slo.as_ref().ok_or_else(|| {
                "Bench result is missing configured request-SLO evidence".to_owned()
            })?;
            Some(RequestSloEvaluation {
                good_requests: evidence.good_requests,
                good_request_ratio: evidence.good_request_ratio,
                goodput: evidence.goodput,
                profiling_duration_seconds: evidence.profiling_duration_seconds,
                profiling_duration_source: evidence.profiling_duration_source.clone(),
                request_count_reconciled: evidence.request_count_reconciled,
                native_aggregate_good_request_count: evidence.native_aggregate_good_request_count,
                native_aggregate_good_request_count_consistent: evidence
                    .native_aggregate_good_request_count_consistent,
                ratio_outcome: if evidence.good_request_ratio >= slo.minimum_good_request_ratio {
                    SloEvaluationOutcome::Passed
                } else {
                    SloEvaluationOutcome::Failed
                },
            })
        }
        None => None,
    };
    let passed = aggregate_evaluations
        .iter()
        .all(|evaluation| evaluation.outcome == SloEvaluationOutcome::Passed)
        && request_evaluation
            .as_ref()
            .is_none_or(|evaluation| evaluation.ratio_outcome == SloEvaluationOutcome::Passed);
    Ok(Some(CaseSloEvaluation {
        aggregate_slos: aggregate_evaluations,
        request_slo: request_evaluation,
        passed,
    }))
}
