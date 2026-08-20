use super::super::result::{BenchResultExpectations, bench_result_error};
use super::super::test_support::{complete_session_evidence, prefill_bench_result};
use super::duplicate_runtime_session_identity;

#[test]
fn linear_session_native_request_identity_is_phase_qualified() {
    let mut evidence = complete_session_evidence();
    evidence.warmup = evidence.profiling.clone();
    let mut warmup_session = evidence.sessions[0].clone();
    warmup_session.phase = "warmup".to_owned();
    warmup_session.runtime_session_id = "warmup-runtime-0".to_owned();
    warmup_session.template_identity = "warmup-template-0".to_owned();
    evidence.sessions.insert(0, warmup_session);
    let mut warmup_turns = evidence.turns.clone();
    for turn in &mut warmup_turns {
        turn.phase = "warmup".to_owned();
        turn.runtime_session_id = "warmup-runtime-0".to_owned();
    }
    evidence.turns.splice(0..0, warmup_turns);

    let mut result = prefill_bench_result();
    result.completed_requests = 2;
    result.session_evidence = Some(evidence);

    assert_eq!(
        bench_result_error(
            &result,
            BenchResultExpectations {
                tpot_applicable: false,
                speed_bench_server_metrics: false,
                sessions: Some((1, 1)),
                agentic_source: None,
                artifact_level: crate::workspace::BenchArtifactLevel::Diagnostic,
                request_count: 2,
                request_slo: None,
                prompt_cache_evidence: false,
            },
        ),
        None
    );
}

#[test]
fn linear_session_runtime_identity_is_unique_across_cases() {
    let first = complete_session_evidence();
    let second = complete_session_evidence();

    assert_eq!(
        duplicate_runtime_session_identity(std::iter::once(&first), &second),
        Some("runtime-0".to_owned())
    );
}

fn performance_expectations(
    request_count: u32,
    sessions: Option<(u32, u32)>,
) -> BenchResultExpectations<'static> {
    BenchResultExpectations {
        tpot_applicable: false,
        speed_bench_server_metrics: false,
        sessions,
        agentic_source: None,
        artifact_level: crate::workspace::BenchArtifactLevel::Performance,
        request_count,
        request_slo: None,
        prompt_cache_evidence: false,
    }
}

#[test]
fn performance_session_evidence_records_raw_derived_dimensions_as_unavailable() {
    let mut evidence = complete_session_evidence();
    evidence.unavailable_dimensions = super::PERFORMANCE_SESSION_UNAVAILABLE_DIMENSIONS
        .iter()
        .map(|dimension| (*dimension).to_owned())
        .collect();
    evidence.native_requests_reconciled = None;
    for turn in &mut evidence.turns {
        turn.pre_template_content_tokens = None;
        turn.preceding_native_session_num = None;
        turn.preceding_terminal_response_receipt_ns = None;
    }
    let mut result = prefill_bench_result();
    result.completed_requests = 2;
    result.session_evidence = Some(evidence);

    assert_eq!(
        bench_result_error(&result, performance_expectations(2, Some((0, 1)))),
        None
    );

    assert!(result.session_evidence.is_some());
    if let Some(evidence) = result.session_evidence.as_mut() {
        evidence.turns[0].observed_prompt_tokens = None;
    }
    let error = bench_result_error(&result, performance_expectations(2, Some((0, 1))));
    assert!(error.is_some_and(|error| error.contains("linear-session")));
}

#[test]
fn diagnostic_session_evidence_rejects_performance_degradation() {
    let mut evidence = complete_session_evidence();
    evidence.turns[1].preceding_terminal_response_receipt_ns = None;
    let mut result = prefill_bench_result();
    result.completed_requests = 2;
    result.session_evidence = Some(evidence);

    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            sessions: Some((0, 1)),
            artifact_level: crate::workspace::BenchArtifactLevel::Diagnostic,
            ..performance_expectations(2, None)
        },
    );

    assert!(error.is_some_and(|error| error.contains("delay evidence")));
}
