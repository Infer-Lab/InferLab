use super::super::result::bench_result_error;
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
        bench_result_error(&result, false, false, Some((1, 1)), None, 2, None),
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
