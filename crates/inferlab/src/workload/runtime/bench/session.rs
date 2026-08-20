//! Linear-session reconciliation. AIPerf owns turn execution; Rust validates
//! its native session and pacing evidence before accepting a case record.

#[cfg(test)]
mod tests;

use inferlab_protocol::{BenchClientResult, BenchSessionResultEvidence, ClientStatus};
use std::collections::{BTreeMap, BTreeSet};

/// Raw-artifact-derived session evidence dimensions recorded as unavailable at
/// the `performance` artifact level
/// ([[RFC-0005:C-BENCH-LINEAR-SESSION-EVIDENCE]]).
pub(crate) const PERFORMANCE_SESSION_UNAVAILABLE_DIMENSIONS: [&str; 4] = [
    "pre_template_content_tokens",
    "max_input_tokens_bound_check",
    "preceding_live_response_pairwise_history",
    "raw_native_request_reconciliation",
];

pub(crate) fn duplicate_runtime_session_identity<'a>(
    existing: impl Iterator<Item = &'a BenchSessionResultEvidence>,
    candidate: &BenchSessionResultEvidence,
) -> Option<String> {
    let mut identities = existing
        .flat_map(|evidence| evidence.sessions.iter())
        .map(|session| session.runtime_session_id.as_str())
        .collect::<BTreeSet<_>>();
    candidate.sessions.iter().find_map(|session| {
        (!identities.insert(session.runtime_session_id.as_str()))
            .then(|| session.runtime_session_id.clone())
    })
}

pub(crate) fn linear_session_result_error(
    result: &BenchClientResult,
    evidence: &BenchSessionResultEvidence,
    expected_warmup_sessions: u32,
    expected_profiling_sessions: u32,
    expected_profiling_requests: u32,
    artifact_level: crate::workspace::BenchArtifactLevel,
) -> Option<String> {
    let raw_expected = artifact_level == crate::workspace::BenchArtifactLevel::Diagnostic;
    let expected_unavailable: Vec<String> = if raw_expected {
        Vec::new()
    } else {
        PERFORMANCE_SESSION_UNAVAILABLE_DIMENSIONS
            .iter()
            .map(|dimension| (*dimension).to_owned())
            .collect()
    };
    if evidence.unavailable_dimensions != expected_unavailable {
        return Some("unavailable dimensions do not match the effective artifact level".to_owned());
    }
    // Diagnostic evidence MUST reconcile raw requests to normalized metric
    // records; performance evidence MUST leave that raw-derived reconciliation
    // absent.
    if evidence.native_requests_reconciled != raw_expected.then_some(true) {
        return Some(
            "native request reconciliation disagrees with the effective artifact level".to_owned(),
        );
    }
    if !evidence.population_slice_reconciled
        || !evidence.sessions_reconciled
        || !evidence.turn_order_reconciled
        || !evidence.inter_turn_delays_reconciled
        || !evidence.counts_reconciled
    {
        return Some("one or more reconciliation conclusions are false".to_owned());
    }
    if evidence.warmup.planned_sessions != expected_warmup_sessions
        || evidence.profiling.planned_sessions != expected_profiling_sessions
        || evidence.profiling.planned_requests != expected_profiling_requests
    {
        return Some("phase plan counts disagree with the resolved case".to_owned());
    }
    for (name, phase) in [
        ("warmup", &evidence.warmup),
        ("profiling", &evidence.profiling),
    ] {
        if !phase.reconciled
            || phase.started_sessions != phase.planned_sessions
            || phase.succeeded_sessions != phase.planned_sessions
            || phase.failed_sessions != 0
            || phase.attempted_requests != phase.planned_requests
            || phase.completed_requests != phase.planned_requests
            || phase.failed_requests != 0
        {
            return Some(format!("{name} phase is not complete and successful"));
        }
    }
    if result.completed_requests != u64::from(evidence.profiling.completed_requests)
        || result.failed_requests != u64::from(evidence.profiling.failed_requests)
    {
        return Some("result counts disagree with profiling session counts".to_owned());
    }

    let expected_session_count = expected_warmup_sessions
        .checked_add(expected_profiling_sessions)
        .map(|count| count as usize);
    if expected_session_count != Some(evidence.sessions.len()) {
        return Some("session outcomes do not cover the admitted sessions".to_owned());
    }
    let expected_turn_count = evidence
        .warmup
        .planned_requests
        .checked_add(evidence.profiling.planned_requests)
        .map(|count| count as usize);
    if expected_turn_count != Some(evidence.turns.len()) {
        return Some("turn evidence does not cover the planned requests".to_owned());
    }

    let mut sessions = BTreeMap::new();
    let mut warmup_outcomes = 0_u32;
    let mut profiling_outcomes = 0_u32;
    for session in &evidence.sessions {
        if session.runtime_session_id.is_empty()
            || session.template_identity.is_empty()
            || session.planned_turns < 2
            || session.attempted_turns != session.planned_turns
            || session.status != ClientStatus::Succeeded
            || session.failure_classification.is_some()
            || session.diagnostic.is_some()
            || session.failing_turn.is_some()
            || session.suppressed_later_turns != 0
        {
            return Some("a session outcome is not complete and successful".to_owned());
        }
        match session.phase.as_str() {
            "warmup" => warmup_outcomes = warmup_outcomes.saturating_add(1),
            "profiling" => profiling_outcomes = profiling_outcomes.saturating_add(1),
            _ => return Some("a session outcome has an unknown phase".to_owned()),
        }
        if sessions
            .insert(session.runtime_session_id.as_str(), session)
            .is_some()
        {
            return Some("runtime session identity is not unique within the case".to_owned());
        }
    }
    if warmup_outcomes != expected_warmup_sessions
        || profiling_outcomes != expected_profiling_sessions
    {
        return Some("session outcomes disagree with phase counts".to_owned());
    }

    let mut native_ids = BTreeSet::new();
    let mut turns_by_session =
        BTreeMap::<&str, Vec<&inferlab_protocol::BenchSessionTurnResult>>::new();
    for turn in &evidence.turns {
        let Some(session) = sessions.get(turn.runtime_session_id.as_str()) else {
            return Some("a turn references an unknown runtime session".to_owned());
        };
        if turn.phase != session.phase
            || turn.native_artifact_name.is_empty()
            || turn.observed_prompt_tokens.is_none()
            || turn.pre_template_content_tokens.is_some() != raw_expected
            || turn.inter_turn_delay_reconciled != Some(true)
            || turn.post_failure_continuation
            || !native_ids.insert((turn.phase.as_str(), turn.native_session_num))
        {
            return Some("a turn does not reconcile to unique native evidence".to_owned());
        }
        turns_by_session
            .entry(turn.runtime_session_id.as_str())
            .or_default()
            .push(turn);
    }
    for (runtime_id, session) in sessions {
        let Some(turns) = turns_by_session.get_mut(runtime_id) else {
            return Some("a session has no turn evidence".to_owned());
        };
        turns.sort_by_key(|turn| turn.turn_index);
        if turns.len() != session.planned_turns as usize {
            return Some("a session turn count disagrees with its outcome".to_owned());
        }
        let mut previous_native_id = None;
        for (expected_index, turn) in turns.iter().enumerate() {
            if turn.turn_index as usize != expected_index {
                return Some("session turn indexes are not contiguous".to_owned());
            }
            if expected_index == 0 {
                if turn.preceding_native_session_num.is_some()
                    || turn.preceding_terminal_response_receipt_ns.is_some()
                    || turn.effective_inter_turn_delay_seconds.is_some()
                {
                    return Some("first turn carries preceding-turn evidence".to_owned());
                }
            } else if raw_expected {
                let (Some(receipt_ns), Some(delay_seconds)) = (
                    turn.preceding_terminal_response_receipt_ns,
                    turn.effective_inter_turn_delay_seconds,
                ) else {
                    return Some("a later turn omits delay evidence".to_owned());
                };
                if turn.preceding_native_session_num != previous_native_id
                    || !delay_seconds.is_finite()
                    || delay_seconds < 0.0
                {
                    return Some("a later turn has invalid predecessor evidence".to_owned());
                }
                let delay_ns = (delay_seconds * 1_000_000_000.0).round();
                if delay_ns > u64::MAX as f64
                    || receipt_ns
                        .checked_add(delay_ns as u64)
                        .is_none_or(|earliest| turn.request_start_ns < earliest)
                {
                    return Some(
                        "a later turn began before its effective delay elapsed".to_owned(),
                    );
                }
            } else {
                // The raw-derived pairwise history is unavailable at the
                // performance artifact level; the client still reconciles the
                // effective delay from normalized record timing.
                if turn.preceding_native_session_num.is_some()
                    || turn.preceding_terminal_response_receipt_ns.is_some()
                {
                    return Some(
                        "a later turn carries raw-derived predecessor evidence without a raw artifact"
                            .to_owned(),
                    );
                }
                match turn.effective_inter_turn_delay_seconds {
                    Some(delay_seconds) if delay_seconds.is_finite() && delay_seconds >= 0.0 => {}
                    _ => return Some("a later turn omits delay evidence".to_owned()),
                }
            }
            previous_native_id = Some(turn.native_session_num);
        }
    }
    None
}
