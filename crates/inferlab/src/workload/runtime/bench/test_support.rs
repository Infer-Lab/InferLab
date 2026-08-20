use inferlab_protocol::{
    BenchClientResult, BenchRuntimeSessionResult, BenchSessionPhaseSummary,
    BenchSessionResultEvidence, BenchSessionTurnResult, ClientStatus,
};
use std::collections::BTreeMap;

pub(super) fn prefill_bench_result() -> BenchClientResult {
    let mut metrics = BTreeMap::from([
        ("request_throughput".to_owned(), 1.0),
        ("output_throughput".to_owned(), 1.0),
        ("total_token_throughput".to_owned(), 1.0),
    ]);
    for family in ["prompt_tokens", "request_latency_ms", "ttft_ms"] {
        for prefix in ["mean", "min", "max", "stddev", "p50", "p90", "p95", "p99"] {
            metrics.insert(format!("{prefix}_{family}"), 1.0);
        }
    }
    BenchClientResult {
        schema_version: 1,
        status: ClientStatus::Succeeded,
        completed_requests: 1,
        failed_requests: 0,
        normalization_schema: "aiperf-summary-v1".to_owned(),
        metrics,
        request_slo: None,
        session_evidence: None,
        agentic_evidence: None,
        prompt_token_reconciliation: Vec::new(),
        prompt_cache_observations: Vec::new(),
        native_command: vec!["fixture-bench".to_owned()],
        native_exit_code: Some(0),
        report_invocations: Vec::new(),
        raw_artifacts: Vec::new(),
        error: None,
    }
}

pub(super) fn complete_session_evidence() -> BenchSessionResultEvidence {
    BenchSessionResultEvidence {
        warmup: BenchSessionPhaseSummary {
            planned_sessions: 0,
            started_sessions: 0,
            succeeded_sessions: 0,
            failed_sessions: 0,
            planned_requests: 0,
            attempted_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            reconciled: true,
        },
        profiling: BenchSessionPhaseSummary {
            planned_sessions: 1,
            started_sessions: 1,
            succeeded_sessions: 1,
            failed_sessions: 0,
            planned_requests: 2,
            attempted_requests: 2,
            completed_requests: 2,
            failed_requests: 0,
            reconciled: true,
        },
        sessions: vec![BenchRuntimeSessionResult {
            phase: "profiling".to_owned(),
            runtime_session_id: "runtime-0".to_owned(),
            template_identity: "template-0".to_owned(),
            planned_turns: 2,
            attempted_turns: 2,
            status: ClientStatus::Succeeded,
            failure_classification: None,
            diagnostic: None,
            failing_turn: None,
            suppressed_later_turns: 0,
        }],
        turns: vec![
            BenchSessionTurnResult {
                phase: "profiling".to_owned(),
                runtime_session_id: "runtime-0".to_owned(),
                turn_index: 0,
                pre_template_content_tokens: Some(1),
                observed_prompt_tokens: Some(4),
                native_session_num: 0,
                preceding_native_session_num: None,
                preceding_terminal_response_receipt_ns: None,
                effective_inter_turn_delay_seconds: None,
                request_start_ns: 100,
                inter_turn_delay_reconciled: Some(true),
                post_failure_continuation: false,
                native_artifact_name: "raw".to_owned(),
            },
            BenchSessionTurnResult {
                phase: "profiling".to_owned(),
                runtime_session_id: "runtime-0".to_owned(),
                turn_index: 1,
                pre_template_content_tokens: Some(3),
                observed_prompt_tokens: Some(6),
                native_session_num: 1,
                preceding_native_session_num: Some(0),
                preceding_terminal_response_receipt_ns: Some(200),
                effective_inter_turn_delay_seconds: Some(0.0),
                request_start_ns: 200,
                inter_turn_delay_reconciled: Some(true),
                post_failure_continuation: false,
                native_artifact_name: "raw".to_owned(),
            },
        ],
        population_slice_reconciled: true,
        sessions_reconciled: true,
        turn_order_reconciled: true,
        inter_turn_delays_reconciled: true,
        native_requests_reconciled: Some(true),
        counts_reconciled: true,
        unavailable_dimensions: Vec::new(),
    }
}
