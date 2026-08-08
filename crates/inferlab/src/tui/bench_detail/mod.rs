mod definition;
mod record;

pub(super) use definition::definition;
pub(super) use record::{
    AgenticSourceProjection, CachePreparationProjection, CaptureProjection, CaseEvidence,
    CaseSloProjection, PopulationSliceProjection, RequestSourceProjection, SessionSourceProjection,
    capture_summary, case_evidence, record_source,
};

#[cfg(test)]
mod tests {
    use super::{CaseEvidence, case_evidence, definition};
    use crate::workspace::BenchDefinition;
    use inferlab_protocol::{BenchAgenticResultEvidence, BenchSessionResultEvidence};

    fn parse(source: &str) -> Result<BenchDefinition, toml::de::Error> {
        toml::from_str(source)
    }

    #[test]
    fn definitions_project_only_the_selected_source_semantics()
    -> Result<(), Box<dyn std::error::Error>> {
        let request = definition(&parse(
            r#"
kind = "serving"
timeout_seconds = 120
concurrency = [1, 8]
prompts_per_concurrency = 10
[request_source]
kind = "random"
input_tokens = { kind = "inclusive_uniform", min = 512, max = 1024 }
output_tokens = 128
prefix_sharing = { shared_prefix_ratio = 0.5 }
[request_source.prompt]
kind = "flat"
"#,
        )?);
        assert_eq!(request.relationship, "requests · random");
        assert!(request.sections.iter().any(|section| {
            section
                .rows
                .iter()
                .any(|(label, value)| label == "Prefix sharing" && value == "50%")
        }));
        assert!(
            !request
                .sections
                .iter()
                .any(|section| { section.rows.iter().any(|(label, _)| label == "Dataset") })
        );

        let session = definition(&parse(
            r#"
kind = "serving"
timeout_seconds = 120
concurrency = [2]
sessions_per_concurrency = 4
[session_source]
dataset = "turns"
profile = "long"
max_input_tokens = 4096
inter_turn_delay_scale = 0.5
"#,
        )?);
        assert_eq!(session.relationship, "linear session · dataset turns");
        assert!(session.sections.iter().any(|section| {
            section
                .rows
                .iter()
                .any(|(label, value)| label == "Sessions / concurrency" && value == "4")
        }));

        let agentic = definition(&parse(
            r#"
kind = "serving"
timeout_seconds = 120
concurrency = [4]
duration_seconds = 300
[agentic_source]
dataset = "semianalysis-agentx"
profile = "agentx-a1"
"#,
        )?);
        assert_eq!(
            agentic.relationship,
            "agentic replay · semianalysis-agentx/agentx-a1"
        );
        assert!(agentic.sections.iter().any(|section| {
            section
                .rows
                .iter()
                .any(|(label, value)| label == "Profiling duration" && value == "300s")
        }));
        assert!(!agentic.sections.iter().any(|section| {
            section.title == "POPULATION"
                || section.rows.iter().any(|(label, _)| {
                    matches!(
                        label.as_str(),
                        "Request rates" | "Request count" | "Reset prefix cache"
                    )
                })
        }));
        assert!(!session.sections.iter().any(|section| {
            section.rows.iter().any(|(label, _)| {
                matches!(
                    label.as_str(),
                    "Request rates" | "Request count" | "Duration" | "Burstiness"
                )
            })
        }));
        assert!(!session.sections.iter().any(|section| {
            section
                .rows
                .iter()
                .any(|(label, _)| label == "Maximum inter-turn delay")
        }));

        let dataset = definition(&parse(
            r#"
kind = "serving"
timeout_seconds = 120
concurrency = [1]
prompts_per_concurrency = 2
[request_source]
kind = "dataset"
dataset = "sharegpt"
max_input_tokens = 4096
"#,
        )?);
        assert!(
            !dataset
                .sections
                .iter()
                .any(|section| { section.rows.iter().any(|(label, _)| label == "Profile") })
        );
        let unprofiled_session = definition(&parse(
            r#"
kind = "serving"
timeout_seconds = 120
concurrency = [1]
sessions_per_concurrency = 1
[session_source]
dataset = "turns"
max_input_tokens = 4096
"#,
        )?);
        assert!(
            !unprofiled_session
                .sections
                .iter()
                .any(|section| { section.rows.iter().any(|(label, _)| label == "Profile") })
        );
        Ok(())
    }

    #[test]
    fn session_and_agentic_results_project_their_distinct_evidence_categories()
    -> Result<(), Box<dyn std::error::Error>> {
        let session = serde_json::from_value::<BenchSessionResultEvidence>(serde_json::json!({
            "warmup": {
                "planned_sessions": 1, "started_sessions": 1, "succeeded_sessions": 1,
                "failed_sessions": 0, "planned_requests": 2, "attempted_requests": 2,
                "completed_requests": 2, "failed_requests": 0, "reconciled": true
            },
            "profiling": {
                "planned_sessions": 2, "started_sessions": 2, "succeeded_sessions": 1,
                "failed_sessions": 1, "planned_requests": 6, "attempted_requests": 5,
                "completed_requests": 4, "failed_requests": 1, "reconciled": true
            },
            "sessions": [{
                "phase": "profiling", "runtime_session_id": "session-2",
                "template_identity": "template-b", "planned_turns": 3,
                "attempted_turns": 2, "status": "failed",
                "failure_classification": "context_limit", "diagnostic": "too long",
                "failing_turn": 1, "suppressed_later_turns": 1
            }],
            "turns": [],
            "population_slice_reconciled": true,
            "sessions_reconciled": true,
            "turn_order_reconciled": true,
            "inter_turn_delays_reconciled": true,
            "native_requests_reconciled": true,
            "counts_reconciled": true
        }))?;
        let (session_sections, _) = case_evidence(CaseEvidence {
            id: Some("c-session"),
            cache_preparation: None,
            slo: None,
            population_slice: None,
            completed_requests: Some(4),
            failed_requests: Some(1),
            normalization_schema: Some("aiperf-summary-v1"),
            request_unavailable: false,
            session: Some(&session),
            session_unavailable: false,
            agentic: None,
            agentic_unavailable: false,
            prompt_token_reconciliation: &[],
            raw_artifacts: &[],
        });
        let session_result = session_sections
            .iter()
            .find(|section| section.title == "LINEAR SESSION RESULT")
            .ok_or("missing session result")?;
        assert!(session_result.rows.iter().any(|(label, value)| {
            label == "Profiling sessions"
                && value == "2 planned · 2 started · 1 succeeded · 1 failed"
        }));
        assert!(session_result.rows.iter().any(|(label, value)| {
            label == "Failed session" && value.contains("context_limit · too long")
        }));
        assert!(
            session_result
                .rows
                .iter()
                .any(|(label, value)| { label == "Warmup reconciled" && value == "yes" })
        );

        let agentic = serde_json::from_value::<BenchAgenticResultEvidence>(serde_json::json!({
            "source": {
                "repository": "https://example.invalid/data.git",
                "expected_revision": "abc", "observed_revision": "abc",
                "filename": "agentx.parquet", "expected_sha256": "deadbeef",
                "observed_sha256": "deadbeef", "cache_path": "cache/agentx.parquet",
                "cache_state_before": "present", "acquisition_outcome": "reused"
            },
            "run": {
                "native_run_id": "run-7", "scenario": "agentx-a1",
                "submission_valid": false,
                "submission_invalid_reasons": ["context_overflow_rate"],
                "warmup_records": 2, "warmup_error_records": 0,
                "warmup_source_coordinate_records": 2, "warmup_succeeded": true,
                "profiling_began_after_warmup_and_drain": true,
                "profiling_records": 20, "source_coordinate_records": 20,
                "distinct_source_traces": 4, "distinct_runtime_conversations": 6,
                "distinct_transport_requests": 20, "cache_bust_records": 1,
                "context_overflow_count": 2, "ordinary_failure_count": 1,
                "branch_stats": {
                    "children_spawned": 5, "children_completed": 4,
                    "children_errored": 1, "children_truncated": 0,
                    "children_delayed": 2, "parents_suspended": 3,
                    "parents_resumed": 3, "parents_failed_due_to_child_error": 1,
                    "joins_suppressed": 1
                },
                "aggregate_artifact": "aggregate.json",
                "raw_records_artifact": "requests.jsonl",
                "unavailable_dimensions": ["root_tree_completion"]
            }
        }))?;
        let (agentic_sections, artifacts) = case_evidence(CaseEvidence {
            id: Some("c-agentic"),
            cache_preparation: None,
            slo: None,
            population_slice: None,
            completed_requests: Some(17),
            failed_requests: Some(3),
            normalization_schema: Some("aiperf-summary-v1"),
            request_unavailable: false,
            session: None,
            session_unavailable: false,
            agentic: Some(&agentic),
            agentic_unavailable: false,
            prompt_token_reconciliation: &[],
            raw_artifacts: &[],
        });
        let agentic_result = agentic_sections
            .iter()
            .find(|section| section.title == "AGENTIC REPLAY RESULT")
            .ok_or("missing agentic result")?;
        assert!(agentic_result.rows.iter().any(|(label, value)| {
            label == "Submission" && value == "invalid · context_overflow_rate"
        }));
        assert!(agentic_result.rows.iter().any(|(label, value)| {
            label == "Branches" && value.contains("5 spawned · 4 completed · 1 errored")
        }));
        assert!(
            agentic_result
                .rows
                .iter()
                .any(|(label, value)| { label == "Warmup source coordinates" && value == "2" })
        );
        assert!(
            agentic_result
                .rows
                .iter()
                .any(|(label, value)| { label == "Profiling source coordinates" && value == "20" })
        );
        assert!(agentic_result.rows.iter().any(|(label, value)| {
            label == "Unavailable dimensions" && value == "root_tree_completion"
        }));
        assert_eq!(
            artifacts,
            [
                "agentic aggregate · aggregate.json",
                "agentic raw records · requests.jsonl"
            ]
        );
        Ok(())
    }
}
