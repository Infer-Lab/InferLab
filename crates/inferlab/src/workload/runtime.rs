//! Runtime facade. Eval and Bench own domain adjudication, preparation owns
//! frozen request populations, and client supervision owns process lifecycle.

mod bench;
mod client;
mod eval;
mod preparation;

use super::domain::{
    BenchPopulation, BenchSessionTemplate, ResolvedBenchRequestSource, ResolvedBenchSource,
    WorkloadEndpointProtocol,
};
use super::record::{
    BenchDatasetRequestSourceEvidence, BenchPopulationPreparationEvidence,
    BenchRequestSourceEvidence, BenchSessionSourceEvidence, ClientCasePaths, ClientProcessEvidence,
    ClientTerminationEvidence, ClientTerminationTrigger, DatasetAcquisitionEvidence,
    DatasetAcquisitionOutcome, EvalCaseEvidence, EvalCaseRecord, WorkloadKind, WorkloadRecord,
    WorkloadRecordSession, WorkloadStatus, write_json,
};
use super::{
    BenchExecutionPlan, BenchPlan, ClientCommandPlan, EvalExecutionPlan, EvalPlan,
    ResolvedWorkloadPlan,
};
use inferlab_runtime::operation_bound::OperationTimingEvidence;
use inferlab_runtime::process_group::LocalProcessGroup;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Child;
use std::time::Duration;

const CLIENT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const CLIENT_TERM_GRACE: Duration = Duration::from_secs(2);
const CLIENT_KILL_GRACE: Duration = Duration::from_secs(2);
const CLIENT_CLEANUP_STATUS_DEADLINE: Duration = Duration::from_secs(2);
const SYNTHETIC_MATERIALIZATION_IDENTITY: &str = "inferlab-synthetic-prompt-target-v3";
struct AdjudicatedClient<T> {
    accepted: AcceptedClient<T>,
    succeeded: bool,
    error: Option<String>,
}

struct ClientRun {
    process: Option<ClientProcessEvidence>,
    error: Option<String>,
    pending_cleanup: Option<PendingClientCleanup>,
    /// Frozen before an early terminal path starts process cleanup. Ordinary
    /// exits leave this empty because result decoding and acceptance still
    /// belong to the measurement-case operation.
    terminal_timing: Option<OperationTimingEvidence>,
}

struct PendingClientCleanup {
    child: Child,
    group: LocalProcessGroup,
    handle_path: PathBuf,
}

/// The lenient result-envelope header: only the version, no field policy, so
/// an evolved envelope still reads far enough to be rejected by version
/// rather than dying in the strict v1 parse ([[RFC-0004:C-MEASUREMENTS]]).
#[derive(Deserialize)]
struct ClientResultEnvelope {
    schema_version: u32,
}

struct AcceptedClient<T> {
    run: ClientRun,
    result: Option<T>,
    decode_error: Option<String>,
    timing: OperationTimingEvidence,
    terminal_timing_frozen: bool,
}

pub(crate) const CLIENT_HANDLE_FILE: &str = "client-handle.json";
const SWEEP_WALK_DEPTH: usize = 6;

/// Durable client process-group handle, recorded at launch so a later run
/// can terminate survivors of an unclean exit by leader start-time
/// identity ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). The owner identity makes
/// "unclean exit" observable: a live handle belongs to a live concurrent
/// run exactly while the owning Inferlab process's identity still matches.
/// Unknown fields are tolerated so an older binary's sweep can still read
/// a newer handle instead of clearing it unparsed.
#[derive(Debug, Deserialize, Serialize)]
struct ClientGroupHandle {
    #[serde(flatten)]
    group: LocalProcessGroup,
    owner_pid: u32,
    owner_start_time_ticks: u64,
}

pub use bench::run_bench;
pub(crate) use bench::skip;
pub use eval::run_eval;

#[cfg(test)]
use super::adaptive::ProbeClassification;
#[cfg(test)]
use super::record::{
    AggregateSloEvaluation, CaseSloEvaluation, SloBoundDirection, SloEvaluationOutcome,
};
#[cfg(test)]
use bench::adaptive::classify_slo_evaluation;
#[cfg(test)]
use bench::result::bench_result_error;
#[cfg(test)]
use bench::session::duplicate_runtime_session_identity;
#[cfg(test)]
use client::{accept_client_result, sweep_stale_client_groups, terminate_client_group};
#[cfg(test)]
use eval::{repeated_eval_result_error, validate_openai_completion_body};

#[cfg(test)]
mod tests {
    use super::validate_openai_completion_body;
    use super::{
        AggregateSloEvaluation, CLIENT_HANDLE_FILE, CaseSloEvaluation, ClientGroupHandle,
        ClientRun, ProbeClassification, SloBoundDirection, SloEvaluationOutcome,
        accept_client_result, bench_result_error, classify_slo_evaluation,
        duplicate_runtime_session_identity, repeated_eval_result_error, sweep_stale_client_groups,
    };
    use crate::bench_metric::{BenchMetric, DistributionFamily, DistributionStatistic};
    use crate::record::RECORDS_DIR;
    use crate::workspace::{EvalDefinition, EvalTaskSource, RequestSlo};
    use inferlab_protocol::{
        BenchClientResult, BenchNativeInvocation, BenchRuntimeSessionResult,
        BenchSessionPhaseSummary, BenchSessionResultEvidence, BenchSessionTurnResult, ClientStatus,
        EvalClientResult, EvalMetricComparison, EvalMetricGate, EvalMetricGateConclusion,
        EvalNormalizedMetric, EvalTrialSummary,
    };
    use inferlab_runtime::operation_bound::OperationBound;
    use inferlab_runtime::process_group::{LocalProcessGroup, process_start_time};
    use serde_json::Value;
    use std::collections::BTreeMap;
    use std::fs;
    use std::os::unix::process::CommandExt;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::time::Duration;

    fn sweep_fixture(tag: &str) -> Result<(PathBuf, PathBuf), String> {
        let root =
            std::env::temp_dir().join(format!("inferlab-sweep-{tag}-{}", std::process::id()));
        let case_dir = root.join(RECORDS_DIR).join("run").join("cases").join("c0");
        fs::create_dir_all(&case_dir).map_err(|error| error.to_string())?;
        Ok((root, case_dir.join(CLIENT_HANDLE_FILE)))
    }

    fn spawn_survivor() -> Result<std::process::Child, String> {
        Command::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|error| error.to_string())
    }

    fn write_handle(path: &PathBuf, pid: u32, ticks: u64, owner: (u32, u64)) -> Result<(), String> {
        let handle = ClientGroupHandle {
            group: LocalProcessGroup::new(pid, pid, ticks).map_err(|error| error.to_string())?,
            owner_pid: owner.0,
            owner_start_time_ticks: owner.1,
        };
        let bytes = serde_json::to_vec(&handle).map_err(|error| error.to_string())?;
        fs::write(path, bytes).map_err(|error| error.to_string())
    }

    /// An owner identity that can never match a live process.
    const DEAD_OWNER: (u32, u64) = (u32::MAX, 1);

    fn own_identity() -> Result<(u32, u64), String> {
        let pid = std::process::id();
        let ticks = process_start_time(pid)
            .map_err(|error| error.to_string())?
            .ok_or("own identity unreadable")?;
        Ok((pid, ticks))
    }

    fn group_alive(pid: u32) -> Result<bool, String> {
        let ticks = process_start_time(pid)
            .map_err(|error| error.to_string())?
            .unwrap_or_default();
        let group = LocalProcessGroup::new(pid, pid, ticks).map_err(|error| error.to_string())?;
        let bound = OperationBound::finite(Duration::from_secs(2));
        group
            .has_live_members(&bound)
            .map_err(|error| error.to_string())
    }

    fn prefill_bench_result() -> BenchClientResult {
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
            native_command: vec!["fixture-bench".to_owned()],
            native_exit_code: Some(0),
            report_invocations: Vec::new(),
            raw_artifacts: Vec::new(),
            error: None,
        }
    }

    fn complete_session_evidence() -> BenchSessionResultEvidence {
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
                    pre_template_content_tokens: 1,
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
                    pre_template_content_tokens: 3,
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
            native_requests_reconciled: true,
            counts_reconciled: true,
        }
    }

    #[test]
    fn prefill_only_bench_does_not_require_tpot() {
        assert_eq!(
            bench_result_error(&prefill_bench_result(), false, false, None, 1, None),
            None
        );
    }

    #[test]
    fn decode_bench_requires_tpot() {
        let error = bench_result_error(&prefill_bench_result(), true, false, None, 1, None);

        assert!(error.is_some_and(|error| error.contains("mean_tpot_ms")));
    }

    #[test]
    fn linear_session_bench_requires_semantically_reconciled_evidence() {
        let mut result = prefill_bench_result();
        result.completed_requests = 2;
        result.session_evidence = Some(complete_session_evidence());

        assert_eq!(
            bench_result_error(&result, false, false, Some((0, 1)), 2, None),
            None
        );

        assert!(result.session_evidence.is_some());
        if let Some(evidence) = result.session_evidence.as_mut() {
            evidence.turns[1].preceding_native_session_num = None;
        }
        let error = bench_result_error(&result, false, false, Some((0, 1)), 2, None);
        assert!(error.is_some_and(|error| error.contains("linear-session")));
    }

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
            bench_result_error(&result, false, false, Some((1, 1)), 2, None),
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

    #[test]
    fn bench_rejects_out_of_range_cache_ratio() {
        let mut result = prefill_bench_result();
        result
            .metrics
            .insert("prompt_cache_read_ratio".to_owned(), 1.01);

        let error = bench_result_error(&result, false, false, None, 1, None);

        assert!(error.is_some_and(|error| error.contains("prompt_cache_read_ratio")));
    }

    #[test]
    fn speed_bench_server_metrics_require_both_acceptance_scalars() {
        let mut result = prefill_bench_result();
        result.metrics.insert("acceptance_length".to_owned(), 2.5);
        result.metrics.insert("acceptance_rate".to_owned(), 0.75);
        result.report_invocations = [
            ("acceptance_length", "accept_length"),
            ("acceptance_rate", "accept_rate"),
        ]
        .into_iter()
        .map(|(purpose, metric)| BenchNativeInvocation {
            purpose: purpose.to_owned(),
            command: vec![
                "aiperf".to_owned(),
                "speed-bench-report".to_owned(),
                "--metric".to_owned(),
                metric.to_owned(),
            ],
            exit_code: Some(0),
            interrupted: false,
            timed_out: false,
        })
        .collect();

        assert_eq!(
            bench_result_error(&result, false, true, None, 1, None),
            None
        );

        result.metrics.remove("acceptance_rate");
        let error = bench_result_error(&result, false, true, None, 1, None);
        assert!(error.is_some_and(|error| error.contains("acceptance_rate")));
    }

    #[test]
    fn complete_all_error_request_slo_result_is_measurement_evidence() {
        let mut result = prefill_bench_result();
        result.completed_requests = 0;
        result.failed_requests = 4;
        result.metrics = BTreeMap::from([
            ("good_request_ratio".to_owned(), 0.0),
            ("goodput".to_owned(), 0.0),
        ]);
        result.request_slo = Some(inferlab_protocol::BenchRequestSloResult {
            good_requests: 0,
            good_request_ratio: 0.0,
            goodput: 0.0,
            profiling_duration_seconds: 2.0,
            profiling_duration_source: "native-profiling-request-window".to_owned(),
            request_count_reconciled: true,
            native_aggregate_good_request_count: None,
            native_aggregate_good_request_count_consistent: None,
        });
        result.native_exit_code = Some(1);
        let slo = RequestSlo {
            request_latency_ms: None,
            ttft_ms: Some(800.0),
            tpot_ms: None,
            minimum_good_request_ratio: 0.99,
        };

        assert_eq!(
            bench_result_error(&result, true, false, None, 4, Some(&slo)),
            None
        );
    }

    #[test]
    fn unavailable_constraint_does_not_erase_an_above_region_failure() {
        let evaluation = CaseSloEvaluation {
            aggregate_slos: vec![
                AggregateSloEvaluation {
                    metric: BenchMetric::PromptCacheReadRatio,
                    direction: SloBoundDirection::AtLeast,
                    bound: 0.5,
                    observed: None,
                    outcome: SloEvaluationOutcome::Unavailable,
                },
                AggregateSloEvaluation {
                    metric: BenchMetric::Distribution {
                        statistic: DistributionStatistic::P99,
                        family: DistributionFamily::Ttft,
                    },
                    direction: SloBoundDirection::AtMost,
                    bound: 100.0,
                    observed: Some(150.0),
                    outcome: SloEvaluationOutcome::Failed,
                },
            ],
            request_slo: None,
            passed: false,
        };

        assert_eq!(
            classify_slo_evaluation(&evaluation),
            ProbeClassification::Above
        );

        let below_evaluation = CaseSloEvaluation {
            aggregate_slos: vec![
                AggregateSloEvaluation {
                    metric: BenchMetric::PromptCacheReadRatio,
                    direction: SloBoundDirection::AtLeast,
                    bound: 0.5,
                    observed: None,
                    outcome: SloEvaluationOutcome::Unavailable,
                },
                AggregateSloEvaluation {
                    metric: BenchMetric::RequestThroughput,
                    direction: SloBoundDirection::AtLeast,
                    bound: 10.0,
                    observed: Some(5.0),
                    outcome: SloEvaluationOutcome::Failed,
                },
            ],
            request_slo: None,
            passed: false,
        };
        assert_eq!(
            classify_slo_evaluation(&below_evaluation),
            ProbeClassification::Below
        );
    }

    #[test]
    fn result_decode_cannot_accept_after_the_owner_deadline() -> Result<(), String> {
        let path = std::env::temp_dir().join(format!(
            "inferlab-late-client-result-{}.json",
            std::process::id()
        ));
        fs::write(&path, br#"{"schema_version":1}"#).map_err(|error| error.to_string())?;
        let bound = OperationBound::finite(Duration::ZERO);
        let accepted = accept_client_result::<Value>(
            &path,
            "fixture client",
            ClientRun {
                process: None,
                error: None,
                pending_cleanup: None,
                terminal_timing: None,
            },
            &bound,
        );
        let _ = fs::remove_file(path);

        if accepted.result.is_some() {
            return Err("late client result was accepted".to_owned());
        }
        if !accepted
            .decode_error
            .as_deref()
            .is_some_and(|error| error.contains("measurement-case deadline"))
        {
            return Err("late client result did not preserve deadline rejection".to_owned());
        }
        Ok(())
    }

    #[test]
    fn repeated_eval_rejects_a_gate_conclusion_that_disagrees_with_its_threshold() {
        let definition = EvalDefinition::LmEval {
            task: EvalTaskSource::BuiltIn("fixture".to_owned()),
            request_body: BTreeMap::new(),
            limit: Some(1),
            few_shot: None,
            seed: Some(1234),
            trials: 2,
            max_tokens: None,
            concurrency: Some(1),
            metric: "exact_match".to_owned(),
            metric_filter: Some("strict".to_owned()),
            threshold: 0.75,
            timeout_seconds: 30,
        };
        let normalized = EvalNormalizedMetric {
            source_identity: "fixture".to_owned(),
            metric: "exact_match".to_owned(),
            filter: Some("strict".to_owned()),
            native_metric_key: "inferlab:pass_rate".to_owned(),
            value: 0.5,
            higher_is_better: true,
        };
        let result = EvalClientResult {
            schema_version: 1,
            status: ClientStatus::Succeeded,
            metrics: BTreeMap::from([("fixture:pass_rate".to_owned(), 0.5)]),
            normalized_metrics: BTreeMap::new(),
            gate: Some(EvalMetricGate {
                metric: normalized,
                threshold: 0.75,
                comparison: EvalMetricComparison::AtLeast,
                conclusion: EvalMetricGateConclusion::Passed,
            }),
            trial_summary: Some(EvalTrialSummary {
                requested_trials: 2,
                issued_trials: 2,
                unissued_trials: 0,
                completed_trials: 2,
                request_failure_trials: 0,
                passed_trials: 1,
                pass_rate: Some(0.5),
                per_trial_metric: "exact_match".to_owned(),
                per_trial_filter: Some("strict".to_owned()),
                higher_is_better: true,
            }),
            native_command: vec!["lm_eval".to_owned()],
            native_exit_code: None,
            native_timed_out: false,
            raw_artifacts: Vec::new(),
            failure_kind: None,
            error: None,
        };

        assert!(
            repeated_eval_result_error(&definition, &result)
                .is_some_and(|error| error.contains("pass-rate threshold semantics"))
        );
    }

    #[test]
    fn termination_covers_the_whole_process_group() -> Result<(), String> {
        // A client whose group contains its own descendants: the leader
        // spawns a grandchild and both share the group created at launch.
        let mut child = Command::new("sh")
            .args(["-c", "sleep 60 & exec sleep 60"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|error| error.to_string())?;
        let pid = child.id();
        let group = LocalProcessGroup::capture_child(&child).map_err(|error| error.to_string())?;
        let evidence = super::terminate_client_group(
            &mut child,
            group,
            super::ClientTerminationTrigger::ResultAccepted,
        );
        let alive = group_alive(pid)?;
        let _ = child.wait();
        if !evidence.verified {
            return Err("group termination was not verified".to_owned());
        }
        if evidence.trigger != super::ClientTerminationTrigger::ResultAccepted {
            return Err("client cleanup did not record its trigger".to_owned());
        }
        if evidence.term_grace_ms != 2_000 || evidence.kill_grace_ms != 2_000 {
            return Err("client cleanup did not record its independent graces".to_owned());
        }
        if alive {
            return Err("descendants survived group termination".to_owned());
        }
        Ok(())
    }

    #[test]
    fn sweep_skips_live_owners_clients() -> Result<(), String> {
        let (root, handle_path) = sweep_fixture("owner")?;
        let mut child = spawn_survivor()?;
        let pid = child.id();
        let ticks = process_start_time(pid)
            .map_err(|error| error.to_string())?
            .ok_or("survivor exited before recording")?;
        write_handle(&handle_path, pid, ticks, own_identity()?)?;
        sweep_stale_client_groups(&root);
        let alive = group_alive(pid)?;
        let handle_kept = handle_path.exists();
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
        let _ = child.wait();
        let _ = fs::remove_dir_all(&root);
        if !alive {
            return Err("sweep terminated a live concurrent run's client".to_owned());
        }
        if !handle_kept {
            return Err("sweep cleared a live concurrent run's handle".to_owned());
        }
        Ok(())
    }

    #[test]
    fn sweep_terminates_identity_matching_survivors() -> Result<(), String> {
        let (root, handle_path) = sweep_fixture("live")?;
        let mut child = spawn_survivor()?;
        let pid = child.id();
        let ticks = process_start_time(pid)
            .map_err(|error| error.to_string())?
            .ok_or("survivor exited before recording")?;
        write_handle(&handle_path, pid, ticks, DEAD_OWNER)?;
        // Reap concurrently: the survivor is this test's child, and the sweep
        // verifies group death, which a zombie would postpone. Real
        // survivors of an unclean exit are reparented to init and reaped.
        let waiter = std::thread::spawn(move || {
            let _ = child.wait();
        });
        sweep_stale_client_groups(&root);
        waiter
            .join()
            .map_err(|_| "waiter thread panicked".to_owned())?;
        if group_alive(pid)? {
            return Err("identity-matching survivor group is still alive".to_owned());
        }
        if handle_path.exists() {
            return Err("swept handle file was not cleared".to_owned());
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn sweep_never_signals_identity_drift() -> Result<(), String> {
        let (root, handle_path) = sweep_fixture("drift")?;
        let mut child = spawn_survivor()?;
        let pid = child.id();
        let ticks = process_start_time(pid)
            .map_err(|error| error.to_string())?
            .ok_or("survivor exited before recording")?;
        write_handle(&handle_path, pid, ticks + 1, DEAD_OWNER)?;
        sweep_stale_client_groups(&root);
        let alive = group_alive(pid)?;
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
        let _ = child.wait();
        if !alive {
            return Err("sweep signalled a group whose identity drifted".to_owned());
        }
        if handle_path.exists() {
            return Err("drifted handle file was not cleared".to_owned());
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn openai_smoke_requires_a_nonempty_choices_array_with_text() {
        assert_eq!(
            validate_openai_completion_body(br#"{"choices":[{"text":"ok"}]}"#),
            Ok(1)
        );
        for body in [
            br#"not-json"#.as_slice(),
            br#"{}"#.as_slice(),
            br#"{"choices":[]}"#.as_slice(),
            br#"{"choices":[{}]}"#.as_slice(),
            br#"{"choices":[{"text":1}]}"#.as_slice(),
        ] {
            assert!(validate_openai_completion_body(body).is_err());
        }
    }
}
