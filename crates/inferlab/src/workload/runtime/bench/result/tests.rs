use super::super::test_support::{complete_session_evidence, prefill_bench_result};
use super::{BenchResultExpectations, bench_result_error};
use crate::workload::domain::{BenchAgenticCatalog, ResolvedBenchAgenticSource};
use crate::workspace::RequestSlo;
use inferlab_protocol::{
    BenchAgenticAcquisitionOutcome, BenchAgenticBranchStats, BenchAgenticResultEvidence,
    BenchAgenticRunEvidence, BenchAgenticSourceVerification, BenchDatasetCacheState,
    BenchNativeInvocation, BenchPromptCacheObservation, RawArtifact,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn expectations(request_count: u32) -> BenchResultExpectations<'static> {
    BenchResultExpectations {
        tpot_applicable: false,
        speed_bench_server_metrics: false,
        sessions: None,
        agentic_source: None,
        artifact_level: crate::workspace::BenchArtifactLevel::Diagnostic,
        request_count,
        request_slo: None,
        prompt_cache_evidence: false,
    }
}

fn agentic_source() -> ResolvedBenchAgenticSource {
    ResolvedBenchAgenticSource {
        dataset: "semianalysis_agentx_062126_256k".to_owned(),
        profile: "inferencex".to_owned(),
        catalog: Box::new(BenchAgenticCatalog {
            repository: "semianalysisai/cc-traces-weka-062126-256k".to_owned(),
            revision: "revision".to_owned(),
            filename: "traces.jsonl".to_owned(),
            sha256: "digest".to_owned(),
            cache_path: None,
            cache_state: None,
            trace_count: 393,
            approximate_bytes: 569_000_000,
            license: "apache-2.0".to_owned(),
            source_format: "weka_kv_cache_tester_agentic_trace_v7_jsonl".to_owned(),
            aiperf_loader: "semianalysis_cc_traces_weka_062126_256k".to_owned(),
            materialization_identity: "loader".to_owned(),
            scenario: "inferencex-agentx-mvp".to_owned(),
            concurrency_semantics: "root_session_tree_lanes".to_owned(),
            replay_semantics: "source_response_inclusive".to_owned(),
            cache_bust: "first_turn_prefix".to_owned(),
            trajectory_start_min: 0.25,
            trajectory_start_max: 0.75,
            global_idle_gap_cap_seconds: 10.0,
            trace_idle_gap_cap_seconds: 300.0,
            cache_warmup_requests_per_lane: 10,
            warmup_grace_seconds: 1800,
            dataset_configuration_timeout_seconds: 1800,
            service_profile_configuration_timeout_seconds: 1800,
            default_duration_seconds: 1800,
            minimum_duration_seconds: 900,
            failure_threshold: 0.1,
            dataset_entries: 393,
            streaming: true,
            ignore_eos: true,
            use_server_token_count: true,
            gpu_telemetry: false,
            server_metric_slice_seconds: 1,
            required_artifacts: vec![
                "aggregate".to_owned(),
                "records".to_owned(),
                "raw_records".to_owned(),
            ],
            unavailable_dimensions: vec!["recycle_identity".to_owned()],
            inferencex_repository: "SemiAnalysisAI/InferenceX".to_owned(),
            inferencex_revision: "inferencex".to_owned(),
            inferencex_reference: "benchmarks/benchmark_lib.sh".to_owned(),
            aiperf_revision: "aiperf".to_owned(),
            aiperf_version: "0.12.0".to_owned(),
        }),
    }
}

fn complete_agentic_evidence(source: &ResolvedBenchAgenticSource) -> BenchAgenticResultEvidence {
    BenchAgenticResultEvidence {
        source: BenchAgenticSourceVerification {
            repository: source.catalog.repository.clone(),
            expected_revision: source.catalog.revision.clone(),
            observed_revision: Some(source.catalog.revision.clone()),
            filename: source.catalog.filename.clone(),
            expected_sha256: source.catalog.sha256.clone(),
            observed_sha256: Some(source.catalog.sha256.clone()),
            cache_path: Some(PathBuf::from("/cache/traces.jsonl")),
            cache_state_before: Some(BenchDatasetCacheState::Present),
            acquisition_outcome: Some(BenchAgenticAcquisitionOutcome::Reused),
        },
        run: Some(Box::new(BenchAgenticRunEvidence {
            native_run_id: "run-1".to_owned(),
            scenario: source.catalog.scenario.clone(),
            submission_valid: true,
            submission_invalid_reasons: Vec::new(),
            warmup_records: 3,
            warmup_error_records: 0,
            warmup_succeeded: true,
            profiling_records: 2,
            distinct_runtime_conversations: 1,
            distinct_transport_requests: 2,
            context_overflow_count: 0,
            ordinary_failure_count: 1,
            branch_stats: BenchAgenticBranchStats {
                children_spawned: 1,
                children_completed: 1,
                children_errored: 0,
                children_truncated: 0,
                children_delayed: 0,
                parents_suspended: 1,
                parents_resumed: 1,
                parents_failed_due_to_child_error: 0,
                joins_suppressed: 0,
            },
            aggregate_artifact: PathBuf::from("aggregate.json"),
            raw_records_artifact: Some(PathBuf::from("raw.jsonl")),
            unavailable_dimensions: source.catalog.unavailable_dimensions.clone(),
            warmup_source_coordinate_records: Some(3),
            source_coordinate_records: Some(2),
            distinct_source_traces: Some(1),
            cache_bust_records: Some(1),
        })),
    }
}

#[test]
fn prefill_only_bench_does_not_require_tpot() {
    assert_eq!(
        bench_result_error(&prefill_bench_result(), expectations(1)),
        None
    );
}

#[test]
fn agentic_bench_defers_failed_request_threshold_to_native_scenario()
-> Result<(), Box<dyn std::error::Error>> {
    let source = agentic_source();
    let evidence = complete_agentic_evidence(&source);
    let run = evidence
        .run
        .as_deref()
        .ok_or("missing agentic run evidence")?;
    let mut result = prefill_bench_result();
    result.completed_requests = 1;
    result.failed_requests = 1;
    result.raw_artifacts = vec![
        RawArtifact {
            name: "aiperf_summary".to_owned(),
            kind: "aiperf-summary".to_owned(),
            path: run.aggregate_artifact.clone(),
        },
        RawArtifact {
            name: "aiperf_records".to_owned(),
            kind: "aiperf-records".to_owned(),
            path: PathBuf::from("records.json"),
        },
        RawArtifact {
            name: "aiperf_raw_records".to_owned(),
            kind: "aiperf-raw-records".to_owned(),
            path: run
                .raw_records_artifact
                .clone()
                .ok_or("complete agentic evidence lacks a raw artifact")?,
        },
    ];
    result.agentic_evidence = Some(Box::new(evidence));

    assert_eq!(
        bench_result_error(
            &result,
            BenchResultExpectations {
                agentic_source: Some(&source),
                ..expectations(0)
            },
        ),
        None
    );

    // Cache-pressure warmup failures are recorded evidence, not a case-failure
    // input: the native scenario outcome remains the acceptance authority.
    if let Some(evidence) = result.agentic_evidence.as_mut()
        && let Some(run) = evidence.run.as_mut()
    {
        run.warmup_error_records = 2;
    }
    assert_eq!(
        bench_result_error(
            &result,
            BenchResultExpectations {
                agentic_source: Some(&source),
                ..expectations(0)
            },
        ),
        None
    );

    if let Some(evidence) = result.agentic_evidence.as_mut() {
        evidence.source.observed_sha256 = Some("wrong".to_owned());
    }
    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            agentic_source: Some(&source),
            ..expectations(0)
        },
    );
    assert!(error.is_some_and(|error| error.contains("source verification")));

    if let Some(evidence) = result.agentic_evidence.as_mut() {
        evidence.source.observed_sha256 = Some(source.catalog.sha256.clone());
        if let Some(run) = evidence.run.as_mut() {
            run.submission_valid = false;
            run.submission_invalid_reasons = vec!["context_overflow_rate_exceeded".to_owned()];
        }
    }
    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            agentic_source: Some(&source),
            ..expectations(0)
        },
    );
    assert!(error.is_some_and(|error| error.contains("scenario submission is invalid")));

    if let Some(evidence) = result.agentic_evidence.as_mut()
        && let Some(run) = evidence.run.as_mut()
    {
        run.submission_valid = true;
        run.submission_invalid_reasons.clear();
    }
    result
        .raw_artifacts
        .retain(|artifact| artifact.name != "aiperf_raw_records");
    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            agentic_source: Some(&source),
            ..expectations(0)
        },
    );
    assert!(error.is_some_and(|error| error.contains("required native artifact")));
    Ok(())
}

#[test]
fn performance_agentic_evidence_records_raw_derived_dimensions_as_unavailable()
-> Result<(), Box<dyn std::error::Error>> {
    let source = agentic_source();
    let mut evidence = complete_agentic_evidence(&source);
    let mut result = prefill_bench_result();
    result.completed_requests = 1;
    result.failed_requests = 1;
    result.raw_artifacts = vec![
        RawArtifact {
            name: "aiperf_summary".to_owned(),
            kind: "aiperf-summary".to_owned(),
            path: PathBuf::from("aggregate.json"),
        },
        RawArtifact {
            name: "aiperf_records".to_owned(),
            kind: "aiperf-records".to_owned(),
            path: PathBuf::from("records.json"),
        },
    ];
    if let Some(run) = evidence.run.as_mut() {
        run.warmup_source_coordinate_records = None;
        run.source_coordinate_records = None;
        run.distinct_source_traces = None;
        run.cache_bust_records = None;
        run.raw_records_artifact = None;
        run.unavailable_dimensions.extend(
            super::PERFORMANCE_AGENTIC_UNAVAILABLE_DIMENSIONS
                .iter()
                .map(|dimension| (*dimension).to_owned()),
        );
    }
    result.agentic_evidence = Some(Box::new(evidence));
    let performance = BenchResultExpectations {
        agentic_source: Some(&source),
        artifact_level: crate::workspace::BenchArtifactLevel::Performance,
        ..expectations(0)
    };

    assert_eq!(bench_result_error(&result, performance), None);

    // The same degraded evidence remains invalid at the diagnostic level.
    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            agentic_source: Some(&source),
            ..expectations(0)
        },
    );
    assert!(error.is_some_and(|error| error.contains("artifact level")));

    // A performance case still requires the aggregate and records artifacts.
    result.raw_artifacts.clear();
    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            agentic_source: Some(&source),
            artifact_level: crate::workspace::BenchArtifactLevel::Performance,
            ..expectations(0)
        },
    );
    assert!(error.is_some_and(|error| error.contains("required native artifact")));
    Ok(())
}

#[test]
fn decode_bench_requires_tpot() {
    let error = bench_result_error(
        &prefill_bench_result(),
        BenchResultExpectations {
            tpot_applicable: true,
            ..expectations(1)
        },
    );

    assert!(error.is_some_and(|error| error.contains("mean_tpot_ms")));
}

#[test]
fn linear_session_bench_requires_semantically_reconciled_evidence() {
    let mut result = prefill_bench_result();
    result.completed_requests = 2;
    result.session_evidence = Some(complete_session_evidence());

    assert_eq!(
        bench_result_error(
            &result,
            BenchResultExpectations {
                sessions: Some((0, 1)),
                ..expectations(2)
            },
        ),
        None
    );

    assert!(result.session_evidence.is_some());
    if let Some(evidence) = result.session_evidence.as_mut() {
        evidence.turns[1].preceding_native_session_num = None;
    }
    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            sessions: Some((0, 1)),
            ..expectations(2)
        },
    );
    assert!(error.is_some_and(|error| error.contains("linear-session")));
}

#[test]
fn bench_rejects_out_of_range_cache_ratio() {
    let mut result = prefill_bench_result();
    result
        .metrics
        .insert("prompt_cache_read_ratio".to_owned(), 1.01);

    let error = bench_result_error(&result, expectations(1));

    assert!(error.is_some_and(|error| error.contains("prompt_cache_read_ratio")));
}

#[test]
fn required_prompt_cache_evidence_reconciles_backend_token_observations() {
    let mut result = prefill_bench_result();
    result.prompt_cache_observations = vec![BenchPromptCacheObservation {
        request_id: 7,
        prompt_tokens: 10,
        cache_read_tokens: 6,
        uncached_prompt_tokens: 4,
        cache_read_ratio: 0.6,
    }];
    for family in ["prompt_cache_read_tokens", "uncached_prompt_tokens"] {
        for statistic in ["mean", "min", "max", "stddev", "p50", "p90", "p95", "p99"] {
            result.metrics.insert(format!("{statistic}_{family}"), 1.0);
        }
    }
    result
        .metrics
        .insert("prompt_cache_read_ratio".to_owned(), 0.6);

    assert_eq!(
        bench_result_error(
            &result,
            BenchResultExpectations {
                prompt_cache_evidence: true,
                ..expectations(1)
            },
        ),
        None
    );

    result.prompt_cache_observations[0].uncached_prompt_tokens = 3;
    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            prompt_cache_evidence: true,
            ..expectations(1)
        },
    );
    assert!(error.is_some_and(|error| error.contains("per-request prompt-cache evidence")));
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
        bench_result_error(
            &result,
            BenchResultExpectations {
                speed_bench_server_metrics: true,
                ..expectations(1)
            },
        ),
        None
    );

    result.metrics.remove("acceptance_rate");
    let error = bench_result_error(
        &result,
        BenchResultExpectations {
            speed_bench_server_metrics: true,
            ..expectations(1)
        },
    );
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
        bench_result_error(
            &result,
            BenchResultExpectations {
                tpot_applicable: true,
                request_slo: Some(&slo),
                ..expectations(4)
            },
        ),
        None
    );
}
