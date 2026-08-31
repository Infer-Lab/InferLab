import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from inferlab_bench_runner import execution
from inferlab_bench_runner.agentic_source import (
    AgenticSourceAcquisition,
    acquire_resolved_agentic_source,
    source_verification_error,
    verify_downloaded_snapshot,
)
from inferlab_bench_runner.aiperf import (
    aiperf_config,
    prepare_aiperf_execution,
)
from inferlab_bench_runner.data_asset import prepare_agentic_data_asset
from inferlab_bench_runner.execution import execute
from inferlab_bench_runner.result_agentic import agentic_result_evidence
from inferlab_measurement_sdk import (
    BenchAgenticAcquisitionOutcome,
    BenchAgenticSourceVerification,
    BenchClientRequest,
    BenchDatasetCacheState,
    CaseDeadline,
    ClientStatus,
    MeasurementDataAssetEffectiveSelectionAgentic,
    MeasurementDataAssetPreparationRequest,
    MeasurementDataAssetReadinessClosed,
    MeasurementDataAssetRemoteMetadataOutcome,
)

from .support import resolved_prompt_input


def agentic_request(tmp_path: Path, *, server_metrics: bool = False) -> BenchClientRequest:
    return BenchClientRequest.model_validate(
        {
            "protocol_version": "9",
            "endpoint": {
                "protocol": "http",
                "host": "127.0.0.1",
                "port": 8000,
                "completions_path": "/v1/completions",
                "chat_completions_path": "/v1/chat/completions",
                "server_metrics": (
                    {
                        "path": "/metrics",
                        "port_name": None,
                        "url": "http://127.0.0.1:8000/metrics",
                    }
                    if server_metrics
                    else None
                ),
            },
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "definition": {
                "agentic_source": {
                    "dataset": "semianalysis_agentx_062126_256k",
                    "profile": "inferencex",
                    "catalog": {
                        "repository": "semianalysisai/cc-traces-weka-062126-256k",
                        "revision": "8fecd2fc56694469f758f0afbbb6335ad3043740",
                        "filename": "traces.jsonl",
                        "sha256": (
                            "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef"
                        ),
                        "cache_path": str(tmp_path / "traces.jsonl"),
                        "cache_state": "missing",
                        "trace_count": 393,
                        "approximate_bytes": 569000000,
                        "license": "apache-2.0",
                        "source_format": "weka_kv_cache_tester_agentic_trace_v7_jsonl",
                        "aiperf_loader": "semianalysis_cc_traces_weka_062126_256k",
                        "materialization_identity": (
                            "aiperf.dataset.loader.semianalysis_cc_traces_weka:"
                            "SemiAnalysisCCTracesWekaLoader"
                        ),
                        "scenario": "inferencex-agentx-mvp",
                        "concurrency_semantics": "root_session_tree_lanes",
                        "replay_semantics": "source_response_inclusive",
                        "cache_bust": "first_turn_prefix",
                        "trajectory_start_min": 0.25,
                        "trajectory_start_max": 0.75,
                        "global_idle_gap_cap_seconds": 10.0,
                        "trace_idle_gap_cap_seconds": 300.0,
                        "cache_warmup_requests_per_lane": 10,
                        "warmup_grace_seconds": 1800,
                        "dataset_configuration_timeout_seconds": 1800,
                        "service_profile_configuration_timeout_seconds": 1800,
                        "default_duration_seconds": 1800,
                        "minimum_duration_seconds": 900,
                        "failure_threshold": 0.10,
                        "dataset_entries": 393,
                        "streaming": True,
                        "ignore_eos": True,
                        "use_server_token_count": True,
                        "gpu_telemetry": False,
                        "server_metric_slice_seconds": 1,
                        "required_artifacts": ["aggregate", "records", "raw_records"],
                        "unavailable_dimensions": ["exported_per_lane_time_origin"],
                        "inferencex_repository": "SemiAnalysisAI/InferenceX",
                        "inferencex_revision": "900b3d8199350a51d731409b690cb79b804b31bf",
                        "inferencex_reference": "benchmarks/benchmark_lib.sh",
                        "aiperf_revision": "0d2aa0572ac685943d38c580675c4a61023581d3",
                        "aiperf_version": "0.12.0",
                    },
                },
                "prompt": resolved_prompt_input({"kind": "server_chat"}),
                "server_metrics": server_metrics,
                "seed": 42,
                "request_body": {},
                "request_slo": None,
                "timeout_seconds": 3600,
                "cache_start": "uncontrolled",
            },
            "case": {
                "load_shape": {"kind": "concurrency_limited", "concurrency": 2},
                "request_count": 0,
                "warmup_request_count": 0,
                "duration_seconds": 900,
            },
            "case_budget_seconds": 3600.0,
            "artifact_dir": str(tmp_path),
        }
    )


def test_agentic_source_preparation_closes_the_release_qualified_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench_request = agentic_request(tmp_path)
    source = bench_request.definition.agentic_source
    assert source is not None
    verification = BenchAgenticSourceVerification(
        repository=source.catalog.repository,
        expected_revision=source.catalog.revision,
        observed_revision=source.catalog.revision,
        filename=source.catalog.filename,
        expected_sha256=source.catalog.sha256,
        observed_sha256=source.catalog.sha256,
        cache_path=str(tmp_path / source.catalog.filename),
        cache_state_before=BenchDatasetCacheState.present,
        acquisition_outcome=BenchAgenticAcquisitionOutcome.reused,
    )
    monkeypatch.setattr(
        "inferlab_bench_runner.data_asset.acquire_resolved_agentic_source",
        lambda source, revision, cache_state: AgenticSourceAcquisition(
            verification=verification, error=None
        ),
    )
    request = MeasurementDataAssetPreparationRequest.model_validate(
        {
            "protocol_version": "9",
            "phase": {
                "kind": "acquire",
                "resolved_revision": source.catalog.revision,
                "cache_state_before": "full_hit",
            },
            "source": {"kind": "agentic", "source": source.model_dump()},
            "artifact_dir": str(tmp_path / "assets"),
        }
    )

    result = prepare_agentic_data_asset(request)

    assert result.status is ClientStatus.succeeded
    assert result.readiness is not None
    assert isinstance(result.readiness.root, MeasurementDataAssetReadinessClosed)
    acquired_source = result.readiness.root.acquired_source.model_dump()
    assert acquired_source["kind"] == "release_qualified"
    assert acquired_source["closure"][0]["sha256"] == source.catalog.sha256
    assert result.cache_stores[0].outcome.value == "full_hit"


def test_agentic_source_resolution_uses_the_release_revision_not_mutable_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench_request = agentic_request(tmp_path)
    source = bench_request.definition.agentic_source
    assert source is not None
    monkeypatch.setattr(
        "inferlab_bench_runner.agentic_source.try_to_load_from_cache",
        lambda **_: None,
    )
    request = MeasurementDataAssetPreparationRequest.model_validate(
        {
            "protocol_version": "9",
            "phase": {"kind": "resolve"},
            "source": {"kind": "agentic", "source": source.model_dump()},
            "artifact_dir": str(tmp_path / "assets"),
        }
    )

    result = prepare_agentic_data_asset(request)

    assert result.status is ClientStatus.succeeded
    assert result.effective_selection is not None
    selection = result.effective_selection.root
    assert isinstance(selection, MeasurementDataAssetEffectiveSelectionAgentic)
    assert selection.observed_revision == source.catalog.revision
    assert result.remote_metadata is MeasurementDataAssetRemoteMetadataOutcome.not_accessed


def test_verified_agentic_cache_hit_uses_local_only_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench_request = agentic_request(tmp_path)
    source = bench_request.definition.agentic_source
    assert source is not None
    snapshot = tmp_path / source.catalog.filename
    snapshot.write_bytes(b"qualified fixture")
    catalog = source.catalog.model_copy(
        update={"sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    )
    source = source.model_copy(update={"catalog": catalog})
    observed_local_only: list[bool] = []

    def download(**kwargs: object) -> str:
        observed_local_only.append(kwargs.get("local_files_only") is True)
        return str(snapshot)

    monkeypatch.setattr(
        "inferlab_bench_runner.agentic_source.hf_hub_download",
        download,
    )

    acquisition = acquire_resolved_agentic_source(
        source,
        catalog.revision,
        BenchDatasetCacheState.present,
    )

    assert acquisition.error is None
    assert observed_local_only == [True]


def test_agentic_config_lowers_the_release_profile_without_an_inferlab_dag(
    tmp_path: Path,
) -> None:
    request_value = agentic_request(tmp_path, server_metrics=True)
    config = aiperf_config(request_value)
    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])
    profiling = cast(dict[str, object], benchmark["profiling"])
    artifacts = cast(dict[str, object], benchmark["artifacts"])

    assert benchmark["scenario"] == "inferencex-agentx-mvp"
    assert dataset == {
        "type": "public",
        "dataset": "semianalysis_cc_traces_weka_062126_256k",
        "entries": 393,
        "sampling": "sequential",
        "traceIdleGapCapSeconds": 300.0,
    }
    assert profiling == {
        "type": "concurrency",
        "concurrency": 2,
        "duration": 900,
        "failedRequestThreshold": 0.10,
        "trajectoryStartMinRatio": 0.25,
        "trajectoryStartMaxRatio": 0.75,
        "systemIdleGapCapSeconds": 10.0,
        "warmupRequestsPerLane": 10,
        "agenticWarmupGracePeriod": 1800,
    }
    assert artifacts["sliceDuration"] == 1
    prepared = prepare_aiperf_execution(request_value, CaseDeadline(3600))
    assert prepared.environment == {
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
    }


def test_agentic_snapshot_verification_binds_revision_and_complete_file_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_bytes(b'{"trace":"qualified"}\n')
    request_value = agentic_request(tmp_path)
    source = request_value.definition.agentic_source
    assert source is not None
    catalog = source.catalog.model_copy(
        update={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
    source = source.model_copy(update={"catalog": catalog})

    evidence = verify_downloaded_snapshot(source, catalog.revision, path)
    assert evidence.expected_sha256 == evidence.observed_sha256
    assert evidence.observed_revision == catalog.revision

    mismatched = verify_downloaded_snapshot(source, "0" * 40, path)
    assert mismatched.observed_revision == "0" * 40
    assert mismatched.expected_revision == catalog.revision


def test_agentic_source_failure_preserves_typed_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_value = agentic_request(tmp_path)
    source = request_value.definition.agentic_source
    assert source is not None
    source_path = tmp_path / "traces.jsonl"
    source_path.write_bytes(b"wrong snapshot")
    verification = verify_downloaded_snapshot(source, source.catalog.revision, source_path)
    error = source_verification_error(source, verification)
    assert error is not None
    monkeypatch.setattr(
        execution,
        "acquire_and_verify_agentic_source",
        lambda _: AgenticSourceAcquisition(verification=verification, error=error),
    )

    result = execute(request_value)

    assert result.status is ClientStatus.failed
    assert result.agentic_evidence is not None
    assert (
        result.agentic_evidence.source.observed_sha256
        == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )
    assert result.agentic_evidence.source.cache_path == str(source_path)
    assert result.agentic_evidence.source.acquisition_outcome is not None
    assert result.agentic_evidence.run is None
    assert result.error is not None
    assert "content digest does not match" in result.error


def test_agentic_result_preserves_public_scenario_coordinates_and_branch_stats(
    tmp_path: Path,
) -> None:
    request_value = agentic_request(tmp_path)
    source = request_value.definition.agentic_source
    assert source is not None
    source_path = tmp_path / "traces.jsonl"
    source_path.write_bytes(b"qualified")
    source = source.model_copy(
        update={
            "catalog": source.catalog.model_copy(
                update={"sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
            )
        }
    )
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "source_trace_id": "trace-1",
                        "source_outer_idx": 2,
                        "source_kind": "parent",
                        "x_request_id": "warmup-request-1",
                        "x_correlation_id": "session-1",
                        "root_correlation_id": "root-1",
                    },
                    "cache_bust_marker": None,
                    "cache_bust_target": None,
                },
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "source_trace_id": "trace-1",
                        "source_outer_idx": 3,
                        "source_inner_idx": None,
                        "source_kind": "parent",
                        "x_request_id": "request-1",
                        "x_correlation_id": "session-1",
                        "root_correlation_id": "root-1",
                    },
                    "cache_bust_marker": " marker ",
                    "cache_bust_target": "first_turn_prefix",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "benchmark_id": "agentx-run-1",
        "metadata": {
            "scenario": "inferencex-agentx-mvp",
            "submission_valid": True,
            "dataset": {
                "loader": "semianalysis_cc_traces_weka_062126_256k",
                "hf_dataset_name": "semianalysisai/cc-traces-weka-062126-256k",
                "num_dataset_entries": 393,
            },
        },
        "context_overflow_count": {"avg": 1.0},
        "skipped_context_overflow_count": {"avg": 1.0},
        "error_request_count": {"avg": 2.0},
        "branch_stats": {
            "children_spawned": 4,
            "children_completed": 3,
            "children_errored": 0,
            "children_truncated": 1,
            "children_delayed": 2,
            "parents_suspended": 2,
            "parents_resumed": 2,
            "parents_failed_due_to_child_error": 0,
            "joins_suppressed": 1,
        },
    }

    evidence = agentic_result_evidence(
        source, summary, tmp_path / "summary.json", tmp_path / "records.jsonl", raw_path
    )
    assert evidence.warmup_records == 1
    assert evidence.warmup_succeeded
    assert evidence.warmup_source_coordinate_records == 1
    assert evidence.profiling_records == 1
    assert evidence.source_coordinate_records == 1
    assert evidence.cache_bust_records == 1
    assert evidence.distinct_source_traces == 1
    assert evidence.distinct_runtime_conversations == 1
    assert evidence.distinct_transport_requests == 1
    assert evidence.context_overflow_count == 1
    assert evidence.ordinary_failure_count == 2
    assert evidence.branch_stats.children_truncated == 1

    invalid_summary = dict(summary)
    invalid_summary["metadata"] = {
        **cast(dict[str, object], summary["metadata"]),
        "submission_valid": False,
        "submission_invalid_reasons": ["context_overflow_rate_exceeded"],
    }
    invalid_evidence = agentic_result_evidence(
        source, invalid_summary, tmp_path / "summary.json", tmp_path / "records.jsonl", raw_path
    )
    assert not invalid_evidence.submission_valid
    assert invalid_evidence.submission_invalid_reasons == ["context_overflow_rate_exceeded"]


def test_agentic_result_records_warmup_failures_as_evidence_not_handoff_state(
    tmp_path: Path,
) -> None:
    request_value = agentic_request(tmp_path)
    source = request_value.definition.agentic_source
    assert source is not None
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "source_trace_id": "trace-1",
                        "source_outer_idx": 2,
                        "source_kind": "parent",
                        "x_request_id": "warmup-request-1",
                        "x_correlation_id": "session-1",
                        "root_correlation_id": "root-1",
                    },
                    "error": {"message": "empty visible content"},
                },
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "source_trace_id": "trace-1",
                        "source_outer_idx": 3,
                        "source_inner_idx": None,
                        "source_kind": "parent",
                        "x_request_id": "request-1",
                        "x_correlation_id": "session-1",
                        "root_correlation_id": "root-1",
                    },
                    "error": None,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "benchmark_id": "agentx-run-1",
        "metadata": {
            "scenario": "inferencex-agentx-mvp",
            "submission_valid": True,
            "dataset": {
                "loader": "semianalysis_cc_traces_weka_062126_256k",
                "hf_dataset_name": "semianalysisai/cc-traces-weka-062126-256k",
                "num_dataset_entries": 393,
            },
        },
        "context_overflow_count": {"avg": 0.0},
        "skipped_context_overflow_count": {"avg": 0.0},
        "error_request_count": {"avg": 0.0},
        "branch_stats": {
            "children_spawned": 0,
            "children_completed": 0,
            "children_errored": 0,
            "children_truncated": 0,
            "children_delayed": 0,
            "parents_suspended": 0,
            "parents_resumed": 0,
            "parents_failed_due_to_child_error": 0,
            "joins_suppressed": 0,
        },
    }

    evidence = agentic_result_evidence(
        source, summary, tmp_path / "summary.json", tmp_path / "records.jsonl", raw_path
    )

    assert evidence.warmup_records == 1
    assert evidence.warmup_error_records == 1
    assert evidence.warmup_succeeded
    assert evidence.profiling_records == 1
    assert evidence.submission_valid
    assert "profiling_began_after_warmup_and_drain" not in evidence.model_dump()


def test_performance_agentic_result_degrades_raw_derived_dimensions(
    tmp_path: Path,
) -> None:
    request_value = agentic_request(tmp_path)
    source = request_value.definition.agentic_source
    assert source is not None
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "x_request_id": "warmup-request-1",
                        "x_correlation_id": "session-1",
                        "was_cancelled": False,
                    },
                    "error": None,
                },
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "x_request_id": "request-1",
                        "x_correlation_id": "session-1",
                        "was_cancelled": False,
                    },
                    "error": None,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "benchmark_id": "agentx-run-1",
        "metadata": {
            "scenario": "inferencex-agentx-mvp",
            "submission_valid": True,
            "dataset": {
                "loader": "semianalysis_cc_traces_weka_062126_256k",
                "hf_dataset_name": "semianalysisai/cc-traces-weka-062126-256k",
                "num_dataset_entries": 393,
            },
        },
        "context_overflow_count": {"avg": 1.0},
        "skipped_context_overflow_count": {"avg": 1.0},
        "error_request_count": {"avg": 2.0},
        "branch_stats": {
            "children_spawned": 4,
            "children_completed": 3,
            "children_errored": 0,
            "children_truncated": 1,
            "children_delayed": 2,
            "parents_suspended": 2,
            "parents_resumed": 2,
            "parents_failed_due_to_child_error": 0,
            "joins_suppressed": 1,
        },
    }

    evidence = agentic_result_evidence(
        source, summary, tmp_path / "summary.json", records_path, None
    )

    assert evidence.warmup_records == 1
    assert evidence.warmup_succeeded
    assert evidence.profiling_records == 1
    assert evidence.distinct_runtime_conversations == 1
    assert evidence.distinct_transport_requests == 1
    assert evidence.context_overflow_count == 1
    assert evidence.ordinary_failure_count == 2
    assert evidence.branch_stats.children_truncated == 1
    assert evidence.warmup_source_coordinate_records is None
    assert evidence.source_coordinate_records is None
    assert evidence.distinct_source_traces is None
    assert evidence.cache_bust_records is None
    assert evidence.raw_records_artifact is None
    assert evidence.unavailable_dimensions == [
        *source.catalog.unavailable_dimensions,
        "source_coordinate_mapping",
        "cache_bust_observations",
        "warmup_source_coordinate_records",
    ]
