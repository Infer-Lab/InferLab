import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from inferlab_bench_runner import result_metrics
from inferlab_bench_runner.aiperf import (
    run_aiperf,
)
from inferlab_bench_runner.bench_client import main
from inferlab_bench_runner.execution import execute
from inferlab_bench_runner.result_metrics import normalize_summary, prompt_cache_evidence
from inferlab_bench_runner.result_policy import warmup_counts
from inferlab_bench_runner.result_records import request_counts
from inferlab_measurement_sdk import (
    BenchClientResult,
    CaseDeadline,
    ClientStatus,
    PromptCacheReadZeroRepresentation,
)

from .support import (
    request,
)


def install_fake_aiperf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, implementation: str
) -> Path:
    aiperf = tmp_path / "aiperf"
    aiperf.write_text(implementation, encoding="utf-8")
    aiperf.chmod(0o755)
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nshift 2\nexec "$(dirname "$0")/aiperf" "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(python))
    return python


def test_normalization_uses_the_versioned_aiperf_summary_mapping() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.12.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))

    assert summary["aiperf_version"] == "0.12.0"

    assert normalize_summary(summary, tpot_applicable=True) == {
        "request_throughput": 23.072953552748583,
        "output_throughput": 46.145907105497166,
        "total_token_throughput": 230.7295355274858,
        "mean_prompt_tokens": 8.0,
        "min_prompt_tokens": 8.0,
        "max_prompt_tokens": 8.0,
        "stddev_prompt_tokens": 0.0,
        "p50_prompt_tokens": 8.0,
        "p90_prompt_tokens": 8.0,
        "p95_prompt_tokens": 8.0,
        "p99_prompt_tokens": 8.0,
        "mean_request_latency_ms": 41.5939565,
        "min_request_latency_ms": 41.148568999999995,
        "max_request_latency_ms": 42.463107,
        "stddev_request_latency_ms": 0.5146282157759619,
        "p50_request_latency_ms": 41.382075,
        "p90_request_latency_ms": 42.1654524,
        "p95_request_latency_ms": 42.3142797,
        "p99_request_latency_ms": 42.43334154,
        "mean_ttft_ms": 41.5939565,
        "min_ttft_ms": 41.148568999999995,
        "max_ttft_ms": 42.463107,
        "stddev_ttft_ms": 0.5146282157759619,
        "p50_ttft_ms": 41.382075,
        "p90_ttft_ms": 42.1654524,
        "p95_ttft_ms": 42.3142797,
        "p99_ttft_ms": 42.43334154,
        "mean_tpot_ms": 0.0,
        "min_tpot_ms": 0.0,
        "max_tpot_ms": 0.0,
        "stddev_tpot_ms": 0.0,
        "p50_tpot_ms": 0.0,
        "p90_tpot_ms": 0.0,
        "p95_tpot_ms": 0.0,
        "p99_tpot_ms": 0.0,
    }

    summary["request_throughput"] = {"avg": math.inf}
    with pytest.raises(ValueError, match=r"request_throughput\.avg"):
        normalize_summary(summary, tpot_applicable=True)


def test_normalization_omits_tpot_for_prefill_only() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.12.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))
    del summary["inter_token_latency"]

    metrics = normalize_summary(summary, tpot_applicable=False)

    assert not any("tpot" in name for name in metrics)
    with pytest.raises(ValueError, match="inter_token_latency"):
        normalize_summary(summary, tpot_applicable=True)


def test_normalization_preserves_optional_weighted_cache_ratio() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.12.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))
    summary["overall_usage_prompt_cache_read_pct"] = {"unit": "%", "avg": 62.5}

    assert normalize_summary(summary, tpot_applicable=True)["prompt_cache_read_ratio"] == 0.625

    summary["overall_usage_prompt_cache_read_pct"] = {"unit": "%", "avg": 101.0}
    with pytest.raises(ValueError, match="overall_usage_prompt_cache_read_pct"):
        normalize_summary(summary, tpot_applicable=True)


def test_required_cache_evidence_uses_backend_usage_and_weighted_token_ratio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(
            json.dumps(
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "session_num": request_id,
                        "was_cancelled": False,
                    },
                    "error": None,
                    "metrics": {
                        "usage_prompt_tokens": {"value": prompt},
                        "usage_prompt_cache_read_tokens": {"value": cache},
                    },
                }
            )
            for request_id, prompt, cache in [(4, 10, 6), (9, 20, 8)]
        )
        + "\n",
        encoding="utf-8",
    )

    observed_distributions: list[tuple[str, list[int]]] = []

    def native_distribution(family: str, values: list[int]) -> SimpleNamespace:
        observed_distributions.append((family, values))
        if values == [6, 8]:
            return SimpleNamespace(
                avg=7.0, min=6.0, max=8.0, std=1.0, p50=7.0, p90=7.8, p95=7.9, p99=7.98
            )
        return SimpleNamespace(
            avg=8.0, min=4.0, max=12.0, std=4.0, p50=8.0, p90=11.2, p95=11.6, p99=11.92
        )

    monkeypatch.setattr(result_metrics, "_aiperf_metric_result", native_distribution)
    observations, metrics, error = prompt_cache_evidence(
        records, required=True, zero_representation=None
    )

    assert error is None
    assert [item.request_id for item in observations] == [4, 9]
    assert [item.uncached_prompt_tokens for item in observations] == [4, 12]
    assert metrics["mean_prompt_cache_read_tokens"] == 7.0
    assert metrics["p90_prompt_cache_read_tokens"] == pytest.approx(7.8)
    assert metrics["prompt_cache_read_ratio"] == pytest.approx(14 / 30)
    assert observed_distributions == [
        ("prompt_cache_read_tokens", [6, 8]),
        ("uncached_prompt_tokens", [4, 12]),
    ]


def test_required_cache_evidence_rejects_missing_backend_cache_usage(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "session_num": 1,
                    "was_cancelled": False,
                },
                "error": None,
                "metrics": {"usage_prompt_tokens": {"value": 10}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, _, error = prompt_cache_evidence(records, required=True, zero_representation=None)

    assert error is not None
    assert "omitted backend cache usage" in error


def test_declared_omitted_zero_cache_usage_normalizes_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "session_num": 3,
                    "was_cancelled": False,
                },
                "error": None,
                "metrics": {"usage_prompt_tokens": {"value": 10}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        result_metrics,
        "_aiperf_metric_result",
        lambda _family, values: SimpleNamespace(
            avg=float(values[0]),
            min=float(values[0]),
            max=float(values[0]),
            std=0.0,
            p50=float(values[0]),
            p90=float(values[0]),
            p95=float(values[0]),
            p99=float(values[0]),
        ),
    )

    observations, metrics, error = prompt_cache_evidence(
        records,
        required=True,
        zero_representation=PromptCacheReadZeroRepresentation.omitted,
    )

    assert error is None
    assert observations[0].cache_read_tokens == 0
    assert observations[0].uncached_prompt_tokens == 10
    assert metrics["prompt_cache_read_ratio"] == 0.0


def test_invalid_summary_preserves_native_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = install_fake_aiperf(tmp_path, monkeypatch, "#!/bin/sh\nprintf 'native output\\n'\n")
    (tmp_path / "inferlab-bench.json").write_text(
        '{"request_throughput":{"avg":"invalid"}}\n', encoding="utf-8"
    )
    (tmp_path / "inferlab-bench.jsonl").write_text(
        '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n',
        encoding="utf-8",
    )

    result = execute(request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1}))

    assert result.status == ClientStatus.failed
    assert result.completed_requests == 1
    assert result.failed_requests == 0
    assert result.native_exit_code == 0
    assert result.native_command[:3] == [
        str(python),
        "-m",
        "inferlab_bench_runner.aiperf_entrypoint",
    ]
    assert {artifact.name for artifact in result.raw_artifacts} >= {
        "aiperf_config",
        "aiperf_summary",
        "aiperf_records",
    }
    assert result.error is not None
    assert "request_throughput.avg" in result.error


def test_requested_server_metrics_fail_when_aiperf_omits_native_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_aiperf(tmp_path, monkeypatch, "#!/bin/sh\nexit 0\n")
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.12.0-summary.json"
    (tmp_path / "profile_export_aiperf.json").write_text(
        summary_fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )
    profiling_record = '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n'
    (tmp_path / "profile_export.jsonl").write_text(profiling_record * 4, encoding="utf-8")

    result = execute(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 1},
            server_metrics=True,
        )
    )

    assert result.status == ClientStatus.failed
    assert result.error is not None
    assert "server_metrics_export.json" in result.error
    assert {artifact.name for artifact in result.raw_artifacts} >= {
        "aiperf_profile_export",
        "aiperf_records",
    }


def test_request_counts_preserve_complete_records_before_a_partial_line(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n{"error":',
        encoding="utf-8",
    )

    completed, failed, error = request_counts(records)

    assert (completed, failed) == (1, 0)
    assert error is not None
    assert "line 2" in error


def test_warmup_counts_use_the_phase_tagged_normalized_records(tmp_path: Path) -> None:
    records = tmp_path / "raw.jsonl"
    records.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "conversation_id": "session_000000",
                        "was_cancelled": False,
                    },
                    "error": None,
                },
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "conversation_id": "session_000001",
                        "was_cancelled": False,
                    },
                    "error": {"message": "backend failed"},
                },
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "conversation_id": "session_000002",
                        "was_cancelled": True,
                    },
                    "error": None,
                },
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "conversation_id": "session_000003",
                        "was_cancelled": False,
                    },
                    "error": None,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    counts = warmup_counts(records, expected=4)

    assert counts.completed == 1
    assert counts.errored == 1
    assert counts.cancelled == 1
    assert counts.missing == 1
    assert counts.observed == 3
    assert counts.parse_error is None


def test_performance_artifact_level_runs_without_raw_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_aiperf(tmp_path, monkeypatch, "#!/bin/sh\nexit 0\n")
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.12.0-summary.json"
    (tmp_path / "inferlab-bench.json").write_text(
        summary_fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )
    profiling_record = '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n'
    warmup_record = (
        '{"metadata":{"benchmark_phase":"warmup","conversation_id":"session_000000",'
        '"was_cancelled":false},"error":null}\n'
    )
    (tmp_path / "inferlab-bench.jsonl").write_text(
        profiling_record * 4 + warmup_record * 2,
        encoding="utf-8",
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 2},
            warmup_request_count=2,
            artifact_level="performance",
        )
    )

    assert result.status == ClientStatus.succeeded
    assert result.completed_requests == 4
    assert result.failed_requests == 0
    artifact_names = {artifact.name for artifact in result.raw_artifacts}
    assert "aiperf_records" in artifact_names
    assert "aiperf_raw_records" not in artifact_names
    assert "aiperf_partial_raw_records" not in artifact_names
    config = cast(
        dict[str, object],
        json.loads((tmp_path / "aiperf-config.json").read_text(encoding="utf-8")),
    )
    benchmark = cast(dict[str, object], config["benchmark"])
    assert cast(dict[str, object], benchmark["artifacts"])["raw"] is False


def test_pinned_aiperf_native_warmup_qualification(tmp_path: Path) -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "aiperf-0.12.0-native-warmup-qualification.json"
    )
    fixture = cast(dict[str, object], json.loads(fixture_path.read_text(encoding="utf-8")))
    config = cast(dict[str, object], fixture["effective_config"])
    dataset = cast(dict[str, object], config["dataset"])
    warmup = cast(dict[str, object], config["warmup"])
    profiling = cast(dict[str, object], config["profiling"])
    native_result = cast(dict[str, object], fixture["native_result"])
    raw_records = cast(list[object], fixture["raw_records"])

    assert fixture["aiperf_version"] == "0.12.0"
    assert fixture["runner_version"] == "0.9.1"
    assert dataset == {"type": "synthetic", "entries": 6, "sampling": "sequential"}
    assert warmup == {"type": "concurrency", "concurrency": 1, "requests": 2}
    assert profiling == {"type": "concurrency", "concurrency": 1, "requests": 4}
    assert native_result == {
        "exit_code": 0,
        "summary_request_count": 4,
        "completed_requests": 4,
        "failed_requests": 0,
    }

    records = tmp_path / "raw.jsonl"
    records.write_text(
        "\n".join(json.dumps(record) for record in raw_records) + "\n",
        encoding="utf-8",
    )
    counts = warmup_counts(records, expected=2)

    assert (
        counts.completed,
        counts.errored,
        counts.cancelled,
        counts.missing,
        counts.observed,
        counts.parse_error,
    ) == (2, 0, 0, 0, 2, None)
    assert request_counts(records) == (4, 0, None)
    assert [
        cast(dict[str, object], cast(dict[str, object], record)["metadata"])["benchmark_phase"]
        for record in raw_records
    ] == ["warmup", "warmup", "profiling", "profiling", "profiling", "profiling"]


def test_incomplete_native_warmup_fails_with_phase_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_aiperf(tmp_path, monkeypatch, "#!/bin/sh\nexit 0\n")
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.12.0-summary.json"
    (tmp_path / "inferlab-bench.json").write_text(
        summary_fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )
    profiling_record = '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n'
    warmup_record = (
        '{"metadata":{"benchmark_phase":"warmup","conversation_id":"session_000000",'
        '"was_cancelled":false},"error":null}\n'
    )
    (tmp_path / "inferlab-bench.jsonl").write_text(
        profiling_record * 4 + warmup_record,
        encoding="utf-8",
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 2},
            warmup_request_count=2,
        )
    )

    assert result.status == ClientStatus.failed
    assert result.completed_requests == 4
    assert result.failed_requests == 0
    assert result.error is not None
    assert "expected=2" in result.error
    assert "completed=1" in result.error
    assert "missing=1" in result.error


def test_request_slo_uses_reconciled_native_records_for_ratio_and_goodput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_aiperf(tmp_path, monkeypatch, "#!/bin/sh\nexit 0\n")
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.12.0-summary.json"
    summary = cast(dict[str, object], json.loads(summary_fixture.read_text(encoding="utf-8")))
    summary["good_request_count"] = {"avg": 2.0, "unit": "requests"}
    (tmp_path / "inferlab-bench.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    records = []
    for session_num, good in enumerate([1, 1, 0, 0]):
        records.append(
            {
                "metadata": {
                    "session_num": session_num,
                    "benchmark_phase": "profiling",
                    "request_start_ns": 1_000_000_000 + session_num * 1_000_000_000,
                    "request_end_ns": 2_000_000_000 + session_num * 1_000_000_000,
                    "was_cancelled": False,
                },
                "metrics": {
                    "time_to_first_token": {"value": 100.0, "unit": "ms"},
                    "good_request_count": {"value": good, "unit": "requests"},
                },
                "error": None,
            }
        )
    (tmp_path / "inferlab-bench.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 1},
            request_slo={"ttft_ms": 800.0, "minimum_good_request_ratio": 0.5},
        )
    )

    assert result.status == ClientStatus.succeeded
    assert (result.completed_requests, result.failed_requests) == (4, 0)
    assert result.metrics["good_request_ratio"] == 0.5
    assert result.metrics["goodput"] == 0.5
    assert result.request_slo is not None
    assert result.request_slo.good_requests == 2
    assert result.request_slo.profiling_duration_seconds == 4.0
    assert result.request_slo.request_count_reconciled is True
    assert result.request_slo.native_aggregate_good_request_count == 2
    assert result.request_slo.native_aggregate_good_request_count_consistent is True


def test_complete_all_error_request_slo_is_service_quality_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_aiperf(tmp_path, monkeypatch, "#!/bin/sh\nexit 1\n")
    records = [
        {
            "metadata": {
                "session_num": session_num,
                "benchmark_phase": "profiling",
                "request_start_ns": 1_000_000_000 + session_num * 1_000_000_000,
                "request_end_ns": 2_000_000_000 + session_num * 1_000_000_000,
                "was_cancelled": False,
            },
            "metrics": {},
            "error": {"message": "backend overload"},
        }
        for session_num in range(4)
    ]
    (tmp_path / "inferlab-bench.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "request_rate_limited", "request_rate": 8.0, "burstiness": None},
            request_slo={"ttft_ms": 800.0, "minimum_good_request_ratio": 0.99},
        )
    )

    assert result.status == ClientStatus.succeeded
    assert (result.completed_requests, result.failed_requests) == (0, 4)
    assert result.metrics == {"good_request_ratio": 0.0, "goodput": 0.0}
    assert result.request_slo is not None
    assert result.request_slo.good_requests == 0
    assert result.native_exit_code == 1
    assert result.error is None


def test_nonzero_exit_with_only_cancelled_requests_is_not_the_inference_error_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_aiperf(tmp_path, monkeypatch, "#!/bin/sh\nexit 1\n")
    records = [
        {
            "metadata": {
                "session_num": session_num,
                "benchmark_phase": "profiling",
                "request_start_ns": 1_000_000_000 + session_num * 1_000_000_000,
                "request_end_ns": 2_000_000_000 + session_num * 1_000_000_000,
                "was_cancelled": True,
            },
            "metrics": {},
            "error": None,
        }
        for session_num in range(4)
    ]
    (tmp_path / "inferlab-bench.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "request_rate_limited", "request_rate": 8.0, "burstiness": None},
            request_slo={"ttft_ms": 800.0, "minimum_good_request_ratio": 0.99},
        )
    )

    assert result.status == ClientStatus.failed
    assert (result.completed_requests, result.failed_requests) == (0, 4)
    assert result.native_exit_code == 1


def test_aiperf_native_guard_consumes_the_case_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return -15 if self.terminated else None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return -15

        def kill(self) -> None:
            raise AssertionError("graceful termination should be sufficient")

    now = [10.0]
    process = FakeProcess()

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        assert command == ["aiperf"]
        assert kwargs == {"stdout": sys.stderr, "stderr": sys.stderr}
        return process

    monkeypatch.setattr("inferlab_bench_runner.aiperf.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "inferlab_bench_runner.aiperf.time.sleep",
        lambda duration: now.__setitem__(0, now[0] + duration),
    )
    monkeypatch.setattr(
        "inferlab_bench_runner.aiperf.subprocess.Popen",
        fake_popen,
    )
    deadline = CaseDeadline(0.1)

    exit_code, interrupted, timed_out = run_aiperf(["aiperf"], deadline)

    assert exit_code == -15
    assert interrupted is False
    assert timed_out is True
    assert process.terminated is True


def test_main_preserves_native_timeout_and_partial_warmup_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return -15 if self.terminated else None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return -15

        def kill(self) -> None:
            raise AssertionError("graceful termination should be sufficient")

    value = request(
        tmp_path / "artifacts",
        {"kind": "concurrency_limited", "concurrency": 2},
        warmup_request_count=2,
    ).model_copy(update={"case_budget_seconds": 0.1})
    input_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(value.model_dump_json(), encoding="utf-8")
    partial_dir = tmp_path / "artifacts" / "raw_records"
    partial_dir.mkdir(parents=True)
    (partial_dir / "raw_records_processor_qual.jsonl").write_text(
        '{"metadata":{"benchmark_phase":"warmup","conversation_id":"session_000000","was_cancelled":false},"error":null}\n',
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "inferlab-bench.jsonl").write_text(
        '{"metadata":{"benchmark_phase":"warmup","conversation_id":"session_000000","was_cancelled":false},"error":null}\n',
        encoding="utf-8",
    )

    now = [10.0]
    monkeypatch.setattr("inferlab_bench_runner.aiperf.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "inferlab_bench_runner.aiperf.time.sleep",
        lambda duration: now.__setitem__(0, now[0] + duration),
    )
    monkeypatch.setattr(
        "inferlab_bench_runner.aiperf.subprocess.Popen",
        lambda command, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench-client", "--input", str(input_path), "--output", str(output_path)],
    )

    assert main() == 0
    result = BenchClientResult.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert result.status == ClientStatus.failed
    assert result.native_command
    assert result.native_exit_code == -15
    assert {artifact.name for artifact in result.raw_artifacts} >= {
        "aiperf_config",
        "inference_request",
        "aiperf_partial_raw_records",
    }
    assert result.error is not None
    assert "measurement-case deadline" in result.error
    assert "expected=2" in result.error
    assert "completed=1" in result.error
