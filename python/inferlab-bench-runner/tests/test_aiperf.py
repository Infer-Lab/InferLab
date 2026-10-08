import json
import sys
import types
from pathlib import Path
from typing import cast

import pytest
from inferlab_bench_runner.aiperf import (
    aiperf_config,
    compute_speed_bench_acceptance,
    inference_request_config,
    speed_bench_category,
)
from inferlab_measurement_sdk import (
    BenchAcceptanceSource,
    BenchClientRequest,
)

from .support import (
    request,
    speed_bench_request,
)


def test_config_maps_one_concurrency_case_to_headless_aiperf(tmp_path: Path) -> None:
    config = aiperf_config(request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1}))
    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])
    tokenizer = cast(dict[str, object], benchmark["tokenizer"])
    runtime = cast(dict[str, object], benchmark["runtime"])

    endpoint = cast(dict[str, object], benchmark["endpoint"])
    timeout = endpoint.pop("timeout")
    assert isinstance(timeout, float)
    assert 0 < timeout <= 120
    assert endpoint == {
        "url": "http://127.0.0.1:8000",
        "path": "/v1/chat/completions",
        "type": "chat",
        "streaming": True,
        "useServerTokenCount": True,
        "extra": {
            "ignore_eos": True,
            "min_tokens": 1000,
            "n": 1,
            "stream_options": {"include_usage": True},
            "temperature": 1.0,
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    assert dataset["prompts"] == {"isl": 8000, "osl": 1000}
    assert dataset["entries"] == 4
    assert "warmup" not in benchmark
    assert benchmark["profiling"] == {
        "type": "concurrency",
        "concurrency": 1,
        "requests": 4,
    }
    assert tokenizer["name"] == "/models/deepseek-v4-flash"
    assert runtime["ui"] == "none"


def test_config_artifact_level_controls_raw_export(tmp_path: Path) -> None:
    diagnostic = request(tmp_path / "diagnostic", {"kind": "concurrency_limited", "concurrency": 1})
    diagnostic_benchmark = cast(dict[str, object], aiperf_config(diagnostic)["benchmark"])
    diagnostic_artifacts = cast(dict[str, object], diagnostic_benchmark["artifacts"])
    assert diagnostic_artifacts == {
        "dir": str(diagnostic.artifact_dir),
        "summary": ["json"],
        "records": ["jsonl"],
        "raw": True,
        "prefix": "inferlab-bench",
    }
    assert inference_request_config(diagnostic)["artifact_level"] == "diagnostic"

    performance = request(
        tmp_path / "performance",
        {"kind": "concurrency_limited", "concurrency": 1},
        artifact_level="performance",
    )
    performance_benchmark = cast(dict[str, object], aiperf_config(performance)["benchmark"])
    performance_artifacts = cast(dict[str, object], performance_benchmark["artifacts"])
    assert performance_artifacts["raw"] is False
    assert performance_artifacts["records"] == ["jsonl"]
    assert performance_artifacts["summary"] == ["json"]
    assert inference_request_config(performance)["artifact_level"] == "performance"


def test_server_side_chat_template_survives_aiperf_config_rendering(tmp_path: Path) -> None:
    template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        request_body={"chat_template": template},
    )

    benchmark = cast(dict[str, object], aiperf_config(value)["benchmark"])
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    extra = cast(dict[str, object], endpoint["extra"])
    assert extra["chat_template"] == "{{ " + json.dumps(template) + " }}"
    assert endpoint["type"] == "chat"
    assert inference_request_config(value)["effective_request_body"] == {
        "chat_template": template,
        "ignore_eos": True,
        "min_tokens": 1000,
        "n": 1,
        "stream_options": {"include_usage": True},
    }


def test_structured_messages_always_derive_the_chat_route(tmp_path: Path) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        request_body={},
    )

    config = aiperf_config(value)
    benchmark = cast(dict[str, object], config["benchmark"])
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert endpoint["url"] == "http://127.0.0.1:8000"
    assert endpoint["path"] == "/v1/chat/completions"
    assert endpoint["type"] == "chat"
    evidence = inference_request_config(value)
    assert evidence["selected_named_route"] == "chat_completions_path"


def test_server_metrics_opt_in_uses_the_resolved_endpoint_and_json_export(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 1},
            server_metrics=True,
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["serverMetrics"] == {
        "enabled": True,
        "urls": ["http://127.0.0.1:8000/metrics"],
        "formats": ["json"],
        "discovery": {"mode": "disabled"},
    }
    artifacts = cast(dict[str, object], benchmark["artifacts"])
    assert "prefix" not in artifacts


def test_server_metrics_can_use_a_separately_allocated_named_port(tmp_path: Path) -> None:
    raw = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        server_metrics=True,
    ).model_dump(mode="json")
    endpoint = cast(dict[str, object], raw["endpoint"])
    server_metrics = cast(dict[str, object], endpoint["server_metrics"])
    server_metrics["port_name"] = "prometheus"
    server_metrics["url"] = "http://127.0.0.1:9000/metrics"

    config = aiperf_config(BenchClientRequest.model_validate(raw))

    benchmark = cast(dict[str, object], config["benchmark"])
    inference_endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert inference_endpoint["url"] == "http://127.0.0.1:8000"
    assert benchmark["serverMetrics"] == {
        "enabled": True,
        "urls": ["http://127.0.0.1:9000/metrics"],
        "formats": ["json"],
        "discovery": {"mode": "disabled"},
    }


def test_server_metrics_aligns_a_v1_metrics_path_without_an_alternative_probe(
    tmp_path: Path,
) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        server_metrics=True,
    )
    raw = value.model_dump(mode="json")
    endpoint = cast(dict[str, object], raw["endpoint"])
    server_metrics = cast(dict[str, object], endpoint["server_metrics"])
    server_metrics["path"] = "/v1/metrics"
    server_metrics["url"] = "http://127.0.0.1:8000/v1/metrics"

    config = aiperf_config(BenchClientRequest.model_validate(raw))

    benchmark = cast(dict[str, object], config["benchmark"])
    aiperf_endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert aiperf_endpoint["url"] == "http://127.0.0.1:8000/v1"
    assert aiperf_endpoint["path"] == "/chat/completions"
    assert benchmark["serverMetrics"] == {
        "enabled": True,
        "urls": ["http://127.0.0.1:8000/v1/metrics"],
        "formats": ["json"],
        "discovery": {"mode": "disabled"},
    }


def test_chat_route_aligns_with_a_v1_metrics_path(tmp_path: Path) -> None:
    raw = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        server_metrics=True,
    ).model_dump(mode="json")
    endpoint = cast(dict[str, object], raw["endpoint"])
    server_metrics = cast(dict[str, object], endpoint["server_metrics"])
    server_metrics["path"] = "/v1/metrics"
    server_metrics["url"] = "http://127.0.0.1:8000/v1/metrics"
    raw["population"] = {
        "path": "/record/population.jsonl",
        "evidence_path": "/record/population-evidence.jsonl",
        "sha256": "1" * 64,
        "entries": 4,
        "tpot_applicable": True,
    }

    config = aiperf_config(BenchClientRequest.model_validate(raw))

    benchmark = cast(dict[str, object], config["benchmark"])
    aiperf_endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert aiperf_endpoint["url"] == "http://127.0.0.1:8000/v1"
    assert aiperf_endpoint["path"] == "/chat/completions"
    assert aiperf_endpoint["type"] == "chat"


def test_server_metrics_rejects_a_path_the_pinned_aiperf_cannot_address_exactly(
    tmp_path: Path,
) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        server_metrics=True,
    )
    raw = value.model_dump(mode="json")
    endpoint = cast(dict[str, object], raw["endpoint"])
    server_metrics = cast(dict[str, object], endpoint["server_metrics"])
    server_metrics["path"] = "/prometheus"
    server_metrics["url"] = "http://127.0.0.1:8000/prometheus"

    with pytest.raises(
        ValueError,
        match="pinned AIPerf cannot address the integration server metrics path exactly",
    ):
        aiperf_config(BenchClientRequest.model_validate(raw))


def test_speed_bench_uses_the_catalog_dataset_format_and_fixed_output_limit(
    tmp_path: Path,
) -> None:
    config = aiperf_config(speed_bench_request(tmp_path))

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["dataset"] == {
        "type": "file",
        "path": str(tmp_path / "population.jsonl"),
        "format": "mooncake_trace",
        "entries": 4,
        "sampling": "sequential",
    }
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    extra = cast(dict[str, object], endpoint["extra"])
    assert extra["min_tokens"] == 128
    assert endpoint["type"] == "chat"
    assert extra["max_tokens"] == 128
    assert "max_completion_tokens" not in extra


def _fake_speed_report(
    monkeypatch: pytest.MonkeyPatch,
    records: dict[str, dict[str, float]],
    summary: dict[str, float | None],
    server: dict[str, float | None],
) -> None:
    """Stand in for AIPerf's SPEED-Bench report module, keyed by metric."""
    module = types.ModuleType("aiperf.analysis.speed_bench_report")
    module.load_profile = lambda run_dir: {"profile": True}  # type: ignore[attr-defined]
    module.load_server_metrics = lambda run_dir: {"server": True}  # type: ignore[attr-defined]
    module.acceptance_from_records = (  # type: ignore[attr-defined]
        lambda run_dir, metric, category: records.get(metric, {})
    )
    module.extract_summary_acceptance = (  # type: ignore[attr-defined]
        lambda profile, metric: summary.get(metric)
    )
    module.extract_accept_length = lambda metrics: server.get("accept_length")  # type: ignore[attr-defined]
    module.extract_accept_rate = lambda metrics: server.get("accept_rate")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiperf.analysis.speed_bench_report", module)


def test_speed_acceptance_takes_the_first_aiperf_source_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_speed_report(
        monkeypatch,
        records={"accept_length": {"coding": 2.34}},
        summary={"accept_length": 9.0, "accept_rate": None},
        server={"accept_length": 9.0, "accept_rate": 0.67},
    )

    metrics, sources, error = compute_speed_bench_acceptance(
        speed_bench_request(tmp_path), tmp_path
    )

    assert error is None
    assert metrics == {"acceptance_length": 2.34, "acceptance_rate": 0.67}
    assert sources == {
        "acceptance_length": BenchAcceptanceSource.records,
        "acceptance_rate": BenchAcceptanceSource.server_metrics,
    }


def test_speed_acceptance_rejects_records_for_another_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_speed_report(
        monkeypatch,
        records={"accept_length": {"weka_main": 2.0}, "accept_rate": {"coding": 1.5}},
        summary={},
        server={},
    )

    metrics, sources, error = compute_speed_bench_acceptance(
        speed_bench_request(tmp_path), tmp_path
    )

    assert metrics == {} and sources == {}
    assert error is not None
    assert "not only 'coding'" in error
    assert "acceptance_rate is outside [0, 1]" in error


def test_speed_acceptance_category_follows_the_profile_filter(tmp_path: Path) -> None:
    raw = speed_bench_request(tmp_path).model_dump(mode="json")
    definition = cast(dict[str, object], raw["definition"])
    source = cast(dict[str, object], definition["request_source"])
    catalog = cast(dict[str, object], source["catalog"])
    source["profile"] = "throughput_8k_mixed"
    catalog["profile"] = "throughput_8k_mixed"
    catalog["source"] = "throughput_8k"
    catalog["configuration"] = "throughput_8k"
    catalog["filter"] = {"field": "category", "value": "mixed"}

    assert speed_bench_category(BenchClientRequest.model_validate(raw)) == "mixed"


def test_config_lowers_explicit_request_slo_to_aiperf_metric_tags(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 1},
            request_slo={
                "request_latency_ms": 5000.0,
                "ttft_ms": 800.0,
                "tpot_ms": 30.0,
                "minimum_good_request_ratio": 0.99,
            },
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["slos"] == {
        "request_latency": 5000.0,
        "time_to_first_token": 800.0,
        "inter_token_latency": 30.0,
    }


def test_request_preserves_both_named_workload_paths(tmp_path: Path) -> None:
    value = request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1})

    assert value.endpoint.completions_path == "/v1/completions"
    assert value.endpoint.chat_completions_path == "/v1/chat/completions"

    evidence = inference_request_config(value)
    assert evidence["selected_named_route"] == "chat_completions_path"
    assert evidence["effective_public_url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert evidence["effective_request_body"] == {
        "ignore_eos": True,
        "min_tokens": 1000,
        "n": 1,
        "stream_options": {"include_usage": True},
        "temperature": 1.0,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_request_evidence_preserves_an_overridden_aiperf_nested_default(
    tmp_path: Path,
) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        {"stream_options": {"include_usage": False, "opaque": "kept"}},
    )

    evidence = inference_request_config(value)

    assert evidence["aiperf_client_defaults"] == {
        "ignore_eos": True,
        "min_tokens": 1000,
        "n": 1,
        "stream_options": {"include_usage": True},
    }
    assert evidence["effective_request_body"] == {
        "ignore_eos": True,
        "min_tokens": 1000,
        "n": 1,
        "stream_options": {"include_usage": False, "opaque": "kept"},
    }
    assert evidence["replaced_defaults"] == [
        {
            "path": "stream_options.include_usage",
            "earlier": True,
            "earlier_authority": "pinned AIPerf chat endpoint",
            "replacement": False,
            "replacement_authority": "effective Bench definition request_body",
        }
    ]


def test_config_maps_vllm_burstiness_to_gamma_smoothness(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {
                "kind": "request_rate_limited",
                "request_rate": 3.5,
                "burstiness": 0.7,
            },
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["profiling"] == {
        "type": "gamma",
        "rate": 3.5,
        "smoothness": 0.7,
        "requests": 4,
    }


def test_config_maps_request_rate_without_burstiness_to_poisson(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {
                "kind": "request_rate_limited",
                "request_rate": 3.5,
                "burstiness": None,
            },
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["profiling"] == {
        "type": "poisson",
        "rate": 3.5,
        "requests": 4,
    }


def test_config_requires_exact_prefix_geometry_to_use_the_frozen_population(
    tmp_path: Path,
) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        request_body={},
        warmup_request_count=2,
        request_source={
            "kind": "random",
            "prompt": {"kind": "flat"},
            "input_tokens": 8000,
            "output_tokens": 1000,
            "prefix_sharing": {"shared_prefix_ratio": 0.75},
        },
    )

    with pytest.raises(ValueError, match="materialized population"):
        aiperf_config(value)


def test_config_lowers_weighted_exact_shapes_to_aiperf_sequence_distribution(
    tmp_path: Path,
) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 1},
            request_source={
                "kind": "random_mixture",
                "shapes": [
                    {"input_tokens": 1024, "output_tokens": 128, "weight": 7},
                    {"input_tokens": 8192, "output_tokens": 1024, "weight": 3},
                ],
                "total_weight": 10,
            },
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["dataset"] == {
        "type": "synthetic",
        "entries": 4,
        "randomSeed": 7,
        "sampling": "sequential",
        "prompts": {
            "isl": 1024,
            "osl": 128,
            "sequenceDistribution": [
                {"isl": 1024, "osl": 128, "probability": 70.0},
                {"isl": 8192, "osl": 1024, "probability": 30.0},
            ],
        },
    }
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    extra = cast(dict[str, object], endpoint["extra"])
    assert extra["ignore_eos"] is True
    assert "min_tokens" not in extra


def test_eos_output_stop_sends_the_limit_as_a_cap(tmp_path: Path) -> None:
    source: dict[str, object] = {
        "kind": "random",
        "input_tokens": 8000,
        "output_tokens": 1000,
        "output_stop": "eos",
        "prefix_sharing": None,
    }
    config = aiperf_config(
        request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1}, request_source=source)
    )
    endpoint = cast(dict[str, object], cast(dict[str, object], config["benchmark"])["endpoint"])
    extra = cast(dict[str, object], endpoint["extra"])

    assert "ignore_eos" not in extra
    assert "min_tokens" not in extra
