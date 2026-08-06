import json
from pathlib import Path
from typing import cast

import pytest
from inferlab_bench_runner.aiperf import (
    aiperf_config,
    inference_request_config,
    parse_speed_bench_report,
    run_speed_bench_reports,
    speed_bench_category,
)
from inferlab_measurement_sdk import (
    BenchClientRequest,
    CaseDeadline,
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
    assert tokenizer["name"] == "/models/dsv4"
    assert runtime["ui"] == "none"


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
        "format": "speed_bench_coding",
        "entries": 4,
        "sampling": "sequential",
    }
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    extra = cast(dict[str, object], endpoint["extra"])
    assert extra["min_tokens"] == 128
    assert endpoint["type"] == "chat"
    assert extra["max_tokens"] == 128
    assert "max_completion_tokens" not in extra


def test_speed_reports_use_pinned_aiperf_cli_and_exact_csv_cells(tmp_path: Path) -> None:
    aiperf = tmp_path / "aiperf"
    aiperf.write_text(
        """#!/bin/sh
metric=''
output=''
while [ \"$#\" -gt 0 ]; do
  case \"$1\" in
    --metric) metric=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    *) shift ;;
  esac
done
if [ \"$metric\" = accept_length ]; then value=2.34; else value=0.67; fi
printf 'Model,coding,Overall\\ndsv4,%s,%s\\n' \"$value\" \"$value\" > \"$output\"
""",
        encoding="utf-8",
    )
    aiperf.chmod(0o755)

    metrics, invocations, error = run_speed_bench_reports(
        speed_bench_request(tmp_path),
        [str(aiperf)],
        tmp_path,
        CaseDeadline(5.0),
    )

    assert error is None
    assert metrics == {"acceptance_length": 2.34, "acceptance_rate": 0.67}
    assert [item.purpose for item in invocations] == [
        "acceptance_length",
        "acceptance_rate",
    ]
    assert all(item.exit_code == 0 for item in invocations)
    assert all("speed-bench-report" in item.command for item in invocations)


def test_speed_report_category_follows_the_catalog_aiperf_format(tmp_path: Path) -> None:
    raw = speed_bench_request(tmp_path).model_dump(mode="json")
    definition = cast(dict[str, object], raw["definition"])
    source = cast(dict[str, object], definition["request_source"])
    catalog = cast(dict[str, object], source["catalog"])
    source["profile"] = "throughput_8k_mixed"
    catalog["profile"] = "throughput_8k_mixed"
    catalog["source"] = "throughput_8k"
    catalog["aiperf_format"] = "speed_bench_throughput_8k_mixed"
    catalog["configuration"] = "throughput_8k"
    catalog["filter"] = {"field": "category", "value": "mixed"}

    assert speed_bench_category(BenchClientRequest.model_validate(raw)) == "throughput_8k_mixed"


def test_speed_reports_attempt_both_native_metrics_after_one_report_fails(
    tmp_path: Path,
) -> None:
    aiperf = tmp_path / "aiperf"
    aiperf.write_text(
        """#!/bin/sh
metric=''
output=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --metric) metric=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    *) shift ;;
  esac
done
if [ "$metric" = accept_length ]; then exit 3; fi
printf 'Model,coding,Overall\\ndsv4,0.67,0.67\\n' > "$output"
""",
        encoding="utf-8",
    )
    aiperf.chmod(0o755)

    metrics, invocations, error = run_speed_bench_reports(
        speed_bench_request(tmp_path),
        [str(aiperf)],
        tmp_path,
        CaseDeadline(5.0),
    )

    assert metrics == {"acceptance_rate": 0.67}
    assert [item.exit_code for item in invocations] == [3, 0]
    assert error == "acceptance_length report exited with 3"


def test_speed_report_rejects_duplicate_model_rows_and_invalid_ranges(tmp_path: Path) -> None:
    report = tmp_path / "report.csv"
    report.write_text(
        "Model,coding,Overall\ndsv4,2.0,2.0\ndsv4,3.0,3.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one row"):
        parse_speed_bench_report(report, "dsv4", "coding", "acceptance_length")

    report.write_text("Model,coding,Overall\ndsv4,1.01,1.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        parse_speed_bench_report(report, "dsv4", "coding", "acceptance_rate")


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
