import json
from pathlib import Path

from inferlab_measurement_sdk import (
    BenchClientRequest,
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return " ".join(f"token{token_id}" for token_id in token_ids)


def resolved_prompt_input(value: dict[str, object]) -> dict[str, object]:
    prompt = dict(value)
    kind = prompt.get("kind")
    if not isinstance(kind, str):
        raise ValueError("test prompt kind must be a string")
    authority = {
        "flat": ("flat_prompt", "completions", "local_flat"),
        "rendered_chat": ("flat_prompt", "completions", "local_template"),
        "server_chat": ("structured_messages", "chat_completions", "server"),
    }.get(kind)
    if authority is None:
        raise ValueError(f"test prompt kind {kind!r} has no resolved authority")
    representation, route, rendering = authority
    prompt.update(
        request_representation=representation,
        route=route,
        rendering_authority=rendering,
    )
    return prompt


def request(
    tmp_path: Path,
    load_shape: dict[str, object],
    request_body: dict[str, object] | None = None,
    warmup_request_count: int = 0,
    output_tokens: int = 1000,
    request_slo: dict[str, float] | None = None,
    request_source: dict[str, object] | None = None,
    server_metrics: bool = False,
    artifact_level: str = "diagnostic",
) -> BenchClientRequest:
    effective_source = (
        dict(request_source)
        if request_source is not None
        else {
            "kind": "random",
            "input_tokens": 8000,
            "output_tokens": output_tokens,
            "prefix_sharing": None,
        }
    )
    prompt = effective_source.pop("prompt", {"kind": "server_chat"})
    if not isinstance(prompt, dict):
        raise ValueError("test prompt must be an object")
    effective_prompt = resolved_prompt_input(prompt)
    return BenchClientRequest.model_validate(
        {
            "protocol_version": "7",
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
                "request_source": effective_source,
                "prompt": effective_prompt,
                "server_metrics": server_metrics,
                "seed": 7,
                "request_body": request_body
                if request_body is not None
                else {
                    "temperature": 1.0,
                    "reasoning_effort": "high",
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                "request_slo": request_slo,
                "timeout_seconds": 120,
                "cache_start": "uncontrolled",
                "artifact_level": artifact_level,
            },
            "case": {
                "load_shape": load_shape,
                "request_count": 4,
                "warmup_request_count": warmup_request_count,
            },
            "case_budget_seconds": 120.0,
            "artifact_dir": str(tmp_path),
        }
    )


def dataset_request(tmp_path: Path, warmup_request_count: int = 0) -> BenchClientRequest:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        warmup_request_count=warmup_request_count,
    )
    raw = value.model_dump(mode="json")
    raw["definition"]["request_source"] = {
        "kind": "dataset",
        "dataset": "sharegpt",
        "profile": None,
        "max_input_tokens": 8192,
        "output_tokens": None,
        "catalog": {
            "dataset": "sharegpt",
            "profile": None,
            "source": "snapshot",
            "upstream_identity": "fixture@1:data.json",
            "url": "https://example.invalid/data.json",
            "sha256": "0" * 64,
            "source_format": "sharegpt-json-array-v1",
            "aiperf_format": "mooncake_trace",
            "configuration": None,
            "split": None,
            "filter": None,
            "license": "Apache-2.0",
            "cache_path": "/cache/source.json",
            "cache_state": "present",
            "materialization_identity": "sharegpt-single-request-v1",
            "provides_output_targets": True,
        },
    }
    population_path = tmp_path / "population.jsonl"
    population_path.parent.mkdir(parents=True, exist_ok=True)
    population_path.write_text(
        "".join(
            json.dumps({"session_id": f"inferlab-{index:08}"}) + "\n"
            for index in range(warmup_request_count + 4)
        ),
        encoding="utf-8",
    )
    raw["population"] = {
        "path": str(population_path),
        "evidence_path": str(population_path),
        "sha256": "1" * 64,
        "entries": warmup_request_count + 4,
        "tpot_applicable": True,
    }
    return BenchClientRequest.model_validate(raw)


def speed_bench_request(tmp_path: Path) -> BenchClientRequest:
    raw = dataset_request(tmp_path).model_dump(mode="json")
    raw["endpoint"]["server_metrics"] = {
        "path": "/metrics",
        "port_name": None,
        "url": "http://127.0.0.1:8000/metrics",
    }
    raw["definition"]["server_metrics"] = True
    raw["definition"]["request_source"] = {
        "kind": "dataset",
        "dataset": "speed_bench",
        "profile": "qualitative_coding",
        "max_input_tokens": 8192,
        "output_tokens": 128,
        "catalog": {
            "dataset": "speed_bench",
            "profile": "qualitative_coding",
            "source": "qualitative",
            "upstream_identity": "fixture@1:qualitative.parquet",
            "url": "https://example.invalid/qualitative.parquet",
            "sha256": "0" * 64,
            "source_format": "huggingface-parquet-v1",
            "aiperf_format": "speed_bench_coding",
            "configuration": "qualitative",
            "split": "test",
            "filter": {"field": "category", "value": "coding"},
            "license": "NVIDIA Evaluation Dataset License",
            "cache_path": "/cache/qualitative.parquet",
            "cache_state": "present",
            "materialization_identity": "speed-bench-first-turn-v1",
            "provides_output_targets": False,
        },
    }
    return BenchClientRequest.model_validate(raw)
