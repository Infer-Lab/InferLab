import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import cast

import pytest
from inferlab_bench_runner.aiperf import (
    aiperf_config,
    aiperf_session_population_layout,
    inference_request_config,
    parse_speed_bench_report,
    prepare_aiperf_execution,
    run_aiperf,
    run_speed_bench_reports,
    speed_bench_category,
)
from inferlab_bench_runner.aiperf_phase_barrier import (
    WarmupExpectation,
    await_capture_open,
    warmup_completion_error,
)
from inferlab_bench_runner.bench_client import main
from inferlab_bench_runner.execution import execute
from inferlab_bench_runner.population import prepare_population
from inferlab_bench_runner.population_sharegpt import (
    effective_inter_turn_delay,
    materialize_conversation,
)
from inferlab_bench_runner.population_synthetic import token_count
from inferlab_bench_runner.result_metrics import normalize_summary
from inferlab_bench_runner.result_policy import warmup_counts
from inferlab_bench_runner.result_population import population_identity_error
from inferlab_bench_runner.result_records import request_counts
from inferlab_bench_runner.result_sessions import session_result_evidence
from inferlab_measurement_sdk import (
    BenchClientRequest,
    BenchClientResult,
    BenchPopulationPreparationRequest,
    CaseDeadline,
    ClientStatus,
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return " ".join(f"token{token_id}" for token_id in token_ids)


class ExactChatTokenizer(FakeTokenizer):
    def get_chat_template(self, chat_template: str | None = None, tools: object = None) -> str:
        assert chat_template == "{{ messages }}"
        assert tools is None
        return chat_template

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert chat_template == "{{ messages }}"
        assert kwargs == {"enable_thinking": True}
        content_tokens = sum(len(message["content"].split()) for message in conversation)
        return list(range(content_tokens + 3))


class DefaultChatTokenizer(FakeTokenizer):
    def get_chat_template(self, chat_template: str | None = None, tools: object = None) -> str:
        assert chat_template is None
        assert tools is None
        return "{{ default_messages }}"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert chat_template == "{{ default_messages }}"
        assert kwargs == {}
        content_tokens = sum(len(message["content"].split()) for message in conversation)
        return list(range(content_tokens + 2))


class UnreachableChatTokenizer(FakeTokenizer):
    def get_chat_template(self, chat_template: str | None = None, tools: object = None) -> str:
        assert chat_template is None
        assert tools is None
        return "{{ unreachable }}"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert chat_template == "{{ unreachable }}"
        assert kwargs == {}
        content_tokens = sum(len(message["content"].split()) for message in conversation)
        return list(range(content_tokens * 2 + 2))


class NonRoundTripTokenizer(FakeTokenizer):
    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert token_ids
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return ""


class PeriodicCorpusTokenizer:
    """Model the short token period that exposed the 0.8.0 fallback regression."""

    _synthetic_corpus = (
        "Reproducible inference measurements need stable prompts, explicit evidence, "
        "and independently selected request shapes. "
    )
    _decoded_prefix = "synthetic_token_"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        words = text.split()
        if words and all(word.startswith(self._decoded_prefix) for word in words):
            return [int(word.removeprefix(self._decoded_prefix)) for word in words]
        if text and len(text) % len(self._synthetic_corpus) == 0:
            repetitions = len(text) // len(self._synthetic_corpus)
            if text == self._synthetic_corpus * repetitions:
                return list(range(20)) * repetitions
        return [int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big") for word in words]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return " ".join(f"{self._decoded_prefix}{token_id}" for token_id in token_ids)


def test_direct_file_entrypoint_resolves_the_staged_runner_package() -> None:
    package_root = Path(__file__).resolve().parents[1]
    runner_source = package_root / "src"
    measurement_source = package_root.parent / "inferlab-measurement-sdk" / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(runner_source), str(measurement_source)))
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            str(runner_source / "inferlab_bench_runner" / "bench_client.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_token_count_accepts_transformers_batch_encoding_shape() -> None:
    assert token_count({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3


def request(
    tmp_path: Path,
    load_shape: dict[str, object],
    request_body: dict[str, object] | None = None,
    warmup_request_count: int = 0,
    output_tokens: int = 1000,
    request_slo: dict[str, float] | None = None,
    request_source: dict[str, object] | None = None,
    server_metrics: bool = False,
) -> BenchClientRequest:
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
                "request_source": request_source
                if request_source is not None
                else {
                    "kind": "random",
                    "input_tokens": 8000,
                    "output_tokens": output_tokens,
                    "prefix_sharing": None,
                },
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
                "reset_prefix_cache": False,
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


def preparation_request(
    tmp_path: Path, source_path: Path, artifact_name: str = "population"
) -> BenchPopulationPreparationRequest:
    return BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": "7",
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "tokenizer_backend": "huggingface",
            "transformers_version": "5.12.1",
            "request_source": {
                "kind": "dataset",
                "dataset": "sharegpt",
                "profile": None,
                "max_input_tokens": 25,
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
                    "cache_path": str(source_path),
                    "cache_state": "present",
                    "materialization_identity": "sharegpt-single-request-v1",
                    "provides_output_targets": True,
                },
            },
            "source_path": str(source_path),
            "required_entries": 2,
            "seed": 7,
            "request_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "artifact_dir": str(tmp_path / artifact_name),
        }
    )


def random_preparation_request(
    tmp_path: Path,
    required_entries: int,
    *,
    request_source: dict[str, object],
    artifact_name: str = "population",
    request_body: dict[str, object] | None = None,
    seed: int = 7,
) -> BenchPopulationPreparationRequest:
    return BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": "7",
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "tokenizer_backend": "huggingface",
            "transformers_version": "5.12.1",
            "request_source": request_source,
            "source_path": None,
            "required_entries": required_entries,
            "seed": seed,
            "request_body": request_body or {},
            "artifact_dir": str(tmp_path / artifact_name),
        }
    )


def session_preparation_request(
    tmp_path: Path, source_path: Path, required_entries: int = 2
) -> BenchPopulationPreparationRequest:
    return BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": "7",
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "tokenizer_backend": "huggingface",
            "transformers_version": "5.12.1",
            "session_source": {
                "dataset": "sharegpt",
                "profile": None,
                "max_input_tokens": 25,
                "output_tokens": None,
                "inter_turn_delay_scale": 0.5,
                "max_inter_turn_delay_seconds": 3.0,
                "catalog": {
                    "dataset": "sharegpt",
                    "profile": None,
                    "source": "snapshot",
                    "upstream_identity": "fixture@1:data.json",
                    "url": "https://example.invalid/data.json",
                    "sha256": "0" * 64,
                    "source_format": "sharegpt-json-array-v1",
                    "configuration": None,
                    "split": None,
                    "filter": None,
                    "license": "Apache-2.0",
                    "cache_path": str(source_path),
                    "cache_state": "present",
                    "materialization_identity": "sharegpt-linear-session-v1",
                    "provides_output_targets": True,
                },
            },
            "source_path": str(source_path),
            "required_entries": required_entries,
            "seed": 7,
            "request_body": {},
            "artifact_dir": str(tmp_path / "sessions"),
        }
    )


def session_request(tmp_path: Path) -> BenchClientRequest:
    population_path = tmp_path / "session-population.jsonl"
    population_path.write_text(
        "".join(
            json.dumps(
                {
                    "type": "multi_turn",
                    "session_id": f"template-{index}",
                    "turns": [
                        {"type": "single_turn", "text": "one", "role": "user"},
                        {"type": "single_turn", "text": "two", "role": "user"},
                    ],
                }
            )
            + "\n"
            for index in range(9)
        ),
        encoding="utf-8",
    )
    return BenchClientRequest.model_validate(
        {
            "protocol_version": "7",
            "endpoint": {
                "protocol": "http",
                "host": "127.0.0.1",
                "port": 8000,
                "completions_path": "/v1/completions",
                "chat_completions_path": "/v1/chat/completions",
                "server_metrics": None,
            },
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "definition": {
                "session_source": {
                    "dataset": "sharegpt",
                    "profile": None,
                    "max_input_tokens": 8192,
                    "output_tokens": None,
                    "inter_turn_delay_scale": 1.0,
                    "max_inter_turn_delay_seconds": None,
                    "catalog": {
                        "dataset": "sharegpt",
                        "profile": None,
                        "source": "snapshot",
                        "upstream_identity": "fixture@1:data.json",
                        "url": "https://example.invalid/data.json",
                        "sha256": "0" * 64,
                        "source_format": "sharegpt-json-array-v1",
                        "configuration": None,
                        "split": None,
                        "filter": None,
                        "license": "Apache-2.0",
                        "cache_path": "/cache/data.json",
                        "cache_state": "present",
                        "materialization_identity": "sharegpt-linear-session-v1",
                        "provides_output_targets": True,
                    },
                },
                "server_metrics": False,
                "seed": 7,
                "request_body": {},
                "request_slo": None,
                "timeout_seconds": 120,
                "reset_prefix_cache": False,
            },
            "population": {
                "path": str(population_path),
                "sha256": "1" * 64,
                "entries": 9,
                "tpot_applicable": True,
                "session_templates": [
                    {"template_identity": f"template-{index}", "turn_count": 2}
                    for index in range(9)
                ],
            },
            "case": {
                "load_shape": {"kind": "concurrency_limited", "concurrency": 2},
                "request_count": 12,
                "warmup_request_count": 4,
                "session_count": 6,
                "warmup_session_count": 2,
            },
            "case_budget_seconds": 120.0,
            "artifact_dir": str(tmp_path),
        }
    )


def test_session_population_freezes_linear_user_turns_without_source_answers(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sharegpt.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "id": f"conversation-{index}",
                    "conversations": [
                        {"from": "human", "value": "first question"},
                        {"from": "gpt", "value": "held out first answer"},
                        {"from": "human", "value": "second question"},
                        {"from": "gpt", "value": "held out second answer"},
                    ],
                }
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )

    result = prepare_population(session_preparation_request(tmp_path, source_path), FakeTokenizer())

    assert result.status == ClientStatus.succeeded
    assert result.population is not None
    assert len(result.population.session_templates) == 2
    assert all(template.turn_count == 2 for template in result.population.session_templates)
    rows = [
        json.loads(line)
        for line in Path(result.population.path).read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["type"] == "multi_turn" for row in rows)
    assert all(len(row["turns"]) == 2 for row in rows)
    assert all(turn["role"] == "user" for row in rows for turn in row["turns"])
    assert all("held out" not in turn["text"] for row in rows for turn in row["turns"])
    assert all(turn["delay"] == 0.0 for row in rows for turn in row["turns"])
    assert result.evidence_path == result.population.path
    assert all(row["_inferlab"]["inter_turn_delay_scale"] == 0.5 for row in rows)
    assert all(
        turn["output_limit_provenance"] == "target_derived"
        for row in rows
        for turn in row["_inferlab"]["turns"]
    )


def test_session_population_identity_includes_effective_delay_controls(tmp_path: Path) -> None:
    source_path = tmp_path / "sharegpt.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "id": "conversation-1",
                    "conversations": [
                        {"from": "human", "value": "first question"},
                        {"from": "gpt", "value": "first answer"},
                        {"from": "human", "value": "second question"},
                        {"from": "gpt", "value": "second answer"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    baseline_request = session_preparation_request(tmp_path, source_path, required_entries=1)
    baseline_request = baseline_request.model_copy(
        update={"artifact_dir": str(tmp_path / "baseline")}
    )
    baseline = prepare_population(baseline_request, FakeTokenizer())

    assert baseline_request.session_source is not None
    changed_source = baseline_request.session_source.model_copy(
        update={"inter_turn_delay_scale": 0.25}
    )
    changed_request = baseline_request.model_copy(
        update={
            "session_source": changed_source,
            "artifact_dir": str(tmp_path / "changed"),
        }
    )
    changed = prepare_population(changed_request, FakeTokenizer())

    assert baseline.population is not None
    assert changed.population is not None
    assert baseline.population.sha256 != changed.population.sha256


def test_effective_inter_turn_delay_scales_then_caps_source_think_time() -> None:
    assert effective_inter_turn_delay(8.0, 0.5, 3.0) == 3.0
    assert effective_inter_turn_delay(8.0, 0.0, None) == 0.0


def test_session_native_config_uses_aiperf_multi_turn_chat_execution(tmp_path: Path) -> None:
    config = aiperf_config(session_request(tmp_path))

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["dataset"] == {
        "type": "file",
        "path": str(tmp_path / "session-population.jsonl"),
        "format": "multi_turn",
        "entries": 9,
        "sampling": "sequential",
    }
    assert benchmark["profiling"] == {
        "type": "concurrency",
        "concurrency": 2,
        "sessions": 6,
    }
    assert benchmark["warmup"] == {
        "type": "concurrency",
        "concurrency": 2,
        "sessions": 2,
    }
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert endpoint["path"] == "/v1/chat/completions"
    assert endpoint["type"] == "chat"


def test_session_population_layout_reserves_positive_warmup_terminal_prefetch() -> None:
    assert aiperf_session_population_layout(0, 6) == (0, 6)
    assert aiperf_session_population_layout(2, 6) == (3, 9)


def test_session_result_reconciles_runtime_sessions_turns_and_delays(tmp_path: Path) -> None:
    value = session_request(tmp_path)
    raw_path = tmp_path / "session-raw.jsonl"
    records_path = tmp_path / "session-records.jsonl"
    raw_lines: list[str] = []
    record_lines: list[str] = []
    timing_sequence = 0
    for phase, start, count in (("warmup", 0, 2), ("profiling", 3, 6)):
        for phase_session in range(count):
            template_index = start + phase_session
            runtime_id = f"runtime-{phase}-{phase_session}"
            previous_end = 0
            for turn_index in range(2):
                native_request = phase_session * 2 + turn_index
                request_start = 1_000_000_000 + timing_sequence * 10_000_000
                if previous_end:
                    request_start = previous_end
                request_end = request_start + 5_000_000
                messages = [{"role": "user", "content": "one"}]
                if turn_index == 1:
                    messages.extend(
                        [
                            {"role": "assistant", "content": "live answer"},
                            {"role": "user", "content": "two"},
                        ]
                    )
                if phase == "warmup":
                    messages.insert(0, {"role": "system", "content": "warmup"})
                content_tokens = sum(len(message["content"].split()) for message in messages)
                observed_prompt_tokens = content_tokens + 3
                metadata = {
                    "benchmark_phase": phase,
                    "conversation_id": f"template-{template_index}",
                    "x_correlation_id": runtime_id,
                    "session_num": native_request,
                    "turn_index": turn_index,
                    "request_start_ns": request_start,
                    "request_end_ns": request_end,
                    "was_cancelled": False,
                }
                raw_lines.append(
                    json.dumps(
                        {
                            "metadata": metadata,
                            "payload": {"messages": messages},
                            "responses": [
                                {
                                    "perf_ns": request_end,
                                    "text": json.dumps(
                                        {
                                            "object": "chat.completion",
                                            "choices": [
                                                {
                                                    "message": {
                                                        "role": "assistant",
                                                        "content": "live answer",
                                                    }
                                                }
                                            ],
                                            "usage": {
                                                "prompt_tokens": observed_prompt_tokens,
                                                "completion_tokens": 2,
                                            },
                                        }
                                    ),
                                }
                            ],
                            "status": 200,
                            "error": None,
                        }
                    )
                )
                if phase == "profiling":
                    record_lines.append(json.dumps({"metadata": metadata, "metrics": {}}))
                timing_sequence += 1
                previous_end = request_end
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    records_path.write_text("\n".join(record_lines) + "\n", encoding="utf-8")

    evidence, error = session_result_evidence(value, records_path, raw_path, FakeTokenizer())

    assert error is None
    assert evidence.warmup.planned_sessions == 2
    assert evidence.warmup.succeeded_sessions == 2
    assert evidence.profiling.planned_sessions == 6
    assert evidence.profiling.completed_requests == 12
    assert len(evidence.sessions) == 8
    assert len(evidence.turns) == 16
    assert evidence.turns[0].pre_template_content_tokens == 2
    assert evidence.turns[0].observed_prompt_tokens == 5
    assert evidence.turns[1].pre_template_content_tokens == 5
    assert evidence.turns[1].observed_prompt_tokens == 8
    assert evidence.turns[1].preceding_native_session_num == 0
    assert evidence.turns[1].inter_turn_delay_reconciled is True
    assert evidence.turns[4].phase == "profiling"
    assert evidence.turns[4].native_session_num == 0
    assert evidence.population_slice_reconciled is True
    assert evidence.counts_reconciled is True

    duplicated_runtime_id = "runtime-warmup-0"
    duplicate_lines: list[str] = []
    for line in raw_lines:
        record = cast(dict[str, object], json.loads(line))
        metadata = cast(dict[str, object], record["metadata"])
        if (
            metadata["benchmark_phase"] == "profiling"
            and metadata["conversation_id"] == "template-3"
        ):
            metadata["x_correlation_id"] = duplicated_runtime_id
        duplicate_lines.append(json.dumps(record))
    raw_path.write_text("\n".join(duplicate_lines) + "\n", encoding="utf-8")

    duplicate_evidence, duplicate_error = session_result_evidence(
        value, records_path, raw_path, FakeTokenizer()
    )

    assert duplicate_error is not None
    assert "duplicates runtime session identity" in duplicate_error
    assert duplicate_evidence.sessions_reconciled is False


def test_session_result_preserves_native_continuation_after_a_failed_turn(
    tmp_path: Path,
) -> None:
    raw = session_request(tmp_path).model_dump(mode="json")
    case = cast(dict[str, object], raw["case"])
    case.update(
        {
            "request_count": 2,
            "warmup_request_count": 0,
            "session_count": 1,
            "warmup_session_count": 0,
        }
    )
    value = BenchClientRequest.model_validate(raw)
    raw_path = tmp_path / "failed-session-raw.jsonl"
    records_path = tmp_path / "failed-session-records.jsonl"
    records = []
    for turn_index in range(2):
        metadata = {
            "benchmark_phase": "profiling",
            "conversation_id": "template-0",
            "x_correlation_id": "runtime-failed",
            "session_num": turn_index,
            "turn_index": turn_index,
            "request_start_ns": 1_000_000_000 + turn_index * 10_000_000,
            "request_end_ns": 1_005_000_000 + turn_index * 10_000_000,
            "was_cancelled": False,
        }
        records.append(
            {
                "metadata": metadata,
                "payload": {"messages": [{"role": "user", "content": "one"}]},
                "responses": [],
                "status": 500 if turn_index == 0 else 200,
                "error": {"message": "backend failed"} if turn_index == 0 else None,
            }
        )
    raw_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    records_path.write_text(
        "\n".join(json.dumps({"metadata": record["metadata"], "metrics": {}}) for record in records)
        + "\n",
        encoding="utf-8",
    )

    evidence, error = session_result_evidence(value, records_path, raw_path, FakeTokenizer())

    assert error is not None
    assert "contains failed sessions" in error
    assert evidence.profiling.failed_sessions == 1
    assert evidence.turn_order_reconciled is True
    assert evidence.sessions[0].failure_classification == "transport_error"
    assert evidence.turns[1].post_failure_continuation is True
    assert evidence.turns[1].preceding_native_session_num is None
    assert evidence.turns[1].preceding_terminal_response_receipt_ns is None
    assert evidence.turns[1].inter_turn_delay_reconciled is None


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


def test_materialization_rolls_back_a_complete_trailing_exchange() -> None:
    entry, reason = materialize_conversation(
        {
            "id": "conversation-1",
            "conversations": [
                {"from": "human", "value": "first question"},
                {"from": "gpt", "value": "first answer"},
                {"from": "human", "value": "second question"},
                {"from": "gpt", "value": "second answer"},
            ],
        },
        0,
        FakeTokenizer(),
        3,
        None,
    )

    assert reason is None
    assert entry is not None
    assert entry.messages == [{"role": "user", "content": "first question"}]
    assert entry.target == "first answer"
    assert entry.kept_messages == 2
    assert entry.removed_messages == 2
    assert entry.input_tokens == 2
    assert entry.output_tokens == 2


def test_prepare_dataset_freezes_one_deterministic_population(tmp_path: Path) -> None:
    source_path = tmp_path / "sharegpt.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "id": f"conversation-{index}",
                    "conversations": [
                        {"from": "human", "value": f"question {index}"},
                        {"from": "gpt", "value": f"answer number {index}"},
                    ],
                }
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )

    first = prepare_population(preparation_request(tmp_path, source_path), FakeTokenizer())
    second = prepare_population(
        preparation_request(tmp_path, source_path, "population-again"), FakeTokenizer()
    )

    assert first.status == ClientStatus.succeeded
    assert first.population is not None
    assert second.population is not None
    assert first.population.sha256 == second.population.sha256
    assert first.population.entries == 2
    assert first.admitted_entries == 4
    assert first.ineligible_entries == 0
    rows = [json.loads(line) for line in Path(first.population.path).read_text().splitlines()]
    assert len(rows) == 2
    assert len({row["messages"][0]["content"] for row in rows}) == 2
    assert all("text_input" not in row for row in rows)
    assert all(row["output_length"] == 3 for row in rows)
    assert all(row["extra"]["min_tokens"] == 3 for row in rows)


def test_uniform_selectors_freeze_a_prefix_stable_population(tmp_path: Path) -> None:
    source: dict[str, object] = {
        "kind": "random",
        "input_tokens": {"kind": "inclusive_uniform", "min": 7, "max": 11},
        "output_tokens": {"kind": "inclusive_uniform", "min": 3, "max": 5},
        "prefix_sharing": None,
    }
    first = prepare_population(
        random_preparation_request(tmp_path, 4, request_source=source, artifact_name="first"),
        FakeTokenizer(),
    )
    larger = prepare_population(
        random_preparation_request(tmp_path, 8, request_source=source, artifact_name="larger"),
        FakeTokenizer(),
    )

    assert first.status == ClientStatus.succeeded
    assert larger.status == ClientStatus.succeeded
    assert first.evidence_path is not None
    assert larger.evidence_path is not None
    first_rows = [json.loads(line) for line in Path(first.evidence_path).read_text().splitlines()]
    larger_rows = [json.loads(line) for line in Path(larger.evidence_path).read_text().splitlines()]
    assert larger_rows[:4] == first_rows
    assert all(7 <= row["selected_prompt_tokens"] <= 11 for row in larger_rows)
    assert all(3 <= row["selected_output_tokens"] <= 5 for row in larger_rows)
    assert all(
        row["pre_template_content_tokens"] == row["selected_prompt_tokens"] for row in larger_rows
    )
    assert all(row["prompt_token_targeting"] == "fallback" for row in larger_rows)


def test_synthetic_population_preserves_structured_messages_and_configured_isl(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            2,
            request_source={
                "kind": "random",
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
        ),
        FakeTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.population is not None
    assert result.evidence_path is not None
    population = [
        json.loads(line) for line in Path(result.population.path).read_text().splitlines()
    ]
    evidence = [json.loads(line) for line in Path(result.evidence_path).read_text().splitlines()]
    assert all("messages" in row and "text_input" not in row for row in population)
    assert all(row["selected_prompt_tokens"] == 8 for row in evidence)
    assert all(
        row["messages"] == population[index]["messages"] for index, row in enumerate(evidence)
    )
    assert all("rendered_prompt" not in row for row in evidence)


def test_synthetic_population_targets_the_complete_local_chat_projection(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            2,
            request_source={
                "kind": "random",
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
            request_body={
                "chat_template": "{{ messages }}",
                "chat_template_kwargs": {"enable_thinking": True},
            },
        ),
        ExactChatTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 2
    assert result.prompt_token_targeting.fallback_entries == 0
    assert result.prompt_token_targeting.fallback_reasons == {}
    assert result.prompt_token_targeting.selected_prompt_tokens.minimum == 8
    assert result.prompt_token_targeting.pre_template_content_tokens.minimum == 5
    assert result.prompt_token_targeting.projection_template is not None
    assert result.prompt_token_targeting.projection_template.source == "request_body"
    assert result.prompt_token_targeting.projection_template.content == "{{ messages }}"
    assert (
        result.prompt_token_targeting.projection_template.sha256
        == hashlib.sha256(b"{{ messages }}").hexdigest()
    )
    assert result.evidence_path is not None
    evidence = [json.loads(line) for line in Path(result.evidence_path).read_text().splitlines()]
    assert all(row["selected_prompt_tokens"] == 8 for row in evidence)
    assert all(row["pre_template_content_tokens"] == 5 for row in evidence)
    assert all(row["locally_predicted_prompt_tokens"] == 8 for row in evidence)
    assert all(row["prompt_token_targeting"] == "exact" for row in evidence)
    assert all(row["prompt_token_fallback_reason"] is None for row in evidence)


def test_synthetic_population_uses_the_tokenizer_default_chat_template(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            1,
            request_source={
                "kind": "random",
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
        ),
        DefaultChatTokenizer(),
    )

    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 1
    assert result.prompt_token_targeting.pre_template_content_tokens.minimum == 6
    assert result.prompt_token_targeting.projection_template is not None
    assert result.prompt_token_targeting.projection_template.source == "tokenizer_default"
    assert result.prompt_token_targeting.projection_template.content == "{{ default_messages }}"


def test_synthetic_population_records_unmodified_fallback_without_a_template_projection(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            1,
            request_source={
                "kind": "random",
                "input_tokens": 8,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
        ),
        FakeTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 0
    assert result.prompt_token_targeting.fallback_entries == 1
    assert result.prompt_token_targeting.fallback_reasons == {
        "chat_template_resolution_unavailable": 1
    }
    assert result.prompt_token_targeting.projection_template is None
    assert result.evidence_path is not None
    evidence = json.loads(Path(result.evidence_path).read_text())
    assert evidence["selected_prompt_tokens"] == 8
    assert evidence["pre_template_content_tokens"] == 8
    assert evidence["locally_predicted_prompt_tokens"] is None
    assert evidence["prompt_token_targeting"] == "fallback"
    assert evidence["prompt_token_fallback_reason"] == "chat_template_resolution_unavailable"


@pytest.mark.parametrize(
    "request_source",
    [
        {
            "kind": "random",
            "input_tokens": 8192,
            "output_tokens": 1,
            "prefix_sharing": None,
        },
        {
            "kind": "random_mixture",
            "shapes": [
                {"input_tokens": 8192, "output_tokens": 1, "weight": 1},
            ],
            "total_weight": 1,
        },
    ],
    ids=["random", "random-mixture"],
)
def test_fallback_population_keeps_long_independent_prompts_distinct(
    tmp_path: Path,
    request_source: dict[str, object],
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            4,
            request_source=request_source,
            seed=0,
        ),
        PeriodicCorpusTokenizer(),
    )

    assert result.population is not None
    population = [
        json.loads(line) for line in Path(result.population.path).read_text().splitlines()
    ]
    contents = [row["messages"][0]["content"] for row in population]
    assert len(set(contents)) == 4
    assert all(len(content.split()) == 8192 for content in contents)
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 0
    assert result.prompt_token_targeting.fallback_entries == 4
    assert result.prompt_token_targeting.fallback_reasons == {
        "chat_template_resolution_unavailable": 4
    }


def test_synthetic_population_keeps_unadjusted_content_when_exact_target_is_unreachable(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            1,
            request_source={
                "kind": "random",
                "input_tokens": 9,
                "output_tokens": 4,
                "prefix_sharing": None,
            },
        ),
        UnreachableChatTokenizer(),
    )

    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.exact_entries == 0
    assert result.prompt_token_targeting.fallback_entries == 1
    assert result.prompt_token_targeting.fallback_reasons == {"exact_prompt_length_unreachable": 1}
    assert result.prompt_token_targeting.projection_template is not None
    assert result.evidence_path is not None
    evidence = json.loads(Path(result.evidence_path).read_text())
    assert evidence["pre_template_content_tokens"] == 9
    assert evidence["locally_predicted_prompt_tokens"] == 20
    assert evidence["prompt_token_targeting"] == "fallback"


def test_synthetic_population_fails_when_the_unadjusted_content_cannot_be_constructed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="could not round-trip a synthetic user-content of 9 tokens",
    ):
        prepare_population(
            random_preparation_request(
                tmp_path,
                1,
                request_source={
                    "kind": "random",
                    "input_tokens": 9,
                    "output_tokens": 4,
                    "prefix_sharing": None,
                },
            ),
            NonRoundTripTokenizer(),
        )


def test_exact_targeting_keeps_the_shared_prefix_and_adjusts_only_the_user_suffix(
    tmp_path: Path,
) -> None:
    result = prepare_population(
        random_preparation_request(
            tmp_path,
            2,
            request_source={
                "kind": "random",
                "input_tokens": 12,
                "output_tokens": 4,
                "prefix_sharing": {
                    "shared_prefix_ratio": 0.5,
                    "shared_prefix_tokens": 6,
                    "unique_suffix_tokens": 6,
                },
            },
            request_body={
                "chat_template": "{{ messages }}",
                "chat_template_kwargs": {"enable_thinking": True},
            },
        ),
        ExactChatTokenizer(),
    )

    assert result.status == ClientStatus.succeeded
    assert result.population is not None
    population = [
        json.loads(line) for line in Path(result.population.path).read_text().splitlines()
    ]
    assert len({row["messages"][0]["content"] for row in population}) == 1
    assert all(len(row["messages"][0]["content"].split()) == 6 for row in population)
    assert all(len(row["messages"][1]["content"].split()) == 3 for row in population)
    assert len({row["messages"][1]["content"] for row in population}) == 2
    assert result.prompt_token_targeting is not None
    assert result.prompt_token_targeting.pre_template_content_tokens.minimum == 9
    assert result.prompt_token_targeting.exact_entries == 2


def test_speed_bench_materialization_filters_without_replacement_and_keeps_only_first_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "qualitative.parquet"
    source_path.write_bytes(b"parquet fixture boundary")
    rows = [
        {
            "question_id": f"{index:032x}",
            "category": category,
            "turns": ["", f"first turn {index}", f"later turn {index}"],
        }
        for index, category in enumerate(["coding", "coding", "math", "coding"])
    ]
    monkeypatch.setattr(
        "inferlab_bench_runner.population._iter_parquet_rows",
        lambda _path: iter(rows),
    )
    request_value = BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": "7",
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "tokenizer_backend": "huggingface",
            "transformers_version": "5.12.1",
            "request_source": {
                "kind": "dataset",
                "dataset": "speed_bench",
                "profile": "qualitative_coding",
                "max_input_tokens": 100,
                "output_tokens": 16,
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
                    "cache_path": str(source_path),
                    "cache_state": "present",
                    "materialization_identity": "speed-bench-first-turn-v1",
                    "provides_output_targets": False,
                },
            },
            "source_path": str(source_path),
            "required_entries": 2,
            "seed": 7,
            "request_body": {},
            "artifact_dir": str(tmp_path / "speed-population"),
        }
    )

    result = prepare_population(request_value, FakeTokenizer())

    assert result.status == ClientStatus.succeeded
    assert result.population is not None
    assert result.evidence_path is not None
    population = [
        json.loads(line) for line in Path(result.population.path).read_text().splitlines()
    ]
    evidence = [json.loads(line) for line in Path(result.evidence_path).read_text().splitlines()]
    assert len({row["question_id"] for row in population}) == 2
    assert all(row["category"] == "coding" for row in population)
    assert all(len(row["messages"]) == 1 for row in population)
    assert all("first turn" in row["messages"][0]["content"] for row in population)
    assert all(row["first_user_turn_index"] == 1 for row in evidence)
    assert all(row["later_turn_count"] == 1 for row in evidence)
    assert all(row["held_out_target"] is None for row in evidence)
    assert all(
        row["messages"] == population[index]["messages"] for index, row in enumerate(evidence)
    )
    assert all("rendered_prompt" not in row for row in evidence)


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


def test_config_maps_native_warmup_before_the_concurrency_profile(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 2},
            warmup_request_count=2,
        )
    )
    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])

    assert dataset["entries"] == 6
    assert dataset["sampling"] == "sequential"
    assert benchmark["warmup"] == {
        "type": "concurrency",
        "concurrency": 2,
        "requests": 2,
    }
    assert benchmark["profiling"] == {
        "type": "concurrency",
        "concurrency": 2,
        "requests": 4,
    }


def test_profile_command_uses_the_release_owned_aiperf_entrypoint(tmp_path: Path) -> None:
    prepared = prepare_aiperf_execution(
        request(tmp_path, {"kind": "concurrency_limited", "concurrency": 2}),
        CaseDeadline(5.0),
    )

    assert prepared.command[:3] == [
        sys.executable,
        "-m",
        "inferlab_bench_runner.aiperf_entrypoint",
    ]
    assert prepared.command[3:] == ["profile", "--config", str(prepared.config_path)]


def test_warmup_gate_requires_the_native_request_phase_to_drain_without_errors() -> None:
    complete = {
        "final_requests_sent": 2,
        "final_requests_completed": 2,
        "final_requests_cancelled": 0,
        "final_request_errors": 0,
        "final_sent_sessions": 2,
        "final_completed_sessions": 2,
        "final_cancelled_sessions": 0,
    }

    assert warmup_completion_error(WarmupExpectation(requests=2, sessions=None), complete) is None
    assert "errors" in (
        warmup_completion_error(
            WarmupExpectation(requests=2, sessions=None),
            {**complete, "final_request_errors": 1},
        )
        or ""
    )


def test_warmup_gate_requires_complete_native_sessions() -> None:
    incomplete = {
        "final_requests_sent": 4,
        "final_requests_completed": 4,
        "final_requests_cancelled": 0,
        "final_request_errors": 0,
        "final_sent_sessions": 2,
        "final_completed_sessions": 1,
        "final_cancelled_sessions": 0,
    }

    error = warmup_completion_error(WarmupExpectation(requests=None, sessions=2), incomplete)

    assert error is not None
    assert "sessions" in error


def test_profile_barrier_waits_for_capture_open_acknowledgement() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    observed: list[bytes] = []

    def acknowledge() -> None:
        connection, _ = listener.accept()
        with connection:
            observed.append(connection.makefile("rb").readline())
            connection.sendall(b"capture-open\n")
        listener.close()

    server = threading.Thread(target=acknowledge)
    server.start()
    await_capture_open(f"{host}:{port}")
    server.join()

    assert observed == [b"profiling-ready\n"]


def test_config_consumes_a_frozen_dataset_population_sequentially(tmp_path: Path) -> None:
    config = aiperf_config(dataset_request(tmp_path))

    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert dataset == {
        "type": "file",
        "path": str(tmp_path / "population.jsonl"),
        "format": "mooncake_trace",
        "entries": 4,
        "sampling": "sequential",
    }
    assert "min_tokens" not in cast(dict[str, object], endpoint["extra"])


def test_native_request_identities_reconcile_to_the_population_slices(tmp_path: Path) -> None:
    profiling_path = tmp_path / "profiling.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    profiling = [
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "session_num": index,
                "conversation_id": f"inferlab-{index + 2:08}",
            }
        }
        for index in range(4)
    ]
    warmup = [
        {
            "metadata": {
                "benchmark_phase": "warmup",
                "session_num": index,
                "conversation_id": f"inferlab-{index:08}",
            }
        }
        for index in range(2)
    ]
    profiling_path.write_text(
        "\n".join(json.dumps(record) for record in profiling) + "\n", encoding="utf-8"
    )
    raw_path.write_text("\n".join(json.dumps(record) for record in warmup) + "\n", encoding="utf-8")
    bench_request = dataset_request(tmp_path, warmup_request_count=2)

    assert population_identity_error(bench_request, profiling_path, raw_path) is None

    profiling[0]["metadata"]["conversation_id"] = "inferlab-00000000"
    profiling_path.write_text(
        "\n".join(json.dumps(record) for record in profiling) + "\n", encoding="utf-8"
    )
    error = population_identity_error(bench_request, profiling_path, raw_path)
    assert error is not None
    assert "expected 'inferlab-00000002'" in error


def test_speed_population_reconciles_upstream_question_identities(tmp_path: Path) -> None:
    bench_request = speed_bench_request(tmp_path)
    assert bench_request.population is not None
    question_ids = [f"{index:032x}" for index in range(4)]
    Path(bench_request.population.path).write_text(
        "".join(json.dumps({"question_id": value}) + "\n" for value in question_ids),
        encoding="utf-8",
    )
    profiling_path = tmp_path / "profiling.jsonl"
    profiling_path.write_text(
        "".join(
            json.dumps(
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "session_num": index,
                        "conversation_id": question_id,
                    }
                }
            )
            + "\n"
            for index, question_id in enumerate(question_ids)
        ),
        encoding="utf-8",
    )

    assert population_identity_error(bench_request, profiling_path, tmp_path / "raw.jsonl") is None


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


def test_config_lowers_shared_prefix_to_one_system_message_prefix(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 1},
            warmup_request_count=2,
            request_source={
                "kind": "random",
                "input_tokens": 8000,
                "output_tokens": 1000,
                "prefix_sharing": {
                    "shared_prefix_ratio": 0.75,
                    "shared_prefix_tokens": 6000,
                    "unique_suffix_tokens": 2000,
                },
            },
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["dataset"] == {
        "type": "synthetic",
        "entries": 6,
        "randomSeed": 7,
        "sampling": "sequential",
        "prompts": {"isl": 2000, "osl": 1000},
        "prefixPrompts": {"sharedSystemLength": 6000},
    }


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
            "sequenceDistribution": [
                {"isl": 1024, "osl": 128, "probability": 70.0},
                {"isl": 8192, "osl": 1024, "probability": 30.0},
            ]
        },
    }
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    extra = cast(dict[str, object], endpoint["extra"])
    assert extra["ignore_eos"] is True
    assert "min_tokens" not in extra


def test_normalization_uses_the_versioned_aiperf_summary_mapping() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))

    assert summary["aiperf_version"] == "0.11.0"

    assert normalize_summary(summary, tpot_applicable=True) == {
        "request_throughput": 7.412500701361551,
        "output_throughput": 118.60001122178481,
        "total_token_throughput": 1126.7001066069556,
        "mean_prompt_tokens": 136.0,
        "min_prompt_tokens": 136.0,
        "max_prompt_tokens": 136.0,
        "stddev_prompt_tokens": 0.0,
        "p50_prompt_tokens": 136.0,
        "p90_prompt_tokens": 136.0,
        "p95_prompt_tokens": 136.0,
        "p99_prompt_tokens": 136.0,
        "mean_request_latency_ms": 134.1524195,
        "min_request_latency_ms": 133.002837,
        "max_request_latency_ms": 135.278056,
        "stddev_request_latency_ms": 0.8202883381654588,
        "p50_request_latency_ms": 134.1643925,
        "p90_request_latency_ms": 135.011908,
        "p95_request_latency_ms": 135.144982,
        "p99_request_latency_ms": 135.2514412,
        "mean_ttft_ms": 33.777362249999996,
        "min_ttft_ms": 32.562256999999995,
        "max_ttft_ms": 34.934636,
        "stddev_ttft_ms": 0.8453862923854322,
        "p50_ttft_ms": 33.806278,
        "p90_ttft_ms": 34.6392266,
        "p95_ttft_ms": 34.78693129999999,
        "p99_ttft_ms": 34.90509506,
        "mean_tpot_ms": 6.691670483333333,
        "min_tpot_ms": 6.685018066666666,
        "max_tpot_ms": 6.696063866666666,
        "stddev_tpot_ms": 0.004665994105672182,
        "p50_tpot_ms": 6.6928,
        "p90_tpot_ms": 6.696056306666667,
        "p95_tpot_ms": 6.696060086666666,
        "p99_tpot_ms": 6.696063110666666,
    }

    summary["request_throughput"] = {"avg": math.inf}
    with pytest.raises(ValueError, match=r"request_throughput\.avg"):
        normalize_summary(summary, tpot_applicable=True)


def test_normalization_omits_tpot_for_prefill_only() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))
    del summary["inter_token_latency"]

    metrics = normalize_summary(summary, tpot_applicable=False)

    assert not any("tpot" in name for name in metrics)
    with pytest.raises(ValueError, match="inter_token_latency"):
        normalize_summary(summary, tpot_applicable=True)


def test_normalization_preserves_optional_weighted_cache_ratio() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))
    summary["overall_usage_prompt_cache_read_pct"] = {"unit": "%", "avg": 62.5}

    assert normalize_summary(summary, tpot_applicable=True)["prompt_cache_read_ratio"] == 0.625

    summary["overall_usage_prompt_cache_read_pct"] = {"unit": "%", "avg": 101.0}
    with pytest.raises(ValueError, match="overall_usage_prompt_cache_read_pct"):
        normalize_summary(summary, tpot_applicable=True)


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
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
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


def test_warmup_counts_use_the_phase_tagged_raw_aiperf_records(tmp_path: Path) -> None:
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


def test_pinned_aiperf_native_warmup_qualification(tmp_path: Path) -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "aiperf-0.11.0-native-warmup-qualification.json"
    )
    fixture = cast(dict[str, object], json.loads(fixture_path.read_text(encoding="utf-8")))
    config = cast(dict[str, object], fixture["effective_config"])
    dataset = cast(dict[str, object], config["dataset"])
    warmup = cast(dict[str, object], config["warmup"])
    profiling = cast(dict[str, object], config["profiling"])
    native_result = cast(dict[str, object], fixture["native_result"])
    raw_records = cast(list[object], fixture["raw_records"])

    assert fixture["aiperf_version"] == "0.11.0"
    assert fixture["runner_version"] == "0.2.0"
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
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    (tmp_path / "inferlab-bench.json").write_text(
        summary_fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )
    profiling_record = '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n'
    (tmp_path / "inferlab-bench.jsonl").write_text(profiling_record * 4, encoding="utf-8")
    (tmp_path / "inferlab-bench_raw.jsonl").write_text(
        '{"metadata":{"benchmark_phase":"warmup","conversation_id":"session_000000","was_cancelled":false},"error":null}\n',
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
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
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
