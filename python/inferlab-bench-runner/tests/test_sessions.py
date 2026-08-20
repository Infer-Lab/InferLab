import json
from pathlib import Path
from typing import cast

from inferlab_bench_runner.aiperf import (
    aiperf_config,
    aiperf_session_population_layout,
)
from inferlab_bench_runner.population import prepare_population
from inferlab_bench_runner.population_sharegpt import (
    effective_inter_turn_delay,
)
from inferlab_bench_runner.result_sessions import session_result_evidence
from inferlab_measurement_sdk import (
    BenchClientRequest,
    BenchPopulationPreparationRequest,
    ClientStatus,
)

from .support import FakeTokenizer, resolved_prompt_input


def session_request(tmp_path: Path, artifact_level: str = "diagnostic") -> BenchClientRequest:
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
                "prompt": resolved_prompt_input({"kind": "server_chat"}),
                "server_metrics": False,
                "seed": 7,
                "request_body": {},
                "request_slo": None,
                "timeout_seconds": 120,
                "cache_start": "uncontrolled",
                "artifact_level": artifact_level,
            },
            "population": {
                "path": str(population_path),
                "evidence_path": str(population_path),
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
            "prompt": resolved_prompt_input({"kind": "server_chat"}),
            "cache_start": "uncontrolled",
            "source_path": str(source_path),
            "required_entries": required_entries,
            "seed": 7,
            "request_body": {},
            "artifact_dir": str(tmp_path / "sessions"),
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


def test_performance_session_result_reads_normalized_records_and_marks_raw_dimensions(
    tmp_path: Path,
) -> None:
    value = session_request(tmp_path, artifact_level="performance")
    records_path = tmp_path / "session-records.jsonl"
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
                record_lines.append(
                    json.dumps(
                        {
                            "metadata": {
                                "benchmark_phase": phase,
                                "conversation_id": f"template-{template_index}",
                                "x_correlation_id": runtime_id,
                                "session_num": native_request,
                                "turn_index": turn_index,
                                "request_start_ns": request_start,
                                "request_end_ns": request_end,
                                "was_cancelled": False,
                            },
                            "metrics": {
                                "input_sequence_length": {
                                    "value": 5 + turn_index,
                                    "unit": "tokens",
                                }
                            },
                            "error": None,
                        }
                    )
                )
                timing_sequence += 1
                previous_end = request_end
    records_path.write_text("\n".join(record_lines) + "\n", encoding="utf-8")

    evidence, error = session_result_evidence(
        value, records_path, tmp_path / "session-raw.jsonl", FakeTokenizer()
    )

    assert error is None
    assert evidence.profiling.completed_requests == 12
    assert len(evidence.sessions) == 8
    assert len(evidence.turns) == 16
    assert evidence.turns[0].pre_template_content_tokens is None
    assert evidence.turns[0].observed_prompt_tokens == 5
    assert evidence.turns[1].pre_template_content_tokens is None
    assert evidence.turns[1].observed_prompt_tokens == 6
    assert evidence.turns[1].preceding_native_session_num is None
    assert evidence.turns[1].preceding_terminal_response_receipt_ns is None
    assert evidence.turns[1].inter_turn_delay_reconciled is True
    assert evidence.turns[1].native_artifact_name == "aiperf_records"
    assert evidence.population_slice_reconciled is True
    assert evidence.inter_turn_delays_reconciled is True
    assert evidence.native_requests_reconciled is None
    assert evidence.counts_reconciled is True
    assert evidence.unavailable_dimensions == [
        "pre_template_content_tokens",
        "max_input_tokens_bound_check",
        "preceding_live_response_pairwise_history",
        "raw_native_request_reconciliation",
    ]
