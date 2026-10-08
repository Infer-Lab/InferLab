import json
from pathlib import Path

import pytest
from inferlab_bench_runner.population import prepare_population
from inferlab_bench_runner.result_population import (
    population_identity_error,
)
from inferlab_measurement_sdk import (
    PROTOCOL_VERSION,
    BenchPopulationPreparationRequest,
    ClientStatus,
)

from .support import (
    FakeTokenizer,
    resolved_prompt_input,
    speed_bench_request,
)


def test_speed_bench_materialization_filters_without_replacement_and_keeps_only_first_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rows in the shape the AIPerf SPEED-Bench materialization writes.
    source_path = tmp_path / "speed_bench_qualitative.jsonl"
    source_path.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": f"{index:032x}",
                    "category": category,
                    "messages": [
                        {"role": "user", "content": ""},
                        {"role": "user", "content": f"first turn {index}"},
                        {"role": "user", "content": f"later turn {index}"},
                    ],
                }
            )
            + "\n"
            for index, category in enumerate(["coding", "coding", "math", "coding"])
        ),
        encoding="utf-8",
    )
    request_value = BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": PROTOCOL_VERSION,
            "model": {"locator": "/models/deepseek-v4-flash", "served_name": "deepseek-v4-flash"},
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
                    "upstream_identity": "aiperf-public-dataset:speed_bench_qualitative",
                    "aiperf_dataset": "speed_bench_qualitative",
                    "sha256": "0" * 64,
                    "source_format": "aiperf-public-dataset-jsonl-v1",
                    "aiperf_format": "mooncake_trace",
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
            "prompt": resolved_prompt_input({"kind": "server_chat"}),
            "cache_start": "uncontrolled",
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
    assert len({row["question_id"] for row in evidence}) == 2
    assert all(row["category"] == "coding" for row in evidence)
    assert all(row["extra"] == {"ignore_eos": True, "min_tokens": 16} for row in population)
    assert all(len(row["messages"]) == 1 for row in population)
    assert all("first turn" in row["messages"][0]["content"] for row in population)
    assert all(row["first_user_turn_index"] == 1 for row in evidence)
    assert all(row["later_turn_count"] == 1 for row in evidence)
    assert all(row["held_out_target"] is None for row in evidence)
    assert all(
        row["messages"] == population[index]["messages"] for index, row in enumerate(evidence)
    )
    assert all("rendered_prompt" not in row for row in evidence)


def test_speed_population_reconciles_frozen_session_identities(tmp_path: Path) -> None:
    bench_request = speed_bench_request(tmp_path)
    assert bench_request.population is not None
    question_ids = [f"inferlab-{index:08}" for index in range(4)]
    Path(bench_request.population.path).write_text(
        "".join(json.dumps({"session_id": value}) + "\n" for value in question_ids),
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

    assert population_identity_error(bench_request, profiling_path) is None
