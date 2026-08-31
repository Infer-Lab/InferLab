import json
from pathlib import Path
from typing import cast

from inferlab_bench_runner.aiperf import (
    aiperf_config,
)
from inferlab_bench_runner.population import prepare_population
from inferlab_bench_runner.population_sharegpt import (
    materialize_conversation,
)
from inferlab_bench_runner.result_population import (
    population_identity_error,
)
from inferlab_measurement_sdk import (
    BenchPopulationPreparationRequest,
    ClientStatus,
)

from .support import (
    FakeTokenizer,
    dataset_request,
    resolved_prompt_input,
)


def preparation_request(
    tmp_path: Path, source_path: Path, artifact_name: str = "population"
) -> BenchPopulationPreparationRequest:
    return BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": "9",
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
            "prompt": resolved_prompt_input({"kind": "server_chat"}),
            "cache_start": "uncontrolled",
            "source_path": str(source_path),
            "required_entries": 2,
            "seed": 7,
            "request_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "artifact_dir": str(tmp_path / artifact_name),
        }
    )


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
    records_path = tmp_path / "records.jsonl"
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
    records_path.write_text(
        "\n".join(json.dumps(record) for record in [*warmup, *profiling]) + "\n",
        encoding="utf-8",
    )
    bench_request = dataset_request(tmp_path, warmup_request_count=2)

    assert population_identity_error(bench_request, records_path) is None

    profiling[0]["metadata"]["conversation_id"] = "inferlab-00000000"
    records_path.write_text(
        "\n".join(json.dumps(record) for record in [*warmup, *profiling]) + "\n",
        encoding="utf-8",
    )
    error = population_identity_error(bench_request, records_path)
    assert error is not None
    assert "expected 'inferlab-00000002'" in error
