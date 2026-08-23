import hashlib
import json
from pathlib import Path

import pytest
from inferlab_bench_runner.population import prepare_population
from inferlab_bench_runner.population_replay import REPLAY_MATERIALIZATION_IDENTITY
from inferlab_measurement_sdk import (
    BenchPopulationPreparationRequest,
    ClientStatus,
)

from .support import FakeTokenizer, resolved_prompt_input


class WordTokenizer(FakeTokenizer):
    """Positional-tokenizer variant whose token ids follow the actual words."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._words)
                self._words.append(word)
            ids.append(self._ids[word])
        return ids

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        return " ".join(self._words[token_id] for token_id in token_ids)


def write_population(path: Path, entries: list[dict[str, object]]) -> bytes:
    payload = b"".join(
        (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for entry in entries
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def flat_entry(index: int, text: str, output: int) -> dict[str, object]:
    return {
        "session_id": f"inferlab-{index:08}",
        "text_input": text,
        "output_length": output,
        "extra": {"ignore_eos": True, "min_tokens": output},
    }


def chat_entry(index: int, content: str, output: int) -> dict[str, object]:
    return {
        "session_id": f"inferlab-{index:08}",
        "messages": [{"role": "user", "content": content}],
        "output_length": output,
        "extra": {"ignore_eos": True, "min_tokens": output},
    }


def preparation_request(
    tmp_path: Path,
    source_path: Path,
    *,
    prompt_kind: str = "flat",
    prefix_sharing: dict[str, object] | None = None,
    cache_start: str = "uncontrolled",
    required: int = 2,
) -> BenchPopulationPreparationRequest:
    return BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": "8",
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "tokenizer_backend": "huggingface",
            "transformers_version": "5.12.1",
            "request_source": {
                "kind": "replay",
                "path": "populations/replay.jsonl",
                "expected_sha256": None,
                "prefix_sharing": prefix_sharing,
            },
            "prompt": resolved_prompt_input({"kind": prompt_kind}),
            "cache_start": cache_start,
            "source_path": str(source_path),
            "required_entries": required,
            "seed": 7,
            "request_body": {},
            "artifact_dir": str(tmp_path / "preparation"),
        }
    )


def test_replay_copies_the_file_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "populations" / "replay.jsonl"
    payload = write_population(
        source,
        [
            flat_entry(0, "alpha beta gamma delta", 128),
            flat_entry(1, "epsilon zeta eta theta", 256),
            flat_entry(2, "iota kappa lambda mu", 128),
        ],
    )

    result = prepare_population(
        preparation_request(tmp_path, source, required=3), tokenizer=FakeTokenizer()
    )

    assert result.status is ClientStatus.succeeded
    assert result.materialization_identity == REPLAY_MATERIALIZATION_IDENTITY
    population = result.population
    assert population is not None
    assert population.entries == 3
    assert population.tpot_applicable
    assert population.sha256 == hashlib.sha256(payload).hexdigest()
    assert Path(population.path).read_bytes() == payload
    assert result.output_tokens is not None
    assert (result.output_tokens.minimum, result.output_tokens.maximum) == (128, 256)
    assert result.input_tokens is not None
    assert (result.input_tokens.minimum, result.input_tokens.maximum) == (4, 4)
    assert result.prompt_token_targeting is None
    assert result.prefix_geometry is None
    evidence_lines = Path(population.evidence_path).read_text(encoding="utf-8").splitlines()
    assert len(evidence_lines) == 3
    first = json.loads(evidence_lines[0])
    assert first["session_id"] == "inferlab-00000000"
    assert first["prompt_kind"] == "flat"
    assert first["output_tokens"] == 128
    assert first["resolved_shared_prefix_tokens"] is None


def test_replay_accepts_server_chat_entries(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(
        source,
        [chat_entry(0, "hello world", 32), chat_entry(1, "again now", 64)],
    )

    result = prepare_population(
        preparation_request(tmp_path, source, prompt_kind="server_chat"),
        tokenizer=FakeTokenizer(),
    )

    assert result.status is ClientStatus.succeeded
    population = result.population
    assert population is not None
    assert population.entries == 2
    assert result.input_tokens is not None
    assert result.input_tokens.minimum == 2


def test_replay_output_one_population_is_tpot_inapplicable(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(source, [flat_entry(0, "one two", 1), flat_entry(1, "three four", 1)])

    result = prepare_population(preparation_request(tmp_path, source), tokenizer=FakeTokenizer())

    assert result.status is ClientStatus.succeeded
    population = result.population
    assert population is not None
    assert not population.tpot_applicable


def test_replay_rejects_mixed_tpot_classes(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(source, [flat_entry(0, "one two", 1), flat_entry(1, "three four", 8)])

    with pytest.raises(ValueError, match="must not mix TPOT"):
        prepare_population(preparation_request(tmp_path, source), tokenizer=FakeTokenizer())


def test_replay_fails_without_repeating_entries_when_insufficient(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(source, [flat_entry(0, "one two", 8)])

    result = prepare_population(
        preparation_request(tmp_path, source, required=4), tokenizer=FakeTokenizer()
    )

    assert result.status is ClientStatus.failed
    assert result.population is None
    assert result.candidate_entries == 1
    assert result.admitted_entries == 1
    assert result.error is not None and "never repeated" in result.error


def test_replay_rejects_malformed_lines(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    source.write_bytes(b'{"session_id": "x", "text_input": "a b", "output_length": 4}\nnot-json\n')

    with pytest.raises(ValueError, match="line 2: invalid JSON"):
        prepare_population(preparation_request(tmp_path, source), tokenizer=FakeTokenizer())


def test_replay_rejects_prompt_shape_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(source, [chat_entry(0, "hello world", 8), chat_entry(1, "again now", 8)])

    with pytest.raises(ValueError, match="text_input"):
        prepare_population(
            preparation_request(tmp_path, source, prompt_kind="flat"),
            tokenizer=FakeTokenizer(),
        )


def test_replay_rejects_invalid_output_length(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(
        source,
        [flat_entry(0, "one two", 8), {"session_id": "bad", "text_input": "three four"}],
    )

    with pytest.raises(ValueError, match="output_length"):
        prepare_population(preparation_request(tmp_path, source), tokenizer=FakeTokenizer())


def test_replay_resolves_prefix_geometry_from_entries(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(
        source,
        [
            flat_entry(0, "shared prefix alpha beta", 16),
            flat_entry(1, "shared prefix gamma", 16),
        ],
    )

    result = prepare_population(
        preparation_request(tmp_path, source, prefix_sharing={"shared_prefix_tokens": 2}),
        tokenizer=WordTokenizer(),
    )

    assert result.status is ClientStatus.succeeded
    geometry = result.prefix_geometry
    assert geometry is not None
    assert geometry.maximum_shared_prefix_tokens == 2
    assert (geometry.shared_prefix_tokens.minimum, geometry.shared_prefix_tokens.maximum) == (2, 2)
    assert len(geometry.canonical_prefix_sha256) == 64
    assert geometry.full_prompt_entries == 0
    assert result.prefix_conditioning is None


def test_replay_ratio_geometry_and_primed_conditioning(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(
        source,
        [
            flat_entry(0, "shared prefix alpha beta", 16),
            flat_entry(1, "shared prefix gamma delta", 16),
        ],
    )

    result = prepare_population(
        preparation_request(
            tmp_path,
            source,
            prefix_sharing={"shared_prefix_ratio": 0.5},
            cache_start="primed",
        ),
        tokenizer=WordTokenizer(),
    )

    assert result.status is ClientStatus.succeeded
    geometry = result.prefix_geometry
    assert geometry is not None
    assert geometry.maximum_shared_prefix_tokens == 2
    conditioning = result.prefix_conditioning
    assert conditioning is not None
    assert conditioning.prompt_tokens == 2
    assert Path(conditioning.path).read_text(encoding="utf-8") == "shared prefix"
    assert conditioning.sha256 == hashlib.sha256(b"shared prefix").hexdigest()


def test_replay_rejects_entries_that_do_not_share_the_declared_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(
        source,
        [
            flat_entry(0, "shared prefix alpha beta", 16),
            flat_entry(1, "different prefix gamma delta", 16),
        ],
    )

    with pytest.raises(ValueError, match="does not share the declared canonical prefix"):
        prepare_population(
            preparation_request(tmp_path, source, prefix_sharing={"shared_prefix_tokens": 1}),
            tokenizer=WordTokenizer(),
        )


def test_replay_rejects_an_entry_shorter_than_its_shared_prefix(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(
        source,
        [
            flat_entry(0, "shared prefix alpha beta", 16),
            flat_entry(1, "shared", 16),
        ],
    )

    with pytest.raises(ValueError, match="fewer than its resolved shared prefix"):
        prepare_population(
            preparation_request(tmp_path, source, prefix_sharing={"shared_prefix_tokens": 2}),
            tokenizer=WordTokenizer(),
        )


def test_replay_rejects_prefix_sharing_for_server_chat(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(source, [chat_entry(0, "hello world", 8), chat_entry(1, "again now", 8)])

    with pytest.raises(ValueError, match="flat or rendered_chat"):
        prepare_population(
            preparation_request(
                tmp_path,
                source,
                prompt_kind="server_chat",
                prefix_sharing={"shared_prefix_tokens": 1},
            ),
            tokenizer=FakeTokenizer(),
        )


def test_replay_rejects_a_zero_length_primed_prefix(tmp_path: Path) -> None:
    source = tmp_path / "replay.jsonl"
    write_population(
        source,
        [flat_entry(0, "one two", 8), flat_entry(1, "three four", 8)],
    )

    with pytest.raises(ValueError, match="zero-length shared prefix"):
        prepare_population(
            preparation_request(
                tmp_path,
                source,
                prefix_sharing={"shared_prefix_ratio": 0.0},
                cache_start="primed",
            ),
            tokenizer=FakeTokenizer(),
        )
