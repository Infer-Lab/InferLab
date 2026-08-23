import hashlib
import json
from pathlib import Path

import pytest
from inferlab_bench_runner.population import prepare_population
from inferlab_bench_runner.population_synthetic import CORPUS_MATERIALIZATION_IDENTITY
from inferlab_measurement_sdk import (
    BenchPopulationPreparationRequest,
    BenchPopulationPreparationResult,
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


def write_corpus(path: Path, words: int) -> bytes:
    payload = (" ".join(f"word{index}" for index in range(words)) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def corpus_request(
    tmp_path: Path,
    source_path: Path,
    *,
    input_tokens: object = 8,
    output_tokens: int = 16,
    prefix_sharing: dict[str, object] | None = None,
    cache_start: str = "uncontrolled",
    required: int = 4,
    seed: int = 7,
    expected_sha256: str | None = None,
    prompt_kind: str = "flat",
    artifact: str = "preparation",
) -> BenchPopulationPreparationRequest:
    return BenchPopulationPreparationRequest.model_validate(
        {
            "protocol_version": "8",
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "tokenizer_backend": "huggingface",
            "transformers_version": "5.12.1",
            "request_source": {
                "kind": "random",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "corpus": {
                    "path": "corpus/corpus.txt",
                    "expected_sha256": expected_sha256,
                },
                "prefix_sharing": prefix_sharing,
            },
            "prompt": resolved_prompt_input({"kind": prompt_kind}),
            "cache_start": cache_start,
            "source_path": str(source_path),
            "required_entries": required,
            "seed": seed,
            "request_body": {},
            "artifact_dir": str(tmp_path / artifact),
        }
    )


def _rows(path: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def population_rows(result: BenchPopulationPreparationResult) -> list[dict[str, object]]:
    population = result.population
    assert population is not None
    return _rows(str(population.path))


def evidence_rows(result: BenchPopulationPreparationResult) -> list[dict[str, object]]:
    population = result.population
    assert population is not None
    return _rows(str(population.evidence_path))


def test_corpus_entries_are_seed_deterministic_exact_slices(tmp_path: Path) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 64)

    first = prepare_population(
        corpus_request(tmp_path, source, required=4, artifact="first"),
        tokenizer=WordTokenizer(),
    )
    second = prepare_population(
        corpus_request(tmp_path, source, required=4, artifact="second"),
        tokenizer=WordTokenizer(),
    )

    assert first.status is ClientStatus.succeeded
    assert first.materialization_identity == CORPUS_MATERIALIZATION_IDENTITY
    rows = population_rows(first)
    assert rows == population_rows(second)
    tokenizer = WordTokenizer()
    corpus_ids = tokenizer.encode(source.read_text(encoding="utf-8"), add_special_tokens=False)
    for row, evidence in zip(rows, evidence_rows(first), strict=True):
        text = row["text_input"]
        assert isinstance(text, str)
        offset = evidence["corpus_slice_offset"]
        length = evidence["corpus_slice_length"]
        assert isinstance(offset, int)
        assert isinstance(length, int)
        assert length == 8
        assert evidence["selected_prompt_tokens"] == 8
        assert evidence["corpus_shared_slice_offset"] is None
        # The recorded slice reconciles exactly with the corpus token stream.
        assert (
            tokenizer.encode(text, add_special_tokens=False) == corpus_ids[offset : offset + length]
        )


def test_corpus_seed_changes_the_drawn_slices(tmp_path: Path) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 64)

    seed7 = prepare_population(
        corpus_request(tmp_path, source, seed=7, artifact="seed7"), tokenizer=WordTokenizer()
    )
    seed11 = prepare_population(
        corpus_request(tmp_path, source, seed=11, artifact="seed11"), tokenizer=WordTokenizer()
    )

    assert population_rows(seed7) != population_rows(seed11)


def test_corpus_population_extension_preserves_the_first_entries(tmp_path: Path) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 64)

    small = prepare_population(
        corpus_request(tmp_path, source, required=2, artifact="small"),
        tokenizer=WordTokenizer(),
    )
    large = prepare_population(
        corpus_request(tmp_path, source, required=6, artifact="large"),
        tokenizer=WordTokenizer(),
    )

    assert population_rows(small) == population_rows(large)[:2]


def test_corpus_extension_preserves_entries_under_a_range_selector(tmp_path: Path) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 64)
    selector: dict[str, object] = {"kind": "inclusive_uniform", "min": 4, "max": 12}

    small = prepare_population(
        corpus_request(tmp_path, source, input_tokens=selector, required=2, artifact="small"),
        tokenizer=WordTokenizer(),
    )
    large = prepare_population(
        corpus_request(tmp_path, source, input_tokens=selector, required=6, artifact="large"),
        tokenizer=WordTokenizer(),
    )

    assert population_rows(small) == population_rows(large)[:2]
    lengths = {len(str(row["text_input"]).split()) for row in population_rows(large)}
    assert lengths <= {4, 5, 6, 7, 8, 9, 10, 11, 12}


def test_corpus_prefix_sharing_draws_one_fixed_slice_and_independent_suffixes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 64)

    result = prepare_population(
        corpus_request(tmp_path, source, prefix_sharing={"shared_prefix_tokens": 4}, required=4),
        tokenizer=WordTokenizer(),
    )

    assert result.status is ClientStatus.succeeded
    tokenizer = WordTokenizer()
    corpus_ids = tokenizer.encode(source.read_text(encoding="utf-8"), add_special_tokens=False)
    rows = population_rows(result)
    entry_ids = [tokenizer.encode(str(row["text_input"]), add_special_tokens=False) for row in rows]
    shared_offsets = set()
    for ids, evidence in zip(entry_ids, evidence_rows(result), strict=True):
        assert len(ids) == 8
        assert evidence["corpus_slice_length"] == 4
        shared_offset = evidence["corpus_shared_slice_offset"]
        assert isinstance(shared_offset, int)
        shared_offsets.add(shared_offset)
        # The shared prefix is one fixed corpus slice; the suffix slice is the
        # entry's independent draw, and both reconcile with the corpus stream.
        assert ids[:4] == corpus_ids[shared_offset : shared_offset + 4]
        suffix_offset = evidence["corpus_slice_offset"]
        assert isinstance(suffix_offset, int)
        assert ids[4:] == corpus_ids[suffix_offset : suffix_offset + 4]
    assert len(shared_offsets) == 1
    assert len({tuple(ids[4:]) for ids in entry_ids}) > 1
    geometry = result.prefix_geometry
    assert geometry is not None
    assert geometry.maximum_shared_prefix_tokens == 4
    assert geometry.full_prompt_entries == 0


def test_corpus_primed_conditioning_uses_the_fixed_corpus_slice(tmp_path: Path) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 64)

    result = prepare_population(
        corpus_request(
            tmp_path,
            source,
            prefix_sharing={"shared_prefix_ratio": 0.5},
            cache_start="primed",
        ),
        tokenizer=WordTokenizer(),
    )

    assert result.status is ClientStatus.succeeded
    conditioning = result.prefix_conditioning
    assert conditioning is not None
    assert conditioning.prompt_tokens == 4
    conditioning_text = Path(conditioning.path).read_text(encoding="utf-8")
    assert conditioning.sha256 == hashlib.sha256(conditioning_text.encode()).hexdigest()
    tokenizer = WordTokenizer()
    corpus_ids = tokenizer.encode(source.read_text(encoding="utf-8"), add_special_tokens=False)
    evidence = evidence_rows(result)[0]
    shared_offset = evidence["corpus_shared_slice_offset"]
    assert isinstance(shared_offset, int)
    assert (
        tokenizer.encode(conditioning_text, add_special_tokens=False)
        == corpus_ids[shared_offset : shared_offset + 4]
    )


def test_corpus_shorter_than_the_largest_selected_target_fails(tmp_path: Path) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 4)

    with pytest.raises(ValueError, match="shorter than the largest selected input-token target 8"):
        prepare_population(corpus_request(tmp_path, source), tokenizer=WordTokenizer())


def test_corpus_digest_mismatch_fails_preparation(tmp_path: Path) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 64)

    with pytest.raises(ValueError, match="SHA-256"):
        prepare_population(
            corpus_request(tmp_path, source, expected_sha256="0" * 64),
            tokenizer=WordTokenizer(),
        )


def test_corpus_requires_a_flat_prompt(tmp_path: Path) -> None:
    source = tmp_path / "corpus" / "corpus.txt"
    write_corpus(source, 64)

    with pytest.raises(ValueError, match="prompt kind flat"):
        prepare_population(
            corpus_request(tmp_path, source, prompt_kind="server_chat"),
            tokenizer=WordTokenizer(),
        )
