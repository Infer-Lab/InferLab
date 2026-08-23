"""Coordinate tokenizer loading and source-owned population materializers."""

import importlib
import importlib.metadata
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    BenchPopulationPreparationRequest,
    BenchPopulationPreparationResult,
    BenchRequestSourceInputDataset,
    BenchRequestSourceInputRandom,
    BenchRequestSourceInputRandomMixture,
    BenchRequestSourceInputReplay,
)

from inferlab_bench_runner.population_replay import prepare_replay_population
from inferlab_bench_runner.population_sharegpt import (
    prepare_sharegpt_population,
    prepare_sharegpt_session_population,
)
from inferlab_bench_runner.population_speed import prepare_speed_bench_population
from inferlab_bench_runner.population_synthetic import write_synthetic_population
from inferlab_bench_runner.population_types import ChatTokenizer


def _iter_parquet_rows(path: Path) -> Iterator[object]:
    parquet = importlib.import_module("pyarrow.parquet")
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        yield from cast(list[object], batch.to_pylist())


def prepare_population(
    request: BenchPopulationPreparationRequest,
    tokenizer: ChatTokenizer | None = None,
) -> BenchPopulationPreparationResult:
    if request.tokenizer_backend != "huggingface":
        raise ValueError(f"unsupported Bench tokenizer backend {request.tokenizer_backend!r}")
    if tokenizer is None:
        tokenizer = load_chat_tokenizer(
            request.model.locator, expected_transformers=request.transformers_version
        )
    if request.session_source is not None:
        if request.request_source is not None:
            raise ValueError("population preparation requires exactly one source boundary")
        return prepare_sharegpt_session_population(request, tokenizer, request.session_source)
    if request.request_source is None:
        raise ValueError("population preparation requires exactly one source boundary")

    source = request.request_source.root
    if isinstance(
        source,
        (BenchRequestSourceInputRandom, BenchRequestSourceInputRandomMixture),
    ):
        return write_synthetic_population(request, tokenizer, source)
    if isinstance(source, BenchRequestSourceInputDataset):
        if source.dataset == "sharegpt":
            return prepare_sharegpt_population(request, tokenizer)
        if source.dataset == "speed_bench":
            return prepare_speed_bench_population(request, tokenizer, source, _iter_parquet_rows)
        raise ValueError(f"unsupported catalog dataset {source.dataset!r}")
    if isinstance(source, BenchRequestSourceInputReplay):
        return prepare_replay_population(request, tokenizer, source)
    raise TypeError(f"unsupported Bench request source {type(source).__name__}")


def load_chat_tokenizer(
    model_locator: str, *, expected_transformers: str | None = None
) -> ChatTokenizer:
    if expected_transformers is not None:
        observed_transformers = importlib.metadata.version("transformers")
        if observed_transformers != expected_transformers:
            raise ValueError(
                "installed Transformers version does not match the resolved Bench toolchain: "
                f"observed={observed_transformers!r}, expected={expected_transformers!r}"
            )
    transformers = importlib.import_module("transformers")
    return cast(
        ChatTokenizer,
        transformers.AutoTokenizer.from_pretrained(model_locator, local_files_only=True),
    )
