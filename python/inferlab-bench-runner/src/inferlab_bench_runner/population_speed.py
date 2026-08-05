"""Materialize the source-owned SPEED-Bench population policy."""

import hashlib
import heapq
from collections.abc import Callable, Iterator
from pathlib import Path

from inferlab_measurement_sdk import (
    BenchPopulationInput,
    BenchPopulationPreparationRequest,
    BenchPopulationPreparationResult,
    BenchRequestSourceInputDataset,
    ClientStatus,
)

from inferlab_bench_runner.chat_tokens import required_messages_content_tokens
from inferlab_bench_runner.population_types import (
    ChatTokenizer,
    MaterializedEntry,
    count_summary,
    json_line,
)


def prepare_speed_bench_population(
    request: BenchPopulationPreparationRequest,
    tokenizer: ChatTokenizer,
    source: BenchRequestSourceInputDataset,
    iter_rows: Callable[[Path], Iterator[object]],
) -> BenchPopulationPreparationResult:
    if request.source_path is None:
        raise ValueError("SPEED-Bench preparation requires a source path")
    if source.output_tokens is None:
        raise ValueError("SPEED-Bench preparation requires fixed output_tokens")
    profile_filter = source.catalog.filter
    if profile_filter is None:
        raise ValueError("SPEED-Bench catalog profile has no filter")
    required = request.required_entries
    selected: list[tuple[int, int, MaterializedEntry]] = []
    candidate_entries = 0
    admitted_entries = 0
    ineligible_reasons: dict[str, int] = {}
    for index, raw in enumerate(iter_rows(Path(request.source_path))):
        candidate_entries += 1
        if not isinstance(raw, dict):
            reason = "invalid_row"
        elif raw.get(profile_filter.field) != profile_filter.value:
            reason = "profile_filter_mismatch"
        else:
            question_id = raw.get("question_id")
            category = raw.get("category")
            turns = raw.get("turns")
            first_turn = (
                next(
                    (
                        (turn_index, turn)
                        for turn_index, turn in enumerate(turns)
                        if isinstance(turn, str) and turn.strip()
                    ),
                    None,
                )
                if isinstance(turns, list)
                else None
            )
            if (
                not isinstance(question_id, str)
                or len(question_id) != 32
                or not isinstance(category, str)
                or not isinstance(turns, list)
                or first_turn is None
            ):
                reason = "invalid_first_turn"
            else:
                turn_index, turn = first_turn
                messages = [{"role": "user", "content": turn}]
                input_tokens = required_messages_content_tokens(messages, tokenizer)
                if input_tokens > source.max_input_tokens:
                    reason = "input_exceeds_maximum"
                else:
                    admitted_entries += 1
                    entry = MaterializedEntry(
                        source_sample_id=question_id,
                        messages=messages,
                        target=None,
                        kept_messages=1,
                        removed_messages=len(turns) - 1,
                        input_tokens=input_tokens,
                        output_tokens=source.output_tokens,
                        category=category,
                        later_turn_count=len(turns) - turn_index - 1,
                        first_user_turn_index=turn_index,
                    )
                    key = int.from_bytes(
                        hashlib.sha256(f"{request.seed}\0{question_id}\0{index}".encode()).digest(),
                        "big",
                    )
                    item = (-key, -index, entry)
                    if len(selected) < required:
                        heapq.heappush(selected, item)
                    elif item > selected[0]:
                        heapq.heapreplace(selected, item)
                    continue
        ineligible_reasons[reason] = ineligible_reasons.get(reason, 0) + 1
    ineligible_entries = candidate_entries - admitted_entries
    if admitted_entries < required:
        return BenchPopulationPreparationResult(
            schema_version=1,
            status=ClientStatus.failed,
            materialization_identity=source.catalog.materialization_identity,
            requested_entries=required,
            candidate_entries=candidate_entries,
            admitted_entries=admitted_entries,
            ineligible_entries=ineligible_entries,
            ineligible_reasons=ineligible_reasons,
            population=None,
            input_tokens=None,
            output_tokens=None,
            evidence_path=None,
            error=f"dataset has {admitted_entries} eligible entries, requires {required}",
        )
    ordered = [item[2] for item in sorted(selected, key=lambda item: (-item[0], -item[1]))]
    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    population_path = artifact_dir / "population.jsonl"
    evidence_path = artifact_dir / "population-evidence.jsonl"
    population_digest = hashlib.sha256()
    with population_path.open("wb") as population_file, evidence_path.open("wb") as evidence_file:
        for population_index, entry in enumerate(ordered):
            population_line = json_line(
                {
                    "question_id": entry.source_sample_id,
                    "category": entry.category,
                    "messages": entry.messages,
                }
            )
            population_file.write(population_line)
            population_digest.update(population_line)
            evidence_file.write(
                json_line(
                    {
                        "population_index": population_index,
                        "question_id": entry.source_sample_id,
                        "profile": source.profile,
                        "category": entry.category,
                        "first_user_content": entry.messages[0]["content"],
                        "first_user_turn_index": entry.first_user_turn_index,
                        "later_turn_count": entry.later_turn_count,
                        "messages": entry.messages,
                        "held_out_target": None,
                        "input_tokens": entry.input_tokens,
                        "selected_output_tokens": entry.output_tokens,
                    }
                )
            )
    input_counts = [entry.input_tokens for entry in ordered]
    output_counts = [entry.output_tokens for entry in ordered]
    return BenchPopulationPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        materialization_identity=source.catalog.materialization_identity,
        requested_entries=required,
        candidate_entries=candidate_entries,
        admitted_entries=admitted_entries,
        ineligible_entries=ineligible_entries,
        ineligible_reasons=ineligible_reasons,
        population=BenchPopulationInput(
            path=str(population_path),
            evidence_path=str(evidence_path),
            sha256=population_digest.hexdigest(),
            entries=required,
            tpot_applicable=True,
        ),
        input_tokens=count_summary(input_counts),
        output_tokens=count_summary(output_counts),
        evidence_path=str(evidence_path),
        error=None,
    )
