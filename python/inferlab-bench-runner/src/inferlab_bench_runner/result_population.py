"""Reconcile a frozen request population with AIPerf-assigned identities."""

import json
from pathlib import Path

from inferlab_measurement_sdk import (
    BenchClientRequest,
    BenchPromptTokenReconciliation,
    BenchRenderingAuthorityInput,
    BenchRequestSourceInputDataset,
    BenchRequestSourceInputRandom,
    BenchRequestSourceInputRandomMixture,
)

from inferlab_bench_runner.result_records import phase_records, profiling_records


def prompt_token_reconciliation(
    request: BenchClientRequest,
    profiling_path: Path,
) -> tuple[list[BenchPromptTokenReconciliation], str | None]:
    source_input = request.definition.request_source
    population = request.population
    if source_input is None or population is None:
        return [], None
    source = source_input.root
    if not isinstance(
        source, (BenchRequestSourceInputRandom, BenchRequestSourceInputRandomMixture)
    ):
        return [], None
    if request.definition.prompt.root.rendering_authority not in (
        BenchRenderingAuthorityInput.local_flat,
        BenchRenderingAuthorityInput.local_template,
    ):
        return [], None
    try:
        evidence_rows = [
            json.loads(line)
            for line in Path(population.evidence_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        return [], f"synthetic population evidence is unreadable: {error}"
    records, parse_error = profiling_records(profiling_path)
    if parse_error is not None:
        return [], parse_error
    reconciliations: list[BenchPromptTokenReconciliation] = []
    first_error: str | None = None
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return reconciliations, "AIPerf profiling record has no metadata"
        if record.get("error") is not None or metadata.get("was_cancelled") is True:
            continue
        native_session_num = metadata.get("session_num")
        if isinstance(native_session_num, bool) or not isinstance(native_session_num, int):
            return reconciliations, "AIPerf profiling record has no integer session_num"
        population_index = request.case.warmup_request_count + native_session_num
        if population_index < 0 or population_index >= len(evidence_rows):
            return reconciliations, (
                f"AIPerf profiling session_num {native_session_num} has no assigned "
                "synthetic population evidence"
            )
        evidence = evidence_rows[population_index]
        planned = evidence.get("selected_prompt_tokens") if isinstance(evidence, dict) else None
        if isinstance(planned, bool) or not isinstance(planned, int) or planned < 0:
            return reconciliations, (
                f"synthetic population entry {population_index} has no selected prompt-token target"
            )
        metrics = record.get("metrics")
        observed_value: object = None
        if isinstance(metrics, dict):
            input_length = metrics.get("input_sequence_length")
            if isinstance(input_length, dict):
                observed_value = input_length.get("value")
        observed = (
            observed_value
            if isinstance(observed_value, int) and not isinstance(observed_value, bool)
            else None
        )
        reconciled = observed == planned
        reconciliations.append(
            BenchPromptTokenReconciliation(
                population_index=population_index,
                native_session_num=native_session_num,
                planned_prompt_tokens=planned,
                observed_prompt_tokens=observed,
                reconciled=reconciled,
            )
        )
        if not reconciled and first_error is None:
            first_error = (
                f"profiling population entry {population_index} planned {planned} prompt tokens, "
                f"backend reported {observed!r}"
            )
    return reconciliations, first_error


def population_identity_error(
    request: BenchClientRequest,
    records_path: Path,
) -> str | None:
    if request.population is None or request.definition.request_source is None:
        return None
    source = request.definition.request_source.root
    identity_field = (
        "question_id"
        if isinstance(source, BenchRequestSourceInputDataset) and source.dataset == "speed_bench"
        else "session_id"
    )
    required_identities = request.case.warmup_request_count + request.case.request_count
    population_ids: list[str] = []
    try:
        population_lines = Path(request.population.path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return f"Bench population identities are unreadable: {error}"
    for line_number, line in enumerate(population_lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            return f"Bench population line {line_number} is invalid JSON: {error}"
        identity = row.get(identity_field) if isinstance(row, dict) else None
        if not isinstance(identity, str) or not identity:
            return f"Bench population line {line_number} has no non-empty {identity_field}"
        population_ids.append(identity)
    if len(population_ids) < required_identities:
        return (
            "Bench population identities do not cover the resolved case: "
            f"required={required_identities}, observed={len(population_ids)}"
        )
    phases = [
        (
            "warmup",
            *phase_records(records_path, "warmup"),
            0,
            request.case.warmup_request_count,
        ),
        (
            "profiling",
            *profiling_records(records_path),
            request.case.warmup_request_count,
            request.case.request_count,
        ),
    ]
    for phase, records, parse_error, population_start, expected_count in phases:
        if parse_error is not None:
            return parse_error
        if len(records) != expected_count:
            return (
                f"AIPerf {phase} identities do not cover the assigned population slice: "
                f"expected={expected_count}, observed={len(records)}"
            )
        observed_session_nums: set[int] = set()
        for record in records:
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                return f"AIPerf {phase} record has no metadata"
            session_num = metadata.get("session_num")
            conversation_id = metadata.get("conversation_id")
            if isinstance(session_num, bool) or not isinstance(session_num, int):
                return f"AIPerf {phase} record has no integer session_num"
            if session_num < 0 or session_num >= expected_count:
                return f"AIPerf {phase} session_num {session_num} is outside its assigned slice"
            if session_num in observed_session_nums:
                return f"AIPerf {phase} records duplicate session_num {session_num}"
            observed_session_nums.add(session_num)
            expected_id = population_ids[population_start + session_num]
            if conversation_id != expected_id:
                return (
                    f"AIPerf {phase} session_num {session_num} references "
                    f"conversation_id {conversation_id!r}, expected {expected_id!r}"
                )
    return None
