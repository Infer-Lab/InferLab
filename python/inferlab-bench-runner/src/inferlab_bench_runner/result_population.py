"""Reconcile a frozen request population with AIPerf-assigned identities."""

import json
from pathlib import Path

from inferlab_measurement_sdk import BenchClientRequest, BenchRequestSourceInputDataset

from inferlab_bench_runner.result_records import profiling_records, raw_phase_records


def population_identity_error(
    request: BenchClientRequest,
    profiling_path: Path,
    raw_path: Path,
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
            *raw_phase_records(raw_path, "warmup"),
            0,
            request.case.warmup_request_count,
        ),
        (
            "profiling",
            *profiling_records(profiling_path),
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
