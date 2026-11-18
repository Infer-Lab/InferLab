"""Read the pinned AIPerf normalized-record artifacts by benchmark phase."""

from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import JsonObject, read_json_lines


def _append_phase_records(
    path: Path, artifact: str, phase: str, records: list[JsonObject]
) -> str | None:
    rows, error = read_json_lines(path, artifact)
    for line_number, record in rows:
        if not isinstance(record, dict):
            return f"{artifact} line {line_number} is not a JSON object"
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return f"{artifact} line {line_number} has no metadata"
        if metadata.get("benchmark_phase") == phase:
            records.append(cast(JsonObject, record))
    return error


def phase_records(path: Path, phase: str) -> tuple[list[JsonObject], str | None]:
    if not path.is_file():
        return [], None
    records: list[JsonObject] = []
    error = _append_phase_records(path, "AIPerf records", phase, records)
    return records, error


def profiling_records(path: Path) -> tuple[list[JsonObject], str | None]:
    return phase_records(path, "profiling")


def raw_phase_records(path: Path, phase: str) -> tuple[list[JsonObject], str | None]:
    records: list[JsonObject] = []
    record_paths = [path] if path.is_file() else sorted(path.glob("raw_records_*.jsonl"))
    for record_path in record_paths:
        error = _append_phase_records(
            record_path, f"AIPerf raw records {record_path.name}", phase, records
        )
        if error is not None:
            return records, error
    return records, None


def request_counts(path: Path) -> tuple[int, int, str | None]:
    records, error = profiling_records(path)
    completed = 0
    failed = 0
    for record in records:
        metadata = record["metadata"]
        cancelled = isinstance(metadata, dict) and metadata.get("was_cancelled") is True
        if record.get("error") is not None or cancelled:
            failed += 1
        else:
            completed += 1
    return completed, failed, error
