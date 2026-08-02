"""Read the pinned AIPerf raw-record artifacts by benchmark phase."""

import json
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import JsonObject


def profiling_records(path: Path) -> tuple[list[JsonObject], str | None]:
    if not path.is_file():
        return [], None
    records: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            return records, f"invalid AIPerf records JSONL line {line_number}: {error}"
        if not isinstance(record, dict):
            return records, f"invalid AIPerf records JSONL object at line {line_number}"
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return records, f"AIPerf records JSONL line {line_number} has no metadata"
        if metadata.get("benchmark_phase") == "profiling":
            records.append(cast(JsonObject, record))
    return records, None


def raw_phase_records(path: Path, phase: str) -> tuple[list[JsonObject], str | None]:
    records: list[JsonObject] = []
    record_paths = [path] if path.is_file() else sorted(path.glob("raw_records_*.jsonl"))
    for record_path in record_paths:
        for line_number, line in enumerate(
            record_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                return records, f"invalid AIPerf raw records JSONL line {line_number}: {error}"
            if not isinstance(record, dict):
                return records, f"invalid AIPerf raw records JSONL object at line {line_number}"
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                return records, f"AIPerf raw records JSONL line {line_number} has no metadata"
            if metadata.get("benchmark_phase") == phase:
                records.append(cast(JsonObject, record))
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
