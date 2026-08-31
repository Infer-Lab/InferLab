"""Validate the public AIPerf 0.12 AgentX evidence boundary."""

from __future__ import annotations

from pathlib import Path

from inferlab_measurement_sdk import (
    BenchAgenticBranchStats,
    BenchAgenticRunEvidence,
    BenchAgenticSourceInput,
    JsonObject,
)

from .result_records import phase_records, raw_phase_records

# Raw-artifact-derived evidence dimensions recorded as unavailable at the
# performance artifact level (RFC-0005:C-BENCH-AGENTIC-TRACE-EVIDENCE). The
# Rust control plane requires exactly the catalog dimensions plus this list,
# in this order, for performance-level evidence.
PERFORMANCE_UNAVAILABLE_DIMENSIONS = [
    "source_coordinate_mapping",
    "cache_bust_observations",
    "warmup_source_coordinate_records",
]

_BRANCH_FIELDS = (
    "children_spawned",
    "children_completed",
    "children_errored",
    "children_truncated",
    "children_delayed",
    "parents_suspended",
    "parents_resumed",
    "parents_failed_due_to_child_error",
    "joins_suppressed",
)


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"AIPerf AgentX field {field!r} is not a non-negative integer")
    return value


def _metric_count(summary: JsonObject, name: str) -> int:
    raw = summary.get(name)
    if raw is None:
        return 0
    if not isinstance(raw, dict):
        raise ValueError(f"AIPerf AgentX metric {name!r} is not an object")
    value = raw.get("avg")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"AIPerf AgentX metric {name!r} has no numeric avg")
    if not float(value).is_integer() or value < 0:
        raise ValueError(f"AIPerf AgentX metric {name!r} is not a non-negative count")
    return int(value)


def _branch_stats(summary: JsonObject) -> BenchAgenticBranchStats:
    raw = summary.get("branch_stats")
    if not isinstance(raw, dict):
        raise ValueError("AIPerf AgentX aggregate has no branch_stats object")
    values = {field: _nonnegative_integer(raw.get(field), field) for field in _BRANCH_FIELDS}
    return BenchAgenticBranchStats.model_validate(values)


def _scenario_metadata(
    summary: JsonObject, source: BenchAgenticSourceInput
) -> tuple[bool, list[str]]:
    raw = summary.get("metadata")
    if not isinstance(raw, dict):
        raise ValueError("AIPerf AgentX aggregate has no metadata object")
    if raw.get("scenario") != source.catalog.scenario:
        raise ValueError("AIPerf AgentX aggregate scenario does not match the release profile")
    submission_valid = raw.get("submission_valid")
    if not isinstance(submission_valid, bool):
        raise ValueError("AIPerf AgentX aggregate has no boolean submission_valid outcome")
    reasons = raw.get("submission_invalid_reasons", [])
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise ValueError("AIPerf AgentX submission_invalid_reasons is not a string list")
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("AIPerf AgentX aggregate has no dataset provenance")
    if (
        dataset.get("loader") != source.catalog.aiperf_loader
        or dataset.get("hf_dataset_name") != source.catalog.repository
        or dataset.get("num_dataset_entries") != source.catalog.dataset_entries
    ):
        raise ValueError("AIPerf AgentX dataset provenance does not match the release profile")
    return submission_valid, reasons


def _request_coordinates(record: JsonObject, index: int, phase: str) -> tuple[str, str, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"AIPerf AgentX {phase} record {index} has no metadata")
    source_trace_id = metadata.get("source_trace_id")
    source_outer_idx = metadata.get("source_outer_idx")
    source_kind = metadata.get("source_kind")
    request_id = metadata.get("x_request_id")
    conversation_id = metadata.get("x_correlation_id")
    root_id = metadata.get("root_correlation_id")
    if (
        not isinstance(source_trace_id, str)
        or isinstance(source_outer_idx, bool)
        or not isinstance(source_outer_idx, int)
        or not isinstance(source_kind, str)
        or not isinstance(request_id, str)
        or not isinstance(conversation_id, str)
        or not isinstance(root_id, str)
    ):
        raise ValueError(
            f"AIPerf AgentX {phase} record {index} lacks source or runtime coordinates"
        )
    return source_trace_id, conversation_id, request_id


def _runtime_identities(record: JsonObject, index: int, phase: str) -> tuple[str, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"AIPerf AgentX {phase} record {index} has no metadata")
    request_id = metadata.get("x_request_id")
    conversation_id = metadata.get("x_correlation_id")
    if not isinstance(request_id, str) or not isinstance(conversation_id, str):
        raise ValueError(f"AIPerf AgentX {phase} record {index} lacks runtime identities")
    return conversation_id, request_id


def _record_failed(record: JsonObject) -> bool:
    metadata = record.get("metadata")
    return record.get("error") is not None or (
        isinstance(metadata, dict) and metadata.get("was_cancelled") is True
    )


def agentic_result_evidence(
    source: BenchAgenticSourceInput,
    summary: JsonObject,
    summary_path: Path,
    records_path: Path,
    raw_records_path: Path | None,
) -> BenchAgenticRunEvidence:
    submission_valid, invalid_reasons = _scenario_metadata(summary, source)
    raw_available = raw_records_path is not None
    if raw_records_path is not None:
        warmup, warmup_error = raw_phase_records(raw_records_path, "warmup")
        profiling, profiling_error = raw_phase_records(raw_records_path, "profiling")
    else:
        # The performance artifact level produces no raw export; phase,
        # runtime identities, terminal outcomes, and counts read the
        # normalized per-request records instead.
        warmup, warmup_error = phase_records(records_path, "warmup")
        profiling, profiling_error = phase_records(records_path, "profiling")
    parse_error = warmup_error or profiling_error
    if parse_error is not None:
        raise ValueError(parse_error)
    if not profiling:
        artifact = "raw artifact" if raw_available else "records artifact"
        raise ValueError(f"AIPerf AgentX {artifact} has no profiling records")
    warmup_coordinates = 0
    if raw_available:
        warmup_coordinates = len(
            [
                _request_coordinates(record, index, "warmup")
                for index, record in enumerate(warmup, start=1)
            ]
        )
    coordinate_records = 0
    cache_bust_records = 0
    source_traces: set[str] = set()
    runtime_conversations: set[str] = set()
    transport_requests: set[str] = set()
    for index, record in enumerate(profiling, start=1):
        if raw_available:
            source_trace, runtime_conversation, transport_request = _request_coordinates(
                record, index, "profiling"
            )
            source_traces.add(source_trace)
            coordinate_records += 1
        else:
            runtime_conversation, transport_request = _runtime_identities(
                record, index, "profiling"
            )
        runtime_conversations.add(runtime_conversation)
        transport_requests.add(transport_request)
        marker = record.get("cache_bust_marker")
        target = record.get("cache_bust_target")
        if marker is not None or target is not None:
            if not isinstance(marker, str) or not marker or target != source.catalog.cache_bust:
                raise ValueError(
                    f"AIPerf AgentX profiling record {index} has invalid cache-bust evidence"
                )
            cache_bust_records += 1

    context_overflow = _metric_count(summary, "context_overflow_count")
    skipped_context_overflow = _metric_count(summary, "skipped_context_overflow_count")
    failures = _metric_count(summary, "error_request_count")
    metric_path_overflow = max(context_overflow - skipped_context_overflow, 0)
    ordinary_failures = max(failures - metric_path_overflow, 0)
    benchmark_id = summary.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise ValueError("AIPerf AgentX aggregate has no native benchmark_id")
    warmup_errors = sum(_record_failed(record) for record in warmup)
    evidence = BenchAgenticRunEvidence(
        native_run_id=benchmark_id,
        scenario=source.catalog.scenario,
        submission_valid=submission_valid,
        submission_invalid_reasons=invalid_reasons,
        warmup_records=len(warmup),
        warmup_error_records=warmup_errors,
        warmup_succeeded=len(profiling) > 0,
        profiling_records=len(profiling),
        distinct_runtime_conversations=len(runtime_conversations),
        distinct_transport_requests=len(transport_requests),
        context_overflow_count=context_overflow,
        ordinary_failure_count=ordinary_failures,
        branch_stats=_branch_stats(summary),
        aggregate_artifact=str(summary_path),
        raw_records_artifact=str(raw_records_path) if raw_available else None,
        unavailable_dimensions=(
            source.catalog.unavailable_dimensions
            if raw_available
            else [
                *source.catalog.unavailable_dimensions,
                *PERFORMANCE_UNAVAILABLE_DIMENSIONS,
            ]
        ),
        warmup_source_coordinate_records=warmup_coordinates if raw_available else None,
        source_coordinate_records=coordinate_records if raw_available else None,
        distinct_source_traces=len(source_traces) if raw_available else None,
        cache_bust_records=cache_bust_records if raw_available else None,
    )
    return evidence
