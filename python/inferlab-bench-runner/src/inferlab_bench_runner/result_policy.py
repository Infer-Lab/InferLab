"""Adjudicate request SLO and warmup policy from native AIPerf evidence."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    BenchRequestSloInput,
    BenchRequestSloResult,
    JsonObject,
)

from inferlab_bench_runner.result_metrics import metric_value
from inferlab_bench_runner.result_records import profiling_records, raw_phase_records


def record_metric_value(metrics: JsonObject, tag: str) -> tuple[float | None, str | None]:
    raw_metric = metrics.get(tag)
    if raw_metric is None:
        return None, None
    if not isinstance(raw_metric, dict):
        return None, f"AIPerf profiling metric {tag!r} is not an object"
    raw_value = raw_metric.get("value")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None, f"AIPerf profiling metric {tag!r} has no numeric value"
    value = float(raw_value)
    if not math.isfinite(value):
        return None, f"AIPerf profiling metric {tag!r} is not finite"
    return value, None


def required_request_metric_tags(slo: BenchRequestSloInput) -> list[str]:
    tags = []
    if slo.request_latency_ms is not None:
        tags.append("request_latency")
    if slo.ttft_ms is not None:
        tags.append("time_to_first_token")
    if slo.tpot_ms is not None:
        tags.append("inter_token_latency")
    return tags


def request_slo_evidence(
    path: Path,
    expected_requests: int,
    slo: BenchRequestSloInput,
    summary: JsonObject | None,
) -> tuple[int, int, BenchRequestSloResult | None, bool, str | None]:
    records, parse_error = profiling_records(path)
    completed = 0
    failed = 0
    good = 0
    identities: set[int] = set()
    starts: list[int] = []
    ends: list[int] = []
    required_tags = required_request_metric_tags(slo)
    every_failed_request_has_inference_error = True
    error = parse_error
    for index, record in enumerate(records, start=1):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            error = f"AIPerf profiling record {index} has no metadata"
            break
        session_num = metadata.get("session_num")
        start = metadata.get("request_start_ns")
        end = metadata.get("request_end_ns")
        cancelled = metadata.get("was_cancelled")
        if isinstance(session_num, bool) or not isinstance(session_num, int):
            error = f"AIPerf profiling record {index} has no integer session_num"
            break
        if session_num in identities:
            error = f"AIPerf profiling records duplicate session_num {session_num}"
            break
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end < start
        ):
            error = f"AIPerf profiling record {session_num} has invalid terminal timestamps"
            break
        if not isinstance(cancelled, bool):
            error = f"AIPerf profiling record {session_num} has no cancellation status"
            break
        identities.add(session_num)
        starts.append(start)
        ends.append(end)
        inference_error = record.get("error") is not None
        if inference_error or cancelled:
            failed += 1
            every_failed_request_has_inference_error &= inference_error
            continue
        completed += 1
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, dict):
            error = f"AIPerf profiling record {session_num} has no metrics object"
            break
        metrics = cast(JsonObject, raw_metrics)
        missing_required = False
        for tag in required_tags:
            value, metric_error = record_metric_value(metrics, tag)
            if metric_error is not None:
                error = f"AIPerf profiling record {session_num}: {metric_error}"
                break
            missing_required |= value is None
        if error is not None:
            break
        good_value, good_error = record_metric_value(metrics, "good_request_count")
        if good_error is not None:
            error = f"AIPerf profiling record {session_num}: {good_error}"
            break
        if missing_required:
            if good_value == 1.0:
                error = (
                    f"AIPerf profiling record {session_num} is marked good "
                    "without every required request metric"
                )
                break
            continue
        if good_value not in (0.0, 1.0):
            error = (
                f"AIPerf profiling record {session_num} requires an integral "
                "good_request_count of zero or one"
            )
            break
        good += int(good_value)
    if error is None and len(identities) != expected_requests:
        error = (
            "AIPerf profiling request count does not match the resolved case: "
            f"expected={expected_requests}, observed={len(identities)}"
        )
    if error is None and completed + failed != expected_requests:
        error = (
            "AIPerf profiling request counts are inconsistent: "
            f"expected={expected_requests}, completed={completed}, failed={failed}"
        )
    if error is not None:
        return completed, failed, None, False, error
    duration = (max(ends) - min(starts)) / 1_000_000_000
    if not math.isfinite(duration) or duration <= 0.0:
        return completed, failed, None, False, "AIPerf profiling request window is not positive"
    native_good: int | None = None
    native_consistent: bool | None = None
    if summary is not None and summary.get("good_request_count") is not None:
        raw_native_good = metric_value(summary, "good_request_count", "avg")
        if not raw_native_good.is_integer() or not 0.0 <= raw_native_good <= completed:
            return (
                completed,
                failed,
                None,
                False,
                "AIPerf aggregate good_request_count is outside the completed-request range",
            )
        native_good = int(raw_native_good)
        native_consistent = native_good == good
        if not native_consistent:
            return (
                completed,
                failed,
                None,
                False,
                "AIPerf aggregate good_request_count disagrees with per-request records",
            )
    evidence = BenchRequestSloResult(
        good_requests=good,
        good_request_ratio=good / expected_requests,
        goodput=good / duration,
        profiling_duration_seconds=duration,
        profiling_duration_source="native-profiling-request-window",
        request_count_reconciled=True,
        native_aggregate_good_request_count=native_good,
        native_aggregate_good_request_count_consistent=native_consistent,
    )
    return completed, failed, evidence, every_failed_request_has_inference_error, None


@dataclass(frozen=True)
class WarmupCounts:
    expected: int
    observed: int
    completed: int
    errored: int
    cancelled: int
    missing: int
    parse_error: str | None


def warmup_counts(path: Path, expected: int) -> WarmupCounts:
    records, parse_error = raw_phase_records(path, "warmup")
    observed = len(records)
    completed = 0
    errored = 0
    cancelled = 0
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            parse_error = "AIPerf raw warmup record has no metadata"
            break
        was_cancelled = metadata.get("was_cancelled")
        if not isinstance(was_cancelled, bool):
            parse_error = "AIPerf raw warmup record has no cancellation status"
            break
        has_error = record.get("error") is not None
        if has_error:
            errored += 1
        if was_cancelled:
            cancelled += 1
        if not has_error and not was_cancelled:
            completed += 1
    return WarmupCounts(
        expected=expected,
        observed=observed,
        completed=completed,
        errored=errored,
        cancelled=cancelled,
        missing=max(expected - observed, 0),
        parse_error=parse_error,
    )


def warmup_error(counts: WarmupCounts) -> str | None:
    if counts.expected == 0:
        return None
    valid = (
        counts.parse_error is None
        and counts.observed == counts.expected
        and counts.completed == counts.expected
        and counts.errored == 0
        and counts.cancelled == 0
    )
    if valid:
        return None
    detail = (
        "AIPerf warmup failed: "
        f"expected={counts.expected}, completed={counts.completed}, "
        f"errored={counts.errored}, cancelled={counts.cancelled}, "
        f"missing={counts.missing}, observed={counts.observed}"
    )
    if counts.parse_error is not None:
        detail = f"{detail}; {counts.parse_error}"
    return detail
