"""Normalize the pinned AIPerf summary into InferLab metric names."""

import importlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from inferlab_measurement_sdk import (
    BenchPromptCacheObservation,
    JsonObject,
    PromptCacheReadZeroRepresentation,
)

from inferlab_bench_runner.result_records import profiling_records

NORMALIZATION_SCHEMA = "aiperf-summary-v1"
SCALAR_METRIC_PATHS: dict[str, tuple[str, str]] = {
    "request_throughput": ("request_throughput", "avg"),
    "output_throughput": ("output_token_throughput", "avg"),
    "total_token_throughput": ("total_token_throughput", "avg"),
}
DISTRIBUTION_SECTIONS = {
    "prompt_tokens": "input_sequence_length",
    "request_latency_ms": "request_latency",
    "ttft_ms": "time_to_first_token",
    "tpot_ms": "inter_token_latency",
}
DISTRIBUTION_STATISTICS = {
    "mean": "avg",
    "min": "min",
    "max": "max",
    "stddev": "std",
    "p50": "p50",
    "p90": "p90",
    "p95": "p95",
    "p99": "p99",
}
CACHE_READ_PERCENT_SECTION = "overall_usage_prompt_cache_read_pct"


class NativeMetricResult(Protocol):
    avg: float
    min: float
    max: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float


def _aiperf_metric_result(family: str, values: list[int]) -> NativeMetricResult:
    numpy = importlib.import_module("numpy")
    metric_dicts = importlib.import_module("aiperf.metrics.metric_dicts")
    asarray = cast(Callable[..., object], numpy.asarray)
    factory = cast(Callable[..., NativeMetricResult], metric_dicts.metric_result_from_array)
    array = asarray(values, dtype="float64")
    return factory(family, family, "tokens", array, float(sum(values)))


def metric_value(summary: JsonObject, section: str, statistic: str) -> float:
    raw_section = summary.get(section)
    if not isinstance(raw_section, dict):
        raise ValueError(f"AIPerf summary has no {section} object")
    raw_value = raw_section.get(statistic)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"AIPerf summary has no numeric {section}.{statistic}")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"AIPerf summary {section}.{statistic} is not finite")
    return value


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


def normalize_summary(summary: JsonObject, tpot_applicable: bool) -> dict[str, float]:
    metrics = {
        target: metric_value(summary, section, statistic)
        for target, (section, statistic) in SCALAR_METRIC_PATHS.items()
    }
    for family, section in DISTRIBUTION_SECTIONS.items():
        if family == "tpot_ms" and not tpot_applicable:
            continue
        for prefix, statistic in DISTRIBUTION_STATISTICS.items():
            metrics[f"{prefix}_{family}"] = metric_value(summary, section, statistic)

    cache_section = summary.get(CACHE_READ_PERCENT_SECTION)
    if cache_section is not None:
        cache_percent = metric_value(summary, CACHE_READ_PERCENT_SECTION, "avg")
        if not 0.0 <= cache_percent <= 100.0:
            raise ValueError(f"AIPerf summary {CACHE_READ_PERCENT_SECTION}.avg is outside [0, 100]")
        metrics["prompt_cache_read_ratio"] = cache_percent / 100.0
    return metrics


def prompt_cache_evidence(
    path: Path,
    required: bool,
    zero_representation: PromptCacheReadZeroRepresentation | None,
    shared_prefix_by_session: dict[int, tuple[int, int | None]] | None = None,
) -> tuple[list[BenchPromptCacheObservation], dict[str, float], str | None]:
    """Per-request backend prompt and cache-read usage. For a generated
    population the observation also carries the entry's shared-prefix tokens,
    its prefix group, and the request's send and first-token times, from which the control
    plane derives reuse opportunity ([[RFC-0004:C-BENCH-CACHE-STATE]])."""
    if not required:
        return [], {}, None
    records, parse_error = profiling_records(path)
    if parse_error is not None:
        return [], {}, parse_error
    observations: list[BenchPromptCacheObservation] = []
    for index, record in enumerate(records, start=1):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return [], {}, f"AIPerf profiling record {index} has no metadata"
        if record.get("error") is not None or metadata.get("was_cancelled") is True:
            continue
        request_id = metadata.get("session_num")
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
            return [], {}, f"AIPerf profiling record {index} has no unsigned session_num"
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, dict):
            return [], {}, f"AIPerf profiling record {request_id} has no metrics object"
        prompt_tokens, prompt_error = record_metric_value(raw_metrics, "usage_prompt_tokens")
        cache_tokens, cache_error = record_metric_value(
            raw_metrics, "usage_prompt_cache_read_tokens"
        )
        if prompt_error is not None or cache_error is not None:
            return [], {}, (f"AIPerf profiling record {request_id}: {prompt_error or cache_error}")
        if prompt_tokens is None:
            return (
                [],
                {},
                (f"AIPerf profiling record {request_id} omitted backend prompt usage"),
            )
        if cache_tokens is None:
            if zero_representation is PromptCacheReadZeroRepresentation.omitted:
                cache_tokens = 0.0
            else:
                return (
                    [],
                    {},
                    (
                        f"AIPerf profiling record {request_id} omitted backend cache usage; "
                        "the server does not report prompt cache-read usage — enable the "
                        "serving integration's cache-read reporting setting and rebuild the "
                        "server"
                    ),
                )
        if not prompt_tokens.is_integer() or prompt_tokens <= 0:
            return [], {}, f"AIPerf profiling record {request_id} has invalid prompt usage"
        if not cache_tokens.is_integer() or cache_tokens < 0 or cache_tokens > prompt_tokens:
            return [], {}, f"AIPerf profiling record {request_id} has invalid cache-read usage"
        prompt = int(prompt_tokens)
        cache = int(cache_tokens)
        planned = shared_prefix_by_session.get(request_id) if shared_prefix_by_session else None
        shared, group = planned if planned is not None else (None, None)
        sent, first_token = _request_times(metadata, raw_metrics)
        observations.append(
            BenchPromptCacheObservation(
                request_id=request_id,
                prompt_tokens=prompt,
                cache_read_tokens=cache,
                uncached_prompt_tokens=prompt - cache,
                cache_read_ratio=cache / prompt,
                shared_prefix_tokens=shared,
                prefix_group=group,
                sent_unix_ns=sent if shared is not None else None,
                first_token_unix_ns=first_token if shared is not None else None,
            )
        )
    if not observations:
        return [], {}, "AIPerf produced no completed profiling cache observations"
    metrics: dict[str, float] = {}
    for family, values in (
        ("prompt_cache_read_tokens", [item.cache_read_tokens for item in observations]),
        ("uncached_prompt_tokens", [item.uncached_prompt_tokens for item in observations]),
    ):
        result = _aiperf_metric_result(family, values)
        for prefix, statistic in DISTRIBUTION_STATISTICS.items():
            metrics[f"{prefix}_{family}"] = float(getattr(result, statistic))
    total_prompt = sum(item.prompt_tokens for item in observations)
    total_cache = sum(item.cache_read_tokens for item in observations)
    metrics["prompt_cache_read_ratio"] = total_cache / total_prompt
    return observations, metrics, None


def draft_acceptance_metrics(path: Path) -> tuple[dict[str, float], str | None]:
    """Token-weighted speculative draft acceptance from per-request usage
    ([[RFC-0004:C-MEASUREMENTS]]): a request contributes when its backend
    reported accepted and rejected prediction tokens; drafted tokens are their
    sum. Acceptance length needs verify-step counts, which usage does not
    carry, so it is not derived here."""
    records, parse_error = profiling_records(path)
    if parse_error is not None:
        return {}, parse_error
    accepted_total = 0
    drafted_total = 0
    for record in records:
        metadata = record.get("metadata")
        if record.get("error") is not None or (
            isinstance(metadata, dict) and metadata.get("was_cancelled") is True
        ):
            continue
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, dict):
            continue
        accepted, accepted_error = record_metric_value(
            raw_metrics, "usage_accepted_prediction_tokens"
        )
        rejected, rejected_error = record_metric_value(
            raw_metrics, "usage_rejected_prediction_tokens"
        )
        if accepted_error is not None or rejected_error is not None:
            return {}, accepted_error or rejected_error
        if accepted is None and rejected is None:
            continue
        if accepted is None or rejected is None:
            return (
                {},
                "AIPerf profiling record reports only one of accepted and rejected draft tokens",
            )
        if not accepted.is_integer() or not rejected.is_integer() or accepted < 0 or rejected < 0:
            return {}, "AIPerf profiling record has invalid draft token usage"
        accepted_total += int(accepted)
        drafted_total += int(accepted + rejected)
    if drafted_total == 0:
        return {}, None
    return {
        "draft_accepted_tokens": float(accepted_total),
        "draft_tokens": float(drafted_total),
        "draft_acceptance_rate": accepted_total / drafted_total,
    }, None


def _request_times(
    metadata: dict[str, object], raw_metrics: dict[str, object]
) -> tuple[int | None, int | None]:
    """AIPerf's wall-clock send time and the first-token time it implies."""
    sent = metadata.get("request_start_ns")
    if isinstance(sent, bool) or not isinstance(sent, int) or sent <= 0:
        return None, None
    ttft, error = record_metric_value(raw_metrics, "time_to_first_token")
    if error is not None or ttft is None or ttft < 0:
        return sent, None
    return sent, sent + round(ttft * 1_000_000)
