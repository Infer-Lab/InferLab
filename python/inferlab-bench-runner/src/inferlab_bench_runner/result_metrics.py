"""Normalize the pinned AIPerf summary into InferLab metric names."""

import math

from inferlab_measurement_sdk import JsonObject

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
