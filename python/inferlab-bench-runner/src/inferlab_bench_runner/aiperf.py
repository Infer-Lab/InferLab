"""Own the pinned AIPerf CLI, configuration, artifact, and report boundary."""

import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    BenchClientRequest,
    BenchLoadInputConcurrencyLimited,
    BenchLoadInputRequestRateLimited,
    BenchLoadInputUnboundedRequestRate,
    BenchNativeInvocation,
    BenchPromptRouteInput,
    BenchRequestSloInput,
    BenchRequestSourceInputDataset,
    BenchRequestSourceInputRandom,
    BenchRequestSourceInputRandomMixture,
    BenchTokenSelectorInput,
    BenchTokenSelectorInput1,
    CaseDeadline,
    JsonObject,
    RawArtifact,
    endpoint_url,
    plain_setting,
)

ARTIFACT_PREFIX = "inferlab-bench"
PROFILE_EXPORT_NAME = "profile_export_aiperf.json"
SERVER_METRICS_EXPORT_NAME = "server_metrics_export.json"
SPEED_REPORT_PATHS = {
    "acceptance_length": ("accept_length", "speed-bench-acceptance-length.csv"),
    "acceptance_rate": ("accept_rate", "speed-bench-acceptance-rate.csv"),
}


@dataclass(frozen=True)
class AiperfRequestPopulation:
    dataset: JsonObject
    tpot_applicable: bool


@dataclass(frozen=True)
class AiperfProfileArtifacts:
    summary: Path
    records: Path
    raw_records: Path


@dataclass(frozen=True)
class PreparedAiperfExecution:
    artifact_dir: Path
    config_path: Path
    request_config_path: Path
    command_prefix: list[str]
    command: list[str]
    population: AiperfRequestPopulation
    profile_artifacts: AiperfProfileArtifacts
    environment: dict[str, str]


def profiling_config(request: BenchClientRequest) -> JsonObject:
    load = request.case.load_shape.root
    requests = request.case.request_count
    agentic_source = request.definition.agentic_source
    if agentic_source is not None:
        if not isinstance(load, BenchLoadInputConcurrencyLimited):
            raise ValueError("agentic replay requires concurrency-limited load")
        if request.case.duration_seconds is None:
            raise ValueError("agentic replay requires an effective duration")
        policy = agentic_source.catalog
        return {
            "type": "concurrency",
            "concurrency": load.concurrency,
            "duration": request.case.duration_seconds,
            "failedRequestThreshold": policy.failure_threshold,
            "trajectoryStartMinRatio": policy.trajectory_start_min,
            "trajectoryStartMaxRatio": policy.trajectory_start_max,
            "systemIdleGapCapSeconds": policy.global_idle_gap_cap_seconds,
            "agenticCacheWarmupDuration": policy.cache_warmup_seconds,
            "agenticWarmupGracePeriod": policy.warmup_grace_seconds,
        }
    if request.case.session_count is not None:
        if not isinstance(load, BenchLoadInputConcurrencyLimited):
            raise ValueError("linear sessions require concurrency-limited load")
        return {
            "type": "concurrency",
            "concurrency": load.concurrency,
            "sessions": request.case.session_count,
        }
    if isinstance(load, BenchLoadInputConcurrencyLimited):
        return {"type": "concurrency", "concurrency": load.concurrency, "requests": requests}
    if isinstance(load, BenchLoadInputRequestRateLimited):
        if load.burstiness is None:
            return {"type": "poisson", "rate": load.request_rate, "requests": requests}
        return {
            "type": "gamma",
            "rate": load.request_rate,
            "smoothness": load.burstiness,
            "requests": requests,
        }
    if isinstance(load, BenchLoadInputUnboundedRequestRate):
        return {"type": "concurrency", "concurrency": requests, "requests": requests}
    raise TypeError(f"unsupported Bench load shape {type(load).__name__}")


def aiperf_client_defaults(request: BenchClientRequest) -> JsonObject:
    defaults: JsonObject = {
        "ignore_eos": True,
        "n": 1,
        "stream_options": {"include_usage": True},
    }
    source_input = request.definition.request_source
    if source_input is None:
        return defaults
    source = source_input.root
    if isinstance(source, BenchRequestSourceInputRandom):
        output_tokens = fixed_tokens(source.output_tokens)
        if output_tokens is not None:
            defaults["min_tokens"] = output_tokens
    elif isinstance(source, BenchRequestSourceInputDataset) and source.output_tokens is not None:
        defaults["min_tokens"] = source.output_tokens
        defaults["max_tokens"] = source.output_tokens
    return defaults


def fixed_tokens(selector: BenchTokenSelectorInput) -> int | None:
    value = selector.root
    if isinstance(value, BenchTokenSelectorInput1):
        return value.root
    return None


def merge_request_body(defaults: JsonObject, fragment: JsonObject) -> JsonObject:
    merged = dict(defaults)
    for key, replacement in fragment.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(replacement, dict):
            merged[key] = merge_request_body(
                cast(JsonObject, current), cast(JsonObject, replacement)
            )
        else:
            merged[key] = replacement
    return merged


def replaced_defaults(
    defaults: JsonObject, fragment: JsonObject, parent: str = ""
) -> list[JsonObject]:
    replacements: list[JsonObject] = []
    for key, replacement in fragment.items():
        if key not in defaults:
            continue
        path = f"{parent}.{key}" if parent else key
        earlier = defaults[key]
        if isinstance(earlier, dict) and isinstance(replacement, dict):
            replacements.extend(
                replaced_defaults(cast(JsonObject, earlier), cast(JsonObject, replacement), path)
            )
        else:
            replacements.append(
                {
                    "path": path,
                    "earlier": earlier,
                    "earlier_authority": "pinned AIPerf chat endpoint",
                    "replacement": replacement,
                    "replacement_authority": "effective Bench definition request_body",
                }
            )
    return replacements


def effective_request_body(request: BenchClientRequest) -> JsonObject:
    fragment: JsonObject = {
        key: plain_setting(value) for key, value in request.definition.request_body.items()
    }
    return merge_request_body(aiperf_client_defaults(request), fragment)


def inference_request_config(request: BenchClientRequest) -> JsonObject:
    definition_body: JsonObject = {
        key: plain_setting(value) for key, value in request.definition.request_body.items()
    }
    selected_name, selected_path, _ = selected_endpoint(request)
    prompt = request.definition.prompt.root
    return {
        "schema_version": 1,
        "selected_named_route": selected_name,
        "effective_public_url": endpoint_url(request.endpoint, selected_path),
        "prompt_authority": {
            "kind": prompt.kind,
            "request_representation": prompt.request_representation.value,
            "route": prompt.route.value,
            "rendering_authority": prompt.rendering_authority.value,
        },
        "definition_request_body": definition_body,
        "aiperf_client_defaults": aiperf_client_defaults(request),
        "effective_request_body": effective_request_body(request),
        "replaced_defaults": replaced_defaults(aiperf_client_defaults(request), definition_body),
    }


def selected_endpoint(request: BenchClientRequest) -> tuple[str, str, str]:
    if request.definition.prompt.root.route == BenchPromptRouteInput.completions:
        return "completions_path", request.endpoint.completions_path, "completions"
    return "chat_completions_path", request.endpoint.chat_completions_path, "chat"


def aiperf_slos(slo: BenchRequestSloInput) -> JsonObject:
    values: JsonObject = {}
    if slo.request_latency_ms is not None:
        values["request_latency"] = slo.request_latency_ms
    if slo.ttft_ms is not None:
        values["time_to_first_token"] = slo.ttft_ms
    if slo.tpot_ms is not None:
        values["inter_token_latency"] = slo.tpot_ms
    return values


def aiperf_session_population_layout(
    warmup_sessions: int, profiling_sessions: int
) -> tuple[int, int]:
    profiling_start = warmup_sessions + (1 if warmup_sessions > 0 else 0)
    return profiling_start, profiling_start + profiling_sessions


def aiperf_endpoint_route(request: BenchClientRequest, selected_path: str) -> tuple[str, str]:
    """Align AIPerf's inference base with its derived trailing /metrics route."""
    endpoint = request.endpoint
    if not request.definition.server_metrics:
        return endpoint_url(endpoint, ""), selected_path

    server_metrics = endpoint.server_metrics
    if server_metrics is None:
        raise ValueError("server metrics requested without an endpoint capability")
    metrics_path = server_metrics.path
    suffix = "/metrics"
    if not metrics_path.endswith(suffix):
        raise ValueError("pinned AIPerf cannot address the integration server metrics path exactly")
    base_path = metrics_path.removesuffix(suffix)
    if base_path and not selected_path.startswith(f"{base_path}/"):
        raise ValueError("pinned AIPerf cannot address the integration server metrics path exactly")
    return endpoint_url(endpoint, base_path), selected_path.removeprefix(base_path)


def resolve_aiperf_population(request: BenchClientRequest) -> AiperfRequestPopulation:
    source_input = request.definition.request_source
    session_source = request.definition.session_source
    agentic_source = request.definition.agentic_source
    selected_sources = sum(
        source is not None for source in (source_input, session_source, agentic_source)
    )
    if selected_sources != 1:
        raise ValueError("Bench request requires exactly one source boundary")
    if agentic_source is not None:
        if request.population is not None:
            raise ValueError("agentic replay does not accept an InferLab population")
        return AiperfRequestPopulation(
            dataset={
                "type": "public",
                "dataset": agentic_source.catalog.aiperf_loader,
                "entries": agentic_source.catalog.dataset_entries,
                "sampling": "sequential",
            },
            tpot_applicable=False,
        )
    if session_source is not None:
        _, entries = aiperf_session_population_layout(
            request.case.warmup_session_count or 0,
            request.case.session_count or 0,
        )
    else:
        entries = request.case.warmup_request_count + request.case.request_count
    if request.population is not None:
        if request.population.entries < entries:
            raise ValueError(
                f"Bench case requires {entries} entries, "
                f"population has {request.population.entries}"
            )
        source = source_input.root if source_input is not None else None
        if session_source is not None:
            source_format = "multi_turn"
        elif isinstance(source, BenchRequestSourceInputDataset):
            source_format = source.catalog.aiperf_format
        else:
            source_format = "mooncake_trace"
        return AiperfRequestPopulation(
            dataset={
                "type": "file",
                "path": str(request.population.path),
                "format": source_format,
                "entries": entries,
                "sampling": "sequential",
            },
            tpot_applicable=request.population.tpot_applicable,
        )
    if source_input is None:
        raise ValueError("linear-session Bench request has no materialized population")
    source = source_input.root
    if isinstance(source, BenchRequestSourceInputRandom):
        input_tokens = fixed_tokens(source.input_tokens)
        output_tokens = fixed_tokens(source.output_tokens)
        if input_tokens is None or output_tokens is None:
            raise ValueError("variable random shapes require a materialized population")
        if source.prefix_sharing is not None or source.shared_system_content is not None:
            raise ValueError("synthetic sharing requires a materialized population")
        dataset: JsonObject = {
            "type": "synthetic",
            "entries": entries,
            "randomSeed": request.definition.seed,
            "sampling": "sequential",
            "prompts": {
                "isl": input_tokens,
                "osl": output_tokens,
            },
        }
        tpot_applicable = output_tokens >= 2
    elif isinstance(source, BenchRequestSourceInputRandomMixture):
        if source.prefix_sharing is not None:
            raise ValueError("synthetic sharing requires a materialized population")
        if source.total_weight <= 0:
            raise ValueError("random_mixture Bench request has no positive total weight")
        probabilities: list[JsonObject] = [
            {
                "isl": shape.input_tokens,
                "osl": shape.output_tokens,
                "probability": (100.0 * float(shape.weight) / float(source.total_weight)),
            }
            for shape in source.shapes
        ]
        dataset = {
            "type": "synthetic",
            "entries": entries,
            "randomSeed": request.definition.seed,
            "sampling": "sequential",
            "prompts": {
                "isl": source.shapes[0].input_tokens,
                "osl": source.shapes[0].output_tokens,
                "sequenceDistribution": probabilities,
            },
        }
        tpot_applicable = bool(source.shapes and source.shapes[0].output_tokens >= 2)
    elif isinstance(source, BenchRequestSourceInputDataset):
        raise ValueError("dataset Bench request has no materialized population")
    else:
        raise TypeError(f"unsupported Bench request source {type(source).__name__}")
    return AiperfRequestPopulation(dataset=dataset, tpot_applicable=tpot_applicable)


def aiperf_config(
    request: BenchClientRequest,
    deadline: CaseDeadline | None = None,
    population: AiperfRequestPopulation | None = None,
) -> JsonObject:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    population = population or resolve_aiperf_population(request)
    endpoint = request.endpoint
    definition = request.definition
    _, selected_path, endpoint_type = selected_endpoint(request)
    url, aiperf_path = aiperf_endpoint_route(request, selected_path)
    endpoint_extra = effective_request_body(request)
    chat_template = endpoint_extra.get("chat_template")
    if endpoint_type == "chat" and isinstance(chat_template, str):
        endpoint_extra["chat_template"] = "{{ " + json.dumps(chat_template) + " }}"
    endpoint_config: JsonObject = {
        "url": url,
        "path": aiperf_path,
        "type": endpoint_type,
        "streaming": True,
        "timeout": deadline.remaining(),
        "useServerTokenCount": True,
        "extra": endpoint_extra,
    }
    artifacts: JsonObject = {
        "dir": str(request.artifact_dir),
        "summary": ["json"],
        "records": ["jsonl"],
        "raw": True,
    }
    if not definition.server_metrics:
        artifacts["prefix"] = ARTIFACT_PREFIX
    benchmark: JsonObject = {
        "model": request.model.served_name,
        "endpoint": endpoint_config,
        "dataset": population.dataset,
        "profiling": profiling_config(request),
        "tokenizer": {"name": request.model.locator},
        "runtime": {"ui": "none", "workers": 1, "recordProcessors": 1},
        "gpuTelemetry": {"enabled": False},
        "serverMetrics": {"enabled": False},
        "artifacts": artifacts,
    }
    if definition.agentic_source is not None:
        benchmark["scenario"] = definition.agentic_source.catalog.scenario
    if definition.server_metrics:
        server_metrics = endpoint.server_metrics
        if server_metrics is None:
            raise ValueError("server metrics requested without an endpoint capability")
        benchmark["serverMetrics"] = {
            "enabled": True,
            "urls": [server_metrics.url],
            "formats": ["json"],
            "discovery": {"mode": "disabled"},
        }
        if definition.agentic_source is not None:
            artifacts["sliceDuration"] = (
                definition.agentic_source.catalog.server_metric_slice_seconds
            )
    if definition.request_slo is not None:
        benchmark["slos"] = aiperf_slos(definition.request_slo)
    warmup_sessions = request.case.warmup_session_count
    if warmup_sessions is not None and warmup_sessions > 0:
        load = request.case.load_shape.root
        if not isinstance(load, BenchLoadInputConcurrencyLimited):
            raise ValueError("native session warmup requires concurrency-limited load")
        benchmark["warmup"] = {
            "type": "concurrency",
            "concurrency": load.concurrency,
            "sessions": warmup_sessions,
        }
    elif request.case.warmup_request_count > 0:
        load = request.case.load_shape.root
        if not isinstance(load, BenchLoadInputConcurrencyLimited):
            raise ValueError("native warmup requires a concurrency-limited Bench case")
        benchmark["warmup"] = {
            "type": "concurrency",
            "concurrency": load.concurrency,
            "requests": request.case.warmup_request_count,
        }
    return {
        "schemaVersion": "2.0",
        "randomSeed": definition.seed,
        "benchmark": benchmark,
    }


def raw_artifacts(
    artifact_dir: Path,
    config_path: Path,
    request_config_path: Path,
    profile_artifacts: AiperfProfileArtifacts,
) -> list[RawArtifact]:
    summary_name, summary_kind = (
        ("aiperf_profile_export", "aiperf-profile-export")
        if profile_artifacts.summary.name == PROFILE_EXPORT_NAME
        else ("aiperf_summary", "aiperf-summary")
    )
    candidates = [
        ("aiperf_config", "aiperf-config", config_path),
        ("inference_request", "inference-request-config", request_config_path),
        (summary_name, summary_kind, profile_artifacts.summary),
        ("aiperf_records", "aiperf-records", profile_artifacts.records),
        ("aiperf_raw_records", "aiperf-raw-records", profile_artifacts.raw_records),
        ("aiperf_partial_raw_records", "directory", artifact_dir / "raw_records"),
        ("aiperf_inputs", "aiperf-inputs", artifact_dir / "inputs.json"),
        ("aiperf_logs", "directory", artifact_dir / "logs"),
        (
            "aiperf_server_metrics_export",
            "aiperf-server-metrics-export",
            artifact_dir / SERVER_METRICS_EXPORT_NAME,
        ),
        (
            "speed_bench_acceptance_length",
            "speed-bench-report-csv",
            artifact_dir / SPEED_REPORT_PATHS["acceptance_length"][1],
        ),
        (
            "speed_bench_acceptance_rate",
            "speed-bench-report-csv",
            artifact_dir / SPEED_REPORT_PATHS["acceptance_rate"][1],
        ),
    ]
    return [
        RawArtifact(name=name, kind=kind, path=str(path))
        for name, kind, path in candidates
        if path.exists()
    ]


def run_aiperf(
    command: list[str],
    deadline: CaseDeadline,
    environment: Mapping[str, str] | None = None,
) -> tuple[int, bool, bool]:
    termination_requested = False
    timed_out = False

    def request_termination(_signal: int, _frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_termination)
    try:
        process_environment = None
        if environment is not None:
            process_environment = {**os.environ, **environment}
        if process_environment is None:
            process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr)
        else:
            process = subprocess.Popen(
                command,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=process_environment,
            )
        while process.poll() is None and not termination_requested and not timed_out:
            try:
                time.sleep(min(0.05, deadline.remaining()))
            except TimeoutError:
                timed_out = True
        if (termination_requested or timed_out) and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
        return process.wait(), termination_requested, timed_out
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def speed_bench_category(request: BenchClientRequest) -> str | None:
    source_input = request.definition.request_source
    if source_input is None:
        return None
    source = source_input.root
    if not isinstance(source, BenchRequestSourceInputDataset) or source.dataset != "speed_bench":
        return None
    prefix = "speed_bench_"
    if not source.catalog.aiperf_format.startswith(prefix):
        raise ValueError("resolved SPEED-Bench profile has an invalid AIPerf dataset format")
    return source.catalog.aiperf_format.removeprefix(prefix)


def parse_speed_bench_report(
    path: Path,
    served_model: str,
    category: str,
    normalized_name: str,
) -> float:
    try:
        with path.open(encoding="utf-8", newline="") as report_file:
            reader = csv.DictReader(report_file)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except (OSError, csv.Error) as error:
        raise ValueError(f"{normalized_name} report {path} is unreadable: {error}") from error
    if fieldnames.count("Model") != 1:
        raise ValueError(f"{normalized_name} report {path} has no unique Model column")
    if fieldnames.count(category) != 1:
        raise ValueError(
            f"{normalized_name} report {path} has no unique category column {category!r}"
        )
    model_rows = [row for row in rows if row.get("Model") == served_model]
    if len(model_rows) != 1 or len(rows) != 1:
        raise ValueError(
            f"{normalized_name} report {path} requires exactly one row for model "
            f"{served_model!r}; matching={len(model_rows)}, total={len(rows)}"
        )
    raw_value = model_rows[0].get(category)
    if raw_value is None or not raw_value.strip():
        raise ValueError(
            f"{normalized_name} report {path} cell for {served_model!r}/{category!r} is empty"
        )
    try:
        value = float(raw_value)
    except ValueError:
        raise ValueError(
            f"{normalized_name} report {path} cell for {served_model!r}/{category!r} "
            f"is not numeric: {raw_value!r}"
        ) from None
    if not math.isfinite(value):
        raise ValueError(
            f"{normalized_name} report {path} cell for {served_model!r}/{category!r} is not finite"
        )
    if normalized_name == "acceptance_length" and value < 1.0:
        raise ValueError(f"acceptance_length report cell is below one: {value}")
    if normalized_name == "acceptance_rate" and not 0.0 <= value <= 1.0:
        raise ValueError(f"acceptance_rate report cell is outside [0, 1]: {value}")
    return value


def run_speed_bench_reports(
    request: BenchClientRequest,
    command_prefix: list[str],
    artifact_dir: Path,
    deadline: CaseDeadline,
) -> tuple[dict[str, float], list[BenchNativeInvocation], str | None]:
    category = speed_bench_category(request)
    if category is None:
        return {}, [], None
    metrics: dict[str, float] = {}
    invocations: list[BenchNativeInvocation] = []
    errors: list[str] = []
    for normalized_name, (report_metric, filename) in SPEED_REPORT_PATHS.items():
        output_path = artifact_dir / filename
        command = [
            *command_prefix,
            "speed-bench-report",
            str(artifact_dir),
            "--output",
            str(output_path),
            "--format",
            "csv",
            "--metric",
            report_metric,
        ]
        try:
            exit_code, interrupted, timed_out = run_aiperf(command, deadline)
        except OSError as error:
            invocations.append(
                BenchNativeInvocation(
                    purpose=normalized_name,
                    command=command,
                    exit_code=None,
                    interrupted=False,
                    timed_out=False,
                )
            )
            errors.append(f"failed to launch {normalized_name} report: {error}")
            continue
        invocations.append(
            BenchNativeInvocation(
                purpose=normalized_name,
                command=command,
                exit_code=exit_code,
                interrupted=interrupted,
                timed_out=timed_out,
            )
        )
        if interrupted:
            return metrics, invocations, f"{normalized_name} report was interrupted"
        if timed_out:
            return metrics, invocations, f"{normalized_name} report reached the case deadline"
        if exit_code != 0:
            errors.append(f"{normalized_name} report exited with {exit_code}")
            continue
        try:
            metrics[normalized_name] = parse_speed_bench_report(
                output_path,
                request.model.served_name,
                category,
                normalized_name,
            )
        except ValueError as error:
            errors.append(str(error))
    return metrics, invocations, "; ".join(errors) or None


def prepare_aiperf_execution(
    request: BenchClientRequest, deadline: CaseDeadline
) -> PreparedAiperfExecution:
    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    population = resolve_aiperf_population(request)
    config_path = artifact_dir / "aiperf-config.json"
    config_path.write_text(
        json.dumps(aiperf_config(request, deadline, population), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    request_config_path = artifact_dir / "inference-request.json"
    request_config_path.write_text(
        json.dumps(inference_request_config(request), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command_prefix = [
        sys.executable,
        "-m",
        "inferlab_bench_runner.aiperf_entrypoint",
    ]
    profile_artifacts = (
        AiperfProfileArtifacts(
            summary=artifact_dir / PROFILE_EXPORT_NAME,
            records=artifact_dir / "profile_export.jsonl",
            raw_records=artifact_dir / "profile_export_raw.jsonl",
        )
        if request.definition.server_metrics
        else AiperfProfileArtifacts(
            summary=artifact_dir / f"{ARTIFACT_PREFIX}.json",
            records=artifact_dir / f"{ARTIFACT_PREFIX}.jsonl",
            raw_records=artifact_dir / f"{ARTIFACT_PREFIX}_raw.jsonl",
        )
    )
    environment: dict[str, str] = {}
    if request.definition.agentic_source is not None:
        catalog = request.definition.agentic_source.catalog
        environment = {
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": str(
                catalog.dataset_configuration_timeout_seconds
            ),
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": str(
                catalog.service_profile_configuration_timeout_seconds
            ),
        }
    return PreparedAiperfExecution(
        artifact_dir=artifact_dir,
        config_path=config_path,
        request_config_path=request_config_path,
        command_prefix=command_prefix,
        command=[*command_prefix, "profile", "--config", str(config_path)],
        population=population,
        profile_artifacts=profile_artifacts,
        environment=environment,
    )
