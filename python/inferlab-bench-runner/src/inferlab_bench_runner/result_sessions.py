"""Reconcile linear-session turn evidence with the frozen population."""

import json
import math
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    BenchArtifactLevelInput,
    BenchClientRequest,
    BenchRuntimeSessionResult,
    BenchSessionPhaseSummary,
    BenchSessionResultEvidence,
    BenchSessionTurnResult,
    ClientStatus,
    JsonObject,
)

from inferlab_bench_runner.aiperf import aiperf_session_population_layout
from inferlab_bench_runner.chat_tokens import ContentTokenizer, messages_content_tokens
from inferlab_bench_runner.result_records import phase_records, profiling_records, raw_phase_records

# Raw-artifact-derived evidence dimensions recorded as unavailable at the
# performance artifact level (RFC-0005:C-BENCH-LINEAR-SESSION-EVIDENCE). The
# Rust control plane requires exactly this list for performance-level evidence.
PERFORMANCE_UNAVAILABLE_DIMENSIONS = [
    "pre_template_content_tokens",
    "max_input_tokens_bound_check",
    "preceding_live_response_pairwise_history",
    "raw_native_request_reconciliation",
]


def _session_population(
    request: BenchClientRequest,
) -> tuple[list[tuple[str, list[float]]], list[str]]:
    errors: list[str] = []
    if request.population is None:
        return [], ["linear-session result has no frozen population"]
    rows: list[tuple[str, list[float]]] = []
    try:
        lines = Path(request.population.path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], [f"linear-session population is unreadable: {error}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"linear-session population line {line_number} is invalid: {error}")
            continue
        if not isinstance(value, dict):
            errors.append(f"linear-session population line {line_number} is not an object")
            continue
        identity = value.get("session_id")
        turns = value.get("turns")
        if not isinstance(identity, str) or not identity or not isinstance(turns, list):
            errors.append(
                f"linear-session population line {line_number} has no session_id or turns"
            )
            continue
        delays: list[float] = []
        valid_turns = True
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                errors.append(
                    f"linear-session population {identity!r} turn {turn_index} is not an object"
                )
                valid_turns = False
                break
            raw_delay = turn.get("delay", 0.0)
            if isinstance(raw_delay, bool) or not isinstance(raw_delay, (int, float)):
                errors.append(
                    f"linear-session population {identity!r} turn {turn_index} has no numeric delay"
                )
                valid_turns = False
                break
            delay = float(raw_delay) / 1000.0
            if not math.isfinite(delay) or delay < 0.0:
                errors.append(
                    f"linear-session population {identity!r} turn {turn_index} has invalid delay"
                )
                valid_turns = False
                break
            delays.append(delay)
        if valid_turns:
            rows.append((identity, delays))
    summaries = request.population.session_templates
    if len(summaries) != len(rows):
        errors.append(
            "linear-session population template summaries do not cover every population row"
        )
    for index, (identity, delays) in enumerate(rows[: len(summaries)]):
        summary = summaries[index]
        if summary.template_identity != identity or summary.turn_count != len(delays):
            errors.append(
                f"linear-session population summary {index} disagrees with its frozen row"
            )
    return rows, errors


def _pre_template_content_tokens(record: JsonObject, tokenizer: ContentTokenizer) -> int | None:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    return messages_content_tokens(payload.get("messages"), tokenizer)


def _response_objects(record: JsonObject) -> list[JsonObject]:
    responses = record.get("responses")
    if not isinstance(responses, list):
        return []
    values: list[JsonObject] = []
    for response in responses:
        if not isinstance(response, dict):
            continue
        candidates: list[str] = []
        text_value = response.get("text")
        if isinstance(text_value, str):
            candidates.append(text_value)
        packets = response.get("packets")
        if isinstance(packets, list):
            for packet in packets:
                if not isinstance(packet, dict):
                    continue
                packet_value = packet.get("value")
                if isinstance(packet_value, str):
                    candidates.append(packet_value)
        for candidate in candidates:
            if candidate in ("", "[DONE]"):
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                values.append(cast(JsonObject, parsed))
    return values


def _observed_prompt_tokens(record: JsonObject) -> int | None:
    observed: int | None = None
    for response in _response_objects(record):
        usage = response.get("usage")
        if not isinstance(usage, dict):
            continue
        value = usage.get("prompt_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        observed = value
    return observed


def _normalized_prompt_tokens(record: JsonObject) -> int | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    input_length = metrics.get("input_sequence_length")
    if not isinstance(input_length, dict):
        return None
    value = input_length.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _response_failure(record: JsonObject, raw_available: bool) -> tuple[str | None, str | None]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("was_cancelled") is True:
        return "cancelled", "native request was cancelled"
    if record.get("error") is not None:
        return "transport_error", str(record.get("error"))
    if not raw_available:
        # Response status and content are raw-artifact dimensions; they are
        # unavailable at the performance artifact level.
        return None, None
    status = record.get("status")
    if isinstance(status, int) and not isinstance(status, bool) and not 200 <= status < 300:
        return "transport_error", f"native request returned HTTP {status}"
    text_parts: list[str] = []
    for response in _response_objects(record):
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        body = (
            choice.get("message")
            if response.get("object") == "chat.completion"
            else choice.get("delta")
        )
        if not isinstance(body, dict):
            continue
        if body.get("tool_calls") or body.get("function_call"):
            return "model_selected_action", "native response selected an unsupported action"
        content = body.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
    if not text_parts:
        return "missing_assistant_text", "native response has no assistant text"
    return None, None


def _phase_session_evidence(
    phase: str,
    planned: list[tuple[str, list[float]]],
    records: list[JsonObject],
    tokenizer: ContentTokenizer,
    max_input_tokens: int,
    native_artifact_name: str,
    raw_available: bool,
) -> tuple[
    BenchSessionPhaseSummary,
    list[BenchRuntimeSessionResult],
    list[BenchSessionTurnResult],
    bool,
    list[str],
]:
    errors: list[str] = []
    turn_order_reconciled = True
    grouped: dict[str, list[JsonObject]] = {}
    native_ids: set[int] = set()
    for record_number, record in enumerate(records, start=1):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"AIPerf {phase} record {record_number} has no metadata")
            continue
        runtime_id = metadata.get("x_correlation_id")
        native_id = metadata.get("session_num")
        if not isinstance(runtime_id, str) or not runtime_id:
            errors.append(f"AIPerf {phase} record {record_number} has no x_correlation_id")
            continue
        if isinstance(native_id, bool) or not isinstance(native_id, int) or native_id < 0:
            errors.append(f"AIPerf {phase} record {record_number} has no native session_num")
            continue
        if native_id in native_ids:
            errors.append(f"AIPerf {phase} duplicates native session_num {native_id}")
        native_ids.add(native_id)
        grouped.setdefault(runtime_id, []).append(record)

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: min(
            cast(int, cast(dict[str, object], record["metadata"])["session_num"])
            for record in item[1]
        ),
    )
    session_results: list[BenchRuntimeSessionResult] = []
    turn_results: list[BenchSessionTurnResult] = []
    succeeded_sessions = 0
    failed_sessions = 0
    completed_requests = 0
    failed_requests = 0
    for session_index, (runtime_id, session_records) in enumerate(ordered_groups):
        session_records.sort(
            key=lambda record: cast(int, cast(dict[str, object], record["metadata"])["session_num"])
        )
        if session_index >= len(planned):
            errors.append(f"AIPerf {phase} admitted an unplanned session {runtime_id!r}")
            template_identity = "unplanned"
            delays: list[float] = []
        else:
            template_identity, delays = planned[session_index]
        planned_turns = len(delays)
        by_turn: dict[int, JsonObject] = {}
        for record in session_records:
            metadata = cast(dict[str, object], record["metadata"])
            turn_index = metadata.get("turn_index")
            conversation_id = metadata.get("conversation_id")
            if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
                errors.append(f"AIPerf {phase} session {runtime_id!r} has an invalid turn_index")
                turn_order_reconciled = False
                continue
            if turn_index in by_turn:
                errors.append(f"AIPerf {phase} session {runtime_id!r} duplicates turn {turn_index}")
                turn_order_reconciled = False
                continue
            if conversation_id != template_identity:
                errors.append(
                    f"AIPerf {phase} session {runtime_id!r} references template "
                    f"{conversation_id!r}, expected {template_identity!r}"
                )
            by_turn[turn_index] = record

        failure_classification: str | None = None
        diagnostic: str | None = None
        failing_turn: int | None = None
        previous_native_id: int | None = None
        previous_end_ns: int | None = None
        for turn_index in sorted(by_turn):
            record = by_turn[turn_index]
            metadata = cast(dict[str, object], record["metadata"])
            native_id = cast(int, metadata["session_num"])
            post_failure_continuation = failure_classification is not None
            request_start_ns = metadata.get("request_start_ns")
            request_end_ns = metadata.get("request_end_ns")
            if (
                isinstance(request_start_ns, bool)
                or not isinstance(request_start_ns, int)
                or request_start_ns < 0
                or isinstance(request_end_ns, bool)
                or not isinstance(request_end_ns, int)
                or request_end_ns < request_start_ns
            ):
                errors.append(
                    f"AIPerf {phase} session {runtime_id!r} turn {turn_index} has invalid timing"
                )
                continue
            content_tokens: int | None = None
            if raw_available:
                content_tokens = _pre_template_content_tokens(record, tokenizer)
                if content_tokens is None:
                    errors.append(
                        f"AIPerf {phase} session {runtime_id!r} turn {turn_index} "
                        "has no structured messages"
                    )
                    content_tokens = 0
            observed_prompt_tokens = (
                _observed_prompt_tokens(record)
                if raw_available
                else _normalized_prompt_tokens(record)
            )
            effective_delay = delays[turn_index] if turn_index < len(delays) else 0.0
            delay_reconciled = (
                None
                if post_failure_continuation
                else turn_index == 0
                or (
                    previous_end_ns is not None
                    and request_start_ns
                    >= previous_end_ns + round(effective_delay * 1_000_000_000.0)
                )
            )
            if delay_reconciled is False:
                errors.append(
                    f"AIPerf {phase} session {runtime_id!r} turn {turn_index} "
                    "began before its delay elapsed"
                )
            turn_results.append(
                BenchSessionTurnResult(
                    phase=phase,
                    runtime_session_id=runtime_id,
                    turn_index=turn_index,
                    pre_template_content_tokens=content_tokens,
                    observed_prompt_tokens=observed_prompt_tokens,
                    native_session_num=native_id,
                    preceding_native_session_num=(previous_native_id if raw_available else None),
                    preceding_terminal_response_receipt_ns=(
                        previous_end_ns if raw_available else None
                    ),
                    effective_inter_turn_delay_seconds=(
                        None if turn_index == 0 else effective_delay
                    ),
                    request_start_ns=request_start_ns,
                    inter_turn_delay_reconciled=delay_reconciled,
                    post_failure_continuation=post_failure_continuation,
                    native_artifact_name=native_artifact_name,
                )
            )
            cancelled = metadata.get("was_cancelled") is True
            if record.get("error") is not None or cancelled:
                failed_requests += 1
            else:
                completed_requests += 1
            turn_failure, turn_diagnostic = _response_failure(record, raw_available)
            if turn_failure is None and observed_prompt_tokens is None:
                turn_failure = "missing_prompt_token_usage"
                turn_diagnostic = "terminal backend response has no prompt-token usage"
            if (
                raw_available
                and content_tokens is not None
                and content_tokens > max_input_tokens
                and turn_failure is None
            ):
                turn_failure = "context_limit_exceeded_after_transport"
                turn_diagnostic = (
                    f"pre-template content tokens {content_tokens} exceed "
                    f"max_input_tokens {max_input_tokens}"
                )
            if turn_failure is not None and failure_classification is None:
                failure_classification = turn_failure
                diagnostic = turn_diagnostic
                failing_turn = turn_index
            if turn_failure is None and not post_failure_continuation:
                previous_native_id = native_id
                previous_end_ns = request_end_ns
            else:
                previous_native_id = None
                previous_end_ns = None

        expected_turns = list(range(planned_turns))
        observed_turns = sorted(by_turn)
        if failure_classification is None and observed_turns != expected_turns:
            failure_classification = "missing_terminal_evidence"
            failing_turn = next(
                (turn for turn in expected_turns if turn not in by_turn),
                None,
            )
            diagnostic = f"planned turns {expected_turns}, observed turns {observed_turns}"
        succeeded = failure_classification is None and observed_turns == expected_turns
        if succeeded:
            succeeded_sessions += 1
        else:
            failed_sessions += 1
        session_results.append(
            BenchRuntimeSessionResult(
                phase=phase,
                runtime_session_id=runtime_id,
                template_identity=template_identity,
                planned_turns=planned_turns,
                attempted_turns=len(by_turn),
                status=ClientStatus.succeeded if succeeded else ClientStatus.failed,
                failure_classification=failure_classification,
                diagnostic=diagnostic,
                failing_turn=failing_turn,
                suppressed_later_turns=max(planned_turns - len(by_turn), 0),
            )
        )

    planned_requests = sum(len(delays) for _, delays in planned)
    started_sessions = len(grouped)
    attempted_requests = len(native_ids)
    reconciled = (
        not errors
        and started_sessions == len(planned)
        and started_sessions == succeeded_sessions + failed_sessions
        and attempted_requests == completed_requests + failed_requests
    )
    return (
        BenchSessionPhaseSummary(
            planned_sessions=len(planned),
            started_sessions=started_sessions,
            succeeded_sessions=succeeded_sessions,
            failed_sessions=failed_sessions,
            planned_requests=planned_requests,
            attempted_requests=attempted_requests,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            reconciled=reconciled,
        ),
        session_results,
        turn_results,
        turn_order_reconciled,
        errors,
    )


def session_result_evidence(
    request: BenchClientRequest,
    profiling_path: Path,
    raw_path: Path,
    tokenizer: ContentTokenizer,
) -> tuple[BenchSessionResultEvidence, str | None]:
    source = request.definition.session_source
    raw_available = request.definition.artifact_level == BenchArtifactLevelInput.diagnostic
    rows, errors = _session_population(request)
    warmup_count = request.case.warmup_session_count or 0
    profiling_count = request.case.session_count or 0
    profiling_start, required = aiperf_session_population_layout(warmup_count, profiling_count)
    population_slice_reconciled = len(rows) >= required and not errors
    warmup_plan = rows[:warmup_count]
    profiling_plan = rows[profiling_start:required]
    if raw_available:
        raw_warmup, warmup_parse_error = raw_phase_records(raw_path, "warmup")
        raw_profiling, profiling_parse_error = raw_phase_records(raw_path, "profiling")
        artifact_name = "aiperf_raw_records" if raw_path.is_file() else "aiperf_partial_raw_records"
    else:
        # The performance artifact level produces no raw export; session
        # identity, phase, turn order, and delay reconciliation read the
        # normalized per-request records instead.
        raw_warmup, warmup_parse_error = phase_records(profiling_path, "warmup")
        raw_profiling, profiling_parse_error = phase_records(profiling_path, "profiling")
        artifact_name = "aiperf_records"
    if warmup_parse_error is not None:
        errors.append(warmup_parse_error)
    if profiling_parse_error is not None:
        errors.append(profiling_parse_error)
    max_input_tokens = source.max_input_tokens if source is not None else 0
    warmup, warmup_sessions, warmup_turns, warmup_turn_order, warmup_errors = (
        _phase_session_evidence(
            "warmup",
            warmup_plan,
            raw_warmup,
            tokenizer,
            max_input_tokens,
            artifact_name,
            raw_available,
        )
    )
    (
        profiling,
        profiling_sessions,
        profiling_turns,
        profiling_turn_order,
        profiling_errors,
    ) = _phase_session_evidence(
        "profiling",
        profiling_plan,
        raw_profiling,
        tokenizer,
        max_input_tokens,
        artifact_name,
        raw_available,
    )
    errors.extend(warmup_errors)
    errors.extend(profiling_errors)
    native_requests_reconciled: bool | None = None
    if raw_available:
        profiling_native_records, native_parse_error = profiling_records(profiling_path)
        if native_parse_error is not None:
            errors.append(native_parse_error)
        raw_native_ids = {
            metadata["session_num"]
            for record in raw_profiling
            if isinstance((metadata := record.get("metadata")), dict)
            and isinstance(metadata.get("session_num"), int)
            and not isinstance(metadata.get("session_num"), bool)
        }
        profiling_native_ids = {
            metadata["session_num"]
            for record in profiling_native_records
            if isinstance((metadata := record.get("metadata")), dict)
            and isinstance(metadata.get("session_num"), int)
            and not isinstance(metadata.get("session_num"), bool)
        }
        native_requests_reconciled = raw_native_ids == profiling_native_ids
        if not native_requests_reconciled:
            errors.append("AIPerf profiling metric records do not reconcile to raw requests")
    sessions = warmup_sessions + profiling_sessions
    turns = warmup_turns + profiling_turns
    runtime_session_ids = [session.runtime_session_id for session in sessions]
    runtime_session_identities_unique = len(runtime_session_ids) == len(set(runtime_session_ids))
    if not runtime_session_identities_unique:
        errors.append("AIPerf duplicates runtime session identity across phases")
    sessions_reconciled = (
        warmup.reconciled and profiling.reconciled and runtime_session_identities_unique
    )
    turn_order_reconciled = warmup_turn_order and profiling_turn_order
    inter_turn_delays_reconciled = all(
        turn.inter_turn_delay_reconciled is not False for turn in turns
    )
    counts_reconciled = (
        warmup.attempted_requests == warmup.completed_requests + warmup.failed_requests
        and profiling.attempted_requests == profiling.completed_requests + profiling.failed_requests
        and native_requests_reconciled is not False
    )
    if warmup.failed_sessions or profiling.failed_sessions:
        errors.append(
            "linear-session phase contains failed sessions: "
            f"warmup={warmup.failed_sessions}, profiling={profiling.failed_sessions}"
        )
    evidence = BenchSessionResultEvidence(
        warmup=warmup,
        profiling=profiling,
        sessions=sessions,
        turns=turns,
        population_slice_reconciled=population_slice_reconciled,
        sessions_reconciled=sessions_reconciled,
        turn_order_reconciled=turn_order_reconciled,
        inter_turn_delays_reconciled=inter_turn_delays_reconciled,
        native_requests_reconciled=native_requests_reconciled,
        counts_reconciled=counts_reconciled,
        unavailable_dimensions=[] if raw_available else PERFORMANCE_UNAVAILABLE_DIMENSIONS,
    )
    return evidence, "; ".join(dict.fromkeys(errors)) or None
