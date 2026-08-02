"""Materialize ShareGPT request and linear-session populations."""

import hashlib
import heapq
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from inferlab_measurement_sdk import (
    BenchPopulationInput,
    BenchPopulationPreparationRequest,
    BenchPopulationPreparationResult,
    BenchRequestSourceInputDataset,
    BenchSessionSourceInput,
    BenchSessionTemplateInput,
    ClientStatus,
    JsonObject,
)

from inferlab_bench_runner.chat_tokens import required_messages_content_tokens
from inferlab_bench_runner.population_types import (
    ChatTokenizer,
    MaterializedEntry,
    count_summary,
    json_line,
)


@dataclass(frozen=True)
class MaterializedSessionTurn:
    user_content: str
    target_sha256: str
    source_inter_turn_delay_seconds: float
    effective_inter_turn_delay_seconds: float
    output_tokens: int
    output_limit_provenance: str


@dataclass(frozen=True)
class MaterializedSession:
    template_identity: str
    source_sample_id: str
    turns: list[MaterializedSessionTurn]
    first_turn_input_tokens: int


def effective_inter_turn_delay(
    source_seconds: float, scale: float, maximum_seconds: float | None
) -> float:
    value = source_seconds * scale
    if maximum_seconds is not None:
        value = min(value, maximum_seconds)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("effective inter-turn delay must be finite and non-negative")
    return value


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as source:
        buffer = ""
        started = False
        finished = False
        eof = False
        while not finished:
            if not eof and len(buffer) < chunk_size:
                chunk = source.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    if eof:
                        raise ValueError(f"{path} is empty")
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"{path} must contain one JSON array")
                buffer = buffer[1:]
                started = True
                continue
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                finished = True
                continue
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if not buffer:
                if eof:
                    raise ValueError(f"{path} has an unterminated JSON array")
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"{path} contains invalid JSON") from None
                chunk = source.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            buffer = buffer[end:]
            yield value
        if buffer.strip():
            raise ValueError(f"{path} has data after its JSON array")


def sharegpt_messages(value: object) -> tuple[str | None, list[dict[str, str]]] | None:
    if not isinstance(value, dict):
        return None
    raw_messages = value.get("conversations")
    if not isinstance(raw_messages, list):
        return None
    messages: list[dict[str, str]] = []
    expected = "user"
    for raw in raw_messages:
        if not isinstance(raw, dict):
            return None
        source_role = raw.get("from")
        content = raw.get("value")
        role = "user" if source_role == "human" else "assistant" if source_role == "gpt" else None
        if role != expected or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": content})
        expected = "assistant" if expected == "user" else "user"
    raw_id = value.get("id")
    return (raw_id if isinstance(raw_id, str) else None, messages)


def materialize_conversation(
    value: object,
    index: int,
    tokenizer: ChatTokenizer,
    max_input_tokens: int,
    fixed_output_tokens: int | None,
) -> tuple[MaterializedEntry | None, str | None]:
    normalized = sharegpt_messages(value)
    if normalized is None:
        return None, "invalid_conversation"
    source_id, messages = normalized
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None, "missing_assistant_target"
    target_index = len(messages) - 1
    while target_index >= 1:
        input_messages = messages[:target_index]
        if not input_messages[-1]["content"].strip():
            target_index -= 2
            continue
        input_tokens = required_messages_content_tokens(input_messages, tokenizer)
        if input_tokens <= max_input_tokens:
            target = messages[target_index]["content"]
            if not target.strip():
                return None, "empty_assistant_target"
            target_tokens = len(tokenizer.encode(target, add_special_tokens=False))
            if target_tokens == 0:
                return None, "empty_assistant_target"
            if fixed_output_tokens is None and target_tokens < 2:
                return None, "assistant_target_shorter_than_two_tokens"
            output_tokens = fixed_output_tokens or target_tokens
            kept_messages = target_index + 1
            return (
                MaterializedEntry(
                    source_sample_id=source_id or f"row-{index}",
                    messages=input_messages,
                    target=target,
                    kept_messages=kept_messages,
                    removed_messages=len(messages) - kept_messages,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                None,
            )
        target_index -= 2
    return None, "input_exceeds_maximum"


def materialize_linear_session(
    value: object,
    index: int,
    tokenizer: ChatTokenizer,
    source: BenchSessionSourceInput,
) -> tuple[MaterializedSession | None, str | None]:
    normalized = sharegpt_messages(value)
    if normalized is None:
        return None, "invalid_conversation"
    source_id, messages = normalized
    if len(messages) < 4 or len(messages) % 2 != 0:
        return None, "fewer_than_two_complete_turns"

    turns: list[MaterializedSessionTurn] = []
    for message_index in range(0, len(messages), 2):
        user = messages[message_index]
        target = messages[message_index + 1]
        if not user["content"].strip():
            return None, "empty_user_message"
        if not target["content"].strip():
            return None, "empty_assistant_target"
        target_tokens = len(tokenizer.encode(target["content"], add_special_tokens=False))
        if target_tokens == 0:
            return None, "empty_assistant_target"
        if source.output_tokens is None and target_tokens < 2:
            return None, "assistant_target_shorter_than_two_tokens"
        source_delay = 0.0
        effective_delay = effective_inter_turn_delay(
            source_delay,
            source.inter_turn_delay_scale,
            source.max_inter_turn_delay_seconds,
        )
        turns.append(
            MaterializedSessionTurn(
                user_content=user["content"],
                target_sha256=hashlib.sha256(target["content"].encode("utf-8")).hexdigest(),
                source_inter_turn_delay_seconds=source_delay,
                effective_inter_turn_delay_seconds=effective_delay,
                output_tokens=source.output_tokens or target_tokens,
                output_limit_provenance=(
                    "fixed_override" if source.output_tokens is not None else "target_derived"
                ),
            )
        )

    first_turn_input_tokens = required_messages_content_tokens(
        [{"role": "user", "content": turns[0].user_content}], tokenizer
    )
    if first_turn_input_tokens > source.max_input_tokens:
        return None, "first_turn_input_exceeds_maximum"
    source_sample_id = source_id or f"row-{index}"
    identity_payload = json.dumps(
        {
            "source_sample_id": source_sample_id,
            "source_index": index,
            "users": [turn.user_content for turn in turns],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    template_identity = f"sharegpt:{hashlib.sha256(identity_payload).hexdigest()}"
    return (
        MaterializedSession(
            template_identity=template_identity,
            source_sample_id=source_sample_id,
            turns=turns,
            first_turn_input_tokens=first_turn_input_tokens,
        ),
        None,
    )


def prepare_sharegpt_population(
    request: BenchPopulationPreparationRequest,
    tokenizer: ChatTokenizer,
) -> BenchPopulationPreparationResult:
    if request.request_source is None:
        raise ValueError("ShareGPT request preparation requires request_source")
    source = request.request_source.root
    if not isinstance(source, BenchRequestSourceInputDataset):
        raise ValueError("ShareGPT preparation requires a dataset request source")
    if source.dataset != "sharegpt":
        raise ValueError(f"unsupported ShareGPT dataset {source.dataset!r}")
    if request.source_path is None:
        raise ValueError("ShareGPT preparation requires a source path")
    required = request.required_entries
    if required <= 0:
        raise ValueError("dataset preparation requires at least one entry")
    selected: list[tuple[int, int, MaterializedEntry]] = []
    candidate_entries = 0
    admitted_entries = 0
    ineligible_reasons: dict[str, int] = {}
    for index, value in enumerate(iter_json_array(Path(request.source_path))):
        candidate_entries += 1
        entry, reason = materialize_conversation(
            value,
            index,
            tokenizer,
            source.max_input_tokens,
            source.output_tokens,
        )
        if entry is None:
            stable_reason = reason or "invalid_conversation"
            ineligible_reasons[stable_reason] = ineligible_reasons.get(stable_reason, 0) + 1
            continue
        admitted_entries += 1
        key = int.from_bytes(
            hashlib.sha256(f"{request.seed}\0{entry.source_sample_id}\0{index}".encode()).digest(),
            "big",
        )
        item = (-key, -index, entry)
        if len(selected) < required:
            heapq.heappush(selected, item)
        elif item > selected[0]:
            heapq.heapreplace(selected, item)
    ineligible_entries = candidate_entries - admitted_entries
    if admitted_entries < required:
        return BenchPopulationPreparationResult(
            schema_version=1,
            status=ClientStatus.failed,
            materialization_identity=source.catalog.materialization_identity,
            requested_entries=required,
            candidate_entries=candidate_entries,
            admitted_entries=admitted_entries,
            ineligible_entries=ineligible_entries,
            ineligible_reasons=ineligible_reasons,
            population=None,
            input_tokens=None,
            output_tokens=None,
            evidence_path=None,
            error=f"dataset has {admitted_entries} eligible entries, requires {required}",
        )
    ordered = [item[2] for item in sorted(selected, key=lambda item: (-item[0], -item[1]))]
    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    population_path = artifact_dir / "population.jsonl"
    evidence_path = artifact_dir / "population-evidence.jsonl"
    population_digest = hashlib.sha256()
    with population_path.open("wb") as population_file, evidence_path.open("wb") as evidence_file:
        for index, entry in enumerate(ordered):
            request_value: JsonObject = {
                "session_id": f"inferlab-{index:08}",
                "messages": entry.messages,
                "output_length": entry.output_tokens,
                "extra": {"ignore_eos": True, "min_tokens": entry.output_tokens},
            }
            population_line = json_line(request_value)
            population_file.write(population_line)
            population_digest.update(population_line)
            evidence_file.write(
                json_line(
                    {
                        "population_index": index,
                        "source_sample_id": entry.source_sample_id,
                        "messages": entry.messages,
                        "held_out_target": entry.target,
                        "held_out_target_sha256": hashlib.sha256(
                            entry.target.encode("utf-8") if entry.target is not None else b""
                        ).hexdigest(),
                        "kept_messages": entry.kept_messages,
                        "removed_messages": entry.removed_messages,
                        "input_tokens": entry.input_tokens,
                        "output_tokens": entry.output_tokens,
                    }
                )
            )
    input_counts = [entry.input_tokens for entry in ordered]
    output_counts = [entry.output_tokens for entry in ordered]
    return BenchPopulationPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        materialization_identity=source.catalog.materialization_identity,
        requested_entries=required,
        candidate_entries=candidate_entries,
        admitted_entries=admitted_entries,
        ineligible_entries=ineligible_entries,
        ineligible_reasons=ineligible_reasons,
        population=BenchPopulationInput(
            path=str(population_path),
            sha256=population_digest.hexdigest(),
            entries=required,
            tpot_applicable=all(value >= 2 for value in output_counts),
        ),
        input_tokens=count_summary(input_counts),
        output_tokens=count_summary(output_counts),
        evidence_path=str(evidence_path),
        error=None,
    )


def prepare_sharegpt_session_population(
    request: BenchPopulationPreparationRequest,
    tokenizer: ChatTokenizer,
    source: BenchSessionSourceInput,
) -> BenchPopulationPreparationResult:
    if source.dataset != "sharegpt":
        raise ValueError(f"unsupported linear-session dataset {source.dataset!r}")
    if request.source_path is None:
        raise ValueError("linear-session preparation requires a source path")
    required = request.required_entries
    if required <= 0:
        raise ValueError("linear-session preparation requires at least one entry")

    selected: list[tuple[int, int, MaterializedSession]] = []
    candidate_entries = 0
    admitted_entries = 0
    ineligible_reasons: dict[str, int] = {}
    for index, value in enumerate(iter_json_array(Path(request.source_path))):
        candidate_entries += 1
        session, reason = materialize_linear_session(value, index, tokenizer, source)
        if session is None:
            stable_reason = reason or "invalid_linear_session"
            ineligible_reasons[stable_reason] = ineligible_reasons.get(stable_reason, 0) + 1
            continue
        admitted_entries += 1
        key = int.from_bytes(
            hashlib.sha256(
                f"{request.seed}\0{source.catalog.materialization_identity}\0{session.template_identity}".encode()
            ).digest(),
            "big",
        )
        item = (-key, -index, session)
        if len(selected) < required:
            heapq.heappush(selected, item)
        elif item > selected[0]:
            heapq.heapreplace(selected, item)

    ineligible_entries = candidate_entries - admitted_entries
    if admitted_entries < required:
        return BenchPopulationPreparationResult(
            schema_version=1,
            status=ClientStatus.failed,
            materialization_identity=source.catalog.materialization_identity,
            requested_entries=required,
            candidate_entries=candidate_entries,
            admitted_entries=admitted_entries,
            ineligible_entries=ineligible_entries,
            ineligible_reasons=ineligible_reasons,
            population=None,
            input_tokens=None,
            output_tokens=None,
            evidence_path=None,
            error=f"dataset has {admitted_entries} eligible linear templates, requires {required}",
        )

    ordered = [item[2] for item in sorted(selected, key=lambda item: (-item[0], -item[1]))]
    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    population_path = artifact_dir / "population.jsonl"
    population_digest = hashlib.sha256()
    with population_path.open("wb") as population_file:
        for population_index, session in enumerate(ordered):
            turns = [
                {
                    "turn_index": turn_index,
                    "user_message": {
                        "role": "user",
                        "content": turn.user_content,
                    },
                    "held_out_target_sha256": turn.target_sha256,
                    "source_inter_turn_delay_seconds": (turn.source_inter_turn_delay_seconds),
                    "effective_inter_turn_delay_seconds": (turn.effective_inter_turn_delay_seconds),
                    "output_tokens": turn.output_tokens,
                    "output_limit_provenance": turn.output_limit_provenance,
                }
                for turn_index, turn in enumerate(session.turns)
            ]
            population_line = json_line(
                {
                    "type": "multi_turn",
                    "session_id": session.template_identity,
                    "turns": [
                        {
                            "type": "single_turn",
                            "text": turn.user_content,
                            "role": "user",
                            "delay": turn.effective_inter_turn_delay_seconds * 1000.0,
                            "output_length": turn.output_tokens,
                            "extra": {
                                "ignore_eos": True,
                                "min_tokens": turn.output_tokens,
                            },
                        }
                        for turn in session.turns
                    ],
                    # AIPerf's multi_turn model permits extension fields and
                    # ignores this namespace when it builds Conversations. It
                    # therefore keeps the AIPerf execution projection and the
                    # Inferlab-owned template evidence in one content-addressed
                    # population artifact without forwarding evidence to the
                    # serving endpoint.
                    "_inferlab": {
                        "population_index": population_index,
                        "template_identity": session.template_identity,
                        "source_sample_id": session.source_sample_id,
                        "first_turn_pre_template_content_tokens": (session.first_turn_input_tokens),
                        "max_input_tokens": source.max_input_tokens,
                        "inter_turn_delay_scale": source.inter_turn_delay_scale,
                        "max_inter_turn_delay_seconds": (source.max_inter_turn_delay_seconds),
                        "turns": turns,
                    },
                }
            )
            population_file.write(population_line)
            population_digest.update(population_line)

    input_counts = [session.first_turn_input_tokens for session in ordered]
    output_counts = [turn.output_tokens for session in ordered for turn in session.turns]
    return BenchPopulationPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        materialization_identity=source.catalog.materialization_identity,
        requested_entries=required,
        candidate_entries=candidate_entries,
        admitted_entries=admitted_entries,
        ineligible_entries=ineligible_entries,
        ineligible_reasons=ineligible_reasons,
        population=BenchPopulationInput(
            path=str(population_path),
            sha256=population_digest.hexdigest(),
            entries=required,
            tpot_applicable=all(value >= 2 for value in output_counts),
            session_templates=[
                BenchSessionTemplateInput(
                    template_identity=session.template_identity,
                    turn_count=len(session.turns),
                )
                for session in ordered
            ],
        ),
        input_tokens=count_summary(input_counts),
        output_tokens=count_summary(output_counts),
        evidence_path=str(population_path),
        error=None,
    )
