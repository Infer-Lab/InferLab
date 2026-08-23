"""Replay one workspace-local frozen population file unchanged.

The file is the sole population authority: entries are validated against the
declared prompt kind, copied byte for byte into the preparation artifact
directory, and never selected, filtered, or transformed
([[RFC-0004:C-BENCH-REQUEST-SOURCES]]).
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from inferlab_measurement_sdk import (
    BenchCacheStartInput,
    BenchPopulationInput,
    BenchPopulationPreparationRequest,
    BenchPopulationPreparationResult,
    BenchPrefixConditioningInput,
    BenchPrefixGeometrySummary,
    BenchPrefixSharingInput1,
    BenchPrefixSharingInput2,
    BenchPromptInputFlat,
    BenchPromptInputRenderedChat,
    BenchRequestSourceInputReplay,
    ClientStatus,
    JsonObject,
)

from inferlab_bench_runner.chat_tokens import required_messages_content_tokens
from inferlab_bench_runner.population_types import (
    ChatTokenizer,
    count_summary,
    decode_exact,
    json_line,
    token_stream_digest,
)

REPLAY_MATERIALIZATION_IDENTITY = "inferlab-replay-population-v1"


@dataclass(frozen=True)
class ReplayEntry:
    line_number: int
    session_id: str
    text_input: str | None
    messages: list[dict[str, str]] | None
    output_length: int


def _parse_entry(value: object, line_number: int, flat_prompt: bool) -> ReplayEntry:
    def malformed(reason: str) -> ValueError:
        return ValueError(f"replay population line {line_number}: {reason}")

    if not isinstance(value, dict):
        raise malformed("entry is not a JSON object")
    session_id = value.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise malformed("entry session_id must be a non-empty string")
    output_length = value.get("output_length")
    if isinstance(output_length, bool) or not isinstance(output_length, int):
        raise malformed("entry output_length must be an integer")
    if output_length < 1:
        raise malformed("entry output_length must be positive")
    text_input = value.get("text_input")
    messages = value.get("messages")
    if flat_prompt:
        if not isinstance(text_input, str) or not text_input:
            raise malformed("flat prompt entries require a non-empty text_input string")
        if messages is not None:
            raise malformed("flat prompt entries must not declare messages")
        return ReplayEntry(line_number, session_id, text_input, None, output_length)
    if text_input is not None:
        raise malformed("server_chat prompt entries must not declare text_input")
    if not isinstance(messages, list) or not messages:
        raise malformed("server_chat prompt entries require a non-empty messages list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise malformed("server_chat message is not a JSON object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise malformed("server_chat messages require string role and content")
        normalized.append({"role": role, "content": content})
    return ReplayEntry(line_number, session_id, None, normalized, output_length)


def _resolved_prefix_tokens(source: BenchRequestSourceInputReplay, input_tokens: int) -> int | None:
    sharing = source.prefix_sharing
    if sharing is None:
        return None
    value = sharing.root
    if isinstance(value, BenchPrefixSharingInput1):
        return value.shared_prefix_tokens
    if isinstance(value, BenchPrefixSharingInput2):
        return math.floor(input_tokens * value.shared_prefix_ratio)
    raise TypeError(f"unsupported prefix sharing {type(value).__name__}")


def prepare_replay_population(
    request: BenchPopulationPreparationRequest,
    tokenizer: ChatTokenizer,
    source: BenchRequestSourceInputReplay,
) -> BenchPopulationPreparationResult:
    if request.source_path is None:
        raise ValueError("replay preparation requires a source path")
    required = request.required_entries
    if required <= 0:
        raise ValueError("replay preparation requires at least one entry")
    source_path = Path(request.source_path)
    source_bytes = source_path.read_bytes()
    prompt = request.prompt.root
    flat_prompt = isinstance(prompt, (BenchPromptInputFlat, BenchPromptInputRenderedChat))
    entries: list[ReplayEntry] = []
    for line_number, raw_line in enumerate(source_bytes.split(b"\n"), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"replay population line {line_number}: invalid JSON ({error})"
            ) from None
        entries.append(_parse_entry(value, line_number, flat_prompt))
    if len(entries) < required:
        return BenchPopulationPreparationResult(
            schema_version=1,
            status=ClientStatus.failed,
            materialization_identity=REPLAY_MATERIALIZATION_IDENTITY,
            requested_entries=required,
            candidate_entries=len(entries),
            admitted_entries=len(entries),
            ineligible_entries=0,
            ineligible_reasons={},
            population=None,
            input_tokens=None,
            output_tokens=None,
            evidence_path=None,
            error=(
                f"replay population has {len(entries)} entries, requires {required}; "
                "entries are never repeated"
            ),
        )
    output_counts = [entry.output_length for entry in entries]
    tpot_classes = {output >= 2 for output in output_counts}
    if len(tpot_classes) > 1:
        raise ValueError(
            "replay population must not mix TPOT-inapplicable (output one) and "
            "TPOT-applicable (output at least two) entries"
        )
    input_counts = [
        (
            len(tokenizer.encode(entry.text_input, add_special_tokens=False))
            if entry.text_input is not None
            else required_messages_content_tokens(entry.messages or [], tokenizer)
        )
        for entry in entries
    ]

    prefix_geometry: BenchPrefixGeometrySummary | None = None
    prefix_conditioning: BenchPrefixConditioningInput | None = None
    resolved_prefix_counts: list[int | None] = [
        _resolved_prefix_tokens(source, input_tokens) for input_tokens in input_counts
    ]
    if source.prefix_sharing is not None:
        if not flat_prompt:
            raise ValueError("replay prefix_sharing requires prompt kind flat or rendered_chat")
        entry_ids = [
            tokenizer.encode(entry.text_input or "", add_special_tokens=False) for entry in entries
        ]
        shared_counts = [shared for shared in resolved_prefix_counts if shared is not None]
        for entry, shared, ids in zip(entries, shared_counts, entry_ids, strict=True):
            if shared > len(ids):
                raise ValueError(
                    f"replay population line {entry.line_number}: entry has "
                    f"{len(ids)} tokens, fewer than its resolved shared prefix {shared}"
                )
        canonical_index = shared_counts.index(max(shared_counts))
        canonical_ids = entry_ids[canonical_index]
        maximum_shared = shared_counts[canonical_index]
        for entry, shared, ids in zip(entries, shared_counts, entry_ids, strict=True):
            if ids[:shared] != canonical_ids[:shared]:
                raise ValueError(
                    f"replay population line {entry.line_number}: entry does not share "
                    "the declared canonical prefix"
                )
        if request.cache_start is BenchCacheStartInput.primed:
            if maximum_shared <= 0:
                raise ValueError("primed replay resolved to a zero-length shared prefix")
            canonical_prefix = decode_exact(
                tokenizer,
                canonical_ids[:maximum_shared],
                "canonical prefix conditioning prompt",
            )
            artifact_dir = Path(request.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            conditioning_path = artifact_dir / "canonical-prefix.txt"
            conditioning_path.write_text(canonical_prefix, encoding="utf-8")
            prefix_conditioning = BenchPrefixConditioningInput(
                path=str(conditioning_path),
                sha256=hashlib.sha256(canonical_prefix.encode()).hexdigest(),
                prompt_tokens=maximum_shared,
            )
        prefix_geometry = BenchPrefixGeometrySummary(
            shared_prefix_tokens=count_summary(shared_counts),
            unique_suffix_tokens=count_summary(
                [
                    input_tokens - shared
                    for input_tokens, shared in zip(input_counts, shared_counts, strict=True)
                ]
            ),
            maximum_shared_prefix_tokens=maximum_shared,
            canonical_prefix_sha256=token_stream_digest(canonical_ids[:maximum_shared]),
            full_prompt_entries=sum(
                shared == input_tokens
                for input_tokens, shared in zip(input_counts, shared_counts, strict=True)
            ),
        )

    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    population_path = artifact_dir / "population.jsonl"
    evidence_path = artifact_dir / "population-evidence.jsonl"
    # The artifact is the replayed file byte for byte.
    population_path.write_bytes(source_bytes)
    population_sha256 = hashlib.sha256(source_bytes).hexdigest()
    with evidence_path.open("wb") as evidence_file:
        for index, entry in enumerate(entries):
            resolved_shared = resolved_prefix_counts[index]
            evidence: JsonObject = {
                "population_index": index,
                "session_id": entry.session_id,
                "prompt_kind": prompt.kind,
                "input_tokens": input_counts[index],
                "output_tokens": entry.output_length,
                "resolved_shared_prefix_tokens": resolved_shared,
                "resolved_unique_suffix_tokens": (
                    input_counts[index] - resolved_shared if resolved_shared is not None else None
                ),
            }
            evidence_file.write(json_line(evidence))
    return BenchPopulationPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        materialization_identity=REPLAY_MATERIALIZATION_IDENTITY,
        requested_entries=required,
        candidate_entries=len(entries),
        admitted_entries=len(entries),
        ineligible_entries=0,
        ineligible_reasons={},
        population=BenchPopulationInput(
            path=str(population_path),
            evidence_path=str(evidence_path),
            sha256=population_sha256,
            entries=len(entries),
            tpot_applicable=all(output >= 2 for output in output_counts),
        ),
        input_tokens=count_summary(input_counts),
        output_tokens=count_summary(output_counts),
        prefix_geometry=prefix_geometry,
        prefix_conditioning=prefix_conditioning,
        evidence_path=str(evidence_path),
        error=None,
    )
