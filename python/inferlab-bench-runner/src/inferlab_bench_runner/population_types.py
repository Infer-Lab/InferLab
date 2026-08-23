"""Shared values for deterministic population materializers."""

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from inferlab_measurement_sdk import BenchTokenCountSummary


class ChatTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: list[int], **kwargs: object) -> str: ...


@dataclass(frozen=True)
class MaterializedEntry:
    source_sample_id: str
    messages: list[dict[str, str]]
    target: str | None
    kept_messages: int
    removed_messages: int
    input_tokens: int
    output_tokens: int
    selected_input_tokens: int | None = None
    category: str | None = None
    later_turn_count: int = 0
    first_user_turn_index: int = 0


def count_summary(values: list[int]) -> BenchTokenCountSummary:
    return BenchTokenCountSummary(
        minimum=min(values), maximum=max(values), mean=sum(values) / len(values)
    )


def json_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def token_stream_digest(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def decode_exact(tokenizer: ChatTokenizer, token_ids: list[int], label: str) -> str:
    if not token_ids:
        return ""
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.encode(text, add_special_tokens=False) != token_ids:
        raise ValueError(f"tokenizer could not round-trip the {label} token stream")
    return text


def common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def unbiased_index(seed: int, population_index: int, label: str, size: int) -> int:
    """Select an index without modulo bias from a stable population identity."""
    if size <= 0:
        raise ValueError("deterministic selection requires a positive range")
    modulus = 1 << 256
    limit = modulus - (modulus % size)
    counter = 0
    while True:
        payload = f"{seed}\0{population_index}\0{label}\0{counter}".encode()
        candidate = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if candidate < limit:
            return candidate % size
        counter += 1
