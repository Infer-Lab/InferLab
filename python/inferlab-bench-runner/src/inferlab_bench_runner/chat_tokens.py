"""Structured-chat content token accounting shared by preparation and reconciliation."""

from collections.abc import Mapping
from typing import Protocol


class ContentTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def content_tokens(value: object, tokenizer: ContentTokenizer) -> int:
    """Count only textual message content, before any chat-template projection."""
    if isinstance(value, str):
        return len(tokenizer.encode(value, add_special_tokens=False))
    if not isinstance(value, list):
        return 0
    total = 0
    for part in value:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str):
            total += len(tokenizer.encode(text, add_special_tokens=False))
    return total


def messages_content_tokens(messages: object, tokenizer: ContentTokenizer) -> int | None:
    if not isinstance(messages, list):
        return None
    total = 0
    for message in messages:
        if not isinstance(message, Mapping):
            return None
        total += content_tokens(message.get("content"), tokenizer)
    return total


def required_messages_content_tokens(
    messages: list[dict[str, str]], tokenizer: ContentTokenizer
) -> int:
    total = messages_content_tokens(messages, tokenizer)
    if total is None:
        raise TypeError("messages must contain structured chat objects")
    return total
