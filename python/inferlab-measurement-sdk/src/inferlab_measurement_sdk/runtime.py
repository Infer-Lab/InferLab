import argparse
import ipaddress
import json
import math
import time
from pathlib import Path
from typing import cast

from ._generated import ClientEndpointInput, ProtocolVersion, SettingValue

PROTOCOL_VERSION: str = ProtocolVersion().root

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, object]


class CaseBudgetExpired(TimeoutError):
    pass


class CaseDeadline:
    def __init__(self, remaining_seconds: float) -> None:
        if not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
            raise CaseBudgetExpired("measurement-case budget expired before client release")
        self._deadline = time.monotonic() + remaining_seconds

    def remaining(self, cap_seconds: float | None = None) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise CaseBudgetExpired("measurement-case budget expired")
        if cap_seconds is None:
            return remaining
        if not math.isfinite(cap_seconds) or cap_seconds <= 0:
            raise ValueError("attempt cap must be finite and positive")
        return min(remaining, cap_seconds)


def plain_setting(value: SettingValue) -> JsonValue:
    root = value.root
    if isinstance(root, list):
        return [plain_setting(item) for item in root]
    if isinstance(root, dict):
        return {key: plain_setting(item) for key, item in root.items()}
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handshake", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-source", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    return parser.parse_args()


def endpoint_url(endpoint: ClientEndpointInput, path: str) -> str:
    try:
        host = (
            f"[{endpoint.host}]"
            if ipaddress.ip_address(endpoint.host).version == 6
            else endpoint.host
        )
    except ValueError:
        host = endpoint.host
    return f"{endpoint.protocol.root}://{host}:{endpoint.port}{path}"


def load_json_object(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(JsonObject, value)


type JsonLine = tuple[int, object]


def parse_json_lines(text: str, artifact: str) -> tuple[list[JsonLine], str | None]:
    """The rows of JSON Lines text with their line numbers, and why reading
    stopped early, if it did.

    Rows split at newline only: U+2028 and the other separators that
    `str.splitlines` honors are legal inside a JSON string. Reading stops at
    the first invalid row, so the rows before it stay available as evidence.
    """
    rows: list[JsonLine] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip(" \t\r"):
            continue
        try:
            rows.append((line_number, json.loads(line)))
        except json.JSONDecodeError as error:
            return rows, f"{artifact} line {line_number} is invalid JSON: {error}"
    return rows, None


def read_json_lines(path: Path, artifact: str) -> tuple[list[JsonLine], str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return [], f"{artifact} is unreadable: {error}"
    return parse_json_lines(text, artifact)
