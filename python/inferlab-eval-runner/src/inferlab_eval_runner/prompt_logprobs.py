from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    CaseDeadline,
    EvalClientRequest,
    EvalDefinitionInputLmEval,
    EvalFailureKind,
    JsonObject,
    RawArtifact,
    endpoint_url,
)

PROMPT_LOGPROB_PROBE_PROMPT: str = "Inferlab prompt logprob probe: 0123456789"


class ProbeTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeTokenization:
    token_ids: list[int]
    offset_mapping: list[tuple[int, int]]


@dataclass(frozen=True)
class PromptLogprobProbeRun:
    failure_kind: EvalFailureKind | None
    error: str | None
    raw_artifacts: list[RawArtifact]


def validate_prompt_logprob_response(
    response: object,
    prompt_tokenization: ProbeTokenization,
) -> tuple[str, EvalFailureKind | None, list[JsonObject], str | None]:
    checks: list[JsonObject] = []

    def checked(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    if not isinstance(response, dict):
        checked("response_shape", False, "response is not a JSON object")
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe response is not a JSON object",
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        checked("response_shape", False, "response must contain exactly one choice object")
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe response must contain exactly one choice",
        )
    choice = choices[0]
    if choice.get("index") != 0 or not isinstance(choice.get("text"), str):
        checked("response_shape", False, "choice must have index 0 and string text")
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe choice has invalid index or text",
        )
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        checked("response_shape", False, "choice has no logprobs object")
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe choice has no logprobs object",
        )
    checked("response_shape", True, "one indexed choice contains text and logprobs")

    arrays = [
        logprobs.get(name) for name in ("tokens", "token_logprobs", "top_logprobs", "text_offset")
    ]
    equal_lengths = (
        all(isinstance(array, list) for array in arrays)
        and len({len(array) for array in arrays if isinstance(array, list)}) == 1
    )
    checked("equal_length_arrays", equal_lengths, "tokens, logprobs, top-logprobs, and offsets")
    if not equal_lengths:
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe arrays are absent or have unequal lengths",
        )
    tokens, token_logprobs, top_logprobs, text_offsets = cast(
        tuple[list[object], list[object], list[object], list[object]], tuple(arrays)
    )
    typed_arrays = (
        bool(tokens)
        and all(isinstance(token, str) for token in tokens)
        and all(
            value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
            for value in token_logprobs
        )
        and all(value is None or isinstance(value, dict) for value in top_logprobs)
        and all(isinstance(offset, int) and not isinstance(offset, bool) for offset in text_offsets)
    )
    checked("array_types", typed_arrays, "token arrays contain the native OpenAI-compatible types")
    if not typed_arrays:
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe arrays contain invalid values",
        )

    prompt_length = len(PROMPT_LOGPROB_PROBE_PROMPT)
    prompt_token_ids = prompt_tokenization.token_ids
    tokenizer_offsets = prompt_tokenization.offset_mapping
    tokenizer_starts = [start for start, _ in tokenizer_offsets]
    tokenizer_covers_prompt = (
        len(tokenizer_offsets) == len(prompt_token_ids)
        and bool(tokenizer_offsets)
        and tokenizer_offsets[0][0] == 0
        and tokenizer_offsets[-1][1] == prompt_length
        and all(
            start < end and end == tokenizer_offsets[index + 1][0]
            for index, (start, end) in enumerate(tokenizer_offsets[:-1])
        )
        and tokenizer_offsets[-1][0] < tokenizer_offsets[-1][1]
    )
    prompt_positions = [
        index for index, offset in enumerate(text_offsets) if cast(int, offset) < prompt_length
    ]
    generated_positions = [
        index for index, offset in enumerate(text_offsets) if cast(int, offset) >= prompt_length
    ]
    text = cast(str, choice["text"])
    aligned = (
        tokenizer_covers_prompt
        and text.startswith(PROMPT_LOGPROB_PROBE_PROMPT)
        and text == "".join(cast(list[str], tokens))
        and "".join(cast(list[str], tokens[: len(prompt_token_ids)])) == PROMPT_LOGPROB_PROBE_PROMPT
        and len(prompt_positions) == len(prompt_token_ids)
        and len(prompt_token_ids) >= 2
        and cast(list[int], text_offsets[: len(prompt_token_ids)]) == tokenizer_starts
        and len(tokens) == len(prompt_token_ids) + 1
        and len(generated_positions) == 1
        and generated_positions[0] == len(prompt_token_ids)
        and cast(int, text_offsets[generated_positions[0]]) == prompt_length
        and all(
            cast(int, text_offsets[index]) <= cast(int, text_offsets[index + 1])
            for index in range(len(text_offsets) - 1)
        )
    )
    checked(
        "tokenizer_alignment",
        aligned,
        f"tokenizer={len(prompt_token_ids)} prompt_positions={len(prompt_positions)}",
    )
    if not aligned:
        return (
            "unsupported",
            EvalFailureKind.probe_tokenizer_alignment,
            checks,
            "prompt-logprob probe echo does not align with the resolved tokenizer",
        )

    prompt_scored = all(
        isinstance(token_logprobs[index], (int, float))
        and not isinstance(token_logprobs[index], bool)
        and math.isfinite(float(cast(int | float, token_logprobs[index])))
        and isinstance(top_logprobs[index], dict)
        for index in prompt_positions[1:]
    )
    checked("prompt_logprobs", prompt_scored, "all continuation-scored prompt positions")
    generated_index = generated_positions[0]
    generated_scored = (
        isinstance(token_logprobs[generated_index], (int, float))
        and not isinstance(token_logprobs[generated_index], bool)
        and math.isfinite(float(cast(int | float, token_logprobs[generated_index])))
        and isinstance(top_logprobs[generated_index], dict)
    )
    checked("generated_logprob", generated_scored, "first generated position")
    if generated_scored and not prompt_scored:
        return (
            "unsupported",
            EvalFailureKind.probe_generated_only_logprobs,
            checks,
            "endpoint returned generated-token logprobs without scored prompt positions",
        )
    if not generated_scored:
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe response has no scored generated position",
        )
    return "supported", None, checks, None


def tokenize_probe_prompt(locator: str, prompt: str, timeout_seconds: float) -> ProbeTokenization:
    script = (
        "import json, sys\n"
        "from transformers import AutoTokenizer\n"
        "tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])\n"
        "encoded = tokenizer(sys.argv[2], add_special_tokens=False, "
        "return_offsets_mapping=True)\n"
        "print(json.dumps({'token_ids': encoded['input_ids'], "
        "'offset_mapping': encoded['offset_mapping']}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, locator, prompt],
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ValueError(f"failed to load resolved tokenizer {locator}: {diagnostic}")
    try:
        encoded = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"resolved tokenizer {locator} returned invalid token JSON: {error}"
        ) from error
    if not isinstance(encoded, dict):
        raise ValueError(f"resolved tokenizer {locator} returned invalid tokenization")
    token_ids = encoded.get("token_ids")
    raw_offsets = encoded.get("offset_mapping")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids
    ):
        raise ValueError(f"resolved tokenizer {locator} returned invalid token identifiers")
    if not isinstance(raw_offsets, list) or not all(
        isinstance(offset, list)
        and len(offset) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in offset)
        for offset in raw_offsets
    ):
        raise ValueError(f"resolved tokenizer {locator} returned invalid offset mapping")
    offsets = [(cast(int, offset[0]), cast(int, offset[1])) for offset in raw_offsets]
    return ProbeTokenization(cast(list[int], token_ids), offsets)


def post_prompt_logprob_probe(
    url: str, body: JsonObject, timeout_seconds: float
) -> tuple[int, bytes]:
    script = (
        "import base64, json, sys\n"
        "import requests\n"
        "response = requests.post(sys.argv[1], json=json.loads(sys.argv[2]), "
        "timeout=float(sys.argv[3]))\n"
        "print(json.dumps({'status': response.status_code, "
        "'content': base64.b64encode(response.content).decode('ascii')}))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, url, json.dumps(body), str(timeout_seconds)],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ProbeTransportError(
            f"prompt-logprob probe transport timed out after {error.timeout} seconds"
        ) from error
    except OSError as error:
        raise ProbeTransportError(f"prompt-logprob probe transport failed: {error}") from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ProbeTransportError(f"prompt-logprob probe transport failed: {diagnostic}")
    try:
        response = json.loads(completed.stdout)
        if not isinstance(response, dict):
            raise ValueError("response envelope is not an object")
        status = response.get("status")
        content = response.get("content")
        if not isinstance(status, int) or isinstance(status, bool) or not isinstance(content, str):
            raise ValueError("response envelope fields are invalid")
        decoded = base64.b64decode(content, validate=True)
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeTransportError(
            f"prompt-logprob probe HTTP client returned an invalid response: {error}"
        ) from error
    return status, decoded


def run_prompt_logprob_probe(
    request: EvalClientRequest,
    definition: EvalDefinitionInputLmEval,
    artifact_dir: Path,
    deadline: CaseDeadline | None = None,
) -> PromptLogprobProbeRun:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    started = time.monotonic()
    timeout_seconds = deadline.remaining(30.0)
    request_body: JsonObject = {
        "model": request.model.served_name,
        "prompt": PROMPT_LOGPROB_PROBE_PROMPT,
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
        "n": 1,
        "echo": True,
        "logprobs": 1,
    }
    evidence_path = artifact_dir / "prompt-logprob-probe.json"
    response_path = artifact_dir / "prompt-logprob-probe-response.json"
    artifacts = [
        RawArtifact(
            name="prompt_logprob_probe",
            kind="prompt-logprob-probe",
            path=str(evidence_path),
        )
    ]
    evidence: JsonObject = {
        "schema_version": 1,
        "effective_request": request_body,
        "effective_timeout_seconds": timeout_seconds,
        "tokenizer": {
            "locator": request.model.locator,
            "backend": "huggingface",
            "tokenized_requests": False,
        },
        "transport_outcome": "not_started",
        "response_status": None,
        "checks": [],
    }

    def finish(
        conclusion: str,
        failure_kind: EvalFailureKind | None,
        error: str | None,
        checks: list[JsonObject],
    ) -> PromptLogprobProbeRun:
        evidence["conclusion"] = conclusion
        evidence["failure_kind"] = failure_kind.value if failure_kind is not None else None
        evidence["error"] = error
        evidence["checks"] = checks
        evidence["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return PromptLogprobProbeRun(failure_kind, error, artifacts)

    tokenizer_started = time.monotonic()
    try:
        prompt_tokenization = tokenize_probe_prompt(
            request.model.locator,
            PROMPT_LOGPROB_PROBE_PROMPT,
            timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        evidence["tokenizer_elapsed_ms"] = round((time.monotonic() - tokenizer_started) * 1000, 3)
        return finish(
            "inconclusive",
            EvalFailureKind.probe_tokenizer,
            f"resolved tokenizer probe timed out after {error.timeout} seconds",
            [{"name": "tokenizer_prompt", "passed": False, "detail": "timeout"}],
        )
    except (OSError, ValueError) as error:
        evidence["tokenizer_elapsed_ms"] = round((time.monotonic() - tokenizer_started) * 1000, 3)
        return finish(
            "inconclusive",
            EvalFailureKind.probe_tokenizer,
            str(error),
            [{"name": "tokenizer_prompt", "passed": False, "detail": str(error)}],
        )
    evidence["tokenizer_elapsed_ms"] = round((time.monotonic() - tokenizer_started) * 1000, 3)
    evidence["prompt_token_count"] = len(prompt_tokenization.token_ids)
    evidence["tokenizer_offset_mapping"] = prompt_tokenization.offset_mapping
    if len(prompt_tokenization.token_ids) < 2:
        return finish(
            "unsupported",
            EvalFailureKind.probe_tokenizer,
            "resolved tokenizer encodes the probe prompt into fewer than two tokens",
            [
                {
                    "name": "tokenizer_prompt",
                    "passed": False,
                    "detail": f"token_count={len(prompt_tokenization.token_ids)}",
                }
            ],
        )

    http_started = time.monotonic()
    try:
        timeout_seconds = deadline.remaining(30.0)
        evidence["effective_timeout_seconds"] = timeout_seconds
        status, raw_response = post_prompt_logprob_probe(
            endpoint_url(request.endpoint, request.endpoint.completions_path),
            request_body,
            timeout_seconds,
        )
    except ProbeTransportError as error:
        evidence["transport_outcome"] = "failed"
        evidence["http_elapsed_ms"] = round((time.monotonic() - http_started) * 1000, 3)
        return finish(
            "inconclusive",
            EvalFailureKind.probe_transport,
            str(error),
            [{"name": "tokenizer_prompt", "passed": True, "detail": "at least two tokens"}],
        )
    evidence["transport_outcome"] = "response_received"
    evidence["http_elapsed_ms"] = round((time.monotonic() - http_started) * 1000, 3)
    evidence["response_status"] = status
    response_path.write_bytes(raw_response)
    artifacts.append(
        RawArtifact(
            name="prompt_logprob_probe_response",
            kind="prompt-logprob-probe-response",
            path=str(response_path),
        )
    )
    if not 200 <= status < 300:
        return finish(
            "inconclusive",
            EvalFailureKind.probe_http,
            f"prompt-logprob probe returned HTTP {status}",
            [{"name": "http_status", "passed": False, "detail": str(status)}],
        )
    try:
        response_object = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return finish(
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            f"prompt-logprob probe returned malformed JSON: {error}",
            [{"name": "json_response", "passed": False, "detail": str(error)}],
        )
    conclusion, failure_kind, checks, validation_error = validate_prompt_logprob_response(
        response_object, prompt_tokenization
    )
    return finish(conclusion, failure_kind, validation_error, checks)
