from __future__ import annotations

import fcntl
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import JsonObject, load_json_object


class TrialEvidenceWriter:
    """Atomically preserve one flat outcome per logical repeated Eval trial."""

    def __init__(
        self,
        path: Path,
        requested_trials: int,
        base_seed: int,
        task_identity: str | None = None,
        threshold: float | None = None,
        *,
        initialize: bool = True,
    ) -> None:
        self.path = path
        self._thread_lock = threading.Lock()
        self._lock_path = path.with_name(f".{path.name}.lock")
        self._task_identity = task_identity
        self._threshold = threshold
        self._planned = [
            {"trial_id": f"trial-{index:04d}", "seed": base_seed + index - 1}
            for index in range(1, requested_trials + 1)
        ]
        if initialize:
            with self._exclusive():
                self._rewrite([])
        elif not self.path.is_file():
            raise ValueError(f"repeated Eval evidence {self.path} is not initialized")

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._thread_lock, self._lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def seed_for(self, trial_id: str) -> int:
        for planned in self._planned:
            if planned["trial_id"] == trial_id:
                seed = planned["seed"]
                if isinstance(seed, int) and not isinstance(seed, bool):
                    return seed
        raise ValueError(f"unknown repeated Eval trial {trial_id!r}")

    def issue(self, trial_id: str, request: JsonObject) -> bool:
        with self._exclusive():
            outcomes = self._outcomes()
            if any(outcome["trial_id"] == trial_id for outcome in outcomes):
                return False
            expected_seed = self.seed_for(trial_id)
            if request.get("seed") != expected_seed:
                raise ValueError(
                    f"repeated Eval trial {trial_id!r} released request seed "
                    f"{request.get('seed')!r}, expected {expected_seed}"
                )
            effective_request = dict(request)
            outcomes.append(
                {
                    "trial_id": trial_id,
                    "seed": expected_seed,
                    "sample_identity": {
                        "task": self._task_identity,
                        "document_index": 0,
                    },
                    "effective_request": effective_request,
                    "response": None,
                    "http_status": None,
                    "generated_response": None,
                    "finish_reason": None,
                    "effective_generation_token_limit": generation_token_limit(effective_request),
                    "completion_token_count": 0,
                    "completion_token_count_source": "none",
                    "maximum_token_hit": None,
                    "binary_score": None,
                    "passed": None,
                    "classified_outcome": None,
                    "failure": "issued request has no completed task classification",
                    "native_sample": None,
                }
            )
            self._rewrite(outcomes)
            return True

    def complete(
        self,
        trial_id: str,
        response: JsonObject,
        tokenizer_count: int | None = None,
        http_status: int | None = None,
    ) -> None:
        with self._exclusive():
            outcomes = self._outcomes()
            outcome = self._outcome(outcomes, trial_id)
            generated, finish_reason = generated_response(response)
            server_count = completion_tokens(response)
            if server_count is not None:
                count = server_count
                count_source = "server-usage"
            elif generated is not None and tokenizer_count is not None:
                count = tokenizer_count
                count_source = "resolved-tokenizer"
            else:
                count = 0
                count_source = "none"
            outcome.update(
                {
                    "response": response,
                    "http_status": http_status,
                    "generated_response": generated,
                    "finish_reason": finish_reason,
                    "completion_token_count": count,
                    "completion_token_count_source": count_source,
                    "maximum_token_hit": (
                        True
                        if finish_reason == "length"
                        else False
                        if finish_reason == "stop"
                        else None
                    ),
                    "failure": "completed endpoint response has no task classification",
                }
            )
            self._rewrite(outcomes)

    def fail(self, trial_id: str, message: str, http_status: int | None = None) -> None:
        with self._exclusive():
            outcomes = self._outcomes()
            outcome = self._outcome(outcomes, trial_id)
            outcome["http_status"] = http_status
            outcome["classified_outcome"] = "request_failure"
            outcome["failure"] = message
            self._rewrite(outcomes)

    def score(
        self,
        trial_id: str,
        score: float,
        native_sample: JsonObject,
    ) -> None:
        if score not in (0.0, 1.0):
            raise ValueError("repeated Eval trial score must be binary zero or one")
        with self._exclusive():
            outcomes = self._outcomes()
            outcome = self._outcome(outcomes, trial_id)
            sample_record = native_sample.get("sample_record")
            task_evidence = (
                sample_record.get("task_evidence") if isinstance(sample_record, dict) else None
            )
            task_outcome = (
                task_evidence.get("classified_outcome") if isinstance(task_evidence, dict) else None
            )
            outcome["binary_score"] = score
            outcome["passed"] = score == 1.0
            if task_outcome in {"passed", "wrong", "unparseable"}:
                outcome["classified_outcome"] = (
                    "truncated" if outcome["maximum_token_hit"] is True else task_outcome
                )
            outcome["failure"] = None
            outcome["native_sample"] = native_sample
            self._rewrite(outcomes)

    def _outcomes(self) -> list[JsonObject]:
        value = load_json_object(self.path)
        raw = value.get("endpoint_outcomes")
        if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
            raise ValueError("repeated Eval evidence has no endpoint outcomes")
        return cast(list[JsonObject], raw)

    @staticmethod
    def _outcome(outcomes: list[JsonObject], trial_id: str) -> JsonObject:
        for outcome in outcomes:
            if outcome["trial_id"] == trial_id:
                return outcome
        raise ValueError(f"repeated Eval trial {trial_id!r} was not issued")

    def _aggregates(self, outcomes: list[JsonObject]) -> JsonObject:
        issued = len(outcomes)
        completed = sum(outcome["binary_score"] in (0.0, 1.0) for outcome in outcomes)
        passed = sum(outcome["binary_score"] == 1.0 for outcome in outcomes)
        return {
            "requested_trials": len(self._planned),
            "issued_trials": issued,
            "unissued_trials": len(self._planned) - issued,
            "completed_trials": completed,
            "request_failure_trials": issued - completed,
            "passed_trials": passed,
            "pass_rate": passed / issued if issued else None,
        }

    def _rewrite(self, outcomes: list[JsonObject]) -> None:
        aggregates = self._aggregates(outcomes)
        pass_rate = aggregates["pass_rate"]
        observed_gate = None
        if isinstance(pass_rate, float) and self._threshold is not None:
            observed_gate = {
                "pass_rate": pass_rate,
                "threshold": self._threshold,
                "comparison": "at_least",
                "conclusion": "passed" if pass_rate >= self._threshold else "failed",
            }
        value = {
            "schema_version": 1,
            "requested_trials": len(self._planned),
            "planned_trials": self._planned,
            "endpoint_outcomes": outcomes,
            "aggregates": aggregates,
            "observed_gate": observed_gate,
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def generation_token_limit(request: JsonObject) -> int | None:
    value = request.get("max_completion_tokens", request.get("max_tokens"))
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def completion_tokens(response: JsonObject) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def generated_response(response: JsonObject) -> tuple[str | None, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None, None
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    typed_finish_reason = finish_reason if isinstance(finish_reason, str) else None
    text = choice.get("text")
    if isinstance(text, str):
        return text, typed_finish_reason
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content, typed_finish_reason
    return None, typed_finish_reason
