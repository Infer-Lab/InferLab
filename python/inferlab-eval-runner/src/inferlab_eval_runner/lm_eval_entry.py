from __future__ import annotations

import argparse
import contextvars
import fcntl
import hashlib
import importlib
import json
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Protocol, cast

from inferlab_measurement_sdk import JsonObject, load_json_object

from inferlab_eval_runner.native_contract import TrialEvidenceWriter, generated_response


class PayloadClientClass(Protocol):
    _create_payload: Callable[..., JsonObject]


class RepeatsConfig(Protocol):
    repeats: int


class RepeatedTask(Protocol):
    OUTPUT_TYPE: str
    config: RepeatsConfig
    instances: object
    build_all_requests: Callable[..., None]


@dataclass(frozen=True)
class PendingPayload:
    client: object
    family: str
    defaults: JsonObject
    effective: JsonObject
    fragment: JsonObject
    evidence: PayloadEvidenceWriter


class RepeatedTrialState:
    def __init__(
        self,
        evidence: TrialEvidenceWriter,
        trial_id: str,
        deadline_monotonic: float,
    ) -> None:
        self.evidence = evidence
        self.trial_id = trial_id
        self.deadline_monotonic = deadline_monotonic
        self._pending: contextvars.ContextVar[PendingPayload | None] = contextvars.ContextVar(
            "inferlab_eval_payload", default=None
        )
        self._client: object | None = None

    def prepare(
        self,
        client: object,
        family: str,
        defaults: JsonObject,
        effective: JsonObject,
        fragment: JsonObject,
        evidence: PayloadEvidenceWriter,
    ) -> None:
        self._pending.set(PendingPayload(client, family, defaults, effective, fragment, evidence))

    def release(self, request: JsonObject) -> RepeatedTrialState:
        pending = self._pending.get()
        if pending is None:
            raise RuntimeError("lm-eval released a repeated request without its payload")
        self._client = pending.client
        if self.evidence.issue(self.trial_id, request):
            pending.evidence.record(
                pending.family,
                pending.defaults,
                pending.effective,
                pending.fragment,
            )
        return self

    def deadline_seconds(self) -> float:
        return self.remaining()

    def failed(self, status: int, text: str) -> None:
        self.evidence.fail(self.current(), f"endpoint returned HTTP {status}: {text}", status)

    def completed(self, status: int, response: JsonObject) -> None:
        trial_id = self.current()
        text, _ = generated_response(response)
        self.evidence.complete(trial_id, response, self.tokenizer_count(trial_id, text), status)

    def transport_failed(self, error: BaseException) -> None:
        # A transport failure surfaces through lm-eval's retry and the
        # attempt's terminal error, which the trial evidence already keeps.
        del error

    def current(self) -> str:
        if self._client is None:
            raise RuntimeError("lm-eval response arrived without a repeated trial")
        return self.trial_id

    def finish(self) -> None:
        self._pending.set(None)

    def remaining(self) -> float:
        remaining = self.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("repeated Eval case deadline expired before request release")
        return remaining

    def tokenizer_count(self, trial_id: str, text: str | None) -> int | None:
        if text is None:
            return None
        if trial_id != self.trial_id:
            raise ValueError(f"unknown repeated Eval trial {trial_id!r}")
        tokenizer = getattr(self._client, "tokenizer", None)
        if not callable(tokenizer):
            raise ValueError("resolved lm-eval client has no callable tokenizer")
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        token_ids = getattr(encoded, "input_ids", None)
        if not isinstance(token_ids, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in token_ids
        ):
            raise ValueError("resolved lm-eval tokenizer returned invalid token ids")
        return len(token_ids)


# Pinned by protocol/fixtures/reserved-request-members.json ([[RFC-0004:C-INFERENCE-REQUESTS]]).
STRUCTURAL_REQUEST_MEMBERS = frozenset(
    {
        "model",
        "prompt",
        "messages",
        "stream",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "stop",
    }
)


def nonstructural_request_body(payload: JsonObject) -> JsonObject:
    return {key: value for key, value in payload.items() if key not in STRUCTURAL_REQUEST_MEMBERS}


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
                    "earlier_authority": "resolved lm-eval task/client payload",
                    "replacement": replacement,
                    "replacement_authority": "effective Eval definition request_body",
                }
            )
    return replacements


@dataclass
class PayloadEvidenceWriter:
    path: Path
    # Concurrent trial processes append to one file, so the next index is
    # counted from the bytes other writers appended since this one last
    # looked, not by re-reading the whole file for every request.
    _counted_offset: int = field(default=0, init=False)
    _counted_records: int = field(default=0, init=False)

    def record(
        self,
        family: str,
        defaults: JsonObject,
        effective: JsonObject,
        fragment: JsonObject,
    ) -> None:
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.exists():
                with self.path.open("rb") as existing:
                    existing.seek(self._counted_offset)
                    for line in existing:
                        if line.strip():
                            self._counted_records += 1
                    self._counted_offset = existing.tell()
            request_index = self._counted_records
            record = {
                "schema_version": 1,
                "request_index": request_index,
                "request_family": family,
                "task_and_client_defaults": nonstructural_request_body(defaults),
                "definition_request_body": fragment,
                "effective_request_body": nonstructural_request_body(effective),
                "replaced_defaults": replaced_defaults(
                    nonstructural_request_body(defaults), fragment
                ),
            }
            line = (json.dumps(record, sort_keys=True) + "\n").encode()
            with self.path.open("ab") as stream:
                stream.write(line)
                self._counted_offset = stream.tell()
            self._counted_records += 1


def merge_request_body(defaults: JsonObject, fragment: JsonObject) -> JsonObject:
    """Recursively apply the operator fragment over one native request body."""
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


def install_request_body(
    fragment: JsonObject,
    evidence: PayloadEvidenceWriter,
    repeated: RepeatedTrialState | None = None,
) -> None:
    """Adapt the pinned local OpenAI clients at their final payload boundary."""
    module = importlib.import_module("lm_eval.models.openai_completions")
    local_completions = cast(PayloadClientClass, module.LocalCompletionsAPI)
    local_chat_completions = cast(PayloadClientClass, module.LocalChatCompletion)
    create_completions = local_completions._create_payload
    create_chat = local_chat_completions._create_payload

    def completions_payload(
        client: object,
        messages: object,
        generate: bool = False,
        gen_kwargs: dict[str, object] | None = None,
        *,
        seed: int,
        eos: object = None,
        **kwargs: object,
    ) -> JsonObject:
        payload = create_completions(
            client,
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )
        effective = merge_request_body(payload, fragment)
        effective["n"] = 1
        effective["stream"] = False
        if repeated is None:
            evidence.record("completions", payload, effective, fragment)
        else:
            effective["seed"] = repeated.evidence.seed_for(repeated.trial_id)
            repeated.prepare(
                client,
                "completions",
                payload,
                effective,
                fragment,
                evidence,
            )
        return effective

    def chat_payload(
        client: object,
        messages: object,
        generate: bool = False,
        gen_kwargs: dict[str, object] | None = None,
        *,
        seed: int,
        eos: object = None,
        **kwargs: object,
    ) -> JsonObject:
        payload = create_chat(
            client,
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )
        effective = merge_request_body(payload, fragment)
        effective["n"] = 1
        effective["stream"] = False
        if repeated is None:
            evidence.record("chat_completions", payload, effective, fragment)
        else:
            effective["seed"] = repeated.evidence.seed_for(repeated.trial_id)
            repeated.prepare(
                client,
                "chat_completions",
                payload,
                effective,
                fragment,
                evidence,
            )
        return effective

    local_completions._create_payload = completions_payload
    local_chat_completions._create_payload = chat_payload


def prepare_repeated_task(task: object, trials: int, metric_filter: str | None) -> None:
    typed_task = cast(RepeatedTask, task)
    if typed_task.OUTPUT_TYPE != "generate_until":
        raise ValueError("trials greater than one require a generate_until lm-eval task")
    del trials, metric_filter
    if typed_task.config.repeats != 1:
        raise ValueError("trials greater than one require task-owned response multiplicity of one")
    original_build = typed_task.build_all_requests

    def build_one_sample(task_self: object, **kwargs: object) -> None:
        original_build(**kwargs)
        instances = getattr(task_self, "instances", None)
        if not isinstance(instances, list) or len(instances) != 1:
            raise ValueError(
                "trials greater than one require a resolved lm-eval task with exactly one sample"
            )
        if getattr(instances[0], "request_type", None) != "generate_until":
            raise ValueError(
                "trials greater than one require one generate_until request per sample"
            )

    typed_task.build_all_requests = MethodType(build_one_sample, task)


def install_repeated_task_loading(trials: int, metric_filter: str | None) -> None:
    tasks_module = importlib.import_module("lm_eval.tasks")
    manager = tasks_module.TaskManager
    original_load = manager.load

    def load(manager_self: object, *args: object, **kwargs: object) -> object:
        loaded = original_load(manager_self, *args, **kwargs)
        tasks = loaded.get("tasks") if isinstance(loaded, dict) else None
        if not isinstance(tasks, dict) or len(tasks) != 1:
            raise ValueError("repeated Eval must resolve to one lm-eval task")
        prepare_repeated_task(next(iter(tasks.values())), trials, metric_filter)
        return loaded

    manager.load = load


class ExchangeAttempt(Protocol):
    """One HTTP attempt's outcome, reported by the transport wrappers."""

    def failed(self, status: int, text: str) -> None: ...

    def completed(self, status: int, response: JsonObject) -> None: ...

    def transport_failed(self, error: BaseException) -> None: ...


class ExchangeObserver(Protocol):
    """What one capture mode does with lm-eval's endpoint exchanges."""

    def deadline_seconds(self) -> float | None: ...

    def release(self, request: JsonObject) -> ExchangeAttempt | None: ...


class SyncResponse(Protocol):
    @property
    def ok(self) -> bool: ...

    @property
    def text(self) -> str: ...

    @property
    def status_code(self) -> int: ...

    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class CapturedSyncResponse:
    def __init__(self, response: SyncResponse, attempt: ExchangeAttempt) -> None:
        self._response = response
        self._attempt = attempt

    @property
    def ok(self) -> bool:
        return self._response.ok

    @property
    def text(self) -> str:
        text = self._response.text
        if not self._response.ok:
            self._attempt.failed(self._response.status_code, text)
        return text

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    def json(self) -> object:
        value = self._response.json()
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            self._attempt.completed(self._response.status_code, cast(JsonObject, value))
        return value


class AsyncResponse(Protocol):
    @property
    def ok(self) -> bool: ...

    @property
    def status(self) -> int: ...

    async def text(self) -> str: ...

    def raise_for_status(self) -> None: ...

    async def json(self) -> object: ...


class CapturedAsyncResponse:
    """The response lm-eval 0.4.13's ``amodel_call`` reads: ``ok``, ``status``,
    ``text()``, ``raise_for_status()``, and ``json()``, each the wrapped
    response's, with the attempt recorded as its body is read."""

    def __init__(self, response: AsyncResponse, attempt: ExchangeAttempt) -> None:
        self._response = response
        self._attempt = attempt

    @property
    def ok(self) -> bool:
        return self._response.ok

    @property
    def status(self) -> int:
        return self._response.status

    async def text(self) -> str:
        text = await self._response.text()
        if not self._response.ok:
            self._attempt.failed(self._response.status, text)
        return text

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    async def json(self) -> object:
        value = await self._response.json()
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            self._attempt.completed(self._response.status, cast(JsonObject, value))
        return value


class AsyncRequestContext(Protocol):
    async def __aenter__(self) -> AsyncResponse: ...

    async def __aexit__(self, error_type: object, error: object, traceback: object) -> object: ...


class AsyncSession(Protocol):
    async def __aenter__(self) -> object: ...

    async def __aexit__(self, error_type: object, error: object, traceback: object) -> object: ...

    def post(
        self,
        url: str,
        *,
        json: JsonObject,
        headers: object,
        timeout: object = None,
    ) -> AsyncRequestContext: ...


class CapturedAsyncRequestContext:
    def __init__(self, context: AsyncRequestContext, attempt: ExchangeAttempt) -> None:
        self._context = context
        self._attempt = attempt

    async def __aenter__(self) -> CapturedAsyncResponse:
        try:
            response = await self._context.__aenter__()
        except BaseException as error:
            self._attempt.transport_failed(error)
            raise
        return CapturedAsyncResponse(response, self._attempt)

    async def __aexit__(self, error_type: object, error: object, traceback: object) -> object:
        return await self._context.__aexit__(error_type, error, traceback)


class CapturedClientSession:
    factory: Callable[..., AsyncSession]
    timeout_factory: Callable[..., object]
    observer: ExchangeObserver

    def __init__(self, connector: object = None, timeout: object = None) -> None:
        deadline = self.observer.deadline_seconds()
        self._session: AsyncSession = self.factory(
            connector=connector,
            timeout=timeout if deadline is None else self.timeout_factory(total=deadline),
        )

    async def __aenter__(self) -> CapturedClientSession:
        await self._session.__aenter__()
        return self

    async def __aexit__(self, error_type: object, error: object, traceback: object) -> object:
        return await self._session.__aexit__(error_type, error, traceback)

    def post(
        self,
        url: str,
        *,
        json: JsonObject,
        headers: object,
    ) -> AsyncRequestContext | CapturedAsyncRequestContext:
        deadline = self.observer.deadline_seconds()
        attempt = self.observer.release(json)
        if deadline is None:
            context = self._session.post(url, json=json, headers=headers)
        else:
            context = self._session.post(
                url, json=json, headers=headers, timeout=self.timeout_factory(total=deadline)
            )
        if attempt is None:
            return context
        return CapturedAsyncRequestContext(context, attempt)


def install_response_capture(observer: ExchangeObserver) -> None:
    """Report every endpoint exchange of lm-eval's API clients to one observer."""
    api_models = importlib.import_module("lm_eval.models.api_models")
    requests_module = api_models.requests
    original_post = requests_module.post

    def post(
        url: str,
        *,
        json: JsonObject,
        headers: object,
        verify: object,
        timeout: object = None,
    ) -> SyncResponse | CapturedSyncResponse:
        # A repeated trial's case deadline replaces lm-eval's client timeout,
        # as on the async path.
        deadline = observer.deadline_seconds()
        attempt = observer.release(json)
        try:
            response = cast(
                SyncResponse,
                original_post(
                    url,
                    json=json,
                    headers=headers,
                    verify=verify,
                    timeout=timeout if deadline is None else deadline,
                ),
            )
        except BaseException as error:
            if attempt is not None:
                attempt.transport_failed(error)
            raise
        if attempt is None:
            return response
        return CapturedSyncResponse(response, attempt)

    requests_module.post = post
    CapturedClientSession.factory = cast(
        Callable[..., AsyncSession], vars(api_models)["ClientSession"]
    )
    CapturedClientSession.timeout_factory = cast(
        Callable[..., object], vars(api_models)["ClientTimeout"]
    )
    CapturedClientSession.observer = observer
    vars(api_models)["ClientSession"] = CapturedClientSession


def install_repeated_response_capture(repeated: RepeatedTrialState) -> None:
    install_response_capture(repeated)
    template_api = importlib.import_module("lm_eval.models.api_models").TemplateAPI
    original_model_call = template_api.model_call
    original_async_model_call = template_api.amodel_call

    def model_call(client: object, *args: object, **kwargs: object) -> object:
        result = original_model_call(client, *args, **kwargs)
        repeated.finish()
        return result

    async def async_model_call(client: object, *args: object, **kwargs: object) -> object:
        result = await cast(Awaitable[object], original_async_model_call(client, *args, **kwargs))
        repeated.finish()
        return result

    template_api.model_call = model_call
    template_api.amodel_call = async_model_call


def context_digest(context: object) -> str:
    """SHA-256 of one request context as lm-eval holds it: a completion
    prompt, or the serialized chat messages of a ``JsonChatStr``."""
    prompt = getattr(context, "prompt", context)
    text = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


class GenerationAttempt:
    def __init__(self, log: GenerationOutcomeLog, contexts: list[str]) -> None:
        self._log = log
        self._contexts = contexts
        self._sent_unix_ns = time.time_ns()

    def _write(self, fields: JsonObject) -> None:
        self._log.write(
            {
                "entry": "attempt",
                "context_sha256": list(self._contexts),
                "sent_unix_ns": self._sent_unix_ns,
                "completed_unix_ns": time.time_ns(),
                **fields,
            }
        )

    def failed(self, status: int, text: str) -> None:
        self._write({"http_status": status, "error": text, "choices": [], "usage": None})

    def completed(self, status: int, response: JsonObject) -> None:
        raw_choices = response.get("choices")
        choices = [
            {key: value for key, value in choice.items() if key != "logprobs"}
            for choice in (raw_choices if isinstance(raw_choices, list) else [])
            if isinstance(choice, dict)
        ]
        self._write(
            {
                "http_status": status,
                "error": None,
                "choices": choices,
                "usage": response.get("usage"),
            }
        )

    def transport_failed(self, error: BaseException) -> None:
        self._write({"http_status": None, "error": repr(error), "choices": [], "usage": None})


class GenerationOutcomeLog:
    """Single-trial generation outcomes, appended as they happen because
    lm-eval writes its sample log only when evaluation completes
    ([[RFC-0005:C-LM-EVAL-EVIDENCE]])."""

    def __init__(self, path: Path) -> None:
        self._stream = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._contexts: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
            "inferlab_generation_contexts", default=None
        )

    def write(self, entry: JsonObject) -> None:
        line = json.dumps(entry, sort_keys=True) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()

    def record_instances(self, requests: Sequence[object]) -> None:
        for request in requests:
            arguments = getattr(request, "arguments", ())
            self.write(
                {
                    "entry": "instance",
                    "task_name": getattr(request, "task_name", None),
                    "doc_id": getattr(request, "doc_id", None),
                    "request_index": getattr(request, "idx", None),
                    "context_sha256": context_digest(arguments[0]) if arguments else None,
                }
            )

    def generating(self, messages: object) -> contextvars.Token[list[str] | None]:
        # lm-eval passes a batch as a list on its concurrent path and as the
        # tuple ``zip`` builds on its synchronous one.
        contexts = messages if isinstance(messages, (list, tuple)) else [messages]
        return self._contexts.set([context_digest(context) for context in contexts])

    def done(self, token: contextvars.Token[list[str] | None]) -> None:
        self._contexts.reset(token)

    def deadline_seconds(self) -> float | None:
        return None

    def release(self, request: JsonObject) -> GenerationAttempt | None:
        del request
        contexts = self._contexts.get()
        return None if contexts is None else GenerationAttempt(self, contexts)


def install_generation_capture(log: GenerationOutcomeLog) -> None:
    install_response_capture(log)
    template_api = importlib.import_module("lm_eval.models.api_models").TemplateAPI
    original_generate_until = template_api.generate_until
    original_model_call = template_api.model_call
    original_async_model_call = template_api.amodel_call

    def generate_until(
        client: object, requests: list[object], *args: object, **kwargs: object
    ) -> object:
        log.record_instances(requests)
        return original_generate_until(client, requests, *args, **kwargs)

    def model_call(client: object, *args: object, **kwargs: object) -> object:
        if not kwargs.get("generate", True):
            return original_model_call(client, *args, **kwargs)
        token = log.generating(kwargs["messages"] if "messages" in kwargs else args[0])
        try:
            return original_model_call(client, *args, **kwargs)
        finally:
            log.done(token)

    async def async_model_call(client: object, *args: object, **kwargs: object) -> object:
        if not kwargs.get("generate", True):
            return await cast(Awaitable[object], original_async_model_call(client, *args, **kwargs))
        token = log.generating(kwargs["messages"] if "messages" in kwargs else args[2])
        try:
            return await cast(Awaitable[object], original_async_model_call(client, *args, **kwargs))
        finally:
            log.done(token)

    template_api.generate_until = generate_until
    template_api.model_call = model_call
    template_api.amodel_call = async_model_call


def install_sample_projection() -> None:
    """Write lm-eval's per-sample log without each sample's document and
    request contexts, which the task identity reproduces and which can
    dominate the record; their hashes stay ([[RFC-0005:C-LM-EVAL-EVIDENCE]])."""
    tracker = importlib.import_module("lm_eval.loggers.evaluation_tracker").EvaluationTracker
    original_save = tracker.save_results_samples

    def save_results_samples(
        tracker_self: object, task_name: str, samples: list[JsonObject]
    ) -> None:
        for sample in samples:
            sample.pop("doc", None)
            arguments = sample.get("arguments")
            if isinstance(arguments, list):
                sample["arguments"] = [
                    (None, *request[1:]) if isinstance(request, (list, tuple)) else request
                    for request in arguments
                ]
        original_save(tracker_self, task_name, samples)

    tracker.save_results_samples = save_results_samples


def initialize_payload_evidence(path: Path, repeated: bool) -> None:
    if repeated:
        if not path.is_file():
            raise ValueError(f"repeated request evidence {path} is not initialized")
        return
    path.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request-config", required=True)
    parser.add_argument("--request-evidence", required=True)
    options, lm_eval_arguments = parser.parse_known_args()
    config = load_json_object(Path(options.request_config))
    raw_fragment = config.get("definition_request_body")
    if not isinstance(raw_fragment, dict) or not all(isinstance(key, str) for key in raw_fragment):
        raise ValueError("inference request config has no definition_request_body object")
    evidence_path = Path(options.request_evidence)
    raw_trials = config.get("trials", 1)
    raw_seed = config.get("base_seed")
    if (
        not isinstance(raw_trials, int)
        or isinstance(raw_trials, bool)
        or raw_trials < 1
        or not isinstance(raw_seed, int)
        or isinstance(raw_seed, bool)
        or raw_seed < 0
    ):
        raise ValueError("inference request config has invalid trials or base_seed")
    initialize_payload_evidence(evidence_path, raw_trials > 1)
    repeated: RepeatedTrialState | None = None
    if raw_trials > 1:
        raw_trial_path = config.get("trial_evidence_path")
        trial_id = config.get("trial_id")
        deadline_monotonic = config.get("deadline_monotonic")
        task_identity = config.get("task_identity")
        metric_filter = config.get("metric_filter")
        if (
            not isinstance(raw_trial_path, str)
            or not isinstance(trial_id, str)
            or not isinstance(deadline_monotonic, (int, float))
            or isinstance(deadline_monotonic, bool)
            or not isinstance(task_identity, str)
        ):
            raise ValueError("repeated inference request config has no trial evidence identity")
        if metric_filter is not None and not isinstance(metric_filter, str):
            raise ValueError("repeated inference request config has invalid metric_filter")
        raw_threshold = config.get("threshold")
        threshold = (
            float(raw_threshold)
            if isinstance(raw_threshold, (int, float)) and not isinstance(raw_threshold, bool)
            else None
        )
        repeated = RepeatedTrialState(
            TrialEvidenceWriter(
                Path(raw_trial_path),
                requested_trials=raw_trials,
                base_seed=raw_seed,
                task_identity=task_identity,
                threshold=threshold,
                initialize=False,
            ),
            trial_id,
            float(deadline_monotonic),
        )
        install_repeated_task_loading(raw_trials, metric_filter)
        install_repeated_response_capture(repeated)
    else:
        outcome_path = config.get("generation_outcome_path")
        if not isinstance(outcome_path, str):
            raise ValueError("inference request config has no generation_outcome_path")
        install_generation_capture(GenerationOutcomeLog(Path(outcome_path)))
    install_sample_projection()
    install_request_body(
        cast(JsonObject, raw_fragment), PayloadEvidenceWriter(evidence_path), repeated
    )

    sys.argv = [sys.argv[0], *lm_eval_arguments]
    module = importlib.import_module("lm_eval.__main__")
    cli_evaluate = cast(Callable[[], None], module.cli_evaluate)
    cli_evaluate()


if __name__ == "__main__":
    main()
