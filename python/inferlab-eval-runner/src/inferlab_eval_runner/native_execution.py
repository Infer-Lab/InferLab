from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    CaseDeadline,
    ClientStatus,
    EvalClientRequest,
    EvalClientResult,
    EvalDefinitionInputLmEval,
    EvalFailureKind,
    EvalTaskSourceInputBundled,
    JsonObject,
    RawArtifact,
    load_json_object,
    plain_setting,
)

from inferlab_eval_runner.native_contract import TrialEvidenceWriter
from inferlab_eval_runner.normalization import (
    lm_eval_result_files,
    lm_eval_sample_files,
    normalize_repeated_lm_eval_result,
    partial_repeated_lm_eval_result,
    preserve_repeated_trial_scores,
    repeated_trial_result_objects,
)
from inferlab_eval_runner.task_resolution import (
    LmEvalRequestTarget,
    lm_eval_task_argument,
    render_mapping,
    repeated_base_seed,
    resolve_lm_eval_target,
)

PROCESS_EVIDENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class EvalCheckpointPublisher:
    callback: Callable[[EvalClientResult], None] | None

    def publish(self, result: EvalClientResult) -> None:
        if self.callback is not None:
            self.callback(result)


def lm_eval_command(
    request: EvalClientRequest,
    definition: EvalDefinitionInputLmEval,
    output_dir: Path,
    resolution: JsonObject,
    request_timeout_seconds: float | None = None,
    *,
    request_config_path: Path | None = None,
    request_evidence_path: Path | None = None,
    seed: int | None = None,
) -> list[str]:
    target = resolve_lm_eval_target(request, definition, resolution)
    request_seed = definition.seed if seed is None else seed
    model_args: dict[str, object] = {
        "model": request.model.served_name,
        "base_url": target.url,
        "timeout": request.case_budget_seconds
        if request_timeout_seconds is None
        else request_timeout_seconds,
        "tokenizer": request.model.locator,
        "tokenized_requests": False,
        "tokenizer_backend": "huggingface",
    }
    if request_seed is not None:
        model_args["seed"] = request_seed
    if definition.trials == 1 and definition.concurrency is not None:
        model_args["num_concurrent"] = definition.concurrency
    request_config_path = request_config_path or output_dir.parent / "inference-request.json"
    request_evidence_path = request_evidence_path or output_dir.parent / "inference-requests.jsonl"
    command = [
        sys.executable,
        "-m",
        "inferlab_eval_runner.lm_eval_entry",
        "--request-config",
        str(request_config_path),
        "--request-evidence",
        str(request_evidence_path),
        "run",
        "--model",
        target.model,
        "--model_args",
        render_mapping(model_args),
        "--tasks",
        lm_eval_task_argument(definition),
        "--output_path",
        str(output_dir),
    ]
    if isinstance(definition.task.root, EvalTaskSourceInputBundled):
        command.extend(["--include_path", str(Path(definition.task.root.path).parent)])
    if definition.limit is not None:
        command.extend(["--limit", str(definition.limit)])
    if definition.few_shot is not None:
        command.extend(["--num_fewshot", str(definition.few_shot)])
    if definition.seed is not None:
        command.extend(["--seed", str(definition.seed)])
    if definition.max_tokens is not None:
        command.extend(["--gen_kwargs", f"max_gen_toks={definition.max_tokens}"])
    if target.apply_chat_template:
        command.append("--apply_chat_template")
    if definition.trials > 1:
        command.append("--log_samples")
    return command


def write_inference_request_config(
    path: Path,
    definition: EvalDefinitionInputLmEval,
    target: LmEvalRequestTarget,
    resolution: JsonObject,
) -> list[RawArtifact]:
    request_body: JsonObject = {
        key: plain_setting(value) for key, value in definition.request_body.items()
    }
    payload_evidence_path = path.with_name("inference-requests.jsonl")
    trial_evidence_path = path.with_name("eval-trials.json")
    payload_evidence_path.write_text("", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_named_route": target.route_name,
                "effective_public_url": target.url,
                "definition_request_body": request_body,
                "trials": definition.trials,
                "base_seed": repeated_base_seed(definition),
                "task_identity": resolution.get("task_identity"),
                "metric_filter": definition.metric_filter,
                "threshold": definition.threshold,
                "trial_evidence_path": str(trial_evidence_path),
                "payload_evidence_path": str(payload_evidence_path),
                "native_model": target.model,
                "apply_chat_template": target.apply_chat_template,
                "prompt_authority": target.prompt_authority,
                "declared_prompt_authority": target.declared_prompt_authority,
                "tokenized_requests": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = [
        RawArtifact(
            name="inference_request",
            kind="inference-request-config",
            path=str(path),
        ),
        RawArtifact(
            name="inference_request_payloads",
            kind="inference-request-payloads",
            path=str(payload_evidence_path),
        ),
    ]
    if definition.trials > 1:
        task_identity = resolution.get("task_identity")
        if not isinstance(task_identity, str):
            raise ValueError("resolved lm-eval task has no identity for repeated evidence")
        TrialEvidenceWriter(
            trial_evidence_path,
            requested_trials=definition.trials,
            base_seed=repeated_base_seed(definition),
            task_identity=task_identity,
            threshold=definition.threshold,
        )
        artifacts.append(
            RawArtifact(
                name="eval_trials",
                kind="eval-trial-evidence",
                path=str(trial_evidence_path),
            )
        )
    return artifacts


def write_repeated_trial_request_config(
    base_path: Path,
    trial_id: str,
    deadline_monotonic: float,
) -> tuple[Path, RawArtifact]:
    config = load_json_object(base_path)
    config["trial_id"] = trial_id
    config["deadline_monotonic"] = deadline_monotonic
    path = base_path.with_name(f"inference-request-{trial_id}.json")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return (
        path,
        RawArtifact(
            name=f"inference_request_{trial_id}",
            kind="inference-request-config",
            path=str(path),
        ),
    )


def emit_captured_output(output: str | bytes | None) -> None:
    text = captured_text(output)
    if text:
        print(text, end="", file=sys.stderr)


def captured_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output


def write_lm_eval_process_evidence(
    path: Path,
    command: list[str],
    *,
    exit_code: int | None,
    timed_out: bool,
    outcome: str | None = None,
    artifact_name: str = "lm_eval_process",
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> RawArtifact:
    with PROCESS_EVIDENCE_LOCK:
        write_lm_eval_process_evidence_unlocked(
            path,
            command,
            exit_code=exit_code,
            timed_out=timed_out,
            outcome=outcome,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    return RawArtifact(name=artifact_name, kind="lm-eval-process", path=str(path))


def write_lm_eval_process_evidence_unlocked(
    path: Path,
    command: list[str],
    *,
    exit_code: int | None,
    timed_out: bool,
    outcome: str | None,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> None:
    value = {
        "schema_version": 1,
        "native_command": command,
        "outcome": outcome or ("timed_out" if timed_out else "exited"),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_path": str(stdout_path) if stdout_path is not None else None,
        "stderr_path": str(stderr_path) if stderr_path is not None else None,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def mark_lm_eval_process_terminating(
    path: Path,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    with PROCESS_EVIDENCE_LOCK:
        try:
            current = load_json_object(path)
        except (OSError, ValueError):
            return
        if current.get("outcome") != "running":
            return
        write_lm_eval_process_evidence_unlocked(
            path,
            command,
            exit_code=None,
            timed_out=False,
            outcome="control_plane_termination",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


@dataclass(frozen=True)
class NativeLmEvalAttempt:
    trial_id: str
    command: list[str]
    output_dir: Path
    process_path: Path
    stdout_path: Path | None = None
    stderr_path: Path | None = None


@dataclass(frozen=True)
class NativeLmEvalAttemptResult:
    attempt: NativeLmEvalAttempt
    returncode: int | None
    timed_out: bool
    started: bool
    error: str | None


def run_native_lm_eval_attempt(
    attempt: NativeLmEvalAttempt,
    deadline: CaseDeadline,
) -> NativeLmEvalAttemptResult:
    try:
        remaining = deadline.remaining()
    except TimeoutError:
        return NativeLmEvalAttemptResult(
            attempt,
            None,
            True,
            False,
            "measurement-case deadline expired before native trial launch",
        )
    if (attempt.stdout_path is None) != (attempt.stderr_path is None):
        raise ValueError("native lm-eval attempt must capture both stdout and stderr or neither")
    if attempt.stdout_path is not None and attempt.stderr_path is not None:
        attempt.stdout_path.write_text("", encoding="utf-8")
        attempt.stderr_path.write_text("", encoding="utf-8")
    write_lm_eval_process_evidence(
        attempt.process_path,
        attempt.command,
        exit_code=None,
        timed_out=False,
        outcome="running",
        stdout_path=attempt.stdout_path,
        stderr_path=attempt.stderr_path,
    )
    try:
        if attempt.stdout_path is not None and attempt.stderr_path is not None:
            with (
                attempt.stdout_path.open("w", encoding="utf-8") as stdout_stream,
                attempt.stderr_path.open("w", encoding="utf-8") as stderr_stream,
            ):
                completed = subprocess.run(
                    attempt.command,
                    check=False,
                    text=True,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    timeout=remaining,
                )
            captured_stdout: str | bytes | None = attempt.stdout_path.read_text(
                encoding="utf-8", errors="replace"
            )
            captured_stderr: str | bytes | None = attempt.stderr_path.read_text(
                encoding="utf-8", errors="replace"
            )
        else:
            completed = subprocess.run(
                attempt.command,
                check=False,
                text=True,
                capture_output=True,
                timeout=remaining,
            )
            captured_stdout = completed.stdout
            captured_stderr = completed.stderr
    except OSError as error:
        write_lm_eval_process_evidence(
            attempt.process_path,
            attempt.command,
            exit_code=None,
            timed_out=False,
            outcome="launch_failed",
            stdout_path=attempt.stdout_path,
            stderr_path=attempt.stderr_path,
        )
        return NativeLmEvalAttemptResult(
            attempt,
            None,
            False,
            False,
            str(error),
        )
    except subprocess.TimeoutExpired as error:
        if attempt.stdout_path is not None and attempt.stderr_path is not None:
            captured_stdout = attempt.stdout_path.read_text(encoding="utf-8", errors="replace")
            captured_stderr = attempt.stderr_path.read_text(encoding="utf-8", errors="replace")
        else:
            captured_stdout = error.stdout
            captured_stderr = error.stderr
        emit_captured_output(captured_stdout)
        emit_captured_output(captured_stderr)
        write_lm_eval_process_evidence(
            attempt.process_path,
            attempt.command,
            exit_code=None,
            timed_out=True,
            stdout_path=attempt.stdout_path,
            stderr_path=attempt.stderr_path,
        )
        return NativeLmEvalAttemptResult(
            attempt,
            None,
            True,
            True,
            f"lm-eval timed out after {error.timeout} seconds",
        )
    emit_captured_output(captured_stdout)
    emit_captured_output(captured_stderr)
    write_lm_eval_process_evidence(
        attempt.process_path,
        attempt.command,
        exit_code=completed.returncode,
        timed_out=False,
        stdout_path=attempt.stdout_path,
        stderr_path=attempt.stderr_path,
    )
    return NativeLmEvalAttemptResult(
        attempt,
        completed.returncode,
        False,
        True,
        None,
    )


def result_file_artifacts(paths: Sequence[Path]) -> list[RawArtifact]:
    return [
        RawArtifact(
            name=f"lm_eval_results_{index}",
            kind="lm-eval-results",
            path=str(path),
        )
        for index, path in enumerate(paths)
    ]


def sample_file_artifacts(paths: Sequence[Path]) -> list[RawArtifact]:
    return [
        RawArtifact(
            name=f"lm_eval_samples_{index}",
            kind="lm-eval-samples",
            path=str(path),
        )
        for index, path in enumerate(paths)
    ]


def repeated_checkpoint(
    publisher: EvalCheckpointPublisher,
    definition: EvalDefinitionInputLmEval,
    resolution: JsonObject,
    evidence_path: Path,
    native_command: list[str],
    raw_artifacts: list[RawArtifact],
    error: str,
) -> None:
    if publisher.callback is None:
        return
    metrics, normalized_metrics, gate, summary = partial_repeated_lm_eval_result(
        definition,
        resolution,
        evidence_path,
    )
    publisher.publish(
        EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics=metrics,
            normalized_metrics=normalized_metrics,
            gate=gate,
            trial_summary=summary,
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=False,
            raw_artifacts=list(raw_artifacts),
            failure_kind=None,
            error=error,
        )
    )


def run_repeated_lm_eval(
    request: EvalClientRequest,
    definition: EvalDefinitionInputLmEval,
    resolution: JsonObject,
    raw_dir: Path,
    request_config_path: Path,
    raw_artifacts: list[RawArtifact],
    publisher: EvalCheckpointPublisher,
    deadline: CaseDeadline,
) -> EvalClientResult:
    evidence_path = request_config_path.with_name("eval-trials.json")
    payload_evidence_path = request_config_path.with_name("inference-requests.jsonl")
    trial_runs: dict[str, NativeLmEvalAttemptResult] = {}
    trial_jobs: list[NativeLmEvalAttempt] = []
    native_command: list[str] = []
    score_errors: list[str] = []

    def refresh_process_artifacts() -> None:
        recorded = {artifact.path for artifact in raw_artifacts}
        for attempt in trial_jobs:
            trial_id = attempt.trial_id
            process_path = attempt.process_path
            if process_path.is_file() and str(process_path) not in recorded:
                raw_artifacts.append(
                    RawArtifact(
                        name=f"lm_eval_process_{trial_id}",
                        kind="lm-eval-process",
                        path=str(process_path),
                    )
                )
                raw_artifacts.extend(
                    [
                        RawArtifact(
                            name=f"lm_eval_stdout_{trial_id}",
                            kind="lm-eval-stdout",
                            path=str(process_path.with_name("stdout.log")),
                        ),
                        RawArtifact(
                            name=f"lm_eval_stderr_{trial_id}",
                            kind="lm-eval-stderr",
                            path=str(process_path.with_name("stderr.log")),
                        ),
                    ]
                )

    def refresh_native_artifacts() -> None:
        recorded = {artifact.path for artifact in raw_artifacts}
        for path, kind, stem in (
            *(
                (path, "lm-eval-results", "lm_eval_results")
                for path in lm_eval_result_files(raw_dir)
            ),
            *(
                (path, "lm-eval-samples", "lm_eval_samples")
                for path in lm_eval_sample_files(raw_dir)
            ),
        ):
            if str(path) in recorded:
                continue
            index = sum(artifact.kind == kind for artifact in raw_artifacts)
            raw_artifacts.append(RawArtifact(name=f"{stem}_{index}", kind=kind, path=str(path)))
            recorded.add(str(path))

    def refresh_available_scores() -> None:
        refresh_process_artifacts()
        refresh_native_artifacts()
        trial_results = repeated_trial_result_objects(raw_dir, tolerate_incomplete=True)
        if not trial_results:
            return
        try:
            preserve_repeated_trial_scores(
                trial_results,
                resolution,
                definition,
                evidence_path,
                strict_completed_scores=False,
            )
        except (OSError, TypeError, ValueError) as error:
            message = str(error)
            if message not in score_errors:
                score_errors.append(message)

    earlier_sigterm = signal.getsignal(signal.SIGTERM)

    def publish_before_termination(signum: int, frame: object) -> None:
        del signum, frame
        for attempt in trial_jobs:
            mark_lm_eval_process_terminating(
                attempt.process_path,
                attempt.command,
                attempt.stdout_path or attempt.process_path.with_name("stdout.log"),
                attempt.stderr_path or attempt.process_path.with_name("stderr.log"),
            )
        refresh_available_scores()
        repeated_checkpoint(
            publisher,
            definition,
            resolution,
            evidence_path,
            native_command,
            raw_artifacts,
            "repeated lm-eval was interrupted during control-plane cleanup",
        )
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, publish_before_termination)
    repeated_checkpoint(
        publisher,
        definition,
        resolution,
        evidence_path,
        native_command,
        raw_artifacts,
        "repeated lm-eval trial planning has started",
    )
    try:
        deadline_monotonic = time.monotonic() + deadline.remaining()
        for index in range(1, definition.trials + 1):
            trial_id = f"trial-{index:04d}"
            output_dir = raw_dir / trial_id
            output_dir.mkdir(parents=True, exist_ok=True)
            config_path, config_artifact = write_repeated_trial_request_config(
                request_config_path,
                trial_id,
                deadline_monotonic,
            )
            raw_artifacts.append(config_artifact)
            process_path = output_dir / "lm-eval-process.json"
            command = lm_eval_command(
                request,
                definition,
                output_dir,
                resolution,
                deadline.remaining(),
                request_config_path=config_path,
                request_evidence_path=payload_evidence_path,
                seed=repeated_base_seed(definition) + index - 1,
            )
            trial_jobs.append(
                NativeLmEvalAttempt(
                    trial_id=trial_id,
                    command=command,
                    output_dir=output_dir,
                    process_path=process_path,
                    stdout_path=output_dir / "stdout.log",
                    stderr_path=output_dir / "stderr.log",
                )
            )
            if not native_command:
                native_command = command
            repeated_checkpoint(
                publisher,
                definition,
                resolution,
                evidence_path,
                native_command,
                raw_artifacts,
                "repeated lm-eval trials are being planned",
            )
        concurrency = definition.concurrency or 1
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending: set[Future[NativeLmEvalAttemptResult]] = {
                executor.submit(
                    run_native_lm_eval_attempt,
                    attempt,
                    deadline,
                )
                for attempt in trial_jobs
            }
            while pending:
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                if not done:
                    repeated_checkpoint(
                        publisher,
                        definition,
                        resolution,
                        evidence_path,
                        native_command,
                        raw_artifacts,
                        "repeated lm-eval trials are still running",
                    )
                    deadline.remaining()
                    continue
                for future in done:
                    trial_run = future.result()
                    trial_runs[trial_run.attempt.trial_id] = trial_run
                refresh_available_scores()
                repeated_checkpoint(
                    publisher,
                    definition,
                    resolution,
                    evidence_path,
                    native_command,
                    raw_artifacts,
                    "repeated lm-eval trials are partially complete",
                )
    except TimeoutError as error:
        refresh_available_scores()
        metrics, normalized_metrics, gate, summary = partial_repeated_lm_eval_result(
            definition, resolution, evidence_path
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics=metrics,
            normalized_metrics=normalized_metrics,
            gate=gate,
            trial_summary=summary,
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=True,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error=str(error),
        )
    finally:
        signal.signal(signal.SIGTERM, earlier_sigterm)

    refresh_available_scores()
    unstarted = [run.attempt.trial_id for run in trial_runs.values() if not run.started]
    timed_out = [run.attempt.trial_id for run in trial_runs.values() if run.timed_out]
    if unstarted or timed_out:
        metrics, normalized_metrics, gate, summary = partial_repeated_lm_eval_result(
            definition, resolution, evidence_path
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics=metrics,
            normalized_metrics=normalized_metrics,
            gate=gate,
            trial_summary=summary,
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=True,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error="repeated lm-eval exceeded its measurement-case deadline",
        )
    evidence = load_json_object(evidence_path)
    raw_outcomes = evidence.get("endpoint_outcomes")
    if not isinstance(raw_outcomes, list) or not all(
        isinstance(outcome, dict) for outcome in raw_outcomes
    ):
        raise ValueError("repeated Eval evidence has no endpoint outcomes")
    issued_ids = {outcome.get("trial_id") for outcome in cast(list[JsonObject], raw_outcomes)}
    pre_inference_failures = [
        run
        for run in trial_runs.values()
        if run.attempt.trial_id not in issued_ids
        and (run.error is not None or run.returncode not in (None, 0))
    ]
    if pre_inference_failures:
        metrics, normalized_metrics, gate, summary = partial_repeated_lm_eval_result(
            definition, resolution, evidence_path
        )
        diagnostic = "; ".join(
            f"{run.attempt.trial_id}: {run.error or f'lm-eval exited with {run.returncode}'}"
            for run in pre_inference_failures
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics=metrics,
            normalized_metrics=normalized_metrics,
            gate=gate,
            trial_summary=summary,
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error=f"repeated lm-eval failed before request release: {diagnostic}",
        )
    try:
        trial_results = repeated_trial_result_objects(raw_dir)
        metrics, normalized_metrics, gate, summary = normalize_repeated_lm_eval_result(
            trial_results,
            resolution,
            definition,
            evidence_path,
        )
    except (OSError, TypeError, ValueError) as error:
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=EvalFailureKind.metric_normalization,
            error=f"lm-eval repeated-result normalization failed: {error}",
        )
    return EvalClientResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        metrics=metrics,
        normalized_metrics=normalized_metrics,
        gate=gate,
        trial_summary=summary,
        native_command=native_command,
        native_exit_code=None,
        native_timed_out=False,
        raw_artifacts=raw_artifacts,
        failure_kind=None,
        error=None,
    )
