import importlib.metadata
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from inferlab_measurement_sdk import (
    CaseDeadline,
    ClientStatus,
    EvalClientRequest,
    EvalClientResult,
    EvalDefinitionInputLmEval,
    EvalFailureKind,
    RawArtifact,
    load_json_object,
    parse_args,
)

from inferlab_eval_runner import native_execution, normalization, prompt_logprobs, task_resolution
from inferlab_eval_runner.task_resolution import (
    lm_eval_task_argument,
)


def run_lm_eval(
    request: EvalClientRequest,
    definition: EvalDefinitionInputLmEval,
    checkpoint: Callable[[EvalClientResult], None] | None = None,
    deadline: CaseDeadline | None = None,
) -> EvalClientResult:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    publisher = native_execution.EvalCheckpointPublisher(checkpoint)
    artifact_dir = Path(request.artifact_dir)
    raw_dir = artifact_dir / "lm-eval-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolution_path = artifact_dir / "task-resolution.json"
    resolution_artifact = RawArtifact(
        name="lm_eval_task_resolution",
        kind="lm-eval-task-resolution",
        path=str(resolution_path),
    )
    raw_dir_artifact = RawArtifact(name="lm_eval_output", kind="directory", path=str(raw_dir))
    request_config_path = artifact_dir / "inference-request.json"
    try:
        prepared = task_resolution.prepare_lm_eval_task(request, definition)
        resolution = prepared.resolution
        resolution_path.write_text(
            json.dumps(resolution, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        resolution_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "task_source": lm_eval_task_argument(definition),
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=[],
            raw_artifacts=[resolution_artifact, raw_dir_artifact],
            failure_kind=EvalFailureKind.task_resolution,
            error=f"lm-eval task resolution failed: {error}",
        )
    raw_artifacts = [resolution_artifact]
    raw_artifacts.extend(
        native_execution.write_inference_request_config(
            request_config_path, definition, prepared.target, resolution
        )
    )
    raw_artifacts.append(raw_dir_artifact)
    if prepared.requires_prompt_logprobs:
        probe = prompt_logprobs.run_prompt_logprob_probe(
            request, definition, artifact_dir, deadline
        )
        raw_artifacts.extend(probe.raw_artifacts)
        if probe.failure_kind is not None:
            return EvalClientResult(
                schema_version=1,
                status=ClientStatus.failed,
                metrics={},
                native_command=[],
                raw_artifacts=raw_artifacts,
                failure_kind=probe.failure_kind,
                error=probe.error,
            )
    if definition.trials > 1:
        return native_execution.run_repeated_lm_eval(
            request,
            definition,
            resolution,
            raw_dir,
            request_config_path,
            raw_artifacts,
            publisher,
            deadline,
        )
    command = native_execution.lm_eval_command(request, raw_dir, resolution, deadline.remaining())
    process_path = artifact_dir / "lm-eval-process.json"
    raw_artifacts.append(
        native_execution.write_lm_eval_process_evidence(
            process_path,
            command,
            exit_code=None,
            timed_out=False,
            outcome="running",
        )
    )
    if publisher.callback is not None:
        publisher.publish(
            EvalClientResult(
                schema_version=1,
                status=ClientStatus.failed,
                metrics={},
                native_command=command,
                native_exit_code=None,
                native_timed_out=False,
                raw_artifacts=raw_artifacts,
                failure_kind=None,
                error="lm-eval native attempt did not finalize",
            )
        )
    attempt = native_execution.run_native_lm_eval_attempt(
        native_execution.NativeLmEvalAttempt(
            trial_id="single",
            command=command,
            output_dir=raw_dir,
            process_path=process_path,
        ),
        deadline,
    )
    if attempt.timed_out:
        result_paths = normalization.lm_eval_result_files(raw_dir)
        raw_artifacts.extend(native_execution.result_file_artifacts(result_paths))
        raw_artifacts.extend(
            native_execution.sample_file_artifacts(normalization.lm_eval_sample_files(raw_dir))
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            native_exit_code=None,
            native_timed_out=True,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error=attempt.error,
        )
    result_paths = normalization.lm_eval_result_files(raw_dir)
    raw_artifacts.extend(native_execution.result_file_artifacts(result_paths))
    raw_artifacts.extend(
        native_execution.sample_file_artifacts(normalization.lm_eval_sample_files(raw_dir))
    )
    if attempt.error is not None or attempt.returncode != 0 or not result_paths:
        message = attempt.error or f"lm-eval exited with {attempt.returncode}"
        if attempt.returncode == 0:
            message = "lm-eval produced no results JSON"
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            native_exit_code=attempt.returncode,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error=message,
        )
    if len(result_paths) != 1:
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            native_exit_code=attempt.returncode,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=EvalFailureKind.metric_normalization,
            error=f"lm-eval produced multiple results JSON files: {len(result_paths)}",
        )
    result_path = result_paths[0]
    try:
        raw_result = load_json_object(result_path)
        metrics, normalized_metrics, gate = normalization.normalize_lm_eval_result(
            raw_result,
            resolution,
            definition,
        )
        trial_summary = None
    except (OSError, TypeError, ValueError) as error:
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            native_exit_code=attempt.returncode,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=EvalFailureKind.metric_normalization,
            error=f"lm-eval result normalization failed: {error}",
        )
    return EvalClientResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        metrics=metrics,
        normalized_metrics=normalized_metrics,
        gate=gate,
        trial_summary=trial_summary,
        native_command=command,
        native_exit_code=attempt.returncode,
        native_timed_out=False,
        raw_artifacts=raw_artifacts,
        failure_kind=None,
        error=None,
    )


def execute(
    request: EvalClientRequest,
    checkpoint: Callable[[EvalClientResult], None] | None = None,
    deadline: CaseDeadline | None = None,
) -> EvalClientResult:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    definition = request.definition.root
    if isinstance(definition, EvalDefinitionInputLmEval):
        return run_lm_eval(request, definition, checkpoint, deadline)
    raise TypeError(f"unsupported Eval definition {type(definition).__name__}")


def write_eval_client_result(path: Path, result: EvalClientResult) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def handle_eval_execution(input_text: str, output: Path) -> EvalClientResult:
    request = EvalClientRequest.model_validate_json(input_text)
    deadline = CaseDeadline(request.case_budget_seconds)
    result = execute(
        request,
        lambda checkpoint: write_eval_client_result(output, checkpoint),
        deadline,
    )
    if result.status == ClientStatus.succeeded:
        deadline.remaining()
    return result


def main() -> int:
    args = parse_args()
    if args.handshake:
        importlib.import_module("tenacity")
        importlib.import_module("transformers")
        print(
            json.dumps(
                {
                    "lm_eval_version": importlib.metadata.version("lm_eval"),
                }
            )
        )
        return 0
    if args.input is None or args.output is None:
        raise ValueError("--input and --output are required")
    output = Path(args.output)
    try:
        result = handle_eval_execution(Path(args.input).read_text(encoding="utf-8"), output)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        try:
            result = EvalClientResult.model_validate_json(output.read_text(encoding="utf-8"))
            result.status = ClientStatus.failed
            result.error = f"{result.error}; Eval runner failed: {error}"
        except (OSError, ValueError):
            result = EvalClientResult(
                schema_version=1,
                status=ClientStatus.failed,
                metrics={},
                native_command=[],
                raw_artifacts=[],
                failure_kind=None,
                error=str(error),
            )
    write_eval_client_result(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
