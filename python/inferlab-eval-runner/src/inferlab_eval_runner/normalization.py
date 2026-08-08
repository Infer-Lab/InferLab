from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    EvalDefinitionInputLmEval,
    EvalMetricComparison,
    EvalMetricGate,
    EvalMetricGateConclusion,
    EvalNormalizedMetric,
    EvalPromptInput,
    EvalTrialSummary,
    JsonObject,
    load_json_object,
)

from inferlab_eval_runner.native_contract import TrialEvidenceWriter
from inferlab_eval_runner.task_resolution import repeated_base_seed


def lm_eval_result_files(output_dir: Path) -> list[Path]:
    return sorted(
        output_dir.rglob("results_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )


def lm_eval_sample_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.rglob("samples_*.jsonl"), key=str)


def repeated_native_sample_reference(
    native_trial_dir: Path,
    definition: EvalDefinitionInputLmEval,
    native_key: str,
    score: float,
    *,
    strict: bool,
) -> tuple[list[Path], JsonObject | None]:
    sample_paths = lm_eval_sample_files(native_trial_dir)

    def unavailable(message: str) -> tuple[list[Path], JsonObject | None]:
        if strict:
            raise ValueError(message)
        return sample_paths, None

    if len(sample_paths) != 1:
        return unavailable(
            f"repeated lm-eval completed trial {native_trial_dir.name!r} must have exactly "
            f"one native samples JSONL artifact, found {len(sample_paths)}"
        )
    metric, separator, native_filter = native_key.partition(",")
    if not separator or metric != definition.metric:
        return unavailable(f"repeated lm-eval metric key {native_key!r} has no native filter")
    try:
        lines = [
            line
            for line in sample_paths[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        return unavailable(
            f"repeated lm-eval native sample for trial {native_trial_dir.name!r} "
            f"is unreadable: {error}"
        )
    candidates = [
        (index, record)
        for index, record in enumerate(records, 1)
        if isinstance(record, dict)
        and record.get("filter") == native_filter
        and isinstance(record.get("metrics"), list)
        and definition.metric in record["metrics"]
    ]
    if len(candidates) != 1:
        return unavailable(
            f"repeated lm-eval completed trial {native_trial_dir.name!r} must have exactly "
            f"one native sample for metric {definition.metric!r} and filter "
            f"{native_filter!r}, found {len(candidates)}"
        )
    line_number, raw_record = candidates[0]
    record = cast(JsonObject, raw_record)
    sample_score = record.get(definition.metric)
    responses = record.get("resps")
    filtered_responses = record.get("filtered_resps")
    doc_id = record.get("doc_id")
    doc = record.get("doc")
    task_evidence = doc.get("_inferlab_task_evidence") if isinstance(doc, dict) else None
    if (
        not isinstance(sample_score, (int, float))
        or isinstance(sample_score, bool)
        or float(sample_score) != score
        or not isinstance(doc_id, int)
        or isinstance(doc_id, bool)
        or not isinstance(responses, list)
        or len(responses) != 1
        or not isinstance(filtered_responses, list)
        or len(filtered_responses) != 1
    ):
        return unavailable(
            f"repeated lm-eval native sample for trial {native_trial_dir.name!r} "
            "does not identify the single scored response"
        )
    reference: JsonObject = {
        "artifact": str(sample_paths[0]),
        "line_number": line_number,
        "doc_id": doc_id,
        "filter": native_filter,
        "metric": definition.metric,
        "score": score,
        "raw_responses": responses,
        "filtered_responses": filtered_responses,
    }
    if isinstance(task_evidence, dict):
        reference["task_evidence"] = task_evidence
    return sample_paths, reference


def resolved_prompt_authority(resolution: JsonObject) -> EvalPromptInput:
    """The authority that produced a metric travels with the metric itself."""
    target = resolution.get("request_target")
    authority = target.get("prompt_authority") if isinstance(target, dict) else None
    if authority not in {"flat", "server_chat"}:
        raise ValueError("resolved lm-eval task has no prompt authority for its normalized metric")
    return EvalPromptInput.model_validate({"kind": authority})


def normalize_lm_eval_result(
    raw: JsonObject,
    resolution: JsonObject,
    definition: EvalDefinitionInputLmEval,
) -> tuple[dict[str, float], dict[str, EvalNormalizedMetric], EvalMetricGate]:
    source_identity = resolution.get("task_identity")
    if not isinstance(source_identity, str):
        raise ValueError("resolved lm-eval task has no metric source identity")
    result_section = raw.get("results")
    if not isinstance(result_section, dict):
        raise ValueError("lm-eval result has no results object")
    selected = result_section.get(source_identity)
    if not isinstance(selected, dict):
        raise ValueError(f"lm-eval result has no task metric source {source_identity!r}")

    if definition.metric_filter is not None:
        native_key = f"{definition.metric},{definition.metric_filter}"
        candidates = [native_key] if native_key in selected else []
    else:
        candidates = sorted(
            key
            for key in selected
            if isinstance(key, str) and key.split(",", 1)[0] == definition.metric
        )
    if not candidates:
        filter_context = (
            f" and filter {definition.metric_filter!r}"
            if definition.metric_filter is not None
            else ""
        )
        raise ValueError(
            f"lm-eval result has no metric {definition.metric!r}{filter_context} "
            f"at task {source_identity!r}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"lm-eval metric {definition.metric!r} is ambiguous at "
            f"task {source_identity!r}: {candidates}"
        )
    native_key = candidates[0]
    value_object = selected[native_key]
    if (
        not isinstance(value_object, (int, float))
        or isinstance(value_object, bool)
        or not math.isfinite(float(value_object))
    ):
        raise ValueError(f"lm-eval metric {native_key!r} is not a finite numeric value")
    value = float(value_object)

    directions = raw.get("higher_is_better")
    source_directions = directions.get(source_identity) if isinstance(directions, dict) else None
    direction = (
        source_directions.get(definition.metric) if isinstance(source_directions, dict) else None
    )
    if not isinstance(direction, bool):
        raise ValueError(
            f"lm-eval metric {definition.metric!r} has no unambiguous comparison direction "
            f"at task {source_identity!r}"
        )

    native_filter = native_key.split(",", 1)[1] if "," in native_key else None
    normalized = EvalNormalizedMetric(
        source_identity=source_identity,
        metric=definition.metric,
        filter=native_filter,
        native_metric_key=native_key,
        value=value,
        higher_is_better=direction,
        prompt_authority=resolved_prompt_authority(resolution),
    )
    comparison = EvalMetricComparison.at_least if direction else EvalMetricComparison.at_most
    passed = value >= definition.threshold if direction else value <= definition.threshold
    gate = EvalMetricGate(
        metric=normalized,
        threshold=definition.threshold,
        comparison=comparison,
        conclusion=(EvalMetricGateConclusion.passed if passed else EvalMetricGateConclusion.failed),
    )
    normalized_key = f"{source_identity}:{native_key}"
    return {normalized_key: value}, {normalized_key: normalized}, gate


def repeated_trial_score(
    raw: JsonObject,
    source_identity: str,
    definition: EvalDefinitionInputLmEval,
    trial_id: str,
) -> tuple[float, str]:
    result_section = raw.get("results")
    selected = result_section.get(source_identity) if isinstance(result_section, dict) else None
    if not isinstance(selected, dict):
        raise ValueError(
            f"repeated lm-eval trial {trial_id!r} has no task metric source {source_identity!r}"
        )
    native_key = (
        f"{definition.metric},{definition.metric_filter}"
        if definition.metric_filter is not None
        else None
    )
    if native_key is None:
        candidates = sorted(
            key
            for key in selected
            if isinstance(key, str) and key.split(",", 1)[0] == definition.metric
        )
        if len(candidates) != 1:
            raise ValueError(
                f"repeated lm-eval metric {definition.metric!r} is absent or ambiguous "
                f"for trial {trial_id!r}: {candidates}"
            )
        native_key = candidates[0]
    if native_key not in selected:
        raise ValueError(
            f"repeated lm-eval completed trial {trial_id!r} has no task score {native_key!r}"
        )
    directions = raw.get("higher_is_better")
    source_directions = directions.get(source_identity) if isinstance(directions, dict) else None
    direction = (
        source_directions.get(definition.metric) if isinstance(source_directions, dict) else None
    )
    if direction is not True:
        raise ValueError(
            f"repeated lm-eval metric {definition.metric!r} must be unambiguously higher-is-better"
        )
    value = selected[native_key]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) not in (0.0, 1.0)
    ):
        raise ValueError(f"repeated lm-eval metric {native_key!r} must yield binary zero or one")
    return float(value), native_key


def preserve_repeated_trial_scores(
    trial_results: dict[str, JsonObject],
    resolution: JsonObject,
    definition: EvalDefinitionInputLmEval,
    evidence_path: Path,
    *,
    strict_completed_scores: bool,
) -> None:
    source_identity = resolution.get("task_identity")
    if not isinstance(source_identity, str):
        raise ValueError("resolved lm-eval task has no metric source identity")
    writer = TrialEvidenceWriter(
        evidence_path,
        requested_trials=definition.trials,
        base_seed=repeated_base_seed(definition),
        task_identity=source_identity,
        threshold=definition.threshold,
        initialize=False,
    )
    errors: list[str] = []
    for trial_id, raw in sorted(trial_results.items()):
        try:
            score, native_key = repeated_trial_score(raw, source_identity, definition, trial_id)
            native_trial_dir = evidence_path.parent / "lm-eval-raw" / trial_id
            sample_paths, sample_reference = repeated_native_sample_reference(
                native_trial_dir,
                definition,
                native_key,
                score,
                strict=strict_completed_scores,
            )
            if sample_reference is None:
                continue
            writer.score(
                trial_id,
                score,
                {
                    "trial_id": trial_id,
                    "task": source_identity,
                    "filter": definition.metric_filter,
                    "metric": definition.metric,
                    "native_metric_key": native_key,
                    "result_artifacts": [
                        str(path) for path in lm_eval_result_files(native_trial_dir)
                    ],
                    "sample_artifacts": [str(path) for path in sample_paths],
                    "sample_record": sample_reference,
                },
            )
        except ValueError as error:
            errors.append(str(error))
    evidence = load_json_object(evidence_path)
    raw_outcomes = evidence.get("endpoint_outcomes")
    if not isinstance(raw_outcomes, list) or not all(
        isinstance(outcome, dict) for outcome in raw_outcomes
    ):
        raise ValueError("repeated Eval evidence has no endpoint outcomes")
    if strict_completed_scores:
        for outcome in cast(list[JsonObject], raw_outcomes):
            if isinstance(outcome.get("response"), dict) and outcome.get("binary_score") is None:
                errors.append(
                    f"repeated lm-eval completed trial {outcome.get('trial_id')!r} "
                    "has no task score"
                )
    if errors:
        raise ValueError("; ".join(errors))


def normalize_repeated_lm_eval_result(
    trial_results: dict[str, JsonObject],
    resolution: JsonObject,
    definition: EvalDefinitionInputLmEval,
    evidence_path: Path,
    *,
    strict_completed_scores: bool = True,
) -> tuple[
    dict[str, float],
    dict[str, EvalNormalizedMetric],
    EvalMetricGate,
    EvalTrialSummary,
]:
    source_identity = resolution.get("task_identity")
    if not isinstance(source_identity, str):
        raise ValueError("resolved lm-eval task has no metric source identity")
    preserve_repeated_trial_scores(
        trial_results,
        resolution,
        definition,
        evidence_path,
        strict_completed_scores=strict_completed_scores,
    )
    evidence = load_json_object(evidence_path)
    raw_outcomes = evidence.get("endpoint_outcomes")
    if not isinstance(raw_outcomes, list) or not all(
        isinstance(outcome, dict) for outcome in raw_outcomes
    ):
        raise ValueError("repeated Eval evidence has no endpoint outcomes")
    outcomes = cast(list[JsonObject], raw_outcomes)
    completed = sum(outcome.get("binary_score") in (0.0, 1.0) for outcome in outcomes)
    passed = sum(outcome.get("binary_score") == 1.0 for outcome in outcomes)
    requested = definition.trials
    issued = len(outcomes)
    if issued > requested:
        raise ValueError("repeated Eval issued more endpoint requests than requested trials")
    if issued == 0:
        raise ValueError("repeated Eval completed without issuing a trial request")
    pass_rate = passed / issued if issued else None
    summary = EvalTrialSummary(
        requested_trials=requested,
        issued_trials=issued,
        unissued_trials=requested - issued,
        completed_trials=completed,
        request_failure_trials=issued - completed,
        passed_trials=passed,
        pass_rate=pass_rate,
        per_trial_metric=definition.metric,
        per_trial_filter=definition.metric_filter,
        higher_is_better=True,
    )
    evidence["aggregates"] = {
        "requested_trials": requested,
        "issued_trials": issued,
        "unissued_trials": requested - issued,
        "completed_trials": completed,
        "request_failure_trials": issued - completed,
        "passed_trials": passed,
        "pass_rate": pass_rate,
    }
    comparison = EvalMetricComparison.at_least
    normalized = EvalNormalizedMetric(
        source_identity=source_identity,
        metric=definition.metric,
        filter=definition.metric_filter,
        native_metric_key="inferlab:pass_rate",
        prompt_authority=resolved_prompt_authority(resolution),
        value=pass_rate if pass_rate is not None else 0.0,
        higher_is_better=True,
    )
    gate = EvalMetricGate(
        metric=normalized,
        threshold=definition.threshold,
        comparison=comparison,
        conclusion=(
            EvalMetricGateConclusion.passed
            if pass_rate is not None and pass_rate >= definition.threshold
            else EvalMetricGateConclusion.failed
        ),
    )
    evidence["observed_gate"] = gate.model_dump(mode="json")
    temporary = evidence_path.with_name(f".{evidence_path.name}.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(evidence_path)
    normalized_key = f"{source_identity}:pass_rate"
    return (
        {normalized_key: normalized.value},
        {normalized_key: normalized},
        gate,
        summary,
    )


def partial_repeated_lm_eval_result(
    definition: EvalDefinitionInputLmEval,
    resolution: JsonObject,
    evidence_path: Path,
) -> tuple[
    dict[str, float],
    dict[str, EvalNormalizedMetric],
    EvalMetricGate | None,
    EvalTrialSummary,
]:
    source_identity = resolution.get("task_identity")
    if not isinstance(source_identity, str):
        raise ValueError("resolved lm-eval task has no metric source identity")
    evidence = load_json_object(evidence_path)
    aggregates = evidence.get("aggregates")
    if not isinstance(aggregates, dict):
        raise ValueError("repeated Eval evidence has no aggregates")

    def count(name: str) -> int:
        value = aggregates.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"repeated Eval aggregate {name!r} is invalid")
        return value

    requested = count("requested_trials")
    issued = count("issued_trials")
    unissued = count("unissued_trials")
    completed = count("completed_trials")
    failures = count("request_failure_trials")
    passed = count("passed_trials")
    raw_pass_rate = aggregates.get("pass_rate")
    pass_rate = (
        float(raw_pass_rate)
        if isinstance(raw_pass_rate, (int, float)) and not isinstance(raw_pass_rate, bool)
        else None
    )
    summary = EvalTrialSummary(
        requested_trials=requested,
        issued_trials=issued,
        unissued_trials=unissued,
        completed_trials=completed,
        request_failure_trials=failures,
        passed_trials=passed,
        pass_rate=pass_rate,
        per_trial_metric=definition.metric,
        per_trial_filter=definition.metric_filter,
        higher_is_better=True,
    )
    if pass_rate is None:
        return {}, {}, None, summary
    normalized_key = f"{source_identity}:pass_rate"
    normalized = EvalNormalizedMetric(
        source_identity=source_identity,
        metric=definition.metric,
        filter=definition.metric_filter,
        native_metric_key="inferlab:pass_rate",
        prompt_authority=resolved_prompt_authority(resolution),
        value=pass_rate,
        higher_is_better=True,
    )
    gate = EvalMetricGate(
        metric=normalized,
        threshold=definition.threshold,
        comparison=EvalMetricComparison.at_least,
        conclusion=(
            EvalMetricGateConclusion.passed
            if pass_rate >= definition.threshold
            else EvalMetricGateConclusion.failed
        ),
    )
    return (
        {normalized_key: pass_rate},
        {normalized_key: normalized},
        gate,
        summary,
    )


def repeated_trial_result_objects(
    raw_dir: Path, *, tolerate_incomplete: bool = False
) -> dict[str, JsonObject]:
    results: dict[str, JsonObject] = {}
    for trial_dir in sorted(raw_dir.glob("trial-*")):
        if not trial_dir.is_dir():
            continue
        paths = lm_eval_result_files(trial_dir)
        if len(paths) > 1:
            raise ValueError(
                f"lm-eval trial {trial_dir.name!r} produced multiple results JSON files: "
                f"{len(paths)}"
            )
        if paths:
            try:
                results[trial_dir.name] = load_json_object(paths[0])
            except (OSError, ValueError):
                if not tolerate_incomplete:
                    raise
    return results
