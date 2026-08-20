"""Coordinate one Bench case deadline across native execution and adjudication."""

import json

from inferlab_measurement_sdk import (
    BenchAgenticResultEvidence,
    BenchAgenticSourceVerification,
    BenchArtifactLevelInput,
    BenchCacheStartInput,
    BenchClientRequest,
    BenchClientResult,
    BenchNativeInvocation,
    BenchRequestSloResult,
    BenchRequestSourceInputRandom,
    BenchRequestSourceInputRandomMixture,
    CaseDeadline,
    ClientStatus,
    JsonObject,
    load_json_object,
)

from .agentic_source import acquire_and_verify_agentic_source
from .aiperf import (
    PROFILE_EXPORT_NAME,
    SERVER_METRICS_EXPORT_NAME,
    prepare_aiperf_execution,
    raw_artifacts,
    run_aiperf,
    run_speed_bench_reports,
    speed_bench_category,
)
from .population import load_chat_tokenizer
from .result_agentic import agentic_result_evidence
from .result_metrics import NORMALIZATION_SCHEMA, normalize_summary, prompt_cache_evidence
from .result_policy import request_slo_evidence, warmup_counts, warmup_error
from .result_population import population_identity_error, prompt_token_reconciliation
from .result_records import request_counts
from .result_sessions import session_result_evidence


def requires_prompt_cache_evidence(request: BenchClientRequest) -> bool:
    if request.definition.cache_start is BenchCacheStartInput.primed:
        return True
    source_input = request.definition.request_source
    if source_input is None:
        return False
    source = source_input.root
    if isinstance(source, BenchRequestSourceInputRandom):
        return source.prefix_sharing is not None or source.shared_system_content is not None
    if isinstance(source, BenchRequestSourceInputRandomMixture):
        return source.prefix_sharing is not None
    return False


def execute(request: BenchClientRequest, deadline: CaseDeadline | None = None) -> BenchClientResult:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    prepared = prepare_aiperf_execution(request, deadline)
    artifact_dir = prepared.artifact_dir
    config_path = prepared.config_path
    request_config_path = prepared.request_config_path
    command = prepared.command
    source_verification: BenchAgenticSourceVerification | None = None
    agentic_evidence: BenchAgenticResultEvidence | None = None
    agentic_source = request.definition.agentic_source
    if agentic_source is not None:
        acquisition = acquire_and_verify_agentic_source(agentic_source)
        source_verification = acquisition.verification
        agentic_evidence = BenchAgenticResultEvidence(
            source=source_verification,
            run=None,
        )
        if acquisition.error is not None:
            return BenchClientResult(
                schema_version=1,
                status=ClientStatus.failed,
                completed_requests=0,
                failed_requests=0,
                normalization_schema=NORMALIZATION_SCHEMA,
                metrics={},
                agentic_evidence=agentic_evidence,
                native_command=command,
                native_exit_code=None,
                raw_artifacts=raw_artifacts(
                    artifact_dir,
                    config_path,
                    request_config_path,
                    prepared.profile_artifacts,
                ),
                error=f"agentic source verification failed: {acquisition.error}",
            )
    try:
        native_exit_code, interrupted, timed_out = run_aiperf(
            command, deadline, prepared.environment
        )
    except OSError as launch_error:
        return BenchClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            completed_requests=0,
            failed_requests=0,
            normalization_schema=NORMALIZATION_SCHEMA,
            metrics={},
            agentic_evidence=agentic_evidence,
            native_command=command,
            native_exit_code=None,
            raw_artifacts=raw_artifacts(
                artifact_dir, config_path, request_config_path, prepared.profile_artifacts
            ),
            error=f"failed to launch AIPerf: {launch_error}",
        )

    summary_path = prepared.profile_artifacts.summary
    records_path = prepared.profile_artifacts.records
    raw_records_path = prepared.profile_artifacts.raw_records
    if not raw_records_path.is_file():
        raw_records_path = artifact_dir / "raw_records"
    summary: JsonObject | None = None
    summary_error: str | None = None
    if summary_path.is_file():
        try:
            summary = load_json_object(summary_path)
        except (OSError, ValueError, json.JSONDecodeError) as load_error:
            summary_error = str(load_error)
    request_slo = request.definition.request_slo
    request_slo_result: BenchRequestSloResult | None = None
    every_failed_request_has_inference_error = False
    if request_slo is None:
        completed_requests, failed_requests, count_error = request_counts(records_path)
    else:
        (
            completed_requests,
            failed_requests,
            request_slo_result,
            every_failed_request_has_inference_error,
            count_error,
        ) = request_slo_evidence(records_path, request.case.request_count, request_slo, summary)
    phase_error = warmup_error(warmup_counts(records_path, request.case.warmup_request_count))
    identity_error = population_identity_error(request, records_path)
    prompt_reconciliation, prompt_reconciliation_error = prompt_token_reconciliation(
        request, records_path
    )
    prompt_cache_observations, prompt_cache_metrics, prompt_cache_error = prompt_cache_evidence(
        records_path,
        requires_prompt_cache_evidence(request),
        request.endpoint.prompt_cache_read_zero_representation,
    )
    session_evidence = None
    session_error: str | None = None
    if request.definition.session_source is not None:
        try:
            tokenizer = load_chat_tokenizer(request.model.locator)
            session_evidence, session_error = session_result_evidence(
                request, records_path, raw_records_path, tokenizer
            )
        except (ImportError, OSError, TypeError, ValueError) as evidence_error:
            session_error = f"linear-session evidence failed: {evidence_error}"
    agentic_error: str | None = None
    if agentic_source is not None and source_verification is not None and summary is not None:
        try:
            run_evidence = agentic_result_evidence(
                agentic_source,
                summary,
                summary_path,
                records_path,
                (
                    raw_records_path
                    if request.definition.artifact_level == BenchArtifactLevelInput.diagnostic
                    else None
                ),
            )
            agentic_evidence = BenchAgenticResultEvidence(
                source=source_verification,
                run=run_evidence,
            )
            if not run_evidence.submission_valid:
                reasons = ", ".join(run_evidence.submission_invalid_reasons) or "unspecified"
                agentic_error = f"AIPerf AgentX scenario submission is invalid: {reasons}"
        except ValueError as evidence_error:
            agentic_error = f"agentic evidence failed: {evidence_error}"
    artifacts = raw_artifacts(
        artifact_dir, config_path, request_config_path, prepared.profile_artifacts
    )
    complete_all_failed = (
        request.definition.session_source is None
        and request_slo_result is not None
        and completed_requests == 0
        and failed_requests == request.case.request_count
        and count_error is None
        and phase_error is None
        and summary_error is None
    )
    complete_all_inference_error = complete_all_failed and every_failed_request_has_inference_error
    if (
        interrupted
        or timed_out
        or (native_exit_code != 0 and not complete_all_inference_error)
        or (summary is None and not complete_all_failed)
        or count_error is not None
        or phase_error is not None
        or identity_error is not None
        or prompt_reconciliation_error is not None
        or prompt_cache_error is not None
        or session_error is not None
        or agentic_error is not None
        or summary_error is not None
    ):
        if interrupted:
            reason = "AIPerf was interrupted"
        elif timed_out:
            reason = "AIPerf reached the measurement-case deadline"
        elif native_exit_code != 0 and not complete_all_inference_error:
            reason = f"AIPerf exited with {native_exit_code}"
        elif summary_error is not None:
            reason = f"AIPerf summary is invalid: {summary_error}"
        elif count_error is not None:
            reason = count_error
        elif phase_error is not None:
            reason = phase_error
        elif identity_error is not None:
            reason = identity_error
        elif prompt_reconciliation_error is not None:
            reason = prompt_reconciliation_error
        elif prompt_cache_error is not None:
            reason = prompt_cache_error
        elif session_error is not None:
            reason = session_error
        elif agentic_error is not None:
            reason = agentic_error
        else:
            reason = "AIPerf produced no summary JSON"
        if count_error is not None and count_error != reason:
            reason = f"{reason}; {count_error}"
        if phase_error is not None and phase_error != reason:
            reason = f"{reason}; {phase_error}"
        if identity_error is not None and identity_error != reason:
            reason = f"{reason}; {identity_error}"
        if prompt_reconciliation_error is not None and prompt_reconciliation_error != reason:
            reason = f"{reason}; {prompt_reconciliation_error}"
        if prompt_cache_error is not None and prompt_cache_error != reason:
            reason = f"{reason}; {prompt_cache_error}"
        if session_error is not None and session_error != reason:
            reason = f"{reason}; {session_error}"
        if agentic_error is not None and agentic_error != reason:
            reason = f"{reason}; {agentic_error}"
        return BenchClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            normalization_schema=NORMALIZATION_SCHEMA,
            metrics={},
            request_slo=request_slo_result,
            session_evidence=session_evidence,
            agentic_evidence=agentic_evidence,
            prompt_token_reconciliation=prompt_reconciliation,
            prompt_cache_observations=prompt_cache_observations,
            native_command=command,
            native_exit_code=native_exit_code,
            raw_artifacts=artifacts,
            error=reason,
        )

    errors: list[str] = []
    metrics: dict[str, float] = {}
    if summary is not None and not complete_all_failed:
        try:
            metrics = normalize_summary(summary, prepared.population.tpot_applicable)
            metrics.update(prompt_cache_metrics)
        except ValueError as normalization_error:
            errors.append(str(normalization_error))
    if request_slo_result is not None:
        metrics["good_request_ratio"] = request_slo_result.good_request_ratio
        metrics["goodput"] = request_slo_result.goodput
    if not errors:
        if request_slo is None and completed_requests == 0:
            errors.append("AIPerf completed no requests")
        elif request_slo is None and agentic_source is None and failed_requests != 0:
            errors.append(f"AIPerf reported {failed_requests} failed requests")
    report_invocations: list[BenchNativeInvocation] = []
    if not errors and request.definition.server_metrics:
        profile_export = artifact_dir / PROFILE_EXPORT_NAME
        server_metrics_export = artifact_dir / SERVER_METRICS_EXPORT_NAME
        if not profile_export.is_file():
            errors.append(f"AIPerf server metrics omitted {PROFILE_EXPORT_NAME}")
        if not server_metrics_export.is_file():
            errors.append(f"AIPerf server metrics omitted {SERVER_METRICS_EXPORT_NAME}")
        if not errors and speed_bench_category(request) is not None:
            report_metrics, report_invocations, report_error = run_speed_bench_reports(
                request,
                prepared.command_prefix,
                artifact_dir,
                deadline,
            )
            metrics.update(report_metrics)
            if report_error is not None:
                errors.append(report_error)
    artifacts = raw_artifacts(
        artifact_dir, config_path, request_config_path, prepared.profile_artifacts
    )
    result_error = "; ".join(errors) or None
    return BenchClientResult(
        schema_version=1,
        status=ClientStatus.failed if result_error else ClientStatus.succeeded,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        normalization_schema=NORMALIZATION_SCHEMA,
        metrics=metrics,
        request_slo=request_slo_result,
        session_evidence=session_evidence,
        agentic_evidence=agentic_evidence,
        prompt_token_reconciliation=prompt_reconciliation,
        prompt_cache_observations=prompt_cache_observations,
        native_command=command,
        native_exit_code=native_exit_code,
        report_invocations=report_invocations,
        raw_artifacts=artifacts,
        error=result_error,
    )
