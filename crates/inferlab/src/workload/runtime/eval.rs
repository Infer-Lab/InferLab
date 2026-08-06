//! Eval execution and Eval-specific result adjudication.

#[cfg(test)]
mod tests;

use super::client::{
    accept_client_result, client_terminal_cause, freeze_adjudicated_timing,
    reject_late_adjudication, remaining_duration, remaining_seconds, run_client,
    sweep_stale_client_groups, wait_for_interrupt,
};
use super::{
    AcceptedClient, AdjudicatedClient, ClientCasePaths, ClientRun, EvalCaseEvidence,
    EvalCaseRecord, EvalExecutionPlan, EvalPlan, ResolvedWorkloadPlan, WorkloadEndpointProtocol,
    WorkloadKind, WorkloadRecord, WorkloadRecordSession, WorkloadStatus, write_json,
};
use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::workload::wire;
use crate::workspace::EvalDefinition;
use inferlab_protocol::{
    ClientStatus, EvalClientRequest, EvalClientResult, EvalMetricComparison,
    EvalMetricGateConclusion, ProtocolVersion, RawArtifact,
};
use inferlab_runtime::operation_bound::{OperationBound, OperationTerminalCause};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

fn adjudicate_eval_client(
    plan: &EvalPlan,
    mut accepted: AcceptedClient<EvalClientResult>,
    bound: &OperationBound,
) -> AdjudicatedClient<EvalClientResult> {
    reject_late_adjudication(&mut accepted, bound);
    let validation_error = accepted
        .result
        .as_ref()
        .and_then(|result| eval_result_error(plan, result));
    let domain_succeeded = accepted.result.as_ref().is_some_and(|result| {
        validation_error.is_none()
            && result.status == ClientStatus::Succeeded
            && eval_passed(plan, result)
    });
    let domain_error = accepted.result.as_ref().and_then(|result| {
        if let Some(error) = validation_error.clone() {
            Some(error)
        } else if result.status == ClientStatus::Failed {
            result.error.clone()
        } else if !domain_succeeded {
            Some("Eval pass rule was not satisfied".to_owned())
        } else {
            None
        }
    });
    reject_late_adjudication(&mut accepted, bound);
    let succeeded =
        accepted.decode_error.is_none() && accepted.result.is_some() && domain_succeeded;
    let error = accepted.decode_error.take().or(domain_error);
    let terminal_cause = client_terminal_cause(&accepted, succeeded);
    freeze_adjudicated_timing(&mut accepted, bound, terminal_cause);
    accepted.run.finish_cleanup();
    AdjudicatedClient {
        accepted,
        succeeded,
        error,
    }
}

pub(crate) fn run_eval(
    root: &Path,
    record_id: &str,
    plan: &EvalPlan,
    server_record_id: &str,
    progress: &Progress,
) -> Result<WorkloadRecord, InferlabError> {
    // Earlier runs' unclean exits leave recorded client groups behind;
    // terminate identity-matching survivors before this run launches its
    // own clients ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    sweep_stale_client_groups(root);
    let resolved = ResolvedWorkloadPlan::Eval(Box::new(plan.clone()));
    let mut session =
        WorkloadRecordSession::begin(root, record_id, WorkloadKind::Eval, &plan.id, resolved)?;
    progress.phase(Phase::named("record created").record(
        record_id,
        root.join(crate::record::RECORDS_DIR).join(record_id),
    ))?;
    let passed = match execute_eval(root, server_record_id, plan, &mut session, progress) {
        Ok(passed) => passed,
        Err(error) => {
            session.record_mut().error = Some(error.to_string());
            false
        }
    };
    session.record_mut().passed = Some(passed);
    session.finish(if passed {
        WorkloadStatus::Succeeded
    } else {
        WorkloadStatus::Failed
    })?;
    Ok(session.into_record())
}

pub(super) fn execute_eval(
    root: &Path,
    server_record_id: &str,
    plan: &EvalPlan,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<bool, InferlabError> {
    let paths = session.case_paths("eval")?;
    let mut capture = if plan.capture {
        let selection = match crate::server::running_profiler_selection(root, server_record_id) {
            Ok(selection) => selection,
            Err(error) => {
                let message = error.to_string();
                session.record_mut().capture = Some(
                    inferlab_profiler::record::CaptureRecord::failed(message.clone()),
                );
                return Err(InferlabError::ProfilingEvidence { message });
            }
        };
        match inferlab_profiler::session::CaptureSession::open(
            server_record_id,
            &session.record_mut().id,
            &["eval".to_owned()],
            selection,
        ) {
            Ok(capture) => Some(capture),
            Err(record) => {
                let record = *record;
                let message = record
                    .error
                    .clone()
                    .unwrap_or_else(|| "failed to open Eval capture".to_owned());
                session.record_mut().capture = Some(record);
                return Err(InferlabError::ProfilingEvidence { message });
            }
        }
    } else {
        None
    };
    let phase = match &plan.execution {
        EvalExecutionPlan::LmEval { .. } => Phase::named("Eval")
            .current_item(&plan.id)
            .log(session.absolute(&paths.stderr)),
        EvalExecutionPlan::NativeOpenAiSmoke => Phase::named("Eval").current_item(&plan.id),
    };
    progress.phase(phase)?;
    let adjudicated: Result<AdjudicatedClient<EvalClientResult>, InferlabError> =
        if let Some(capture) = capture.as_mut() {
            capture.run_window("eval", || {
                let bound = OperationBound::finite(Duration::from_secs(eval_timeout_seconds(plan)));
                let run = run_eval_operation(root, plan, session, &paths, &bound)?;
                let accepted = accept_client_result::<EvalClientResult>(
                    &session.absolute(&paths.result),
                    "Eval client",
                    run,
                    &bound,
                );
                Ok(adjudicate_eval_client(plan, accepted, &bound))
            })
        } else {
            let bound = OperationBound::finite(Duration::from_secs(eval_timeout_seconds(plan)));
            let run = run_eval_operation(root, plan, session, &paths, &bound)?;
            let accepted = accept_client_result::<EvalClientResult>(
                &session.absolute(&paths.result),
                "Eval client",
                run,
                &bound,
            );
            Ok(adjudicate_eval_client(plan, accepted, &bound))
        };
    let AdjudicatedClient {
        accepted,
        succeeded: case_passed,
        error,
    } = adjudicated?;
    let result = accepted.result;
    let native_started = result
        .as_ref()
        .is_some_and(|result| !result.native_command.is_empty())
        && matches!(&plan.execution, EvalExecutionPlan::LmEval { .. });
    let native_terminal = result.as_ref().filter(|result| {
        native_started && (result.native_exit_code.is_some() || result.native_timed_out)
    });
    let native_timed_out = native_terminal.map(|result| result.native_timed_out);
    let native_interrupted = native_terminal.map(|_| false);
    session.push_eval_case(EvalCaseRecord {
        id: "eval".to_owned(),
        status: if case_passed {
            WorkloadStatus::Succeeded
        } else {
            WorkloadStatus::Failed
        },
        request: paths.request,
        result: paths.result,
        stdout: matches!(&plan.execution, EvalExecutionPlan::LmEval { .. }).then_some(paths.stdout),
        stderr: matches!(&plan.execution, EvalExecutionPlan::LmEval { .. }).then_some(paths.stderr),
        process: accepted.run.process,
        timing: accepted.timing,
        evidence: EvalCaseEvidence {
            metrics: result.as_ref().map(|result| result.metrics.clone()),
            normalized_metrics: result
                .as_ref()
                .map(|result| result.normalized_metrics.clone())
                .unwrap_or_default(),
            eval_gate: result.as_ref().and_then(|result| result.gate.clone()),
            eval_trial_summary: result
                .as_ref()
                .and_then(|result| result.trial_summary.clone()),
            native_timed_out,
            native_interrupted,
            failure_kind: result.as_ref().and_then(|result| result.failure_kind),
        },
        native_command: result.as_ref().map(|result| result.native_command.clone()),
        native_exit_code: result.as_ref().and_then(|result| result.native_exit_code),
        raw_artifacts: result.as_ref().map(|result| result.raw_artifacts.clone()),
        error,
    })?;
    if capture.is_some() {
        progress.phase(Phase::named("profiler finalization").current_item(&plan.id))?;
    }
    let capture_record = capture.map(inferlab_profiler::session::CaptureSession::finish);
    let capture_succeeded = capture_record
        .as_ref()
        .is_none_or(inferlab_profiler::record::CaptureRecord::succeeded);
    if let Some(message) = capture_record
        .as_ref()
        .filter(|record| !record.succeeded())
        .and_then(|record| record.error.clone())
    {
        session.record_mut().error = Some(message);
    }
    session.record_mut().capture = capture_record;
    Ok(case_passed && capture_succeeded)
}

pub(super) fn run_eval_operation(
    workspace_root: &Path,
    plan: &EvalPlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    bound: &OperationBound,
) -> Result<ClientRun, InferlabError> {
    match &plan.execution {
        EvalExecutionPlan::NativeOpenAiSmoke => run_openai_smoke(plan, session, paths, bound),
        EvalExecutionPlan::LmEval {
            command,
            bundled_task,
            ..
        } => {
            let request = EvalClientRequest {
                protocol_version: ProtocolVersion::V7,
                workspace_root: workspace_root.to_path_buf(),
                workspace_source_exclusions: plan.workspace_source_exclusions.clone(),
                endpoint: wire::endpoint_input(&plan.endpoint),
                model: wire::model_input(&plan.model),
                definition: wire::eval_definition_input(&plan.definition, bundled_task.as_deref())?,
                case_budget_seconds: remaining_seconds(bound),
                artifact_dir: paths.artifact_dir.clone(),
            };
            run_client(command, &request, session, paths, bound)
        }
    }
}

#[derive(Serialize)]
struct OpenAiCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f64,
    stream: bool,
    n: u32,
}

#[derive(Serialize)]
struct OpenAiSmokeRequestEvidence<'a> {
    method: &'static str,
    url: &'a str,
    body: &'a OpenAiCompletionRequest<'a>,
}

struct OpenAiSmokeResponse {
    status: u16,
    body: Result<Vec<u8>, String>,
}

enum OpenAiSmokeError {
    Interrupted,
    Message(String),
}

pub(super) fn run_openai_smoke(
    plan: &EvalPlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    bound: &OperationBound,
) -> Result<ClientRun, InferlabError> {
    let EvalDefinition::OpenAiSmoke {
        prompt,
        max_tokens,
        timeout_seconds,
    } = &plan.definition
    else {
        return Err(InferlabError::InvalidConfig {
            message: "native OpenAI smoke execution requires an openai-smoke definition".to_owned(),
        });
    };
    let scheme = match plan.endpoint.protocol {
        WorkloadEndpointProtocol::Http => "http",
    };
    let url = format!(
        "{scheme}://{}:{}{}",
        plan.endpoint.host, plan.endpoint.port, plan.endpoint.completions_path
    );
    let body = OpenAiCompletionRequest {
        model: &plan.model.served_name,
        prompt,
        max_tokens: *max_tokens,
        temperature: 0.0,
        stream: false,
        n: 1,
    };
    write_json(
        &session.absolute(&paths.request),
        &OpenAiSmokeRequestEvidence {
            method: "POST",
            url: &url,
            body: &body,
        },
    )?;

    let started = Instant::now();
    let response = execute_openai_smoke_request(&url, &body, bound, *timeout_seconds);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut metrics = BTreeMap::from([("elapsed_ms".to_owned(), elapsed_ms)]);
    let mut raw_artifacts = Vec::new();
    let mut error = None;

    match response {
        Ok(response) => {
            metrics.insert("http_status".to_owned(), f64::from(response.status));
            let body = match response.body {
                Ok(body) => {
                    metrics.insert("response_bytes".to_owned(), body.len() as f64);
                    let response_path = paths.artifact_dir.join("openai-response.json");
                    fs::create_dir_all(&paths.artifact_dir).map_err(|source| {
                        InferlabError::RecordIo {
                            path: paths.artifact_dir.clone(),
                            source,
                        }
                    })?;
                    fs::write(&response_path, &body).map_err(|source| InferlabError::RecordIo {
                        path: response_path.clone(),
                        source,
                    })?;
                    raw_artifacts.push(RawArtifact {
                        name: "response".to_owned(),
                        kind: "openai-response".to_owned(),
                        path: response_path,
                    });
                    Some(body)
                }
                Err(message) => {
                    error = Some(message);
                    None
                }
            };
            if !(200..300).contains(&response.status) {
                error = Some(format!(
                    "OpenAI smoke completion returned HTTP {}",
                    response.status
                ));
            } else if let Some(body) = body {
                match validate_openai_completion_body(&body) {
                    Ok(choices_count) => {
                        metrics.insert("choices_count".to_owned(), choices_count as f64);
                        metrics.insert("completed".to_owned(), 1.0);
                    }
                    Err(message) => error = Some(message),
                }
            }
        }
        Err(OpenAiSmokeError::Interrupted) => {
            return Ok(ClientRun {
                process: None,
                error: Some("OpenAI smoke interrupted".to_owned()),
                pending_cleanup: None,
                terminal_timing: Some(bound.timing(
                    "before_builtin_request_or_client_release",
                    OperationTerminalCause::Interrupted,
                )),
            });
        }
        Err(OpenAiSmokeError::Message(message)) => error = Some(message),
    }

    let result = EvalClientResult {
        schema_version: 1,
        status: if error.is_none() {
            ClientStatus::Succeeded
        } else {
            ClientStatus::Failed
        },
        metrics,
        normalized_metrics: BTreeMap::new(),
        gate: None,
        trial_summary: None,
        native_command: vec!["POST".to_owned(), url],
        native_exit_code: None,
        native_timed_out: false,
        raw_artifacts,
        failure_kind: None,
        error,
    };
    if bound.is_expired() {
        return Ok(ClientRun {
            process: None,
            error: Some(format!(
                "OpenAI smoke timed out after {timeout_seconds} seconds"
            )),
            pending_cleanup: None,
            terminal_timing: None,
        });
    }
    write_json(&session.absolute(&paths.result), &result)?;
    if bound.is_expired() {
        return Ok(ClientRun {
            process: None,
            error: Some(format!(
                "OpenAI smoke timed out after {timeout_seconds} seconds"
            )),
            pending_cleanup: None,
            terminal_timing: None,
        });
    }
    Ok(ClientRun {
        process: None,
        error: None,
        pending_cleanup: None,
        terminal_timing: None,
    })
}

fn execute_openai_smoke_request(
    url: &str,
    body: &OpenAiCompletionRequest<'_>,
    bound: &OperationBound,
    configured_timeout_seconds: u64,
) -> Result<OpenAiSmokeResponse, OpenAiSmokeError> {
    let timeout = remaining_duration(bound).ok_or_else(|| {
        OpenAiSmokeError::Message(format!(
            "OpenAI smoke timed out after {configured_timeout_seconds} seconds"
        ))
    })?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| {
            OpenAiSmokeError::Message(format!(
                "failed to initialize OpenAI smoke HTTP runtime: {error}"
            ))
        })?;
    runtime.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|error| {
                OpenAiSmokeError::Message(format!(
                    "failed to initialize OpenAI smoke HTTP client: {error}"
                ))
            })?;
        let request = async {
            let response = client.post(url).json(body).send().await.map_err(|error| {
                OpenAiSmokeError::Message(smoke_request_error(error, configured_timeout_seconds))
            })?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .await
                .map(|body| body.to_vec())
                .map_err(|error| smoke_request_error(error, configured_timeout_seconds));
            Ok(OpenAiSmokeResponse { status, body })
        };
        tokio::select! {
            result = request => result,
            () = wait_for_interrupt() => Err(OpenAiSmokeError::Interrupted),
        }
    })
}

pub(super) fn smoke_request_error(error: reqwest::Error, timeout_seconds: u64) -> String {
    if error.is_timeout() {
        format!("OpenAI smoke timed out after {timeout_seconds} seconds")
    } else {
        format!("OpenAI smoke request failed: {error}")
    }
}

pub(super) fn validate_openai_completion_body(body: &[u8]) -> Result<usize, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| format!("OpenAI completion response was not valid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "OpenAI completion response was not a JSON object".to_owned())?;
    let choices = object
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| "OpenAI completion response had no choices array".to_owned())?;
    let first = choices
        .first()
        .and_then(Value::as_object)
        .ok_or_else(|| "OpenAI completion response choices array was empty".to_owned())?;
    first
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| "OpenAI completion response first choice had no string text".to_owned())?;
    Ok(choices.len())
}

pub(super) fn eval_timeout_seconds(plan: &EvalPlan) -> u64 {
    match &plan.definition {
        EvalDefinition::OpenAiSmoke {
            timeout_seconds, ..
        }
        | EvalDefinition::LmEval {
            timeout_seconds, ..
        } => *timeout_seconds,
    }
}

pub(super) fn eval_passed(plan: &EvalPlan, result: &EvalClientResult) -> bool {
    match &plan.definition {
        EvalDefinition::OpenAiSmoke { .. } => true,
        EvalDefinition::LmEval { .. } => result
            .gate
            .as_ref()
            .is_some_and(|gate| gate.conclusion == EvalMetricGateConclusion::Passed),
    }
}

pub(super) fn eval_result_error(plan: &EvalPlan, result: &EvalClientResult) -> Option<String> {
    repeated_eval_result_error(&plan.definition, result)
}

pub(super) fn repeated_eval_result_error(
    definition: &EvalDefinition,
    result: &EvalClientResult,
) -> Option<String> {
    let EvalDefinition::LmEval {
        trials,
        metric,
        metric_filter,
        threshold,
        ..
    } = definition
    else {
        return None;
    };
    let Some(summary) = result.trial_summary.as_ref() else {
        return (*trials > 1 && result.status == ClientStatus::Succeeded)
            .then(|| "repeated Eval result is missing its trial summary".to_owned());
    };
    if summary.requested_trials != *trials {
        return Some(format!(
            "repeated Eval result requested {} trials but its definition requested {trials}",
            summary.requested_trials
        ));
    }
    if summary.issued_trials.checked_add(summary.unissued_trials) != Some(*trials) {
        return Some(
            "repeated Eval result issued and unissued trial counts do not reconstruct requested trials"
                .to_owned(),
        );
    }
    if summary
        .completed_trials
        .checked_add(summary.request_failure_trials)
        != Some(summary.issued_trials)
    {
        return Some(
            "repeated Eval result completed and request-failure counts do not reconstruct issued trials"
                .to_owned(),
        );
    }
    if summary.passed_trials > summary.completed_trials {
        return Some("repeated Eval result passed trials exceed completed trials".to_owned());
    }
    if summary.per_trial_metric != *metric
        || summary.per_trial_filter.as_ref() != metric_filter.as_ref()
        || !summary.higher_is_better
    {
        return Some(
            "repeated Eval result does not preserve the definition's binary higher-is-better metric contract"
                .to_owned(),
        );
    }
    let expected_pass_rate = (summary.issued_trials > 0)
        .then(|| f64::from(summary.passed_trials) / f64::from(summary.issued_trials));
    if summary
        .pass_rate
        .zip(expected_pass_rate)
        .is_some_and(|(actual, expected)| {
            !actual.is_finite() || (actual - expected).abs() > f64::EPSILON
        })
        || summary.pass_rate.is_some() != expected_pass_rate.is_some()
    {
        return Some(
            "repeated Eval result pass rate is not passed trials divided by issued trials"
                .to_owned(),
        );
    }
    match (summary.pass_rate, result.gate.as_ref()) {
        (None, None) => {}
        (None, Some(_)) => {
            return Some("repeated Eval result has a gate without an issued trial".to_owned());
        }
        (Some(_), None) => {
            return Some("repeated Eval result is missing its observed pass-rate gate".to_owned());
        }
        (Some(pass_rate), Some(gate)) => {
            let expected_conclusion = if pass_rate >= *threshold {
                EvalMetricGateConclusion::Passed
            } else {
                EvalMetricGateConclusion::Failed
            };
            if (gate.metric.value - pass_rate).abs() > f64::EPSILON
                || !gate.metric.value.is_finite()
                || gate.metric.metric != *metric
                || gate.metric.filter.as_ref() != metric_filter.as_ref()
                || gate.metric.native_metric_key != "inferlab:pass_rate"
                || !gate.metric.higher_is_better
                || !gate.threshold.is_finite()
                || (gate.threshold - *threshold).abs() > f64::EPSILON
                || gate.comparison != EvalMetricComparison::AtLeast
                || gate.conclusion != expected_conclusion
            {
                return Some(
                    "repeated Eval result gate does not preserve its pass-rate threshold semantics"
                        .to_owned(),
                );
            }
        }
    }
    None
}
