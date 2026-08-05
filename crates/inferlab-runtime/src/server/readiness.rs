use super::{
    ProcessHandle, ProcessObserver, ProcessStatus, ReadinessAttemptEvidence, ReadinessEvidence,
    ReadinessFailure, ReadinessFailureKind, ReadinessObserver, SystemProcessRuntime,
    TargetRegistryMatchEvidence,
};
use crate::interrupt;
use crate::operation_bound::{AttemptBound, OperationBound, OperationTerminalCause, Remaining};
use crate::plan::{ProcessEndpointPlan, ReadinessPlan, TargetRegistryExpectedTarget};
use std::thread;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_PROBE_INTERVAL: Duration = Duration::from_secs(5);
const READINESS_START_BOUNDARY: &str = "after_process_spawn_before_readiness_attempt";

#[derive(Debug, thiserror::Error)]
pub(super) enum ReadinessProbeError {
    #[error("readiness operation deadline expired")]
    Deadline,
    #[error("bounded readiness attempt was unexpectedly unbounded")]
    UnexpectedUnbounded,
    #[error("{label} request failed: {source}")]
    Request {
        label: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("{label} returned HTTP {status}")]
    HttpStatus { label: String, status: u16 },
    #[error("{label} returned invalid JSON: {source}")]
    InvalidJson {
        label: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("target registry observation mismatch: {details}")]
    RegistryMismatch { details: String },
}

pub(super) fn ensure_alive(status: ProcessStatus) -> Result<(), ReadinessFailure> {
    if !status.queried {
        return Err(readiness_failure(
            ReadinessFailureKind::Exited,
            status
                .error
                .unwrap_or_else(|| "failed to query server process group".to_owned()),
        ));
    }
    if !status.alive {
        return Err(readiness_failure(
            ReadinessFailureKind::Exited,
            status
                .error
                .unwrap_or_else(|| "server process group exited before readiness".to_owned()),
        ));
    }
    Ok(())
}

fn readiness_failure(kind: ReadinessFailureKind, message: String) -> ReadinessFailure {
    ReadinessFailure {
        kind,
        message,
        timing: None,
        diagnostic_attempts: Vec::new(),
    }
}

pub(super) fn timed_readiness_failure(
    mut failure: ReadinessFailure,
    bound: &OperationBound,
    terminal_cause: OperationTerminalCause,
    diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
) -> ReadinessFailure {
    failure.timing = Some(bound.timing(READINESS_START_BOUNDARY, terminal_cause));
    failure.diagnostic_attempts = diagnostic_attempts;
    failure
}

pub(super) fn wait_http_ready<R: ProcessObserver>(
    runtime: &R,
    handle: &ProcessHandle,
    endpoint: &ProcessEndpointPlan,
    path: &str,
    attempt_timeout_seconds: u64,
    bound: &OperationBound,
    on_probe_failure: &mut dyn FnMut(&str),
) -> Result<ReadinessEvidence, ReadinessFailure> {
    // A capture-armed server carries no readiness deadline
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]); the loop still terminates on
    // readiness, process-group exit, or interruption.
    let url = format!("http://{}:{}{}", endpoint.host, endpoint.port, path);
    let mut attempts = 0_u32;
    let mut diagnostic_attempts = Vec::new();
    // The probe cadence backs off from POLL_INTERVAL to a cap: sub-second
    // detection for ordinary startups without tens of thousands of no-op
    // probes across a capture-armed unbounded wait. The sleep is clamped to
    // the remaining deadline so a configured timeout fires within one
    // interval.
    let mut probe_interval = POLL_INTERVAL;
    loop {
        ensure_readiness_active(bound, "no readiness probe completed").map_err(|failure| {
            timed_readiness_failure(
                failure,
                bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts.clone(),
            )
        })?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                bound,
                OperationTerminalCause::Interrupted,
                diagnostic_attempts,
            ));
        }
        let status_attempt = readiness_attempt(bound, attempt_timeout_seconds);
        let status_effective_bound_ms = status_attempt.configured_ms().unwrap_or_default();
        let status_bound = status_attempt.into_operation_bound();
        let status = runtime.status_with_bound(handle, &status_bound);
        diagnostic_attempts = vec![process_status_evidence(&status, status_effective_bound_ms)];
        if !status.queried && status_bound.is_expired() && !bound.is_expired() {
            let last_error = status
                .error
                .as_deref()
                .unwrap_or("process status attempt deadline expired");
            on_probe_failure(last_error);
            sleep_within_readiness(bound, probe_interval);
            probe_interval = (probe_interval * 2).min(MAX_PROBE_INTERVAL);
            continue;
        }
        ensure_readiness_active(
            bound,
            "the server process status attempt did not complete in time",
        )
        .map_err(|failure| {
            timed_readiness_failure(
                failure,
                bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts.clone(),
            )
        })?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                bound,
                OperationTerminalCause::Interrupted,
                diagnostic_attempts,
            ));
        }
        ensure_alive(status).map_err(|failure| {
            timed_readiness_failure(
                failure,
                bound,
                OperationTerminalCause::Failed,
                diagnostic_attempts.clone(),
            )
        })?;
        attempts = attempts.saturating_add(1);
        let attempt = probe_http_attempt(
            &endpoint.host,
            endpoint.port,
            path,
            bound,
            attempt_timeout_seconds,
        );
        let effective_bound_ms = attempt.effective_bound_ms;
        let last_error = match attempt.outcome {
            Ok(()) => {
                diagnostic_attempts.push(ReadinessAttemptEvidence {
                    operation: "http_readiness".to_owned(),
                    effective_bound_ms,
                    succeeded: true,
                    error: None,
                });
                let ready_unix_ms = unix_time_millis().map_err(|failure| {
                    timed_readiness_failure(
                        failure,
                        bound,
                        OperationTerminalCause::Failed,
                        diagnostic_attempts.clone(),
                    )
                })?;
                ensure_readiness_active(
                    bound,
                    "the readiness response completed after the deadline",
                )
                .map_err(|failure| {
                    timed_readiness_failure(
                        failure,
                        bound,
                        OperationTerminalCause::TimedOut,
                        diagnostic_attempts.clone(),
                    )
                })?;
                return Ok(ReadinessEvidence::Http {
                    url,
                    attempts,
                    ready_unix_ms,
                    timing: bound
                        .timing(READINESS_START_BOUNDARY, OperationTerminalCause::Succeeded),
                    diagnostic_attempts,
                });
            }
            Err(error) => {
                let error = error.to_string();
                diagnostic_attempts.push(ReadinessAttemptEvidence {
                    operation: "http_readiness".to_owned(),
                    effective_bound_ms,
                    succeeded: false,
                    error: Some(error.clone()),
                });
                error
            }
        };
        on_probe_failure(&last_error);
        if bound.is_expired() {
            let timeout_seconds = readiness_timeout_seconds(bound);
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Timeout,
                    format!(
                        "server did not become ready within {timeout_seconds} seconds; last probe error: {last_error}"
                    ),
                ),
                bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts,
            ));
        }
        sleep_within_readiness(bound, probe_interval);
        probe_interval = (probe_interval * 2).min(MAX_PROBE_INTERVAL);
    }
}

pub(super) struct HttpTargetRegistryProbe<'a> {
    pub(super) readiness_path: &'a str,
    pub(super) registry_path: &'a str,
    pub(super) targets_field: &'a str,
    pub(super) target_url_field: &'a str,
    pub(super) target_role_field: &'a str,
    pub(super) target_healthy_field: &'a str,
    pub(super) target_bootstrap_port_field: &'a str,
    pub(super) expected_targets: &'a [TargetRegistryExpectedTarget],
}

fn sleep_within_readiness(bound: &OperationBound, cadence: Duration) {
    match bound.remaining() {
        Remaining::Finite(remaining) => thread::sleep(cadence.min(remaining)),
        Remaining::Expired => {}
        Remaining::Unbounded => thread::sleep(cadence),
    }
}

fn process_status_evidence(
    status: &ProcessStatus,
    effective_bound_ms: u64,
) -> ReadinessAttemptEvidence {
    let succeeded = status.queried && status.alive;
    ReadinessAttemptEvidence {
        operation: "process_status".to_owned(),
        effective_bound_ms,
        succeeded,
        error: if succeeded {
            None
        } else {
            Some(
                status
                    .error
                    .clone()
                    .unwrap_or_else(|| "server process group is not alive".to_owned()),
            )
        },
    }
}

fn ensure_readiness_active(
    bound: &OperationBound,
    last_error: &str,
) -> Result<(), ReadinessFailure> {
    if !bound.is_expired() {
        return Ok(());
    }
    Err(readiness_failure(
        ReadinessFailureKind::Timeout,
        format!(
            "server did not become ready within {} seconds; last probe error: {last_error}",
            readiness_timeout_seconds(bound)
        ),
    ))
}

fn readiness_timeout_seconds(bound: &OperationBound) -> u64 {
    bound.configured_ms().unwrap_or_default() / 1_000
}

fn attempt_remaining(attempt: &AttemptBound) -> Result<Duration, ReadinessProbeError> {
    match attempt.remaining() {
        Remaining::Finite(remaining) => Ok(remaining),
        Remaining::Expired => Err(ReadinessProbeError::Deadline),
        Remaining::Unbounded => Err(ReadinessProbeError::UnexpectedUnbounded),
    }
}

pub(super) fn wait_http_target_registry_ready(
    status: impl Fn(&OperationBound) -> ProcessStatus,
    endpoint: &ProcessEndpointPlan,
    probe: HttpTargetRegistryProbe<'_>,
    attempt_timeout_seconds: u64,
    bound: &OperationBound,
    on_probe_failure: &mut dyn FnMut(&str),
) -> Result<ReadinessEvidence, ReadinessFailure> {
    let readiness_url = format!(
        "http://{}:{}{}",
        endpoint.host, endpoint.port, probe.readiness_path
    );
    let registry_url = format!(
        "http://{}:{}{}",
        endpoint.host, endpoint.port, probe.registry_path
    );
    let mut attempts = 0_u32;
    let mut diagnostic_attempts = Vec::new();
    let mut probe_interval = POLL_INTERVAL;
    loop {
        ensure_readiness_active(bound, "no readiness probe completed").map_err(|failure| {
            timed_readiness_failure(
                failure,
                bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts.clone(),
            )
        })?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                bound,
                OperationTerminalCause::Interrupted,
                diagnostic_attempts,
            ));
        }
        let status_attempt = readiness_attempt(bound, attempt_timeout_seconds);
        let status_effective_bound_ms = status_attempt.configured_ms().unwrap_or_default();
        let status_bound = status_attempt.into_operation_bound();
        let process_status = status(&status_bound);
        diagnostic_attempts = vec![process_status_evidence(
            &process_status,
            status_effective_bound_ms,
        )];
        if !process_status.queried && status_bound.is_expired() && !bound.is_expired() {
            let last_error = process_status
                .error
                .as_deref()
                .unwrap_or("process status attempt deadline expired");
            on_probe_failure(last_error);
            sleep_within_readiness(bound, probe_interval);
            probe_interval = (probe_interval * 2).min(MAX_PROBE_INTERVAL);
            continue;
        }
        ensure_readiness_active(
            bound,
            "the server process status attempt did not complete in time",
        )
        .map_err(|failure| {
            timed_readiness_failure(
                failure,
                bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts.clone(),
            )
        })?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                bound,
                OperationTerminalCause::Interrupted,
                diagnostic_attempts,
            ));
        }
        ensure_alive(process_status).map_err(|failure| {
            timed_readiness_failure(
                failure,
                bound,
                OperationTerminalCause::Failed,
                diagnostic_attempts.clone(),
            )
        })?;
        attempts = attempts.saturating_add(1);
        let public_attempt = probe_http_attempt(
            &endpoint.host,
            endpoint.port,
            probe.readiness_path,
            bound,
            attempt_timeout_seconds,
        );
        let public_effective_bound_ms = public_attempt.effective_bound_ms;
        let last_error = match public_attempt.outcome {
            Ok(()) => {
                let registry_attempt = probe_target_registry_attempt(
                    &endpoint.host,
                    endpoint.port,
                    &probe,
                    bound,
                    attempt_timeout_seconds,
                );
                let registry_effective_bound_ms = registry_attempt.effective_bound_ms;
                match registry_attempt.outcome {
                    Ok(matched_targets) => {
                        diagnostic_attempts.extend([
                            ReadinessAttemptEvidence {
                                operation: "public_http_readiness".to_owned(),
                                effective_bound_ms: public_effective_bound_ms,
                                succeeded: true,
                                error: None,
                            },
                            ReadinessAttemptEvidence {
                                operation: "target_registry".to_owned(),
                                effective_bound_ms: registry_effective_bound_ms,
                                succeeded: true,
                                error: None,
                            },
                        ]);
                        let ready_unix_ms = unix_time_millis().map_err(|failure| {
                            timed_readiness_failure(
                                failure,
                                bound,
                                OperationTerminalCause::Failed,
                                diagnostic_attempts.clone(),
                            )
                        })?;
                        ensure_readiness_active(
                            bound,
                            "the target registry response completed after the deadline",
                        )
                        .map_err(|failure| {
                            timed_readiness_failure(
                                failure,
                                bound,
                                OperationTerminalCause::TimedOut,
                                diagnostic_attempts.clone(),
                            )
                        })?;
                        return Ok(ReadinessEvidence::HttpTargetRegistry {
                            readiness_url,
                            registry_url,
                            attempts,
                            ready_unix_ms,
                            matched_targets,
                            timing: bound.timing(
                                READINESS_START_BOUNDARY,
                                OperationTerminalCause::Succeeded,
                            ),
                            diagnostic_attempts,
                        });
                    }
                    Err(error) => {
                        let error = error.to_string();
                        diagnostic_attempts.extend([
                            ReadinessAttemptEvidence {
                                operation: "public_http_readiness".to_owned(),
                                effective_bound_ms: public_effective_bound_ms,
                                succeeded: true,
                                error: None,
                            },
                            ReadinessAttemptEvidence {
                                operation: "target_registry".to_owned(),
                                effective_bound_ms: registry_effective_bound_ms,
                                succeeded: false,
                                error: Some(error.clone()),
                            },
                        ]);
                        error
                    }
                }
            }
            Err(error) => {
                let error = error.to_string();
                diagnostic_attempts.push(ReadinessAttemptEvidence {
                    operation: "public_http_readiness".to_owned(),
                    effective_bound_ms: public_effective_bound_ms,
                    succeeded: false,
                    error: Some(error.clone()),
                });
                format!("public readiness probe failed: {error}")
            }
        };
        on_probe_failure(&last_error);
        if bound.is_expired() {
            let timeout_seconds = readiness_timeout_seconds(bound);
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Timeout,
                    format!(
                        "server did not become ready within {timeout_seconds} seconds; last probe error: {last_error}"
                    ),
                ),
                bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts,
            ));
        }
        sleep_within_readiness(bound, probe_interval);
        probe_interval = (probe_interval * 2).min(MAX_PROBE_INTERVAL);
    }
}

fn probe_target_registry_attempt(
    host: &str,
    port: u16,
    probe: &HttpTargetRegistryProbe<'_>,
    bound: &OperationBound,
    attempt_timeout_seconds: u64,
) -> ProbeAttempt<Vec<TargetRegistryMatchEvidence>> {
    let response = probe_http_json_attempt(
        host,
        port,
        probe.registry_path,
        "target registry",
        bound,
        attempt_timeout_seconds,
    );
    let effective_bound_ms = response.effective_bound_ms;
    let outcome = response
        .outcome
        .and_then(|response| match_target_registry(&response, probe, bound));
    ProbeAttempt {
        effective_bound_ms,
        outcome,
    }
}

pub(super) fn match_target_registry(
    response: &serde_json::Value,
    probe: &HttpTargetRegistryProbe<'_>,
    bound: &OperationBound,
) -> Result<Vec<TargetRegistryMatchEvidence>, ReadinessProbeError> {
    readiness_remaining(bound)?;
    let targets = response
        .get(probe.targets_field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| ReadinessProbeError::RegistryMismatch {
            details: format!(
                "target registry response has no array field {:?}",
                probe.targets_field
            ),
        })?;
    let mut evidence = Vec::with_capacity(probe.expected_targets.len());
    for expected in probe.expected_targets {
        readiness_remaining(bound)?;
        let matches: Vec<&serde_json::Map<String, serde_json::Value>> = targets
            .iter()
            .filter_map(serde_json::Value::as_object)
            .filter(|target| {
                target
                    .get(probe.target_url_field)
                    .and_then(serde_json::Value::as_str)
                    == Some(expected.url.as_str())
                    && target
                        .get(probe.target_role_field)
                        .and_then(serde_json::Value::as_str)
                        == Some(expected.role.as_str())
            })
            .collect();
        let target = match matches.as_slice() {
            [] => {
                return Err(ReadinessProbeError::RegistryMismatch {
                    details: format!(
                        "target registry has no {:?} target at {:?}",
                        expected.role, expected.url
                    ),
                });
            }
            [target] => *target,
            _ => {
                return Err(ReadinessProbeError::RegistryMismatch {
                    details: format!(
                        "target registry has multiple {:?} targets at {:?}",
                        expected.role, expected.url
                    ),
                });
            }
        };
        let healthy = target
            .get(probe.target_healthy_field)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| ReadinessProbeError::RegistryMismatch {
                details: format!(
                    "target registry entry for {:?} at {:?} has no boolean {:?} field",
                    expected.role, expected.url, probe.target_healthy_field
                ),
            })?;
        if !healthy {
            return Err(ReadinessProbeError::RegistryMismatch {
                details: format!(
                    "target registry entry for {:?} at {:?} is not healthy",
                    expected.role, expected.url
                ),
            });
        }
        let bootstrap_port = match target.get(probe.target_bootstrap_port_field) {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => {
                let port = value.as_u64().and_then(|port| u16::try_from(port).ok());
                Some(port.ok_or_else(|| ReadinessProbeError::RegistryMismatch {
                    details: format!(
                        "target registry entry for {:?} at {:?} has invalid {:?}",
                        expected.role, expected.url, probe.target_bootstrap_port_field
                    ),
                })?)
            }
        };
        if let Some(expected_port) = expected.bootstrap_port
            && bootstrap_port != Some(expected_port)
        {
            return Err(ReadinessProbeError::RegistryMismatch {
                details: format!(
                    "target registry entry for {:?} at {:?} has bootstrap port {bootstrap_port:?}, expected {expected_port}",
                    expected.role, expected.url
                ),
            });
        }
        evidence.push(TargetRegistryMatchEvidence {
            url: expected.url.clone(),
            role: expected.role.clone(),
            healthy,
            bootstrap_port,
        });
    }
    readiness_remaining(bound)?;
    Ok(evidence)
}

struct ProbeAttempt<T> {
    effective_bound_ms: u64,
    outcome: Result<T, ReadinessProbeError>,
}

fn readiness_attempt(bound: &OperationBound, attempt_timeout_seconds: u64) -> AttemptBound {
    bound.attempt(Some(Duration::from_secs(attempt_timeout_seconds)))
}

#[cfg(test)]
pub(super) fn probe_http(
    host: &str,
    port: u16,
    path: &str,
    bound: &OperationBound,
    attempt_timeout_seconds: u64,
) -> Result<(), ReadinessProbeError> {
    probe_http_attempt(host, port, path, bound, attempt_timeout_seconds).outcome
}

fn probe_http_attempt(
    host: &str,
    port: u16,
    path: &str,
    bound: &OperationBound,
    attempt_timeout_seconds: u64,
) -> ProbeAttempt<()> {
    let attempt = readiness_attempt(bound, attempt_timeout_seconds);
    let effective_bound_ms = attempt.configured_ms().unwrap_or_default();
    let outcome = (|| {
        let url = format!("http://{host}:{port}{path}");
        let timeout = attempt_remaining(&attempt)?;
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .connect_timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|source| readiness_request_error("readiness", source))?;
        let mut response = client
            .get(&url)
            .send()
            .map_err(|source| readiness_request_error("readiness", source))?;
        let status = response.status().as_u16();
        response
            .copy_to(&mut std::io::sink())
            .map_err(|source| readiness_request_error("readiness", source))?;
        if (200..300).contains(&status) {
            Ok(())
        } else {
            Err(ReadinessProbeError::HttpStatus {
                label: "readiness".to_owned(),
                status,
            })
        }
    })();
    ProbeAttempt {
        effective_bound_ms,
        outcome,
    }
}

#[cfg(test)]
pub(super) fn probe_http_json(
    host: &str,
    port: u16,
    path: &str,
    label: &str,
    bound: &OperationBound,
    attempt_timeout_seconds: u64,
) -> Result<serde_json::Value, ReadinessProbeError> {
    probe_http_json_attempt(host, port, path, label, bound, attempt_timeout_seconds).outcome
}

fn probe_http_json_attempt(
    host: &str,
    port: u16,
    path: &str,
    label: &str,
    bound: &OperationBound,
    attempt_timeout_seconds: u64,
) -> ProbeAttempt<serde_json::Value> {
    let attempt = readiness_attempt(bound, attempt_timeout_seconds);
    let effective_bound_ms = attempt.configured_ms().unwrap_or_default();
    let outcome = (|| {
        let url = format!("http://{host}:{port}{path}");
        let timeout = attempt_remaining(&attempt)?;
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .connect_timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|source| readiness_request_error(label, source))?;
        let response = client
            .get(&url)
            .send()
            .map_err(|source| readiness_request_error(label, source))?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(ReadinessProbeError::HttpStatus {
                label: label.to_owned(),
                status,
            });
        }
        let body = response
            .bytes()
            .map_err(|source| readiness_request_error(label, source))?;
        let value =
            serde_json::from_slice(&body).map_err(|source| ReadinessProbeError::InvalidJson {
                label: label.to_owned(),
                source,
            })?;
        readiness_remaining(bound)?;
        Ok(value)
    })();
    ProbeAttempt {
        effective_bound_ms,
        outcome,
    }
}

fn readiness_request_error(label: &str, source: reqwest::Error) -> ReadinessProbeError {
    if source.is_timeout() {
        ReadinessProbeError::Deadline
    } else {
        ReadinessProbeError::Request {
            label: label.to_owned(),
            source,
        }
    }
}

fn readiness_remaining(bound: &OperationBound) -> Result<(), ReadinessProbeError> {
    match bound.remaining() {
        Remaining::Expired => Err(ReadinessProbeError::Deadline),
        Remaining::Finite(_) | Remaining::Unbounded => Ok(()),
    }
}

pub(super) fn unix_time_millis() -> Result<u64, ReadinessFailure> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(crate::operation_bound::duration_millis)
        .map_err(|error| {
            readiness_failure(
                ReadinessFailureKind::Exited,
                format!("system clock is before Unix epoch: {error}"),
            )
        })
}

pub(super) fn wait_process_alive_ready(
    status: impl Fn(&OperationBound) -> ProcessStatus,
    attempt_timeout_seconds: u64,
    bound: &OperationBound,
    on_probe_failure: &mut dyn FnMut(&str),
) -> Result<ReadinessEvidence, ReadinessFailure> {
    loop {
        ensure_readiness_active(bound, "no process status attempt completed").map_err(
            |failure| {
                timed_readiness_failure(
                    failure,
                    bound,
                    OperationTerminalCause::TimedOut,
                    Vec::new(),
                )
            },
        )?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                bound,
                OperationTerminalCause::Interrupted,
                Vec::new(),
            ));
        }
        let attempt = readiness_attempt(bound, attempt_timeout_seconds);
        let effective_bound_ms = attempt.configured_ms().unwrap_or_default();
        let attempt_bound = attempt.into_operation_bound();
        let process_status = status(&attempt_bound);
        let diagnostic_attempts =
            vec![process_status_evidence(&process_status, effective_bound_ms)];
        if !process_status.queried && attempt_bound.is_expired() && !bound.is_expired() {
            on_probe_failure(
                process_status
                    .error
                    .as_deref()
                    .unwrap_or("process status attempt deadline expired"),
            );
            sleep_within_readiness(bound, POLL_INTERVAL);
            continue;
        }
        ensure_readiness_active(
            bound,
            "the server process status attempt did not complete in time",
        )
        .map_err(|failure| {
            timed_readiness_failure(
                failure,
                bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts.clone(),
            )
        })?;
        ensure_alive(process_status).map_err(|failure| {
            timed_readiness_failure(
                failure,
                bound,
                OperationTerminalCause::Failed,
                diagnostic_attempts.clone(),
            )
        })?;
        return Ok(ReadinessEvidence::ProcessAlive {
            ready_unix_ms: unix_time_millis().map_err(|failure| {
                timed_readiness_failure(
                    failure,
                    bound,
                    OperationTerminalCause::Failed,
                    diagnostic_attempts.clone(),
                )
            })?,
            timing: bound.timing(READINESS_START_BOUNDARY, OperationTerminalCause::Succeeded),
            diagnostic_attempts,
        });
    }
}

impl ReadinessObserver for SystemProcessRuntime {
    fn wait_ready(
        &self,
        handle: &ProcessHandle,
        endpoint: &ProcessEndpointPlan,
        readiness: &ReadinessPlan,
        bound: &OperationBound,
        on_probe_failure: &mut dyn FnMut(&str),
    ) -> Result<ReadinessEvidence, ReadinessFailure> {
        match readiness {
            ReadinessPlan::ProcessAlive {
                attempt_timeout_seconds,
                ..
            } => wait_process_alive_ready(
                |bound| self.status_with_bound(handle, bound),
                *attempt_timeout_seconds,
                bound,
                on_probe_failure,
            ),
            ReadinessPlan::Http {
                path,
                attempt_timeout_seconds,
                ..
            } => wait_http_ready(
                self,
                handle,
                endpoint,
                path,
                *attempt_timeout_seconds,
                bound,
                on_probe_failure,
            ),
            ReadinessPlan::HttpTargetRegistry {
                readiness_path,
                registry_path,
                targets_field,
                target_url_field,
                target_role_field,
                target_healthy_field,
                target_bootstrap_port_field,
                expected_targets,
                attempt_timeout_seconds,
                ..
            } => wait_http_target_registry_ready(
                |bound| self.status_with_bound(handle, bound),
                endpoint,
                HttpTargetRegistryProbe {
                    readiness_path,
                    registry_path,
                    targets_field,
                    target_url_field,
                    target_role_field,
                    target_healthy_field,
                    target_bootstrap_port_field,
                    expected_targets,
                },
                *attempt_timeout_seconds,
                bound,
                on_probe_failure,
            ),
        }
    }
}
