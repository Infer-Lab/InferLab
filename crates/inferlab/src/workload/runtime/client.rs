//! Shared measurement-client deadline, process-group, cleanup, and result
//! acceptance boundary. It is independent of Eval and Bench domain policy.

use super::{
    AcceptedClient, CLIENT_CLEANUP_STATUS_DEADLINE, CLIENT_HANDLE_FILE, CLIENT_KILL_GRACE,
    CLIENT_POLL_INTERVAL, CLIENT_TERM_GRACE, ClientCasePaths, ClientCommandPlan, ClientGroupHandle,
    ClientProcessEvidence, ClientResultEnvelope, ClientRun, ClientTerminationEvidence,
    ClientTerminationTrigger, PendingClientCleanup, SWEEP_WALK_DEPTH, WorkloadRecordSession,
    write_json,
};
use crate::InferlabError;
use inferlab_runtime::interrupt;
use inferlab_runtime::operation_bound::{OperationBound, OperationTerminalCause, Remaining};
use inferlab_runtime::process_group::{LocalProcessGroup, TerminationSignal, process_start_time};
use serde::{Serialize, de::DeserializeOwned};
use std::fs::{self, File};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub(super) async fn wait_for_interrupt() {
    loop {
        if interrupt::received() {
            return;
        }
        tokio::time::sleep(CLIENT_POLL_INTERVAL).await;
    }
}

pub(super) fn remaining_duration(bound: &OperationBound) -> Option<Duration> {
    match bound.remaining() {
        Remaining::Finite(duration) => Some(duration),
        Remaining::Expired | Remaining::Unbounded => None,
    }
}

pub(super) fn remaining_seconds(bound: &OperationBound) -> f64 {
    remaining_duration(bound)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

pub(super) fn reject_late_adjudication<T>(
    accepted: &mut AcceptedClient<T>,
    bound: &OperationBound,
) {
    if accepted.terminal_timing_frozen {
        return;
    }
    let rejection = if interrupt::received() {
        if let Some(process) = accepted.run.process.as_mut() {
            process.interrupted = true;
        }
        Some("client result was not adjudicated before interruption".to_owned())
    } else if bound.is_expired() {
        if let Some(process) = accepted.run.process.as_mut() {
            process.timed_out = true;
        }
        Some("client result was not adjudicated before the measurement-case deadline".to_owned())
    } else {
        None
    };
    if let Some(rejection) = rejection {
        accepted.result = None;
        accepted.decode_error = Some(
            accepted
                .decode_error
                .take()
                .map(|error| format!("{rejection}; {error}"))
                .unwrap_or(rejection),
        );
    }
}

pub(super) fn freeze_adjudicated_timing<T>(
    accepted: &mut AcceptedClient<T>,
    bound: &OperationBound,
    terminal_cause: OperationTerminalCause,
) {
    if accepted.terminal_timing_frozen {
        accepted.timing.terminal_cause = terminal_cause;
    } else {
        accepted.timing = bound.timing("before_builtin_request_or_client_release", terminal_cause);
    }
}

pub(super) fn client_terminal_cause<T>(
    accepted: &AcceptedClient<T>,
    succeeded: bool,
) -> OperationTerminalCause {
    if accepted.terminal_timing_frozen {
        accepted.timing.terminal_cause
    } else if succeeded {
        OperationTerminalCause::Succeeded
    } else if accepted
        .run
        .process
        .as_ref()
        .is_some_and(|process| process.interrupted)
    {
        OperationTerminalCause::Interrupted
    } else if accepted
        .run
        .process
        .as_ref()
        .is_some_and(|process| process.timed_out)
    {
        OperationTerminalCause::TimedOut
    } else {
        OperationTerminalCause::Failed
    }
}

impl ClientRun {
    pub(super) fn finish_cleanup(&mut self) {
        let Some(mut pending) = self.pending_cleanup.take() else {
            return;
        };
        let termination = cleanup_remaining_client_group(&mut pending.child, pending.group);
        let verified = termination
            .as_ref()
            .is_none_or(|evidence| evidence.verified);
        if let Some(process) = self.process.as_mut() {
            process.termination = termination;
        }
        if verified {
            let _ = fs::remove_file(pending.handle_path);
        }
    }
}

pub(super) fn run_client(
    command: &ClientCommandPlan,
    request: &impl Serialize,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    bound: &OperationBound,
) -> Result<ClientRun, InferlabError> {
    let request_path = session.absolute(&paths.request);
    let result_path = session.absolute(&paths.result);
    let stdout_path = session.absolute(&paths.stdout);
    let stderr_path = session.absolute(&paths.stderr);
    write_json(&request_path, request)?;
    if bound.is_expired() {
        return Ok(ClientRun {
            process: None,
            error: Some("client exceeded its measurement-case budget before release".to_owned()),
            pending_cleanup: None,
            terminal_timing: None,
        });
    }
    let (program, args) =
        command
            .argv
            .split_first()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: "resolved client command is empty".to_owned(),
            })?;
    let stdout = File::create(&stdout_path).map_err(|source| InferlabError::RecordIo {
        path: stdout_path,
        source,
    })?;
    let stderr = File::create(&stderr_path).map_err(|source| InferlabError::RecordIo {
        path: stderr_path,
        source,
    })?;
    let mut child = match Command::new(program)
        .args(args)
        .args(["--input", &request_path.to_string_lossy()])
        .args(["--output", &result_path.to_string_lossy()])
        .current_dir(&command.cwd)
        .env_clear()
        .envs(&command.env)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .process_group(0)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return Ok(ClientRun {
                process: None,
                error: Some(format!("failed to launch client {program:?}: {error}")),
                pending_cleanup: None,
                terminal_timing: None,
            });
        }
    };
    // The durable process-group handle precedes the client's first
    // experiment effect so an unclean exit stays recoverable
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let handle_path = request_path.with_file_name(CLIENT_HANDLE_FILE);
    let group = match LocalProcessGroup::capture_child(&child) {
        Ok(group) => group,
        Err(message) => {
            let terminal_timing = bound.timing(
                "before_builtin_request_or_client_release",
                OperationTerminalCause::Failed,
            );
            let fallback_group = LocalProcessGroup::unverified(child.id());
            let termination = terminate_client_group(
                &mut child,
                fallback_group,
                ClientTerminationTrigger::LaunchFailure,
            );
            return Ok(ClientRun {
                process: Some(ClientProcessEvidence {
                    exit_code: None,
                    timed_out: false,
                    interrupted: false,
                    termination: Some(termination),
                }),
                error: Some(format!(
                    "failed to capture the client process-group identity: {message}"
                )),
                pending_cleanup: None,
                terminal_timing: Some(terminal_timing),
            });
        }
    };
    if let Err(message) = record_client_group_handle(group, &handle_path) {
        let terminal_timing = bound.timing(
            "before_builtin_request_or_client_release",
            OperationTerminalCause::Failed,
        );
        let termination =
            terminate_client_group(&mut child, group, ClientTerminationTrigger::LaunchFailure);
        return Ok(ClientRun {
            process: Some(ClientProcessEvidence {
                exit_code: None,
                timed_out: false,
                interrupted: false,
                termination: Some(termination),
            }),
            error: Some(message),
            pending_cleanup: None,
            terminal_timing: Some(terminal_timing),
        });
    }
    let wait_for_client = || -> Result<ClientRun, InferlabError> {
        loop {
            if interrupt::received() {
                let terminal_timing = bound.timing(
                    "before_builtin_request_or_client_release",
                    OperationTerminalCause::Interrupted,
                );
                let termination = terminate_client_group(
                    &mut child,
                    group,
                    ClientTerminationTrigger::Interruption,
                );
                return Ok(ClientRun {
                    process: Some(ClientProcessEvidence {
                        exit_code: None,
                        timed_out: false,
                        interrupted: true,
                        termination: Some(termination),
                    }),
                    error: Some("client interrupted".to_owned()),
                    pending_cleanup: None,
                    terminal_timing: Some(terminal_timing),
                });
            }
            if bound.is_expired() {
                let terminal_timing = bound.timing(
                    "before_builtin_request_or_client_release",
                    OperationTerminalCause::TimedOut,
                );
                let termination =
                    terminate_client_group(&mut child, group, ClientTerminationTrigger::Timeout);
                return Ok(ClientRun {
                    process: Some(ClientProcessEvidence {
                        exit_code: None,
                        timed_out: true,
                        interrupted: false,
                        termination: Some(termination),
                    }),
                    error: Some("client exceeded its measurement-case budget".to_owned()),
                    pending_cleanup: None,
                    terminal_timing: Some(terminal_timing),
                });
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Ok(ClientRun {
                        process: Some(ClientProcessEvidence {
                            exit_code: status.code(),
                            timed_out: false,
                            interrupted: false,
                            termination: None,
                        }),
                        error: (!status.success())
                            .then(|| format!("client exited with status {status}")),
                        pending_cleanup: Some(PendingClientCleanup {
                            child,
                            group,
                            handle_path: handle_path.clone(),
                        }),
                        terminal_timing: None,
                    });
                }
                Ok(None) => match bound.remaining() {
                    Remaining::Finite(remaining) => {
                        thread::sleep(CLIENT_POLL_INTERVAL.min(remaining));
                    }
                    Remaining::Expired => {}
                    Remaining::Unbounded => thread::sleep(CLIENT_POLL_INTERVAL),
                },
                Err(error) => {
                    let terminal_timing = bound.timing(
                        "before_builtin_request_or_client_release",
                        OperationTerminalCause::Failed,
                    );
                    let termination = terminate_client_group(
                        &mut child,
                        group,
                        ClientTerminationTrigger::WaitFailure,
                    );
                    return Ok(ClientRun {
                        process: Some(ClientProcessEvidence {
                            exit_code: None,
                            timed_out: false,
                            interrupted: false,
                            termination: Some(termination),
                        }),
                        error: Some(format!("failed to wait for client: {error}")),
                        pending_cleanup: None,
                        terminal_timing: Some(terminal_timing),
                    });
                }
            }
        }
    };
    let run = wait_for_client()?;
    let termination_verified = run
        .process
        .as_ref()
        .is_some_and(|process| process.termination.as_ref().is_none_or(|t| t.verified));
    if run.pending_cleanup.is_none() && termination_verified {
        let _ = fs::remove_file(&handle_path);
    }
    Ok(run)
}

pub(super) fn cleanup_remaining_client_group(
    child: &mut Child,
    group: LocalProcessGroup,
) -> Option<ClientTerminationEvidence> {
    let started = Instant::now();
    let bound = OperationBound::finite(CLIENT_CLEANUP_STATUS_DEADLINE);
    match group.has_live_members(&bound) {
        Ok(true) => {
            let status_elapsed_ms =
                inferlab_runtime::operation_bound::duration_millis(started.elapsed());
            let mut evidence =
                terminate_client_group(child, group, ClientTerminationTrigger::ResultAccepted);
            evidence.elapsed_ms = evidence.elapsed_ms.saturating_add(status_elapsed_ms);
            evidence.status_deadline_ms =
                inferlab_runtime::operation_bound::duration_millis(CLIENT_CLEANUP_STATUS_DEADLINE);
            Some(evidence)
        }
        Ok(false) => None,
        Err(error) => Some(ClientTerminationEvidence {
            trigger: ClientTerminationTrigger::ResultAccepted,
            elapsed_ms: inferlab_runtime::operation_bound::duration_millis(started.elapsed()),
            status_deadline_ms: inferlab_runtime::operation_bound::duration_millis(
                CLIENT_CLEANUP_STATUS_DEADLINE,
            ),
            term_grace_ms: inferlab_runtime::operation_bound::duration_millis(CLIENT_TERM_GRACE),
            kill_grace_ms: inferlab_runtime::operation_bound::duration_millis(CLIENT_KILL_GRACE),
            term_sent: false,
            kill_sent: false,
            verified: false,
            error: Some(error.to_string()),
        }),
    }
}

pub(super) fn terminate_client_group(
    child: &mut Child,
    group: LocalProcessGroup,
    trigger: ClientTerminationTrigger,
) -> ClientTerminationEvidence {
    let started = Instant::now();
    let mut errors = Vec::new();
    let term_bound = OperationBound::finite(CLIENT_TERM_GRACE);
    let term = group.send_signal(TerminationSignal::Term, &term_bound);
    let term_sent = term.succeeded();
    if let Some(error) = term.error {
        errors.push(error);
    }
    let mut verified =
        match group.wait_until_stopped(Some(child), &term_bound, CLIENT_POLL_INTERVAL) {
            Ok(verified) => verified,
            Err(error) => {
                errors.push(error.to_string());
                false
            }
        };
    let mut kill_sent = false;
    if !verified {
        let kill_bound = OperationBound::finite(CLIENT_KILL_GRACE);
        let kill = group.send_signal(TerminationSignal::Kill, &kill_bound);
        kill_sent = kill.succeeded();
        if let Some(error) = kill.error {
            errors.push(error);
        }
        verified = match group.wait_until_stopped(Some(child), &kill_bound, CLIENT_POLL_INTERVAL) {
            Ok(verified) => verified,
            Err(error) => {
                errors.push(error.to_string());
                false
            }
        };
    }
    if !verified {
        errors.push(format!(
            "process group {} is still alive after SIGKILL",
            group.process_group
        ));
    }
    ClientTerminationEvidence {
        trigger,
        elapsed_ms: inferlab_runtime::operation_bound::duration_millis(started.elapsed()),
        status_deadline_ms: 0,
        term_grace_ms: inferlab_runtime::operation_bound::duration_millis(CLIENT_TERM_GRACE),
        kill_grace_ms: inferlab_runtime::operation_bound::duration_millis(CLIENT_KILL_GRACE),
        term_sent,
        kill_sent,
        verified,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

pub(super) fn accept_client_result<T: DeserializeOwned>(
    path: &Path,
    client: &'static str,
    mut run: ClientRun,
    bound: &OperationBound,
) -> AcceptedClient<T> {
    let frozen_terminal_cause = run
        .terminal_timing
        .as_ref()
        .map(|timing| timing.terminal_cause);
    let (mut result, mut decode_error) =
        decode_client_result(path, client, run.process.as_ref(), run.error.as_deref());
    let terminal_rejection = if run.error.is_some() {
        None
    } else if interrupt::received() {
        if let Some(process) = run.process.as_mut() {
            process.interrupted = true;
        }
        Some("client result was not accepted before interruption".to_owned())
    } else if bound.is_expired() {
        if let Some(process) = run.process.as_mut() {
            process.timed_out = true;
        }
        Some("client result was not accepted before the measurement-case deadline".to_owned())
    } else {
        None
    };
    if let Some(message) = terminal_rejection {
        result = None;
        decode_error = Some(
            decode_error
                .map(|error| format!("{message}; {error}"))
                .unwrap_or(message),
        );
    }
    let terminal_cause = frozen_terminal_cause.unwrap_or_else(|| {
        if run
            .process
            .as_ref()
            .is_some_and(|process| process.interrupted)
        {
            OperationTerminalCause::Interrupted
        } else if run
            .process
            .as_ref()
            .is_some_and(|process| process.timed_out)
            || bound.is_expired()
        {
            OperationTerminalCause::TimedOut
        } else if decode_error.is_some() || run.error.is_some() {
            OperationTerminalCause::Failed
        } else {
            OperationTerminalCause::Succeeded
        }
    });
    let terminal_timing_frozen = run.terminal_timing.is_some();
    let mut timing = run.terminal_timing.take().unwrap_or_else(|| {
        bound.timing("before_builtin_request_or_client_release", terminal_cause)
    });
    timing.start_boundary = "before_builtin_request_or_client_release".to_owned();
    timing.terminal_cause = terminal_cause;
    AcceptedClient {
        run,
        result,
        decode_error,
        timing,
        terminal_timing_frozen,
    }
}

pub(super) fn decode_client_result<T: DeserializeOwned>(
    path: &Path,
    client: &'static str,
    process: Option<&ClientProcessEvidence>,
    run_error: Option<&str>,
) -> (Option<T>, Option<String>) {
    let process_error = run_error.map(str::to_owned).or_else(|| {
        process
            .is_some_and(|process| process.exit_code != Some(0))
            .then(|| "client did not exit successfully".to_owned())
    });
    match fs::read(path) {
        Ok(bytes) => {
            // The version gates before the strict DTO parse; a header that
            // does not even yield a version falls through so the strict parse
            // names the precise JSON defect.
            if let Ok(envelope) = serde_json::from_slice::<ClientResultEnvelope>(&bytes)
                && envelope.schema_version != 1
            {
                let message = format!(
                    "{client} returned unsupported result schema version {}",
                    envelope.schema_version
                );
                return (
                    None,
                    Some(
                        process_error
                            .map(|process_error| format!("{process_error}; {message}"))
                            .unwrap_or(message),
                    ),
                );
            }
            match serde_json::from_slice(&bytes) {
                Ok(result) => (Some(result), process_error),
                Err(error) => (
                    None,
                    Some(
                        process_error
                            .map(|process_error| {
                                format!("{process_error}; invalid client result JSON: {error}")
                            })
                            .unwrap_or_else(|| format!("invalid client result JSON: {error}")),
                    ),
                ),
            }
        }
        Err(error) => (
            None,
            Some(
                process_error
                    .map(|process_error| {
                        format!("{process_error}; failed to read client result: {error}")
                    })
                    .unwrap_or_else(|| format!("failed to read client result: {error}")),
            ),
        ),
    }
}

pub(super) fn record_client_group_handle(
    group: LocalProcessGroup,
    path: &Path,
) -> Result<(), String> {
    let owner_pid = std::process::id();
    let owner_start_time_ticks = process_start_time(owner_pid)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "the owning process's identity could not be recorded".to_owned())?;
    let handle = ClientGroupHandle {
        group,
        owner_pid,
        owner_start_time_ticks,
    };
    write_json(path, &handle)
        .map_err(|error| format!("failed to record the client process-group handle: {error}"))
}

pub(super) fn process_identity_matches(pid: u32, ticks: u64) -> bool {
    process_start_time(pid)
        .ok()
        .flatten()
        .is_some_and(|current| current == ticks)
}

/// Terminate identity-matching client process groups recorded by earlier
/// runs that exited uncleanly, then clear their handles. A handle whose
/// leader start-time no longer matches is cleared without signalling
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(crate) fn sweep_stale_client_groups(root: &Path) {
    let mut handles = Vec::new();
    collect_client_handles(&root.join(crate::record::RECORDS_DIR), 0, &mut handles);
    for path in handles {
        let handle = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ClientGroupHandle>(&bytes).ok());
        if let Some(handle) = handle {
            // A live owner means a live concurrent run, not an unclean
            // exit: its clients are not this run's to touch.
            if process_identity_matches(handle.owner_pid, handle.owner_start_time_ticks) {
                continue;
            }
            if handle.group.identity_matches() {
                let term_bound = OperationBound::finite(CLIENT_TERM_GRACE);
                let _ = handle
                    .group
                    .send_signal(TerminationSignal::Term, &term_bound);
                let mut gone = handle
                    .group
                    .wait_until_stopped(None, &term_bound, CLIENT_POLL_INTERVAL)
                    .unwrap_or(false);
                if !gone && handle.group.identity_matches() {
                    let kill_bound = OperationBound::finite(CLIENT_KILL_GRACE);
                    let _ = handle
                        .group
                        .send_signal(TerminationSignal::Kill, &kill_bound);
                    gone = handle
                        .group
                        .wait_until_stopped(None, &kill_bound, CLIENT_POLL_INTERVAL)
                        .unwrap_or(false);
                }
                if !gone {
                    // Keep the handle: the next run must still be able to
                    // discharge the termination it could not verify.
                    continue;
                }
            }
        }
        let _ = fs::remove_file(&path);
    }
}

pub(super) fn collect_client_handles(dir: &Path, depth: usize, into: &mut Vec<PathBuf>) {
    if depth > SWEEP_WALK_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_client_handles(&path, depth + 1, into);
        } else if entry.file_name() == CLIENT_HANDLE_FILE {
            into.push(path);
        }
    }
}
