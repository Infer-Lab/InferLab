use crate::plan::{
    CaptureWindowActionPlan, CaptureWindowHttpMethodPlan, ProfilerControl, ProfilerLaunch,
    env_prefix,
};
use crate::poll::{Poll, poll_until};
use crate::record::{
    ARM_START_BOUNDARY, CONTROL_START_BOUNDARY, CaptureActionRecord, CaptureHttpFailureKind,
    MEASUREMENT_FINALIZATION_START, ProfilerTargetRecord,
};
use inferlab_runtime::operation_bound::{OperationBound, OperationTerminalCause, Remaining};
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Output;
use std::time::Duration;

const INITIAL_REPORT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_REPORT_POLL_INTERVAL: Duration = Duration::from_secs(2);

pub(crate) fn prepare_output(
    target: &ProfilerTargetRecord,
    parent: &Path,
    bound: &OperationBound,
) -> CaptureActionRecord {
    command_action(
        target,
        "prepare-output",
        vec![
            "mkdir".to_owned(),
            "-p".to_owned(),
            parent.display().to_string(),
        ],
        bound,
        ARM_START_BOUNDARY,
        CommandActionMode::Operation,
    )
}

pub(crate) fn arm_range_collection(
    target: &ProfilerTargetRecord,
    output: &Path,
    range_count: usize,
    bound: &OperationBound,
) -> CaptureActionRecord {
    command_action(
        target,
        "start-range-collection",
        nsys_start_argv(target, output, range_count),
        bound,
        ARM_START_BOUNDARY,
        CommandActionMode::Operation,
    )
}

pub(crate) fn verify_report(
    target: &ProfilerTargetRecord,
    path: &Path,
    bound: &OperationBound,
    start_boundary: &str,
) -> CaptureActionRecord {
    poll_until(
        bound,
        INITIAL_REPORT_POLL_INTERVAL,
        MAX_REPORT_POLL_INTERVAL,
        || {
            let action = check_report(target, path, bound, start_boundary);
            if action.succeeded() {
                Poll::Done(action)
            } else {
                Poll::Pending(action)
            }
        },
    )
}

pub(crate) fn check_report(
    target: &ProfilerTargetRecord,
    path: &Path,
    bound: &OperationBound,
    start_boundary: &str,
) -> CaptureActionRecord {
    command_action(
        target,
        "verify-report",
        vec![
            "test".to_owned(),
            "-f".to_owned(),
            path.display().to_string(),
        ],
        bound,
        start_boundary,
        CommandActionMode::Cleanup,
    )
}

pub(crate) fn inspect_collection_state(
    target: &ProfilerTargetRecord,
    bound: &OperationBound,
    start_boundary: &str,
) -> CaptureActionRecord {
    let mut argv = env_prefix(&target.escapes.env);
    argv.extend([
        target.executable.clone(),
        "sessions".to_owned(),
        "list".to_owned(),
        "--output-format=json".to_owned(),
    ]);
    command_action(
        target,
        "inspect-collection-state",
        argv,
        bound,
        start_boundary,
        CommandActionMode::Cleanup,
    )
}

pub(crate) fn stop_collection(
    target: &ProfilerTargetRecord,
    bound: &OperationBound,
    start_boundary: &str,
) -> CaptureActionRecord {
    let mut argv = env_prefix(&target.escapes.env);
    argv.extend([
        target.executable.clone(),
        "stop".to_owned(),
        format!("--session={}", target.session),
    ]);
    command_action(
        target,
        "stop-collection",
        argv,
        bound,
        start_boundary,
        CommandActionMode::Cleanup,
    )
}

pub(crate) fn start_windows(
    targets: &[&ProfilerTargetRecord],
    deadline_seconds: u64,
) -> Vec<CaptureActionRecord> {
    let process_ids = targets
        .iter()
        .map(|target| match &target.control {
            ProfilerControl::Http { process_id, .. } => process_id.as_str(),
        })
        .collect::<BTreeSet<_>>();
    window_actions_for(targets, true, &process_ids, deadline_seconds)
}

pub(crate) fn stop_windows(
    targets: &[&ProfilerTargetRecord],
    process_ids: &BTreeSet<&str>,
    deadline_seconds: u64,
) -> Vec<CaptureActionRecord> {
    window_actions_for(targets, false, process_ids, deadline_seconds)
}

/// Engine-trace window closing is a dispatch into the one global finalization
/// budget ([[RFC-0004:C-WORKLOAD-PROFILING]]): the close request is
/// dispatched when the measured phase ends and its response consumption draws
/// the shared budget rather than a per-action control budget. A delivery
/// failure (transport error or a prompt error status) is window-closing
/// control failure evidence; a slow or absent response is neutral
/// flush-pending evidence, and coverage verification is the sole completion
/// verdict.
pub(crate) fn close_engine_trace_windows(
    targets: &[&ProfilerTargetRecord],
    process_ids: &BTreeSet<&str>,
    bound: &OperationBound,
) -> Vec<CaptureActionRecord> {
    let mut seen = BTreeSet::new();
    targets
        .iter()
        .filter_map(|target| match &target.control {
            ProfilerControl::Http {
                process_id, stop, ..
            } if process_ids.contains(process_id.as_str()) && seen.insert(process_id.clone()) => {
                Some(engine_trace_close_action(process_id, stop, bound))
            }
            _ => None,
        })
        .collect()
}

fn engine_trace_close_action(
    process_id: &str,
    action: &CaptureWindowActionPlan,
    bound: &OperationBound,
) -> CaptureActionRecord {
    let record = |status: Option<u16>,
                  failure_kind: Option<CaptureHttpFailureKind>,
                  flush_pending: bool,
                  error: Option<String>,
                  succeeded: bool,
                  terminal_cause: OperationTerminalCause| {
        CaptureActionRecord::Http {
            process_id: process_id.to_owned(),
            operation: "stop-range".to_owned(),
            method: Some(action.method),
            path: Some(action.path.clone()),
            url: action.effective_url.clone(),
            body: action.body.clone(),
            status,
            failure_kind,
            flush_pending,
            error,
            succeeded,
            timing: bound.timing(MEASUREMENT_FINALIZATION_START, terminal_cause),
        }
    };
    // With no budget left the dispatch cannot complete; record the pending
    // flush instead of pretending a request went out.
    let timeout = match bound.remaining() {
        Remaining::Finite(remaining) => Some(remaining),
        Remaining::Expired => {
            return record(
                None,
                None,
                true,
                None,
                true,
                OperationTerminalCause::TimedOut,
            );
        }
        Remaining::Unbounded => None,
    };
    let result = send_control_request(action, timeout, None);
    match result {
        Ok(status) if status.is_success() => record(
            Some(status.as_u16()),
            None,
            false,
            None,
            true,
            OperationTerminalCause::Succeeded,
        ),
        // A prompt error status is a delivery failure: window-closing control
        // failure evidence, adjudicated by coverage verification.
        Ok(status) => record(
            Some(status.as_u16()),
            None,
            false,
            None,
            false,
            OperationTerminalCause::Failed,
        ),
        // A slow or absent response is neutral flush-pending evidence: no
        // error, no deadline failure kind, no capture failure by itself.
        Err(error) if error.is_response_wait_expiry() => record(
            None,
            None,
            true,
            None,
            true,
            OperationTerminalCause::TimedOut,
        ),
        Err(error) => record(
            None,
            Some(error.failure_kind()),
            false,
            Some(error.record_message()),
            false,
            OperationTerminalCause::Failed,
        ),
    }
}

fn window_actions_for(
    targets: &[&ProfilerTargetRecord],
    start: bool,
    process_ids: &BTreeSet<&str>,
    deadline_seconds: u64,
) -> Vec<CaptureActionRecord> {
    let mut seen = BTreeSet::new();
    targets
        .iter()
        .filter_map(|target| match &target.control {
            ProfilerControl::Http {
                process_id,
                start: start_action,
                stop: stop_action,
                ..
            } if process_ids.contains(process_id.as_str()) && seen.insert(process_id.clone()) => {
                let action = if start { start_action } else { stop_action };
                Some(http_action(
                    process_id,
                    if start { "start-range" } else { "stop-range" },
                    action,
                    deadline_seconds,
                ))
            }
            _ => None,
        })
        .collect()
}

fn http_action(
    process_id: &str,
    operation: &str,
    action: &CaptureWindowActionPlan,
    deadline_seconds: u64,
) -> CaptureActionRecord {
    let url = action.effective_url.clone();
    let deadline = Duration::from_secs(deadline_seconds);
    let bound = OperationBound::finite(deadline);
    let result = (|| {
        let client_timeout = finite_remaining(&bound)?;
        let request_timeout = finite_remaining(&bound)?;
        send_control_request(action, Some(client_timeout), Some(request_timeout))
    })();
    match result {
        Ok(response) => CaptureActionRecord::Http {
            process_id: process_id.to_owned(),
            operation: operation.to_owned(),
            method: Some(action.method),
            path: Some(action.path.clone()),
            url,
            body: action.body.clone(),
            status: Some(response.as_u16()),
            failure_kind: None,
            flush_pending: false,
            error: None,
            succeeded: response.is_success(),
            timing: bound.timing(
                CONTROL_START_BOUNDARY,
                if response.is_success() {
                    OperationTerminalCause::Succeeded
                } else {
                    OperationTerminalCause::Failed
                },
            ),
        },
        Err(error) => CaptureActionRecord::Http {
            process_id: process_id.to_owned(),
            operation: operation.to_owned(),
            method: Some(action.method),
            path: Some(action.path.clone()),
            url,
            body: action.body.clone(),
            status: None,
            failure_kind: Some(error.failure_kind()),
            flush_pending: false,
            error: Some(error.record_message()),
            succeeded: false,
            timing: bound.timing(
                CONTROL_START_BOUNDARY,
                if bound.is_expired() {
                    OperationTerminalCause::TimedOut
                } else {
                    OperationTerminalCause::Failed
                },
            ),
        },
    }
}

/// The one window-control request construction shared by the per-action
/// control budget and the engine-trace close dispatch: build a client under
/// the caller's timeout classification, POST with an optional JSON body, then
/// consume the complete response body so the deadline covers the full
/// response wait. A `None` timeout leaves the client unbounded.
fn send_control_request(
    action: &CaptureWindowActionPlan,
    client_timeout: Option<Duration>,
    request_timeout: Option<Duration>,
) -> Result<reqwest::StatusCode, CaptureHttpError> {
    let mut client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy();
    if let Some(timeout) = client_timeout {
        client = client.timeout(timeout).connect_timeout(timeout);
    }
    let client = client
        .build()
        .map_err(|source| CaptureHttpError::Request { source })?;
    let request = match (action.method, action.body.as_ref()) {
        (CaptureWindowHttpMethodPlan::Post, Some(body)) => {
            let payload = serde_json::to_vec(body)
                .map_err(|source| CaptureHttpError::Serialization { source })?;
            client
                .post(&action.effective_url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(payload)
        }
        (CaptureWindowHttpMethodPlan::Post, None) => client.post(&action.effective_url),
    };
    let request = match request_timeout {
        Some(timeout) => request.timeout(timeout),
        None => request,
    };
    let response = request
        .send()
        .map_err(|source| CaptureHttpError::Request { source })?;
    let status = response.status();
    response
        .bytes()
        .map_err(|source| CaptureHttpError::Request { source })?;
    Ok(status)
}

#[derive(Debug, thiserror::Error)]
enum CaptureHttpError {
    #[error("profiler control deadline expired")]
    Deadline,
    #[error("failed to serialize profiler control request: {source}")]
    Serialization {
        #[source]
        source: serde_json::Error,
    },
    #[error("profiler control request failed: {source}")]
    Request {
        #[source]
        source: reqwest::Error,
    },
}

impl CaptureHttpError {
    /// The engine-trace close classification: the response wait expired
    /// against the owning budget, which is neutral flush-pending evidence
    /// rather than a delivery failure ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    fn is_response_wait_expiry(&self) -> bool {
        match self {
            Self::Deadline => true,
            Self::Request { source } => source.is_timeout(),
            Self::Serialization { .. } => false,
        }
    }

    fn failure_kind(&self) -> CaptureHttpFailureKind {
        match self {
            Self::Deadline => CaptureHttpFailureKind::Deadline,
            Self::Serialization { .. } => CaptureHttpFailureKind::Serialization,
            Self::Request { source } if source.is_timeout() => CaptureHttpFailureKind::Deadline,
            Self::Request { source } if source.is_connect() || source.is_builder() => {
                CaptureHttpFailureKind::Transport
            }
            Self::Request { .. } => CaptureHttpFailureKind::InvalidResponse,
        }
    }

    fn record_message(&self) -> String {
        if self.failure_kind() == CaptureHttpFailureKind::Deadline {
            "profiler control deadline expired".to_owned()
        } else {
            self.to_string()
        }
    }
}

fn finite_remaining(bound: &OperationBound) -> Result<Duration, CaptureHttpError> {
    match bound.remaining() {
        Remaining::Finite(remaining) => Ok(remaining),
        Remaining::Expired | Remaining::Unbounded => Err(CaptureHttpError::Deadline),
    }
}

fn nsys_start_argv(
    target: &ProfilerTargetRecord,
    output: &Path,
    range_count: usize,
) -> Vec<String> {
    let escapes = &target.escapes;
    let mut argv = env_prefix(&escapes.env);
    argv.push(target.executable.clone());
    argv.push("start".to_owned());
    argv.extend(escapes.start_options.iter().cloned());
    argv.extend([
        format!("--session={}", target.session),
        format!("--sample={}", escapes.sampling.as_deref().unwrap_or("none")),
        format!(
            "--cpuctxsw={}",
            escapes.context_switch.as_deref().unwrap_or("none")
        ),
        "--force-overwrite=true".to_owned(),
        "--export=none".to_owned(),
        format!("--output={}", output.display()),
        "--capture-range=cudaProfilerApi".to_owned(),
        format!("--capture-range-end=repeat:{range_count}:async"),
    ]);
    argv
}

fn command_action(
    target: &ProfilerTargetRecord,
    operation: &str,
    argv: Vec<String>,
    bound: &OperationBound,
    start_boundary: &str,
    mode: CommandActionMode,
) -> CaptureActionRecord {
    let output = target_output(target, &argv, bound, mode);
    match output {
        Ok(output) => CaptureActionRecord::Command {
            target_id: target.process_id.clone(),
            operation: operation.to_owned(),
            argv,
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            succeeded: output.status.success(),
            timing: bound.timing(
                start_boundary,
                if output.status.success() {
                    OperationTerminalCause::Succeeded
                } else {
                    OperationTerminalCause::Failed
                },
            ),
            cleanup: None,
        },
        Err(error) => {
            let timing = bound.timing(start_boundary, error.terminal_cause);
            CaptureActionRecord::Command {
                target_id: target.process_id.clone(),
                operation: operation.to_owned(),
                argv,
                exit_code: None,
                stdout: String::new(),
                stderr: error.to_string(),
                succeeded: false,
                timing,
                cleanup: error.cleanup,
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CommandActionMode {
    Operation,
    Cleanup,
}

#[derive(Debug, thiserror::Error)]
#[error("{failure}")]
pub(crate) struct TargetCommandError {
    #[source]
    failure: TargetCommandFailure,
    terminal_cause: OperationTerminalCause,
    cleanup: Option<inferlab_runtime::container::CommandCleanupEvidence>,
}

#[derive(Debug, thiserror::Error)]
enum TargetCommandFailure {
    #[error("profiler command deadline expired")]
    Deadline,
    #[error("profiler command was interrupted")]
    Interrupted,
    #[error("failed to launch profiler command: {source}")]
    Launch {
        #[source]
        source: std::io::Error,
    },
    #[error("profiler command failed: {source}")]
    Io {
        #[source]
        source: std::io::Error,
    },
    #[error("profiler command wait failed: {source}")]
    WaitCleanup {
        #[source]
        source: std::io::Error,
    },
}

pub(crate) fn target_output(
    target: &ProfilerTargetRecord,
    argv: &[String],
    bound: &OperationBound,
    mode: CommandActionMode,
) -> Result<Output, TargetCommandError> {
    let local_argv;
    let (command, cwd) = match &target.launch {
        ProfilerLaunch::Local => (argv, Some(target.command_cwd.as_path())),
        ProfilerLaunch::Ssh { target: ssh_target } => {
            let script = ssh_control_script(&target.command_cwd, argv);
            local_argv = inferlab_runtime::ssh::ssh_argv(ssh_target, &script);
            (local_argv.as_slice(), None)
        }
    };
    let outcome = match mode {
        CommandActionMode::Operation => {
            inferlab_runtime::container::run_with_bound(command, cwd, None, bound, None)
        }
        CommandActionMode::Cleanup => {
            inferlab_runtime::container::run_cleanup_with_bound(command, cwd, None, bound, None)
        }
    };
    match outcome {
        Ok(inferlab_runtime::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Ok(inferlab_runtime::container::BoundedWait::Expired { cleanup, .. }) => {
            Err(TargetCommandError {
                failure: TargetCommandFailure::Deadline,
                terminal_cause: OperationTerminalCause::TimedOut,
                cleanup,
            })
        }
        Ok(inferlab_runtime::container::BoundedWait::Interrupted { cleanup, .. }) => {
            Err(TargetCommandError {
                failure: TargetCommandFailure::Interrupted,
                terminal_cause: OperationTerminalCause::Interrupted,
                cleanup: Some(cleanup),
            })
        }
        Err(inferlab_runtime::container::BoundedError::Launch(source)) => Err(TargetCommandError {
            failure: TargetCommandFailure::Launch { source },
            terminal_cause: OperationTerminalCause::Failed,
            cleanup: None,
        }),
        Err(
            inferlab_runtime::container::BoundedError::Stdin(source)
            | inferlab_runtime::container::BoundedError::Wait(source),
        ) => Err(TargetCommandError {
            failure: TargetCommandFailure::Io { source },
            terminal_cause: OperationTerminalCause::Failed,
            cleanup: None,
        }),
        Err(inferlab_runtime::container::BoundedError::WaitCleanup {
            source, cleanup, ..
        }) => Err(TargetCommandError {
            failure: TargetCommandFailure::WaitCleanup { source },
            terminal_cause: OperationTerminalCause::Failed,
            cleanup: Some(cleanup),
        }),
    }
}

fn ssh_control_script(cwd: &Path, argv: &[String]) -> String {
    format!(
        "cd {} && exec {}",
        inferlab_runtime::shell::shell_quote(&cwd.to_string_lossy()),
        argv.iter()
            .map(|argument| inferlab_runtime::shell::shell_quote(argument))
            .collect::<Vec<_>>()
            .join(" ")
    )
}

#[cfg(test)]
mod tests {
    use super::{
        CommandActionMode, command_action, engine_trace_close_action, http_action,
        ssh_control_script,
    };
    use crate::plan::{
        CaptureWindowActionPlan, CaptureWindowControlEndpointPlan, CaptureWindowHttpMethodPlan,
        NsysEscapes, ProfilerControl, ProfilerFinalization, ProfilerLaunch, WindowControlKind,
    };
    use crate::record::{
        ARM_START_BOUNDARY, CaptureActionRecord, CaptureHttpFailureKind,
        MEASUREMENT_FINALIZATION_START, ProfilerTargetRecord,
    };
    use inferlab_protocol::EndpointAssignment;
    use inferlab_runtime::operation_bound::{
        OperationBound, OperationBudgetEvidence, OperationTerminalCause,
    };
    use std::error::Error;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn target(command_cwd: PathBuf) -> ProfilerTargetRecord {
        let action = CaptureWindowActionPlan {
            method: CaptureWindowHttpMethodPlan::Post,
            path: "/profile".to_owned(),
            body: None,
            effective_url: "http://127.0.0.1:1/profile".to_owned(),
        };
        ProfilerTargetRecord {
            process_id: "prefill-0".to_owned(),
            role_id: "prefill".to_owned(),
            replica_id: "prefill".to_owned(),
            replica_index: 0,
            rank: 0,
            rank_count: 1,
            device_count: 1,
            mechanism: inferlab_protocol::CaptureMechanism::ManagedCollection,
            trace_storage: None,
            session: "inferlab-serve-prefill-0".to_owned(),
            executable: "nsys".to_owned(),
            launch: ProfilerLaunch::Local,
            finalization: ProfilerFinalization::NsysStop,
            control: ProfilerControl::Http {
                window_control_endpoint: CaptureWindowControlEndpointPlan::ReplicaEntry,
                process_id: "prefill-0".to_owned(),
                endpoint: EndpointAssignment {
                    host: "127.0.0.1".to_owned(),
                    port: 1,
                },
                start: action.clone(),
                stop: action,
            },
            supported_window_controls: vec![WindowControlKind::FrameworkRange],
            command_cwd,
            runtime_root: PathBuf::from("profiles"),
            launch_prefix: Vec::new(),
            escapes: NsysEscapes::default(),
        }
    }

    #[test]
    fn finalization_command_records_its_own_deadline_after_business_work()
    -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let bound = OperationBound::finite(Duration::from_millis(50));
        let action = command_action(
            &target(temp.path().to_path_buf()),
            "fixture-finalization",
            vec!["sh".to_owned(), "-c".to_owned(), "sleep 5".to_owned()],
            &bound,
            MEASUREMENT_FINALIZATION_START,
            CommandActionMode::Cleanup,
        );
        let CaptureActionRecord::Command {
            timing, cleanup, ..
        } = action
        else {
            return Err("finalization fixture returned non-command evidence".into());
        };
        assert_eq!(
            timing.budget,
            OperationBudgetEvidence::Finite { configured_ms: 50 }
        );
        assert_eq!(timing.terminal_cause, OperationTerminalCause::TimedOut);
        assert!(timing.elapsed_ms >= 50 && timing.elapsed_ms < 500);
        assert!(cleanup.is_some_and(|cleanup| {
            cleanup.verified
                && cleanup.trigger == inferlab_runtime::container::CommandCleanupTrigger::Deadline
                && cleanup.kill_attempted
        }));
        Ok(())
    }

    #[test]
    fn sequential_arm_commands_share_one_owner_budget() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let target = target(temp.path().to_path_buf());
        let bound = OperationBound::finite(Duration::from_millis(150));
        let first = command_action(
            &target,
            "fixture-arm-one",
            vec!["sh".to_owned(), "-c".to_owned(), "sleep 0.1".to_owned()],
            &bound,
            ARM_START_BOUNDARY,
            CommandActionMode::Operation,
        );
        let second = command_action(
            &target,
            "fixture-arm-two",
            vec!["sh".to_owned(), "-c".to_owned(), "sleep 0.1".to_owned()],
            &bound,
            ARM_START_BOUNDARY,
            CommandActionMode::Operation,
        );

        assert!(first.succeeded());
        assert!(!second.succeeded());
        Ok(())
    }

    #[test]
    fn sequential_finalization_commands_share_one_owner_budget() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let target = target(temp.path().to_path_buf());
        let bound = OperationBound::finite(Duration::from_millis(150));
        let first = command_action(
            &target,
            "fixture-finalize-one",
            vec!["sh".to_owned(), "-c".to_owned(), "sleep 0.1".to_owned()],
            &bound,
            MEASUREMENT_FINALIZATION_START,
            CommandActionMode::Cleanup,
        );
        let second = command_action(
            &target,
            "fixture-finalize-two",
            vec!["sh".to_owned(), "-c".to_owned(), "sleep 0.1".to_owned()],
            &bound,
            MEASUREMENT_FINALIZATION_START,
            CommandActionMode::Cleanup,
        );

        assert!(first.succeeded());
        assert!(!second.succeeded());
        Ok(())
    }

    #[test]
    fn control_request_deadline_covers_the_complete_response_body() -> Result<(), Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 1_024];
            let _ = stream.read(&mut request)?;
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n")?;
            std::thread::sleep(Duration::from_millis(2_100));
            stream.write_all(b"x")
        });
        let action = CaptureWindowActionPlan {
            method: CaptureWindowHttpMethodPlan::Post,
            path: "/profile".to_owned(),
            body: None,
            effective_url: format!("http://{address}/profile"),
        };

        let record = http_action("serve", "start-range", &action, 3);

        assert!(record.succeeded());
        let CaptureActionRecord::Http { timing, .. } = record else {
            return Err("control fixture returned non-HTTP evidence".into());
        };
        assert!(timing.elapsed_ms >= 2_000);
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    fn engine_trace_stop(url: String) -> CaptureWindowActionPlan {
        CaptureWindowActionPlan {
            method: CaptureWindowHttpMethodPlan::Post,
            path: "/stop_profile".to_owned(),
            body: None,
            effective_url: url,
        }
    }

    /// A close response that outlasts the shared finalization budget is
    /// neutral flush-pending evidence: succeeded, no error, no failure kind,
    /// no status ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn engine_trace_close_records_flush_pending_when_the_response_outlasts_the_budget()
    -> Result<(), Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 1_024];
            let _ = stream.read(&mut request)?;
            std::thread::sleep(Duration::from_millis(600));
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")?;
            Ok(())
        });
        let bound = OperationBound::finite(Duration::from_millis(150));

        let record = engine_trace_close_action(
            "serve",
            &engine_trace_stop(format!("http://{address}/stop_profile")),
            &bound,
        );

        let CaptureActionRecord::Http {
            succeeded,
            flush_pending,
            failure_kind,
            status,
            error,
            timing,
            ..
        } = record
        else {
            return Err("engine-trace close fixture returned non-HTTP evidence".into());
        };
        assert!(succeeded);
        assert!(flush_pending);
        assert_eq!(failure_kind, None);
        assert_eq!(status, None);
        assert_eq!(error, None);
        assert_eq!(timing.terminal_cause, OperationTerminalCause::TimedOut);
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    /// A refused close connection is a delivery failure: window-closing
    /// control failure evidence adjudicated by coverage
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn engine_trace_close_records_a_delivery_failure_on_connection_refusal()
    -> Result<(), Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        drop(listener);
        let bound = OperationBound::finite(Duration::from_secs(1));

        let record = engine_trace_close_action(
            "serve",
            &engine_trace_stop(format!("http://{address}/stop_profile")),
            &bound,
        );

        let CaptureActionRecord::Http {
            succeeded,
            flush_pending,
            failure_kind,
            error,
            timing,
            ..
        } = record
        else {
            return Err("engine-trace close fixture returned non-HTTP evidence".into());
        };
        assert!(!succeeded);
        assert!(!flush_pending);
        assert_eq!(failure_kind, Some(CaptureHttpFailureKind::Transport));
        assert!(error.is_some());
        assert_eq!(timing.terminal_cause, OperationTerminalCause::Failed);
        Ok(())
    }

    /// A prompt error status is a delivery failure, not flush-pending
    /// evidence ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn engine_trace_close_records_a_prompt_error_status_as_a_delivery_failure()
    -> Result<(), Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 1_024];
            let _ = stream.read(&mut request)?;
            stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")?;
            Ok(())
        });
        let bound = OperationBound::finite(Duration::from_secs(1));

        let record = engine_trace_close_action(
            "serve",
            &engine_trace_stop(format!("http://{address}/stop_profile")),
            &bound,
        );

        let CaptureActionRecord::Http {
            succeeded,
            flush_pending,
            failure_kind,
            status,
            error,
            timing,
            ..
        } = record
        else {
            return Err("engine-trace close fixture returned non-HTTP evidence".into());
        };
        assert!(!succeeded);
        assert!(!flush_pending);
        assert_eq!(failure_kind, None);
        assert_eq!(status, Some(500));
        assert_eq!(error, None);
        assert_eq!(timing.terminal_cause, OperationTerminalCause::Failed);
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    /// A slow close response consumed inside the shared budget is ordinary
    /// success evidence drawn from the finalization budget, not the
    /// per-action control budget ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn engine_trace_close_consumes_a_slow_response_inside_the_shared_budget()
    -> Result<(), Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 1_024];
            let _ = stream.read(&mut request)?;
            std::thread::sleep(Duration::from_millis(100));
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")?;
            Ok(())
        });
        let bound = OperationBound::finite(Duration::from_secs(5));

        let record = engine_trace_close_action(
            "serve",
            &engine_trace_stop(format!("http://{address}/stop_profile")),
            &bound,
        );

        let CaptureActionRecord::Http {
            succeeded,
            flush_pending,
            status,
            timing,
            ..
        } = record
        else {
            return Err("engine-trace close fixture returned non-HTTP evidence".into());
        };
        assert!(succeeded);
        assert!(!flush_pending);
        assert_eq!(status, Some(200));
        assert_eq!(timing.terminal_cause, OperationTerminalCause::Succeeded);
        assert_eq!(
            timing.budget,
            OperationBudgetEvidence::Finite {
                configured_ms: 5_000
            }
        );
        assert_eq!(timing.start_boundary, MEASUREMENT_FINALIZATION_START);
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    #[test]
    fn report_verification_waits_for_async_completion_within_the_owner_budget()
    -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let target = target(temp.path().to_path_buf());
        let report = temp.path().join("trace.1.nsys-rep");
        let publication = report.clone();
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            std::fs::write(publication, b"fixture")
        });

        let bound = OperationBound::finite(Duration::from_secs(1));
        let record = super::verify_report(&target, &report, &bound, MEASUREMENT_FINALIZATION_START);

        writer.join().map_err(|_| "report writer panicked")??;
        assert!(record.succeeded());
        Ok(())
    }

    #[test]
    fn ssh_control_script_quotes_escape_values_with_metacharacters() {
        let script = ssh_control_script(
            Path::new("/work dir"),
            &[
                "env".to_owned(),
                "--".to_owned(),
                "NSYS_OPTS=a b;c".to_owned(),
                "nsys".to_owned(),
                "start".to_owned(),
            ],
        );
        assert_eq!(
            script,
            "cd '/work dir' && exec 'env' '--' 'NSYS_OPTS=a b;c' 'nsys' 'start'"
        );
    }

    #[test]
    fn ssh_control_script_routes_session_inspection_through_the_target_cwd() {
        let script = ssh_control_script(
            Path::new("/remote workspace"),
            &[
                "env".to_owned(),
                "--".to_owned(),
                "NSYS_HOME=/opt/nsys custom".to_owned(),
                "nsys-custom".to_owned(),
                "sessions".to_owned(),
                "list".to_owned(),
                "--output-format=json".to_owned(),
            ],
        );
        assert_eq!(
            script,
            "cd '/remote workspace' && exec 'env' '--' 'NSYS_HOME=/opt/nsys custom' 'nsys-custom' 'sessions' 'list' '--output-format=json'"
        );
    }
}
