use crate::plan::{
    CaptureWindowActionPlan, CaptureWindowHttpMethodPlan, ProfilerControl, ProfilerLaunch,
    ProfilerTargetRecord, env_prefix,
};
use crate::record::{CaptureActionRecord, CaptureHttpFailureKind};
use inferlab_runtime::operation_bound::{OperationBound, OperationTerminalCause};
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Output;
use std::time::Duration;

const PROFILER_ARM_COMMAND_DEADLINE: Duration = Duration::from_secs(60);
const PROFILER_REPORT_VERIFICATION_DEADLINE: Duration = Duration::from_secs(30);

pub(crate) fn prepare_output(target: &ProfilerTargetRecord, parent: &Path) -> CaptureActionRecord {
    command_action(
        target,
        "prepare-output",
        vec![
            "mkdir".to_owned(),
            "-p".to_owned(),
            parent.display().to_string(),
        ],
        PROFILER_ARM_COMMAND_DEADLINE,
        CommandActionMode::Operation,
    )
}

pub(crate) fn arm_range_collection(
    target: &ProfilerTargetRecord,
    output: &Path,
    range_count: usize,
) -> CaptureActionRecord {
    command_action(
        target,
        "start-range-collection",
        nsys_start_argv(target, output, range_count),
        PROFILER_ARM_COMMAND_DEADLINE,
        CommandActionMode::Operation,
    )
}

pub(crate) fn verify_report(target: &ProfilerTargetRecord, path: &Path) -> CaptureActionRecord {
    command_action(
        target,
        "verify-report",
        vec![
            "test".to_owned(),
            "-f".to_owned(),
            path.display().to_string(),
        ],
        PROFILER_REPORT_VERIFICATION_DEADLINE,
        CommandActionMode::Cleanup,
    )
}

pub(crate) fn inspect_collection_state(
    target: &ProfilerTargetRecord,
    deadline: Duration,
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
        deadline,
        CommandActionMode::Cleanup,
    )
}

pub(crate) fn stop_collection(
    target: &ProfilerTargetRecord,
    deadline: Duration,
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
        deadline,
        CommandActionMode::Cleanup,
    )
}

pub(crate) fn start_windows(targets: &[ProfilerTargetRecord]) -> Vec<CaptureActionRecord> {
    let process_ids = targets
        .iter()
        .map(|target| match &target.control {
            ProfilerControl::Http { process_id, .. } => process_id.as_str(),
        })
        .collect::<BTreeSet<_>>();
    window_actions_for(targets, true, &process_ids)
}

pub(crate) fn stop_windows(
    targets: &[ProfilerTargetRecord],
    process_ids: &BTreeSet<&str>,
) -> Vec<CaptureActionRecord> {
    window_actions_for(targets, false, process_ids)
}

fn window_actions_for(
    targets: &[ProfilerTargetRecord],
    start: bool,
    process_ids: &BTreeSet<&str>,
) -> Vec<CaptureActionRecord> {
    let mut seen = BTreeSet::new();
    targets
        .iter()
        .filter_map(|target| match &target.control {
            ProfilerControl::Http {
                process_id,
                start: start_action,
                stop: stop_action,
                deadline_seconds,
                ..
            } if process_ids.contains(process_id.as_str()) && seen.insert(process_id.clone()) => {
                let action = if start { start_action } else { stop_action };
                Some(http_action(
                    process_id,
                    if start { "start-range" } else { "stop-range" },
                    action,
                    *deadline_seconds,
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
        let client = reqwest::blocking::Client::builder()
            .timeout(deadline)
            .connect_timeout(deadline.min(Duration::from_secs(2)))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|source| CaptureHttpError::Request { source })?;
        let request = match (action.method, action.body.as_ref()) {
            (CaptureWindowHttpMethodPlan::Post, Some(body)) => {
                let payload = serde_json::to_vec(body)
                    .map_err(|source| CaptureHttpError::Serialization { source })?;
                client
                    .post(&url)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(payload)
            }
            (CaptureWindowHttpMethodPlan::Post, None) => client.post(&url),
        };
        request
            .send()
            .map_err(|source| CaptureHttpError::Request { source })
    })();
    match result {
        Ok(response) => CaptureActionRecord::Http {
            process_id: process_id.to_owned(),
            operation: operation.to_owned(),
            method: Some(action.method),
            path: Some(action.path.clone()),
            url,
            body: action.body.clone(),
            status: Some(response.status().as_u16()),
            failure_kind: None,
            error: None,
            succeeded: response.status().is_success(),
            timing: bound.timing(
                &format!("before_profiler_{operation}"),
                if response.status().is_success() {
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
            error: Some(error.record_message()),
            succeeded: false,
            timing: bound.timing(
                &format!("before_profiler_{operation}"),
                if bound.is_expired() {
                    OperationTerminalCause::TimedOut
                } else {
                    OperationTerminalCause::Failed
                },
            ),
        },
    }
}

#[derive(Debug, thiserror::Error)]
enum CaptureHttpError {
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
    fn failure_kind(&self) -> CaptureHttpFailureKind {
        match self {
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
    deadline: Duration,
    mode: CommandActionMode,
) -> CaptureActionRecord {
    let bound = OperationBound::finite(deadline);
    let output = target_output(target, &argv, &bound, mode);
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
                &format!("before_profiler_{operation}"),
                if output.status.success() {
                    OperationTerminalCause::Succeeded
                } else {
                    OperationTerminalCause::Failed
                },
            ),
            cleanup: None,
        },
        Err(error) => {
            let mut timing = bound.timing(
                &format!("before_profiler_{operation}"),
                error.terminal_cause,
            );
            timing.elapsed_ms = error.operation_elapsed_ms;
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
    operation_elapsed_ms: u64,
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
        Ok(inferlab_runtime::container::BoundedWait::Expired {
            operation_elapsed_ms,
            cleanup,
            ..
        }) => Err(TargetCommandError {
            failure: TargetCommandFailure::Deadline,
            terminal_cause: OperationTerminalCause::TimedOut,
            operation_elapsed_ms,
            cleanup,
        }),
        Ok(inferlab_runtime::container::BoundedWait::Interrupted {
            operation_elapsed_ms,
            cleanup,
            ..
        }) => Err(TargetCommandError {
            failure: TargetCommandFailure::Interrupted,
            terminal_cause: OperationTerminalCause::Interrupted,
            operation_elapsed_ms,
            cleanup: Some(cleanup),
        }),
        Err(inferlab_runtime::container::BoundedError::Launch(source)) => Err(TargetCommandError {
            failure: TargetCommandFailure::Launch { source },
            terminal_cause: OperationTerminalCause::Failed,
            operation_elapsed_ms: bound.elapsed_ms(),
            cleanup: None,
        }),
        Err(
            inferlab_runtime::container::BoundedError::Stdin(source)
            | inferlab_runtime::container::BoundedError::Wait(source),
        ) => Err(TargetCommandError {
            failure: TargetCommandFailure::Io { source },
            terminal_cause: OperationTerminalCause::Failed,
            operation_elapsed_ms: bound.elapsed_ms(),
            cleanup: None,
        }),
        Err(inferlab_runtime::container::BoundedError::WaitCleanup {
            source,
            operation_elapsed_ms,
            cleanup,
        }) => Err(TargetCommandError {
            failure: TargetCommandFailure::WaitCleanup { source },
            terminal_cause: OperationTerminalCause::Failed,
            operation_elapsed_ms,
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
    use super::{CommandActionMode, command_action, ssh_control_script};
    use crate::plan::{
        CaptureWindowActionPlan, CaptureWindowControlEndpointPlan, CaptureWindowHttpMethodPlan,
        NsysEscapes, ProfilerControl, ProfilerFinalization, ProfilerLaunch, ProfilerTargetRecord,
        WindowControlKind,
    };
    use crate::record::CaptureActionRecord;
    use inferlab_protocol::EndpointAssignment;
    use inferlab_runtime::operation_bound::{OperationBudgetEvidence, OperationTerminalCause};
    use std::error::Error;
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
                deadline_seconds: 60,
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
        let action = command_action(
            &target(temp.path().to_path_buf()),
            "fixture-finalization",
            vec!["sh".to_owned(), "-c".to_owned(), "sleep 5".to_owned()],
            Duration::from_millis(50),
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
