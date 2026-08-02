use crate::plan::{ProfilerLaunch, ProfilerTargetRecord};
use crate::transport::{CommandActionMode, TargetCommandError, target_output};
use inferlab_runtime::operation_bound::{OperationBound, Remaining, duration_millis};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

const PROFILER_AGENT_DISCOVERY_DEADLINE: Duration = Duration::from_secs(10);
const PROFILER_AGENT_TERM_GRACE: Duration = Duration::from_secs(2);
const PROFILER_AGENT_KILL_GRACE: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilerCleanupRecord {
    pub trigger: ProfilerCleanupTrigger,
    pub session: String,
    pub strategy: String,
    pub elapsed_ms: u64,
    pub discovery_deadline_ms: u64,
    pub term_grace_ms: u64,
    pub kill_grace_ms: u64,
    pub pids: Vec<u32>,
    pub already_exited: bool,
    pub term_sent: bool,
    pub kill_sent: bool,
    pub verified: bool,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfilerCleanupTrigger {
    StartupRollback,
    Recovery,
    Stop,
}

#[must_use]
pub fn cleanup_target_agent(
    target: &ProfilerTargetRecord,
    trigger: ProfilerCleanupTrigger,
) -> ProfilerCleanupRecord {
    let started = Instant::now();
    let strategy = strategy(target);
    let pattern = format!("nsys --start-agent --session-name {}", target.session);
    let discovery_bound = OperationBound::finite(PROFILER_AGENT_DISCOVERY_DEADLINE);
    let output = target_output(
        target,
        &["pgrep".to_owned(), "-f".to_owned(), pattern],
        &discovery_bound,
        CommandActionMode::Cleanup,
    );
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            return cleanup_error(
                target,
                trigger,
                started,
                format!("failed to launch pgrep: {error}"),
            );
        }
    };
    if output.status.code() == Some(1) {
        return ProfilerCleanupRecord {
            trigger,
            session: target.session.clone(),
            strategy: strategy.to_owned(),
            elapsed_ms: duration_millis(started.elapsed()),
            discovery_deadline_ms: duration_millis(PROFILER_AGENT_DISCOVERY_DEADLINE),
            term_grace_ms: duration_millis(PROFILER_AGENT_TERM_GRACE),
            kill_grace_ms: duration_millis(PROFILER_AGENT_KILL_GRACE),
            pids: Vec::new(),
            already_exited: true,
            term_sent: false,
            kill_sent: false,
            verified: true,
            error: None,
        };
    }
    if !output.status.success() {
        return cleanup_error(
            target,
            trigger,
            started,
            format!(
                "pgrep failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        );
    }
    let pids = match String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(pids) => pids,
        Err(error) => {
            return cleanup_error(
                target,
                trigger,
                started,
                format!("pgrep returned an invalid PID: {error}"),
            );
        }
    };
    let term_bound = OperationBound::finite(PROFILER_AGENT_TERM_GRACE);
    let mut errors = Vec::new();
    let term_sent = match signal_pids(target, &pids, "-TERM", &term_bound) {
        Ok(sent) => sent,
        Err(error) => {
            errors.push(error.to_string());
            false
        }
    };
    let stopped_after_term = wait_for_pids(target, &pids, &term_bound);
    let (kill_sent, verified) = match stopped_after_term {
        Ok(true) => (false, true),
        Ok(false) => {
            let kill_bound = OperationBound::finite(PROFILER_AGENT_KILL_GRACE);
            let kill_sent = match signal_pids(target, &pids, "-KILL", &kill_bound) {
                Ok(sent) => sent,
                Err(error) => {
                    errors.push(error.to_string());
                    false
                }
            };
            match wait_for_pids(target, &pids, &kill_bound) {
                Ok(true) => (kill_sent, true),
                Ok(false) => {
                    errors.push("Nsight Systems session agent remained alive".to_owned());
                    (kill_sent, false)
                }
                Err(error) => {
                    errors.push(error.to_string());
                    (kill_sent, false)
                }
            }
        }
        Err(error) => {
            errors.push(error.to_string());
            (false, false)
        }
    };
    ProfilerCleanupRecord {
        trigger,
        session: target.session.clone(),
        strategy: strategy.to_owned(),
        elapsed_ms: duration_millis(started.elapsed()),
        discovery_deadline_ms: duration_millis(PROFILER_AGENT_DISCOVERY_DEADLINE),
        term_grace_ms: duration_millis(PROFILER_AGENT_TERM_GRACE),
        kill_grace_ms: duration_millis(PROFILER_AGENT_KILL_GRACE),
        pids,
        already_exited: false,
        term_sent,
        kill_sent,
        verified,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

fn cleanup_error(
    target: &ProfilerTargetRecord,
    trigger: ProfilerCleanupTrigger,
    started: Instant,
    error: String,
) -> ProfilerCleanupRecord {
    ProfilerCleanupRecord {
        trigger,
        session: target.session.clone(),
        strategy: strategy(target).to_owned(),
        elapsed_ms: duration_millis(started.elapsed()),
        discovery_deadline_ms: duration_millis(PROFILER_AGENT_DISCOVERY_DEADLINE),
        term_grace_ms: duration_millis(PROFILER_AGENT_TERM_GRACE),
        kill_grace_ms: duration_millis(PROFILER_AGENT_KILL_GRACE),
        pids: Vec::new(),
        already_exited: false,
        term_sent: false,
        kill_sent: false,
        verified: false,
        error: Some(error),
    }
}

fn strategy(target: &ProfilerTargetRecord) -> &'static str {
    match &target.launch {
        ProfilerLaunch::Local => "local-pgrep-command-line",
        ProfilerLaunch::Ssh { .. } => "ssh-pgrep-command-line",
    }
}

#[derive(Debug, thiserror::Error)]
enum ProfilerCleanupCommandError {
    #[error("failed to send {signal} to profiler PID {pid}: {source}")]
    Signal {
        pid: u32,
        signal: String,
        #[source]
        source: TargetCommandError,
    },
    #[error("sending {signal} to profiler PID {pid} exited with {status}: {stderr}")]
    SignalExit {
        pid: u32,
        signal: String,
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error("failed to inspect profiler PID {pid}: {source}")]
    Inspect {
        pid: u32,
        #[source]
        source: TargetCommandError,
    },
    #[error("inspecting profiler PID {pid} exited with {status}: {stderr}")]
    InspectExit {
        pid: u32,
        status: std::process::ExitStatus,
        stderr: String,
    },
}

fn signal_pids(
    target: &ProfilerTargetRecord,
    pids: &[u32],
    signal: &str,
    bound: &OperationBound,
) -> Result<bool, ProfilerCleanupCommandError> {
    let mut succeeded = true;
    let mut failure = None;
    for pid in pids {
        let output = match target_output(
            target,
            &[
                "kill".to_owned(),
                signal.to_owned(),
                "--".to_owned(),
                pid.to_string(),
            ],
            bound,
            CommandActionMode::Cleanup,
        ) {
            Ok(output) => output,
            Err(source) => {
                succeeded = false;
                failure.get_or_insert_with(|| ProfilerCleanupCommandError::Signal {
                    pid: *pid,
                    signal: signal.to_owned(),
                    source,
                });
                continue;
            }
        };
        if !output.status.success() {
            succeeded = false;
            failure.get_or_insert_with(|| ProfilerCleanupCommandError::SignalExit {
                pid: *pid,
                signal: signal.to_owned(),
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            });
        }
    }
    if let Some(error) = failure {
        Err(error)
    } else {
        Ok(succeeded)
    }
}

fn wait_for_pids(
    target: &ProfilerTargetRecord,
    pids: &[u32],
    bound: &OperationBound,
) -> Result<bool, ProfilerCleanupCommandError> {
    loop {
        let mut any_alive = false;
        for pid in pids {
            any_alive |= target_pid_alive(target, *pid, bound)?;
        }
        if !any_alive {
            return Ok(true);
        }
        if bound.is_expired() {
            return Ok(false);
        }
        match bound.remaining() {
            Remaining::Finite(remaining) => {
                thread::sleep(Duration::from_millis(100).min(remaining));
            }
            Remaining::Expired => return Ok(false),
            Remaining::Unbounded => thread::sleep(Duration::from_millis(100)),
        }
    }
}

fn target_pid_alive(
    target: &ProfilerTargetRecord,
    pid: u32,
    bound: &OperationBound,
) -> Result<bool, ProfilerCleanupCommandError> {
    match &target.launch {
        ProfilerLaunch::Local => Ok(Path::new(&format!("/proc/{pid}")).exists()),
        ProfilerLaunch::Ssh { .. } => {
            let output = target_output(
                target,
                &[
                    "kill".to_owned(),
                    "-0".to_owned(),
                    "--".to_owned(),
                    pid.to_string(),
                ],
                bound,
                CommandActionMode::Cleanup,
            )
            .map_err(|source| ProfilerCleanupCommandError::Inspect { pid, source })?;
            match output.status.code() {
                Some(0) => Ok(true),
                Some(1) => Ok(false),
                _ => Err(ProfilerCleanupCommandError::InspectExit {
                    pid,
                    status: output.status,
                    stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
                }),
            }
        }
    }
}
