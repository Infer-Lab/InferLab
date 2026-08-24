mod cleanup;
mod launch;
mod observation;
mod readiness;

use crate::operation_bound::{OperationBound, OperationTimingEvidence};
use crate::plan::{CommandPlan, LaunchFilePlan, LaunchPlan, ProcessEndpointPlan, ReadinessPlan};
use crate::process_group::{SignalEvidence, process_start_time};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(test)]
use crate::operation_bound::OperationTerminalCause;
#[cfg(test)]
use crate::plan::TargetRegistryExpectedTarget;
#[cfg(test)]
use crate::shell::shell_quote_path;
use cleanup::removal_summary;
#[cfg(test)]
use cleanup::terminate_local;
#[cfg(test)]
use launch::{materialize_local_launch_files, remote_launch_file_script, spawn_local};
#[cfg(test)]
use observation::{run_status_command, verified_local_status};
#[cfg(test)]
use readiness::{
    HttpTargetRegistryProbe, match_target_registry, probe_http, probe_http_json,
    wait_http_target_registry_ready, wait_process_alive_ready,
};
#[cfg(test)]
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::io::{Read, Write};
#[cfg(test)]
use std::os::unix::process::CommandExt;
#[cfg(test)]
use std::process::{Command, Output, Stdio};
#[cfg(test)]
use std::thread;
#[cfg(test)]
use std::time::Instant;

pub const REMOTE_LOG_SYNC_DEADLINE: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostProcessHandle {
    pub leader_pid: u32,
    pub process_group: u32,
    pub leader_start_time_ticks: u64,
    /// The container this process launched, when the command is a
    /// containerized substitution: the daemon-owned cleanup handle the
    /// process-group kill cannot reach ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
}

impl HostProcessHandle {
    fn new(leader_pid: u32, container: Option<String>) -> Result<Self, String> {
        if leader_pid == 0 {
            return Err("host process-group handle requires a non-zero leader pid".to_owned());
        }
        let leader_start_time_ticks = process_start_time(leader_pid)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                format!("host process {leader_pid} exited before its identity could be recorded")
            })?;
        Ok(Self {
            leader_pid,
            process_group: leader_pid,
            leader_start_time_ticks,
            container,
        })
    }

    fn validate(&self) -> Result<(), String> {
        validate_process_identity(
            self.leader_pid,
            self.process_group,
            self.leader_start_time_ticks,
            "host",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SshProcessHandle {
    pub target: String,
    pub leader_pid: u32,
    pub process_group: u32,
    pub leader_start_time_ticks: u64,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    /// The container this process launched, when the command is a
    /// containerized substitution: the daemon-owned cleanup handle the
    /// process-group kill cannot reach ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
}

impl SshProcessHandle {
    fn validate(&self) -> Result<(), String> {
        if self.target.is_empty() {
            return Err("SSH process handle requires a target".to_owned());
        }
        validate_process_identity(
            self.leader_pid,
            self.process_group,
            self.leader_start_time_ticks,
            "SSH",
        )
    }
}

fn validate_process_identity(
    leader_pid: u32,
    process_group: u32,
    leader_start_time_ticks: u64,
    kind: &str,
) -> Result<(), String> {
    if leader_pid == 0 || process_group == 0 || leader_start_time_ticks == 0 {
        return Err(format!("{kind} process-group handle requires non-zero ids"));
    }
    if leader_pid != process_group {
        return Err(format!(
            "{kind} process-group handle requires leader_pid to equal process_group"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ProcessHandle {
    Local(HostProcessHandle),
    Ssh(SshProcessHandle),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CleanupTrigger {
    StartupRollback,
    Stop,
    Recovery,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupEvidence {
    pub trigger: CleanupTrigger,
    pub elapsed_ms: u64,
    pub status_deadline_ms: u64,
    pub term_grace_ms: u64,
    pub kill_grace_ms: u64,
    pub reap_grace_ms: Option<u64>,
    pub remote_deadline_ms: Option<u64>,
    pub verified: bool,
    pub already_exited: bool,
    pub forced: bool,
    pub signals: Vec<SignalEvidence>,
    pub error: Option<String>,
    /// Confirmed removal of the process's container on its launch machine
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]); present when the cleanup path
    /// attempted to remove a known container — from a running server's
    /// handle, or from a launch failure whose command already named one
    /// before any handle existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_removal: Option<ContainerRemovalEvidence>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerRemovalEvidence {
    pub container: String,
    pub elapsed_ms: u64,
    pub operation_elapsed_ms: u64,
    pub deadline_ms: u64,
    pub client_cleanup: Option<crate::container::CommandCleanupEvidence>,
    pub confirmed: bool,
    pub already_absent: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessStatus {
    pub queried: bool,
    pub alive: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRegistryMatchEvidence {
    pub url: String,
    pub role: String,
    pub healthy: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadinessEvidence {
    Http {
        url: String,
        attempts: u32,
        ready_unix_ms: u64,
        timing: OperationTimingEvidence,
        diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
    },
    HttpTargetRegistry {
        readiness_url: String,
        registry_url: String,
        attempts: u32,
        ready_unix_ms: u64,
        matched_targets: Vec<TargetRegistryMatchEvidence>,
        timing: OperationTimingEvidence,
        diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
    },
    ProcessAlive {
        ready_unix_ms: u64,
        timing: OperationTimingEvidence,
        diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadinessAttemptEvidence {
    pub operation: String,
    pub effective_bound_ms: u64,
    pub succeeded: bool,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessFailureKind {
    Exited,
    Interrupted,
    Timeout,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadinessFailure {
    pub kind: ReadinessFailureKind,
    pub message: String,
    pub timing: Option<OperationTimingEvidence>,
    pub diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
}

pub struct ProcessSpec<'a> {
    pub launch: &'a LaunchPlan,
    pub command: &'a CommandPlan,
    pub launch_files: &'a [LaunchFilePlan],
    pub cache_root: &'a Path,
    pub stdout: &'a Path,
    pub stderr: &'a Path,
    pub remote_dir: &'a Path,
    /// The resolver-assigned container name when the command is a
    /// containerized substitution.
    pub container: Option<&'a str>,
}

#[derive(Debug, thiserror::Error)]
pub enum ServerLaunchError {
    #[error("{message}")]
    Preparation { message: String },
    #[error("failed to {operation} {path}: {source}")]
    FileIo {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to launch {program:?}: {source}")]
    Process {
        program: String,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Ssh(#[from] crate::ssh::SshError),
    #[error("{operation} exited with {status}: {diagnostics}")]
    Exit {
        operation: String,
        status: std::process::ExitStatus,
        diagnostics: String,
    },
    #[error("SSH launch on {target:?} returned non-UTF-8 identity: {source}")]
    NonUtf8Identity {
        target: String,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error("SSH launch on {target:?} returned no process id")]
    MissingProcessId { target: String },
    #[error("SSH launch on {target:?} returned no process start time")]
    MissingStartTime { target: String },
    #[error("SSH launch on {target:?} returned invalid process id {value:?}: {source}")]
    InvalidProcessId {
        target: String,
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error("SSH launch on {target:?} returned invalid process start time {value:?}: {source}")]
    InvalidStartTime {
        target: String,
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error("SSH launch on {target:?} returned an invalid process identity: {details}")]
    InvalidIdentity { target: String, details: String },
    #[error("existing launch file target {path} is not a regular file", path = path.display())]
    NotRegularFile { path: PathBuf },
    #[error(
        "existing launch file {path} does not match declared digest {expected}; found {actual}",
        path = path.display()
    )]
    FileDigestMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
}

#[derive(Debug)]
pub struct LaunchFailure {
    pub error: ServerLaunchError,
    pub ownership_unknown: bool,
    /// The structured outcome of removing the container this launch may
    /// have created, when the failure attempted one; the record's cleanup
    /// evidence carries the actual container and reason rather than a
    /// generic note ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub container_removal: Option<Box<ContainerRemovalEvidence>>,
    pub cleanup: Option<Box<CleanupEvidence>>,
    cleanup_note: Option<String>,
}

impl LaunchFailure {
    pub fn before_launch(message: String) -> Self {
        Self {
            error: ServerLaunchError::Preparation { message },
            ownership_unknown: false,
            container_removal: None,
            cleanup: None,
            cleanup_note: None,
        }
    }

    fn from_error(error: ServerLaunchError) -> Self {
        Self {
            error,
            ownership_unknown: false,
            container_removal: None,
            cleanup: None,
            cleanup_note: None,
        }
    }

    pub fn message(&self) -> String {
        let mut message = self.error.to_string();
        if let Some(note) = &self.cleanup_note {
            message = format!("{message}; {note}");
        } else if let Some(cleanup) = &self.cleanup
            && let Some(error) = &cleanup.error
        {
            message = format!("{message}; local launch cleanup was not verified: {error}");
        }
        if let Some(removal) = &self.container_removal {
            message = format!("{message}; {}", removal_summary(removal));
        }
        message
    }

    pub fn unresolved_ownership(message: String) -> Self {
        Self {
            error: ServerLaunchError::Preparation { message },
            ownership_unknown: true,
            container_removal: None,
            cleanup: None,
            cleanup_note: None,
        }
    }
}

pub trait ProcessLauncher {
    fn spawn(&self, spec: ProcessSpec<'_>) -> Result<ProcessHandle, LaunchFailure>;
}

pub trait ProcessObserver {
    fn status(&self, handle: &ProcessHandle) -> ProcessStatus;
    fn status_with_bound(&self, handle: &ProcessHandle, bound: &OperationBound) -> ProcessStatus;
    fn sync_logs(
        &self,
        handle: &ProcessHandle,
        stdout: &Path,
        stderr: &Path,
        cleanup: bool,
    ) -> Result<(), LogSyncError>;
}

pub trait ReadinessObserver {
    fn wait_ready(
        &self,
        handle: &ProcessHandle,
        endpoint: &ProcessEndpointPlan,
        readiness: &ReadinessPlan,
        bound: &OperationBound,
        on_probe_failure: &mut dyn FnMut(&str),
    ) -> Result<ReadinessEvidence, ReadinessFailure>;
}

pub trait ProcessCleanup {
    fn terminate(
        &self,
        handle: &ProcessHandle,
        trigger: CleanupTrigger,
        on_container_removal: &mut dyn FnMut(&str),
    ) -> CleanupEvidence;
}

pub trait ServerRuntime:
    ProcessLauncher + ProcessObserver + ReadinessObserver + ProcessCleanup
{
}

impl<T> ServerRuntime for T where
    T: ProcessLauncher + ProcessObserver + ReadinessObserver + ProcessCleanup
{
}

#[derive(Debug, thiserror::Error)]
pub enum ProcessCommandError {
    #[error("{operation} deadline expired")]
    Deadline { operation: String },
    #[error("{operation} was interrupted")]
    Interrupted { operation: String },
    #[error("failed to launch {operation}: {source}")]
    Launch {
        operation: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to launch {operation}: {source}")]
    Ssh {
        operation: String,
        #[source]
        source: crate::ssh::SshError,
    },
    #[error("{operation} failed: {source}")]
    Io {
        operation: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{operation} wait failed: {source}; child cleanup: {cleanup}")]
    WaitCleanup {
        operation: String,
        #[source]
        source: std::io::Error,
        cleanup: String,
    },
    #[error("{operation} exited with {status}: {stderr}")]
    Exit {
        operation: String,
        status: std::process::ExitStatus,
        stderr: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum LogSyncError {
    #[error("failed to read remote log {path}: {source}")]
    ReadRemote {
        path: PathBuf,
        #[source]
        source: ProcessCommandError,
    },
    #[error("failed to read remote log {path}: command exited with {status}: {stderr}")]
    RemoteExit {
        path: PathBuf,
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error("failed to write local log {path}: {source}")]
    WriteLocal {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemProcessRuntime;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::LaunchFilePlan;
    use std::cell::Cell;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    #[test]
    fn expired_readiness_owner_prevents_a_fresh_network_attempt() {
        let bound = OperationBound::finite(Duration::ZERO);
        let error = probe_http("127.0.0.1", 9, "/ready", &bound, 30)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();

        assert_eq!(error, "readiness operation deadline expired");
    }

    #[test]
    fn readiness_attempt_deadline_bounds_a_trickled_status_line() -> Result<(), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let server = thread::spawn(move || -> Result<(), String> {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            let response = format!("HTTP/1.1 200 {}\r\n", " ".repeat(96));
            for byte in response.bytes() {
                if stream.write_all(&[byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        });

        let started = Instant::now();
        let error = probe_http("127.0.0.1", port, "/ready", &OperationBound::unbounded(), 1)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        let elapsed = started.elapsed();
        server
            .join()
            .map_err(|_| "trickle fixture panicked".to_owned())??;

        assert!(error.contains("deadline expired"), "{error}");
        assert!(
            elapsed < Duration::from_secs(2),
            "a one-second attempt lasted {elapsed:?}"
        );
        Ok(())
    }

    #[test]
    fn finite_readiness_accepts_a_response_after_250_milliseconds() -> Result<(), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let server = thread::spawn(move || -> Result<(), String> {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            thread::sleep(Duration::from_millis(350));
            stream
                .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                .map_err(|error| error.to_string())
        });

        let started = Instant::now();
        probe_http(
            "127.0.0.1",
            port,
            "/ready",
            &OperationBound::finite(Duration::from_secs(2)),
            1,
        )
        .map_err(|error| error.to_string())?;
        let elapsed = started.elapsed();
        server
            .join()
            .map_err(|_| "delayed readiness fixture panicked".to_owned())??;

        assert!(elapsed >= Duration::from_millis(250), "elapsed {elapsed:?}");
        assert!(elapsed < Duration::from_secs(2), "elapsed {elapsed:?}");
        Ok(())
    }

    #[test]
    fn readiness_attempt_deadline_includes_the_complete_response_body() -> Result<(), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let server = thread::spawn(move || -> Result<(), String> {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nx")
                .map_err(|error| error.to_string())?;
            thread::sleep(Duration::from_millis(1_500));
            Ok(())
        });

        let started = Instant::now();
        let error = probe_http("127.0.0.1", port, "/ready", &OperationBound::unbounded(), 1)
            .err()
            .map(|error| error.to_string());
        let elapsed = started.elapsed();
        server
            .join()
            .map_err(|_| "readiness body fixture panicked".to_owned())??;

        assert!(
            error.is_some_and(|error| error == "readiness operation deadline expired"),
            "readiness accepted an incomplete response body"
        );
        assert!(
            elapsed < Duration::from_millis(1_500),
            "elapsed {elapsed:?}"
        );
        Ok(())
    }

    #[test]
    fn registry_attempt_deadline_bounds_a_trickled_body() -> Result<(), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let server = thread::spawn(move || -> Result<(), String> {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .map_err(|error| error.to_string())?;
            let body = format!("{}{{}}", " ".repeat(96));
            for byte in body.bytes() {
                if stream.write_all(&[byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        });

        let started = Instant::now();
        let error = probe_http_json(
            "127.0.0.1",
            port,
            "/workers",
            "target registry",
            &OperationBound::unbounded(),
            1,
        )
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
        let elapsed = started.elapsed();
        server
            .join()
            .map_err(|_| "registry trickle fixture panicked".to_owned())??;

        assert!(error.contains("deadline expired"), "{error}");
        assert!(
            elapsed < Duration::from_secs(2),
            "a one-second registry attempt lasted {elapsed:?}"
        );
        Ok(())
    }

    #[test]
    fn expired_readiness_owner_rejects_a_registry_match() {
        let expected = vec![TargetRegistryExpectedTarget {
            url: "http://decode:30001".to_owned(),
            role: "decode".to_owned(),
            bootstrap_port: None,
        }];
        let response = serde_json::json!({
            "workers": [{
                "url": "http://decode:30001",
                "worker_type": "decode",
                "is_healthy": true
            }]
        });

        let error = match_target_registry(
            &response,
            &target_registry_probe(&expected),
            &OperationBound::finite(Duration::ZERO),
        )
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();

        assert_eq!(error, "readiness operation deadline expired");
    }

    #[test]
    fn process_status_command_cannot_outlive_the_readiness_owner() {
        let started = Instant::now();
        let error = run_status_command(
            &["sh", "-c", "sleep 5"],
            &[],
            &OperationBound::finite(Duration::from_millis(50)),
        )
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();

        assert_eq!(error, "process status attempt deadline expired");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "bounded process status did not stop promptly"
        );
    }

    #[test]
    fn finite_readiness_retries_an_expired_process_status_attempt() -> Result<(), String> {
        let calls = Cell::new(0_u32);
        let mut failures = Vec::new();
        let bound = OperationBound::finite(Duration::from_secs(3));
        let evidence = wait_process_alive_ready(
            |_| {
                calls.set(calls.get() + 1);
                if calls.get() == 1 {
                    thread::sleep(Duration::from_millis(1_050));
                    ProcessStatus {
                        queried: false,
                        alive: false,
                        error: Some("process status attempt deadline expired".to_owned()),
                    }
                } else {
                    alive_status()
                }
            },
            1,
            &bound,
            &mut |failure| failures.push(failure.to_owned()),
        )
        .map_err(|failure| failure.message)?;

        assert_eq!(calls.get(), 2);
        assert_eq!(failures, ["process status attempt deadline expired"]);
        let ReadinessEvidence::ProcessAlive {
            timing,
            diagnostic_attempts,
            ..
        } = evidence
        else {
            return Err("process-alive readiness returned the wrong evidence kind".to_owned());
        };
        assert_eq!(
            timing.start_boundary,
            "after_process_spawn_before_readiness_attempt"
        );
        assert_eq!(diagnostic_attempts.len(), 1);
        assert!(diagnostic_attempts[0].succeeded);
        assert!((1..=1_000).contains(&diagnostic_attempts[0].effective_bound_ms));
        Ok(())
    }

    #[test]
    fn unbounded_process_status_command_does_not_acquire_a_timeout() -> Result<(), String> {
        let output = run_status_command(
            &["sh", "-c", "sleep 0.1; printf alive"],
            &[],
            &OperationBound::unbounded(),
        )
        .map_err(|error| error.to_string())?;

        assert!(output.status.success());
        assert_eq!(output.stdout, b"alive");
        Ok(())
    }

    fn launch_file(root: &Path, text: &str, name: &str) -> LaunchFilePlan {
        let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
        let relative_path = format!("launch-files/{sha256}/{name}");
        LaunchFilePlan {
            resolved_path: root.join(&relative_path),
            relative_path,
            text: text.to_owned(),
            sha256,
        }
    }

    fn run_script_with_input(script: &str, input: &[u8]) -> Result<Output, String> {
        match crate::container::run_with_bound(
            &["bash", "-c", script],
            &[],
            None,
            Some(input),
            &OperationBound::unbounded(),
            None,
        ) {
            Ok(crate::container::BoundedWait::Exited {
                status,
                stdout,
                stderr,
            }) => Ok(Output {
                status,
                stdout,
                stderr,
            }),
            Ok(crate::container::BoundedWait::Expired { .. }) => {
                Err("unbounded launch-file fixture expired".to_owned())
            }
            Ok(crate::container::BoundedWait::Interrupted { .. }) => {
                Err("launch-file fixture was interrupted".to_owned())
            }
            Err(_) => Err("launch-file fixture failed".to_owned()),
        }
    }

    fn target_registry_endpoint(
        registry_body: String,
    ) -> Result<(ProcessEndpointPlan, thread::JoinHandle<Result<(), String>>), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
                let mut request_line = String::new();
                let mut reader =
                    BufReader::new(stream.try_clone().map_err(|error| error.to_string())?);
                reader
                    .read_line(&mut request_line)
                    .map_err(|error| error.to_string())?;
                loop {
                    let mut header = String::new();
                    reader
                        .read_line(&mut header)
                        .map_err(|error| error.to_string())?;
                    if header == "\r\n" || header.is_empty() {
                        break;
                    }
                }
                let body = if request_line.starts_with("GET /workers ") {
                    registry_body.as_bytes()
                } else {
                    b""
                };
                let mut response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .into_bytes();
                response.extend_from_slice(body);
                stream
                    .write_all(&response)
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        });
        Ok((
            ProcessEndpointPlan {
                host: "127.0.0.1".to_owned(),
                port,
            },
            server,
        ))
    }

    fn target_registry_probe<'a>(
        expected_targets: &'a [TargetRegistryExpectedTarget],
    ) -> HttpTargetRegistryProbe<'a> {
        HttpTargetRegistryProbe {
            readiness_path: "/readiness",
            registry_path: "/workers",
            targets_field: "workers",
            target_url_field: "url",
            target_role_field: "worker_type",
            target_healthy_field: "is_healthy",
            target_bootstrap_port_field: "bootstrap_port",
            expected_targets,
        }
    }

    fn alive_status() -> ProcessStatus {
        ProcessStatus {
            queried: true,
            alive: true,
            error: None,
        }
    }

    #[test]
    fn local_launch_file_publication_reuses_the_immutable_target() -> Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let launch_file = launch_file(
            root.path(),
            "worker: \u{2603}\nmode: context\n",
            "worker.yaml",
        );

        materialize_local_launch_files(std::slice::from_ref(&launch_file))
            .map_err(|error| error.to_string())?;
        let first_metadata =
            fs::metadata(&launch_file.resolved_path).map_err(|error| error.to_string())?;
        materialize_local_launch_files(std::slice::from_ref(&launch_file))
            .map_err(|error| error.to_string())?;
        let second_metadata =
            fs::metadata(&launch_file.resolved_path).map_err(|error| error.to_string())?;

        assert_eq!(
            fs::read_to_string(&launch_file.resolved_path).map_err(|error| error.to_string())?,
            launch_file.text
        );
        assert_eq!(first_metadata.ino(), second_metadata.ino());
        assert_eq!(second_metadata.permissions().mode() & 0o222, 0);
        Ok(())
    }

    #[test]
    fn local_launch_file_mismatch_fails_before_spawn_without_replacing_it() -> Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let cache = root.path().join("cache");
        let launch_file = launch_file(&cache, "expected\n", "worker.yaml");
        let parent = launch_file
            .resolved_path
            .parent()
            .ok_or_else(|| "launch file has no parent".to_owned())?;
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        fs::write(&launch_file.resolved_path, "stale\n").map_err(|error| error.to_string())?;
        let marker = root.path().join("spawned");
        let command = CommandPlan {
            argv: vec![
                "sh".to_owned(),
                "-c".to_owned(),
                format!("printf launched > {}", shell_quote_path(&marker)),
            ],
            env: BTreeMap::new(),
            explicit_env: Vec::new(),
            pass_env: Vec::new(),
            cwd: root.path().to_path_buf(),
        };
        let launch_files = vec![launch_file.clone()];

        let result = spawn_local(ProcessSpec {
            launch: &LaunchPlan::Local,
            command: &command,
            launch_files: &launch_files,
            cache_root: &cache,
            stdout: &root.path().join("stdout.log"),
            stderr: &root.path().join("stderr.log"),
            remote_dir: &root.path().join("remote"),
            container: None,
        });

        let failure = match result {
            Err(failure) => failure,
            Ok(handle) => {
                let _ = terminate_local(&handle, CleanupTrigger::StartupRollback);
                return Err("mismatched launch file unexpectedly spawned a process".to_owned());
            }
        };
        assert!(!failure.ownership_unknown, "{failure:?}");
        assert!(failure.message().contains("does not match"), "{failure:?}");
        assert!(!marker.exists());
        assert_eq!(
            fs::read_to_string(&launch_file.resolved_path).map_err(|error| error.to_string())?,
            "stale\n"
        );
        Ok(())
    }

    #[test]
    fn remote_launch_file_script_publishes_stdin_without_replacing_targets() -> Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let published = launch_file(
            root.path(),
            "worker: \u{96ea}\nmode: context\n",
            "worker.yaml",
        );
        let script = remote_launch_file_script(&published).map_err(|error| error.to_string())?;

        let first = run_script_with_input(&script, published.text.as_bytes())?;
        assert!(
            first.status.success(),
            "{}",
            String::from_utf8_lossy(&first.stderr)
        );
        let first_metadata =
            fs::metadata(&published.resolved_path).map_err(|error| error.to_string())?;
        let first_inode = first_metadata.ino();
        assert_eq!(first_metadata.permissions().mode() & 0o222, 0);
        let reused = run_script_with_input(&script, published.text.as_bytes())?;
        assert!(
            reused.status.success(),
            "{}",
            String::from_utf8_lossy(&reused.stderr)
        );
        assert_eq!(
            fs::read_to_string(&published.resolved_path).map_err(|error| error.to_string())?,
            published.text
        );
        assert_eq!(
            fs::metadata(&published.resolved_path)
                .map_err(|error| error.to_string())?
                .ino(),
            first_inode
        );

        let corrupt = launch_file(root.path(), "expected\n", "corrupt.yaml");
        let corrupt_parent = corrupt
            .resolved_path
            .parent()
            .ok_or_else(|| "launch file has no parent".to_owned())?;
        fs::create_dir_all(corrupt_parent).map_err(|error| error.to_string())?;
        fs::write(&corrupt.resolved_path, "stale\n").map_err(|error| error.to_string())?;
        let rejected = run_script_with_input(
            &remote_launch_file_script(&corrupt).map_err(|error| error.to_string())?,
            corrupt.text.as_bytes(),
        )?;
        assert!(!rejected.status.success());
        assert_eq!(
            fs::read_to_string(&corrupt.resolved_path).map_err(|error| error.to_string())?,
            "stale\n"
        );
        Ok(())
    }

    #[test]
    fn target_registry_readiness_records_all_expected_targets() -> Result<(), String> {
        let (endpoint, server) = target_registry_endpoint(
            serde_json::json!({
                "workers": [
                    {
                        "url": "http://prefill:30000",
                        "worker_type": "prefill",
                        "is_healthy": true,
                        "bootstrap_port": 8998
                    },
                    {
                        "url": "http://decode:30001",
                        "worker_type": "decode",
                        "is_healthy": true
                    }
                ]
            })
            .to_string(),
        )?;
        let expected = vec![
            TargetRegistryExpectedTarget {
                url: "http://prefill:30000".to_owned(),
                role: "prefill".to_owned(),
                bootstrap_port: Some(8998),
            },
            TargetRegistryExpectedTarget {
                url: "http://decode:30001".to_owned(),
                role: "decode".to_owned(),
                bootstrap_port: None,
            },
        ];

        let bound = OperationBound::unbounded();
        let evidence = wait_http_target_registry_ready(
            |_| alive_status(),
            &endpoint,
            target_registry_probe(&expected),
            1,
            &bound,
            &mut |_| {},
        )
        .map_err(|failure| failure.message)?;
        server
            .join()
            .map_err(|_| "target registry fixture panicked".to_owned())??;

        let record_value = serde_json::to_value(&evidence).map_err(|error| error.to_string())?;
        assert_eq!(record_value["kind"], "http_target_registry");
        assert_eq!(
            record_value["matched_targets"].as_array().map(Vec::len),
            Some(2)
        );
        let ReadinessEvidence::HttpTargetRegistry {
            readiness_url,
            registry_url,
            attempts,
            matched_targets,
            timing,
            diagnostic_attempts,
            ready_unix_ms: _,
        } = evidence
        else {
            return Err("target registry readiness returned the wrong evidence kind".to_owned());
        };
        assert_eq!(
            readiness_url,
            format!("http://127.0.0.1:{}/readiness", endpoint.port)
        );
        assert_eq!(
            registry_url,
            format!("http://127.0.0.1:{}/workers", endpoint.port)
        );
        assert_eq!(attempts, 1);
        assert_eq!(
            timing.budget,
            crate::operation_bound::OperationBudgetEvidence::Unbounded
        );
        assert_eq!(
            timing.terminal_cause,
            crate::operation_bound::OperationTerminalCause::Succeeded
        );
        assert_eq!(diagnostic_attempts.len(), 3);
        assert!(diagnostic_attempts.iter().all(|attempt| {
            attempt.succeeded && (1..=1_000).contains(&attempt.effective_bound_ms)
        }));
        assert_eq!(
            matched_targets,
            vec![
                TargetRegistryMatchEvidence {
                    url: "http://prefill:30000".to_owned(),
                    role: "prefill".to_owned(),
                    healthy: true,
                    bootstrap_port: Some(8998),
                },
                TargetRegistryMatchEvidence {
                    url: "http://decode:30001".to_owned(),
                    role: "decode".to_owned(),
                    healthy: true,
                    bootstrap_port: None,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn finite_target_registry_attempts_record_the_resolved_attempt_budget() -> Result<(), String> {
        let (endpoint, server) = target_registry_endpoint(
            serde_json::json!({
                "workers": [{
                    "url": "http://decode:30001",
                    "worker_type": "decode",
                    "is_healthy": true
                }]
            })
            .to_string(),
        )?;
        let expected = vec![TargetRegistryExpectedTarget {
            url: "http://decode:30001".to_owned(),
            role: "decode".to_owned(),
            bootstrap_port: None,
        }];

        let bound = OperationBound::finite(Duration::from_secs(2));
        let evidence = wait_http_target_registry_ready(
            |_| alive_status(),
            &endpoint,
            target_registry_probe(&expected),
            1,
            &bound,
            &mut |_| {},
        )
        .map_err(|failure| failure.message)?;
        server
            .join()
            .map_err(|_| "target registry fixture panicked".to_owned())??;

        let ReadinessEvidence::HttpTargetRegistry {
            timing,
            diagnostic_attempts,
            ..
        } = evidence
        else {
            return Err("target registry readiness returned the wrong evidence kind".to_owned());
        };
        assert_eq!(
            timing.budget,
            crate::operation_bound::OperationBudgetEvidence::Finite {
                configured_ms: 2_000,
            }
        );
        assert_eq!(diagnostic_attempts.len(), 3);
        assert!(diagnostic_attempts.iter().all(|attempt| {
            attempt.succeeded && (1..=1_000).contains(&attempt.effective_bound_ms)
        }));
        Ok(())
    }

    #[test]
    fn target_registry_readiness_rejects_partial_registration() -> Result<(), String> {
        let (endpoint, server) = target_registry_endpoint(
            serde_json::json!({
                "workers": [{
                    "url": "http://prefill:30000",
                    "worker_type": "prefill",
                    "is_healthy": true,
                    "bootstrap_port": 8998
                }]
            })
            .to_string(),
        )?;
        let expected = vec![
            TargetRegistryExpectedTarget {
                url: "http://prefill:30000".to_owned(),
                role: "prefill".to_owned(),
                bootstrap_port: Some(8998),
            },
            TargetRegistryExpectedTarget {
                url: "http://decode:30001".to_owned(),
                role: "decode".to_owned(),
                bootstrap_port: None,
            },
        ];

        let mut probe_failures = Vec::new();
        let bound = OperationBound::finite(Duration::from_secs(1));
        let failure = match wait_http_target_registry_ready(
            |_| alive_status(),
            &endpoint,
            target_registry_probe(&expected),
            1,
            &bound,
            &mut |failure| probe_failures.push(failure.to_owned()),
        ) {
            Err(failure) => failure,
            Ok(evidence) => {
                return Err(format!(
                    "partial target registration unexpectedly became ready: {evidence:?}"
                ));
            }
        };
        server
            .join()
            .map_err(|_| "target registry fixture panicked".to_owned())??;

        assert_eq!(failure.kind, ReadinessFailureKind::Timeout);
        let timing = failure
            .timing
            .as_ref()
            .ok_or_else(|| "readiness timeout has no timing evidence".to_owned())?;
        assert_eq!(
            timing.budget,
            crate::operation_bound::OperationBudgetEvidence::Finite {
                configured_ms: 1_000,
            }
        );
        assert_eq!(timing.terminal_cause, OperationTerminalCause::TimedOut);
        assert!(probe_failures.iter().any(|failure| {
            failure.contains("target registry has no \"decode\" target at \"http://decode:30001\"")
        }));
        Ok(())
    }

    #[test]
    fn termination_waits_for_the_group_after_the_launcher_exits() -> Result<(), String> {
        let mut child = Command::new("sh")
            .args([
                "-c",
                "trap 'exit 0' TERM; sh -c 'trap \"\" TERM; exec sleep 30' & wait",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|error| error.to_string())?;
        let handle = HostProcessHandle::new(child.id(), None)?;
        thread::sleep(Duration::from_millis(100));
        let reaper = thread::spawn(move || child.wait());

        let cleanup = terminate_local(&handle, CleanupTrigger::Stop);
        if !cleanup.verified {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{}", handle.process_group)])
                .status();
        }
        let _ = reaper.join();

        assert!(cleanup.verified, "{cleanup:?}");
        assert!(cleanup.forced);
        assert!(cleanup.elapsed_ms >= cleanup.term_grace_ms);
        assert_eq!(cleanup.status_deadline_ms, 2_000);
        assert_eq!(cleanup.term_grace_ms, 2_000);
        assert_eq!(cleanup.kill_grace_ms, 10_000);
        Ok(())
    }

    #[test]
    fn rejects_a_reused_process_identity() -> Result<(), Box<dyn std::error::Error>> {
        let pid = std::process::id();
        let actual = process_start_time(pid)
            .map_err(std::io::Error::other)?
            .ok_or_else(|| std::io::Error::other("test process has no /proc identity"))?;
        let recorded = if actual == u64::MAX {
            actual - 1
        } else {
            actual + 1
        };
        let status = verified_local_status(&HostProcessHandle {
            leader_pid: pid,
            process_group: pid,
            leader_start_time_ticks: recorded,
            container: None,
        });

        assert!(status.queried);
        assert!(!status.alive);
        assert!(
            status
                .error
                .is_some_and(|error| error.contains("pid was reused"))
        );
        Ok(())
    }
}
