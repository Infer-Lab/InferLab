use crate::operation_bound::{OperationBound, Remaining};
use serde::{Deserialize, Serialize};
use std::process::{Child, Output};
use std::thread;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProcessGroupError {
    #[error("local process-group identity requires non-zero identifiers")]
    ZeroIdentity,
    #[error("local process-group leader must equal its process group")]
    LeaderMismatch,
    #[error("process {pid} exited before its identity could be recorded")]
    ExitedBeforeCapture { pid: u32 },
    #[error("failed to reap process-group leader: {source}")]
    Reap {
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read {path}: {source}")]
    StatRead {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid process stat for pid {pid}")]
    InvalidStat { pid: u32 },
    #[error("process stat for pid {pid} has no start time")]
    MissingStartTime { pid: u32 },
    #[error("invalid process start time for pid {pid}: {source}")]
    InvalidStartTime {
        pid: u32,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error("process-group query exited with {status}: {stderr}")]
    QueryExit {
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error("process-group cleanup command deadline expired")]
    Deadline,
    #[error("process-group cleanup command was interrupted")]
    Interrupted,
    #[error("process-group cleanup command failed to launch: {source}")]
    Launch {
        #[source]
        source: std::io::Error,
    },
    #[error("process-group cleanup command failed: {source}")]
    CommandIo {
        #[source]
        source: std::io::Error,
    },
    #[error("process-group cleanup command wait failed: {source}; cleanup verification: {cleanup}")]
    WaitCleanup {
        #[source]
        source: std::io::Error,
        cleanup: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationSignal {
    Term,
    Kill,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignalEvidence {
    pub signal: TerminationSignal,
    pub process_group: u32,
    pub exit_code: Option<i32>,
    pub stderr: Option<String>,
    pub error: Option<String>,
}

impl SignalEvidence {
    pub fn succeeded(&self) -> bool {
        self.error.is_none() && self.exit_code == Some(0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LocalProcessGroup {
    pub leader_pid: u32,
    pub process_group: u32,
    pub leader_start_time_ticks: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifiedStatus {
    Alive,
    Exited,
    Reused,
    LeaderMissingWithMembers,
}

impl LocalProcessGroup {
    pub const fn unverified(process_group: u32) -> Self {
        Self {
            leader_pid: process_group,
            process_group,
            leader_start_time_ticks: 0,
        }
    }

    pub fn new(
        leader_pid: u32,
        process_group: u32,
        leader_start_time_ticks: u64,
    ) -> Result<Self, ProcessGroupError> {
        if leader_pid == 0 || process_group == 0 {
            return Err(ProcessGroupError::ZeroIdentity);
        }
        if leader_pid != process_group {
            return Err(ProcessGroupError::LeaderMismatch);
        }
        Ok(Self {
            leader_pid,
            process_group,
            leader_start_time_ticks,
        })
    }

    pub fn capture_child(child: &Child) -> Result<Self, ProcessGroupError> {
        let leader_pid = child.id();
        let leader_start_time_ticks = process_start_time(leader_pid)?
            .ok_or(ProcessGroupError::ExitedBeforeCapture { pid: leader_pid })?;
        Self::new(leader_pid, leader_pid, leader_start_time_ticks)
    }

    pub fn identity_matches(&self) -> bool {
        process_start_time(self.leader_pid)
            .ok()
            .flatten()
            .is_some_and(|current| current == self.leader_start_time_ticks)
    }

    pub fn verified_status(
        &self,
        bound: &OperationBound,
    ) -> Result<VerifiedStatus, ProcessGroupError> {
        let leader_start = process_start_time(self.leader_pid)?;
        match leader_start {
            Some(actual) if actual != self.leader_start_time_ticks => Ok(VerifiedStatus::Reused),
            Some(_) => {
                let alive = self.has_live_members(bound)?;
                Ok(if alive {
                    VerifiedStatus::Alive
                } else {
                    VerifiedStatus::Exited
                })
            }
            None => {
                let alive = self.has_live_members(bound)?;
                Ok(if alive {
                    VerifiedStatus::LeaderMissingWithMembers
                } else {
                    VerifiedStatus::Exited
                })
            }
        }
    }

    pub fn send_signal(&self, signal: TerminationSignal, bound: &OperationBound) -> SignalEvidence {
        let signal_argument = match signal {
            TerminationSignal::Term => "-TERM",
            TerminationSignal::Kill => "-KILL",
        };
        let target = format!("-{}", self.process_group);
        match cleanup_output(&["kill", signal_argument, "--", &target], bound) {
            Ok(output) => SignalEvidence {
                signal,
                process_group: self.process_group,
                exit_code: output.status.code(),
                stderr: Some(String::from_utf8_lossy(&output.stderr).trim().to_owned()),
                error: None,
            },
            Err(error) => SignalEvidence {
                signal,
                process_group: self.process_group,
                exit_code: None,
                stderr: None,
                error: Some(error.to_string()),
            },
        }
    }

    pub fn wait_until_stopped(
        &self,
        mut child: Option<&mut Child>,
        bound: &OperationBound,
        poll_interval: Duration,
    ) -> Result<bool, ProcessGroupError> {
        loop {
            if let Some(child) = child.as_deref_mut() {
                child
                    .try_wait()
                    .map_err(|source| ProcessGroupError::Reap { source })?;
            }
            if bound.is_expired() {
                return Ok(false);
            }
            match self.has_live_members(bound) {
                Ok(false) => return Ok(true),
                Ok(true) => {}
                Err(_) if bound.is_expired() => return Ok(false),
                Err(error) => return Err(error),
            }
            match bound.remaining() {
                Remaining::Finite(remaining) => {
                    thread::sleep(poll_interval.min(remaining));
                }
                Remaining::Expired => return Ok(false),
                Remaining::Unbounded => thread::sleep(poll_interval),
            }
        }
    }

    pub fn has_live_members(&self, bound: &OperationBound) -> Result<bool, ProcessGroupError> {
        let output = cleanup_output(&["ps", "-eo", "pid=,pgid=,stat="], bound)?;
        if !output.status.success() {
            return Err(ProcessGroupError::QueryExit {
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            });
        }
        let process_group = self.process_group.to_string();
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let _pid = fields.next()?;
                let group = fields.next()?;
                let state = fields.next()?;
                Some((group, state))
            })
            .any(|(group, state)| group == process_group && !state.starts_with('Z')))
    }
}

pub fn process_start_time(pid: u32) -> Result<Option<u64>, ProcessGroupError> {
    let path = format!("/proc/{pid}/stat");
    let stat = match std::fs::read_to_string(&path) {
        Ok(stat) => stat,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(ProcessGroupError::StatRead { path, source }),
    };
    let command_end = stat
        .rfind(')')
        .ok_or(ProcessGroupError::InvalidStat { pid })?;
    let start_time = stat[command_end + 1..]
        .split_whitespace()
        .nth(19)
        .ok_or(ProcessGroupError::MissingStartTime { pid })?
        .parse::<u64>()
        .map_err(|source| ProcessGroupError::InvalidStartTime { pid, source })?;
    Ok(Some(start_time))
}

fn cleanup_output(argv: &[&str], bound: &OperationBound) -> Result<Output, ProcessGroupError> {
    match crate::container::run_cleanup_with_bound(argv, None, None, bound, None) {
        Ok(crate::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Ok(crate::container::BoundedWait::Expired { .. }) => Err(ProcessGroupError::Deadline),
        Ok(crate::container::BoundedWait::Interrupted { .. }) => {
            Err(ProcessGroupError::Interrupted)
        }
        Err(crate::container::BoundedError::Launch(source)) => {
            Err(ProcessGroupError::Launch { source })
        }
        Err(
            crate::container::BoundedError::Stdin(error)
            | crate::container::BoundedError::Wait(error),
        ) => Err(ProcessGroupError::CommandIo { source: error }),
        Err(crate::container::BoundedError::WaitCleanup {
            source, cleanup, ..
        }) => Err(ProcessGroupError::WaitCleanup {
            source,
            cleanup: cleanup.error.unwrap_or_else(|| {
                if cleanup.verified {
                    "verified"
                } else {
                    "unverified"
                }
                .to_owned()
            }),
        }),
    }
}
