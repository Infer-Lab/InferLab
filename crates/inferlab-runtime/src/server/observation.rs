use super::{
    HostProcessHandle, LogSyncError, ProcessCommandError, ProcessHandle, ProcessObserver,
    ProcessStatus, REMOTE_LOG_SYNC_DEADLINE, SshProcessHandle, SystemProcessRuntime,
};
use crate::operation_bound::OperationBound;
use crate::process_group::process_start_time;
use crate::shell::shell_quote_path;
use crate::ssh::{SSH_ENV_REMOVE, ssh_argv, ssh_output};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

pub(super) fn verified_local_status(handle: &HostProcessHandle) -> ProcessStatus {
    verified_local_status_under(handle, None, false)
}

pub(super) fn verified_local_status_with_bound(
    handle: &HostProcessHandle,
    bound: &OperationBound,
) -> ProcessStatus {
    verified_local_status_under(handle, Some(bound), false)
}

pub(super) fn verified_local_status_under(
    handle: &HostProcessHandle,
    bound: Option<&OperationBound>,
    cleanup: bool,
) -> ProcessStatus {
    if let Err(error) = handle.validate() {
        return status_error(error);
    }
    match process_start_time(handle.leader_pid) {
        Ok(Some(actual)) if actual != handle.leader_start_time_ticks => ProcessStatus {
            queried: true,
            alive: false,
            error: Some(format!(
                "managed process {} exited and its pid was reused: recorded start time {}, observed {}",
                handle.leader_pid, handle.leader_start_time_ticks, actual
            )),
        },
        Ok(Some(_)) => {
            match process_group_has_live_members_under(handle.process_group, bound, cleanup) {
                Ok(alive) => ProcessStatus {
                    queried: true,
                    alive,
                    error: None,
                },
                Err(error) => status_error(error.to_string()),
            }
        }
        Ok(None) => {
            match process_group_has_live_members_under(handle.process_group, bound, cleanup) {
                Ok(false) => ProcessStatus {
                    queried: true,
                    alive: false,
                    error: None,
                },
                Ok(true) => status_error(format!(
                    "process-group {} still has members but recorded leader {} no longer exists; ownership cannot be verified",
                    handle.process_group, handle.leader_pid
                )),
                Err(error) => status_error(error.to_string()),
            }
        }
        Err(error) => status_error(error.to_string()),
    }
}

pub(super) fn verified_ssh_status(handle: &SshProcessHandle) -> ProcessStatus {
    verified_ssh_status_under(handle, None)
}

pub(super) fn verified_ssh_status_with_bound(
    handle: &SshProcessHandle,
    bound: &OperationBound,
) -> ProcessStatus {
    verified_ssh_status_under(handle, Some(bound))
}

pub(super) fn verified_ssh_status_under(
    handle: &SshProcessHandle,
    bound: Option<&OperationBound>,
) -> ProcessStatus {
    if let Err(error) = handle.validate() {
        return status_error(error);
    }
    let script = format!(
        "set -eu; pid={}; expected={}; if [ -r /proc/$pid/stat ]; then actual=$(awk '{{print $22}}' /proc/$pid/stat); if [ \"$actual\" != \"$expected\" ]; then printf 'stale %s\\n' \"$actual\"; exit 4; fi; elif {}; then printf 'unknown leader-missing\\n'; exit 5; else printf 'dead\\n'; exit 3; fi; if {}; then printf 'alive\\n'; exit 0; fi; printf 'dead\\n'; exit 3",
        handle.leader_pid,
        handle.leader_start_time_ticks,
        remote_group_alive_script(&handle.process_group.to_string()),
        remote_group_alive_script(&handle.process_group.to_string()),
    );
    let output = match bound {
        Some(bound) => {
            run_status_command(&ssh_argv(&handle.target, &script), SSH_ENV_REMOVE, bound)
        }
        None => ssh_output(&handle.target, &script).map_err(|source| ProcessCommandError::Ssh {
            operation: "process status command".to_owned(),
            source,
        }),
    };
    match output {
        Ok(output) if output.status.success() => ProcessStatus {
            queried: true,
            alive: true,
            error: None,
        },
        Ok(output) if output.status.code() == Some(3) => ProcessStatus {
            queried: true,
            alive: false,
            error: None,
        },
        Ok(output) if output.status.code() == Some(4) => ProcessStatus {
            queried: true,
            alive: false,
            error: Some(format!(
                "managed SSH process {} exited and its pid was reused: {}",
                handle.leader_pid,
                String::from_utf8_lossy(&output.stdout).trim()
            )),
        },
        Ok(output) if output.status.code() == Some(5) => status_error(format!(
            "SSH process-group {} ownership could not be verified: {}",
            handle.process_group,
            String::from_utf8_lossy(&output.stdout).trim()
        )),
        Ok(output) => status_error(format!(
            "SSH status exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(error) => status_error(error.to_string()),
    }
}

fn status_error(error: String) -> ProcessStatus {
    ProcessStatus {
        queried: false,
        alive: false,
        error: Some(error),
    }
}

pub(super) fn remote_group_alive_script(group: &str) -> String {
    format!(
        "ps -eo pgid=,stat= | awk -v pgid={group} '$1 == pgid && $2 !~ /^Z/ {{ found=1 }} END {{ exit !found }}'"
    )
}

pub(super) fn fetch_remote_file(
    target: &str,
    remote: &Path,
    local: &Path,
    bound: &OperationBound,
    cleanup: bool,
) -> Result<(), LogSyncError> {
    let argv = ssh_argv(target, &format!("cat -- {}", shell_quote_path(remote)));
    let output = if cleanup {
        run_cleanup_command(&argv, SSH_ENV_REMOVE, bound, "remote log synchronization")
    } else {
        run_status_command(&argv, SSH_ENV_REMOVE, bound)
    }
    .map_err(|source| LogSyncError::ReadRemote {
        path: remote.to_path_buf(),
        source,
    })?;
    if !output.status.success() {
        return Err(LogSyncError::RemoteExit {
            path: remote.to_path_buf(),
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    fs::write(local, output.stdout).map_err(|source| LogSyncError::WriteLocal {
        path: local.to_path_buf(),
        source,
    })
}

fn process_group_has_live_members_under(
    process_group: u32,
    bound: Option<&OperationBound>,
    cleanup: bool,
) -> Result<bool, ProcessCommandError> {
    let argv = ["ps", "-eo", "pid=,pgid=,stat="];
    let output = match bound {
        Some(bound) if cleanup => run_cleanup_command(&argv, &[], bound, "process cleanup status"),
        Some(bound) => run_status_command(&argv, &[], bound),
        None => Command::new(argv[0])
            .args(&argv[1..])
            .output()
            .map_err(|source| ProcessCommandError::Launch {
                operation: "process-group query".to_owned(),
                source,
            }),
    }?;
    if !output.status.success() {
        return Err(ProcessCommandError::Exit {
            operation: "process-group query".to_owned(),
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let process_group = process_group.to_string();
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

pub(super) fn run_status_command<S: AsRef<std::ffi::OsStr>>(
    argv: &[S],
    env_remove: &[&str],
    bound: &OperationBound,
) -> Result<Output, ProcessCommandError> {
    let operation = "process status command";
    match crate::container::run_with_bound(argv, env_remove, None, None, bound, None) {
        Ok(crate::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Ok(crate::container::BoundedWait::Expired { kill, .. }) => {
            kill.map_err(|source| ProcessCommandError::Io {
                operation: "process status cleanup".to_owned(),
                source,
            })?;
            Err(ProcessCommandError::Deadline {
                operation: "process status attempt".to_owned(),
            })
        }
        Ok(crate::container::BoundedWait::Interrupted { kill, .. }) => {
            kill.map_err(|source| ProcessCommandError::Io {
                operation: "process status cleanup".to_owned(),
                source,
            })?;
            Err(ProcessCommandError::Interrupted {
                operation: "process status attempt".to_owned(),
            })
        }
        Err(crate::container::BoundedError::Launch(source)) => Err(ProcessCommandError::Launch {
            operation: operation.to_owned(),
            source,
        }),
        Err(
            crate::container::BoundedError::Stdin(error)
            | crate::container::BoundedError::Wait(error),
        ) => Err(ProcessCommandError::Io {
            operation: operation.to_owned(),
            source: error,
        }),
        Err(crate::container::BoundedError::WaitCleanup {
            source, cleanup, ..
        }) => Err(ProcessCommandError::WaitCleanup {
            operation: operation.to_owned(),
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

pub(super) fn run_cleanup_command<S: AsRef<std::ffi::OsStr>>(
    argv: &[S],
    env_remove: &[&str],
    bound: &OperationBound,
    operation: &str,
) -> Result<Output, ProcessCommandError> {
    match crate::container::run_cleanup_with_bound(argv, env_remove, None, None, bound, None) {
        Ok(crate::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Ok(crate::container::BoundedWait::Expired { kill, .. }) => {
            kill.map_err(|source| ProcessCommandError::Io {
                operation: format!("{operation} child cleanup"),
                source,
            })?;
            Err(ProcessCommandError::Deadline {
                operation: operation.to_owned(),
            })
        }
        Ok(crate::container::BoundedWait::Interrupted { kill, .. }) => {
            kill.map_err(|source| ProcessCommandError::Io {
                operation: format!("{operation} child cleanup"),
                source,
            })?;
            Err(ProcessCommandError::Interrupted {
                operation: operation.to_owned(),
            })
        }
        Err(crate::container::BoundedError::Launch(source)) => Err(ProcessCommandError::Launch {
            operation: operation.to_owned(),
            source,
        }),
        Err(
            crate::container::BoundedError::Stdin(error)
            | crate::container::BoundedError::Wait(error),
        ) => Err(ProcessCommandError::Io {
            operation: operation.to_owned(),
            source: error,
        }),
        Err(crate::container::BoundedError::WaitCleanup {
            source, cleanup, ..
        }) => Err(ProcessCommandError::WaitCleanup {
            operation: operation.to_owned(),
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

impl ProcessObserver for SystemProcessRuntime {
    fn status(&self, handle: &ProcessHandle) -> ProcessStatus {
        match handle {
            ProcessHandle::Local(handle) => verified_local_status(handle),
            ProcessHandle::Ssh(handle) => verified_ssh_status(handle),
        }
    }

    fn status_with_bound(&self, handle: &ProcessHandle, bound: &OperationBound) -> ProcessStatus {
        match handle {
            ProcessHandle::Local(handle) => verified_local_status_with_bound(handle, bound),
            ProcessHandle::Ssh(handle) => verified_ssh_status_with_bound(handle, bound),
        }
    }

    fn sync_logs(
        &self,
        handle: &ProcessHandle,
        stdout: &Path,
        stderr: &Path,
        cleanup: bool,
    ) -> Result<(), LogSyncError> {
        match handle {
            ProcessHandle::Local(_) => Ok(()),
            ProcessHandle::Ssh(handle) => {
                let bound = OperationBound::finite(REMOTE_LOG_SYNC_DEADLINE);
                fetch_remote_file(&handle.target, &handle.stdout, stdout, &bound, cleanup)?;
                fetch_remote_file(&handle.target, &handle.stderr, stderr, &bound, cleanup)
            }
        }
    }
}
