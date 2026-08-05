//! The one authority for the SSH client invocation shape, shared by every
//! module that reaches a remote machine.

use crate::shell::shell_quote;
use crate::{
    container::{BoundedError, BoundedWait, CommandCleanupEvidence, run_with_bound},
    operation_bound::OperationBound,
};
use std::process::Output;
use thiserror::Error;

const SSH_OPTIONS: &[&str] = &["-o", "BatchMode=yes", "--"];

/// The full SSH argv for bounded execution through an owning operation or
/// cleanup deadline.
pub fn ssh_argv(target: &str, script: &str) -> Vec<String> {
    // Keep the remote account's interactive login initialization, which owns
    // tool discovery and declared pass-through values, but replace that shell
    // before the InferLab script runs. The non-login replacement preserves the
    // initialized environment without running `.bash_logout`, whose exit status
    // must not replace the remote operation's result.
    let remote_command = format!("exec \"$BASH\" -c {}", shell_quote(script));
    let mut argv: Vec<String> = ["ssh"]
        .into_iter()
        .map(str::to_owned)
        .chain(SSH_OPTIONS.iter().map(|option| (*option).to_owned()))
        .collect();
    argv.extend([
        target.to_owned(),
        "bash".to_owned(),
        "-lic".to_owned(),
        shell_quote(&remote_command),
    ]);
    argv
}

#[derive(Debug, Error)]
pub enum SshError {
    #[error("failed to launch SSH for {target:?}: {source}")]
    Launch {
        target: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to {operation} for SSH target {target:?}: {source}")]
    Io {
        target: String,
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "failed while waiting for SSH target {target:?} after {operation_elapsed_ms} ms: {source}; child cleanup: {cleanup:?}"
    )]
    WaitCleanup {
        target: String,
        operation_elapsed_ms: u64,
        #[source]
        source: std::io::Error,
        cleanup: Box<CommandCleanupEvidence>,
    },
    #[error(
        "SSH for target {target:?} was interrupted after {operation_elapsed_ms} ms; child cleanup: {cleanup:?}"
    )]
    Interrupted {
        target: String,
        operation_elapsed_ms: u64,
        cleanup: Box<CommandCleanupEvidence>,
    },
    #[error(
        "failed to clean up interrupted SSH for target {target:?} after {operation_elapsed_ms} ms: {source}; child cleanup: {cleanup:?}"
    )]
    InterruptCleanup {
        target: String,
        operation_elapsed_ms: u64,
        #[source]
        source: std::io::Error,
        cleanup: Box<CommandCleanupEvidence>,
    },
    #[error(
        "SSH supervisor for target {target:?} unexpectedly exhausted an unbounded operation after {operation_elapsed_ms} ms; child cleanup: {cleanup:?}"
    )]
    UnexpectedDeadline {
        target: String,
        operation_elapsed_ms: u64,
        cleanup: Option<Box<CommandCleanupEvidence>>,
    },
}

pub fn ssh_output(target: &str, script: &str) -> Result<Output, SshError> {
    run_ssh(target, script, None)
}

pub fn ssh_output_with_input(target: &str, script: &str, input: &[u8]) -> Result<Output, SshError> {
    run_ssh(target, script, Some(input))
}

fn run_ssh(target: &str, script: &str, input: Option<&[u8]>) -> Result<Output, SshError> {
    let argv = ssh_argv(target, script);
    match run_with_bound(&argv, None, input, &OperationBound::unbounded(), None) {
        Ok(BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Ok(BoundedWait::Expired {
            kill,
            operation_elapsed_ms,
            cleanup,
        }) => {
            kill.map_err(|source| SshError::Io {
                target: target.to_owned(),
                operation: "clean up SSH after unexpected deadline",
                source,
            })?;
            Err(SshError::UnexpectedDeadline {
                target: target.to_owned(),
                operation_elapsed_ms,
                cleanup: cleanup.map(Box::new),
            })
        }
        Ok(BoundedWait::Interrupted {
            kill,
            operation_elapsed_ms,
            cleanup,
        }) => match kill {
            Ok(()) => Err(SshError::Interrupted {
                target: target.to_owned(),
                operation_elapsed_ms,
                cleanup: Box::new(cleanup),
            }),
            Err(source) => Err(SshError::InterruptCleanup {
                target: target.to_owned(),
                operation_elapsed_ms,
                source,
                cleanup: Box::new(cleanup),
            }),
        },
        Err(BoundedError::Launch(source)) => Err(SshError::Launch {
            target: target.to_owned(),
            source,
        }),
        Err(BoundedError::Stdin(source)) => Err(SshError::Io {
            target: target.to_owned(),
            operation: "write SSH stdin",
            source,
        }),
        Err(BoundedError::Wait(source)) => Err(SshError::Io {
            target: target.to_owned(),
            operation: "wait for SSH",
            source,
        }),
        Err(BoundedError::WaitCleanup {
            source,
            operation_elapsed_ms,
            cleanup,
        }) => Err(SshError::WaitCleanup {
            target: target.to_owned(),
            operation_elapsed_ms,
            source,
            cleanup: Box::new(cleanup),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::ssh_argv;
    use std::error::Error;
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::process::{Command, Output};

    #[test]
    fn ssh_argv_leaves_connection_policy_to_openssh_configuration() {
        let argv = ssh_argv("fixture-target", "exit 0");

        assert_eq!(
            &argv[..5],
            ["ssh", "-o", "BatchMode=yes", "--", "fixture-target"]
        );
        assert!(!argv.iter().any(|argument| {
            argument.starts_with("ConnectTimeout=")
                || argument.starts_with("ServerAliveInterval=")
                || argument.starts_with("ServerAliveCountMax=")
        }));
    }

    fn run_remote_command(home: &Path, script: &str) -> Result<Output, Box<dyn Error>> {
        let argv = ssh_argv("fixture-target", script);
        let separator = argv
            .iter()
            .position(|argument| argument == "--")
            .ok_or_else(|| io::Error::other("SSH argv has no option separator"))?;
        let remote_command = argv
            .get(separator + 2..)
            .ok_or_else(|| io::Error::other("SSH argv has no remote command"))?
            .join(" ");
        Ok(Command::new("sh")
            .args(["-c", &remote_command])
            .env("HOME", home)
            .output()?)
    }

    #[test]
    fn login_shell_teardown_does_not_override_remote_script_result() -> Result<(), Box<dyn Error>> {
        let home = tempfile::tempdir()?;
        fs::write(
            home.path().join(".bash_profile"),
            "export INFERLAB_LOGIN_MARKER=loaded\n",
        )?;
        fs::write(
            home.path().join(".bash_logout"),
            "printf 'logout-ran\\n' >&2\nfalse\n",
        )?;

        let alive = run_remote_command(
            home.path(),
            "set -eu; printf '%s\\n' \"$INFERLAB_LOGIN_MARKER\"; printf 'probe-stderr\\n' >&2; exit 0",
        )?;
        assert!(alive.status.success());
        assert_eq!(String::from_utf8(alive.stdout)?, "loaded\n");
        let alive_stderr = String::from_utf8(alive.stderr)?;
        assert!(alive_stderr.contains("probe-stderr"));
        assert!(!alive_stderr.contains("logout-ran"));

        let dead = run_remote_command(home.path(), "set -eu; printf 'dead\\n'; exit 3")?;
        assert_eq!(dead.status.code(), Some(3));
        assert_eq!(String::from_utf8(dead.stdout)?, "dead\n");
        Ok(())
    }
}
