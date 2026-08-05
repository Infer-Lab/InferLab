//! The one authority for the SSH client invocation shape, shared by every
//! module that reaches a remote machine.

use crate::shell::shell_quote;
use std::process::{Command, Output};
use thiserror::Error;

const SSH_OPTIONS: &[&str] = &[
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=2",
    "--",
];

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
}

pub fn ssh_output(target: &str, script: &str) -> Result<Output, SshError> {
    let argv = ssh_argv(target, script);
    Command::new(&argv[0])
        .args(&argv[1..])
        .output()
        .map_err(|source| SshError::Launch {
            target: target.to_owned(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use super::ssh_argv;
    use std::error::Error;
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::process::{Command, Output};

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
