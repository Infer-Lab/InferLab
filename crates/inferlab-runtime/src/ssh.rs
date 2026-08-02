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
    let mut argv: Vec<String> = ["ssh"]
        .into_iter()
        .map(str::to_owned)
        .chain(SSH_OPTIONS.iter().map(|option| (*option).to_owned()))
        .collect();
    argv.extend([
        target.to_owned(),
        "bash".to_owned(),
        "-lic".to_owned(),
        shell_quote(script),
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
