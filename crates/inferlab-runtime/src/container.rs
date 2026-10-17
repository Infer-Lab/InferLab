//! One authority for docker container lifecycle mechanics
//! ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): bounded client invocation with
//! concurrently drained pipes, and confirmed container removal on the
//! launch machine. Callers own their argv assembly, evidence shapes, and
//! operator-facing messages — this module owns only the mechanics that
//! have actually failed in the field: the pipe-capacity deadlock, the
//! deadline kill, and the container that outlives its docker client.

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

use crate::operation_bound::{OperationBound, Remaining, duration_millis};

const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long a command's output may stay open after the command exits.
pub const COMMAND_IO_DRAIN_GRACE: Duration = Duration::from_secs(5);

/// The `--gpus` argv pair for a device spec (a single index or a
/// comma-joined list). The literal quotes are part of the value: docker's
/// `--gpus` parser splits an unquoted `device=0,1` on the comma and exposes
/// only the first device (verified on real hardware in v1.5).
pub fn docker_device_args(spec: &str) -> [String; 2] {
    ["--gpus".to_owned(), format!("\"device={spec}\"")]
}

/// The `--mount` argv pair for a read-only host bind. The explicit long
/// form, not the `-v` shorthand: at least one site docker proxy mis-parses
/// the shorthand's `:ro` suffix on same-path binds and silently drops the
/// mount (verified on real hardware), while the long form passes through.
pub fn docker_bind_mount_readonly(source: &str, target: &str) -> [String; 2] {
    [
        "--mount".to_owned(),
        format!("type=bind,source={source},target={target},readonly"),
    ]
}

/// A bounded invocation that ran to an observable end.
pub enum BoundedWait {
    Exited {
        status: ExitStatus,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    /// The effective wait bound expired: the client was killed and reaped.
    /// The caller remains responsible for deciding whether the owning
    /// operation expired or a shorter subordinate attempt cap fired.
    Expired {
        kill: std::io::Result<()>,
        operation_elapsed_ms: u64,
        cleanup: Option<CommandCleanupEvidence>,
    },
    /// A new operator interruption arrived after this command started. The
    /// owned process group was terminated and reaped before returning.
    Interrupted {
        kill: std::io::Result<()>,
        operation_elapsed_ms: u64,
        cleanup: CommandCleanupEvidence,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommandCleanupEvidence {
    pub trigger: CommandCleanupTrigger,
    pub elapsed_ms: u64,
    /// The wait until every thread of the killed command's process group
    /// exited ([[RFC-0009:C-COMPLETION-WAITS]]).
    pub exit_wait: Option<Box<crate::completion::CompletionWaitEvidence>>,
    pub io_drain_grace_ms: u64,
    pub kill_attempted: bool,
    pub verified: bool,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandCleanupTrigger {
    Deadline,
    Interruption,
    WaitFailure,
}

/// Where a bounded invocation failed before producing a wait outcome.
pub enum BoundedError {
    Launch(std::io::Error),
    Stdin(std::io::Error),
    Wait(std::io::Error),
    WaitCleanup {
        source: std::io::Error,
        operation_elapsed_ms: u64,
        cleanup: CommandCleanupEvidence,
    },
}

/// Spawn `argv`, optionally write `stdin_payload`, and wait under the
/// operation's remaining budget and optional attempt cap while draining
/// stdout and stderr concurrently: a child whose
/// output exceeds the pipe capacity would otherwise block writing and
/// never exit, turning a large response or traceback into a timeout.
/// `env_remove` names inherited environment variables the child must not
/// see (for example [`crate::ssh::SSH_ENV_REMOVE`] for a host SSH client).
pub fn run_with_bound<S: AsRef<std::ffi::OsStr>>(
    argv: &[S],
    env_remove: &[&str],
    cwd: Option<&Path>,
    stdin_payload: Option<&[u8]>,
    bound: &OperationBound,
    attempt_cap: Option<Duration>,
) -> Result<BoundedWait, BoundedError> {
    run_with_bound_mode(
        argv,
        env_remove,
        cwd,
        stdin_payload,
        bound,
        attempt_cap,
        true,
        crate::interrupt::received,
    )
}

pub fn run_cleanup_with_bound<S: AsRef<std::ffi::OsStr>>(
    argv: &[S],
    env_remove: &[&str],
    cwd: Option<&Path>,
    stdin_payload: Option<&[u8]>,
    bound: &OperationBound,
    attempt_cap: Option<Duration>,
) -> Result<BoundedWait, BoundedError> {
    run_with_bound_mode(
        argv,
        env_remove,
        cwd,
        stdin_payload,
        bound,
        attempt_cap,
        false,
        crate::interrupt::received,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_with_bound_mode<S: AsRef<std::ffi::OsStr>, F: FnMut() -> bool>(
    argv: &[S],
    env_remove: &[&str],
    cwd: Option<&Path>,
    stdin_payload: Option<&[u8]>,
    bound: &OperationBound,
    attempt_cap: Option<Duration>,
    interruptible: bool,
    mut interrupted: F,
) -> Result<BoundedWait, BoundedError> {
    let attempt = bound.attempt(attempt_cap);
    if matches!(attempt.remaining(), Remaining::Expired) {
        return Ok(BoundedWait::Expired {
            kill: Ok(()),
            operation_elapsed_ms: bound.elapsed_ms(),
            cleanup: None,
        });
    }
    let Some(program) = argv.first() else {
        return Err(BoundedError::Launch(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "external command argv is empty",
        )));
    };
    let mut command = Command::new(program);
    command
        .args(&argv[1..])
        .stdin(if stdin_payload.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    for variable in env_remove {
        command.env_remove(variable);
    }
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let mut child = command.spawn().map_err(BoundedError::Launch)?;
    let stdin_write = if let Some(payload) = stdin_payload {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| BoundedError::Stdin(std::io::Error::other("stdin was not piped")))?;
        let payload = payload.to_owned();
        Some(std::thread::spawn(move || {
            stdin.write_all(&payload).and_then(|()| stdin.flush())
        }))
    } else {
        None
    };
    let mut stdout_pipe = child.stdout.take();
    let stdout_drain = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buffer);
        }
        buffer
    });
    let mut stderr_pipe = child.stderr.take();
    let stderr_drain = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buffer);
        }
        buffer
    });
    let status = loop {
        if interruptible && interrupted() {
            let operation_elapsed_ms = bound.elapsed_ms();
            let (kill, cleanup) = terminate_group_and_finish_io(
                &mut child,
                stdin_write,
                stdout_drain,
                stderr_drain,
                CommandCleanupTrigger::Interruption,
            );
            return Ok(BoundedWait::Interrupted {
                kill,
                operation_elapsed_ms,
                cleanup,
            });
        }
        let Some(wait) = wait_slice(&attempt) else {
            let operation_elapsed_ms = bound.elapsed_ms();
            let (kill, cleanup) = terminate_group_and_finish_io(
                &mut child,
                stdin_write,
                stdout_drain,
                stderr_drain,
                CommandCleanupTrigger::Deadline,
            );
            return Ok(BoundedWait::Expired {
                kill,
                operation_elapsed_ms,
                cleanup: Some(cleanup),
            });
        };
        match child.wait_timeout(wait) {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => {
                let operation_elapsed_ms = bound.elapsed_ms();
                let (_, cleanup) = terminate_group_and_finish_io(
                    &mut child,
                    stdin_write,
                    stdout_drain,
                    stderr_drain,
                    CommandCleanupTrigger::WaitFailure,
                );
                return Err(BoundedError::WaitCleanup {
                    source: error,
                    operation_elapsed_ms,
                    cleanup,
                });
            }
        }
    };
    while !io_finished(&stdin_write, &stdout_drain, &stderr_drain) {
        if interruptible && interrupted() {
            let operation_elapsed_ms = bound.elapsed_ms();
            let (kill, cleanup) = terminate_group_and_finish_io(
                &mut child,
                stdin_write,
                stdout_drain,
                stderr_drain,
                CommandCleanupTrigger::Interruption,
            );
            return Ok(BoundedWait::Interrupted {
                kill,
                operation_elapsed_ms,
                cleanup,
            });
        }
        let Some(wait) = wait_slice(&attempt) else {
            let operation_elapsed_ms = bound.elapsed_ms();
            let (kill, cleanup) = terminate_group_and_finish_io(
                &mut child,
                stdin_write,
                stdout_drain,
                stderr_drain,
                CommandCleanupTrigger::Deadline,
            );
            return Ok(BoundedWait::Expired {
                kill,
                operation_elapsed_ms,
                cleanup: Some(cleanup),
            });
        };
        std::thread::sleep(wait);
    }
    let stdin_result = join_writer_result(stdin_write);
    let stdout = join_drain(stdout_drain);
    let stderr = join_drain(stderr_drain);
    stdin_result.map_err(BoundedError::Stdin)?;
    let stdout = stdout?;
    let stderr = stderr?;
    Ok(BoundedWait::Exited {
        status,
        stdout,
        stderr,
    })
}

fn wait_slice(attempt: &crate::operation_bound::AttemptBound) -> Option<Duration> {
    match attempt.remaining() {
        Remaining::Finite(remaining) => Some(remaining.min(INTERRUPT_POLL_INTERVAL)),
        Remaining::Expired => None,
        Remaining::Unbounded => Some(INTERRUPT_POLL_INTERVAL),
    }
}

fn io_finished(
    writer: &Option<std::thread::JoinHandle<std::io::Result<()>>>,
    stdout: &std::thread::JoinHandle<Vec<u8>>,
    stderr: &std::thread::JoinHandle<Vec<u8>>,
) -> bool {
    writer.as_ref().is_none_or(|writer| writer.is_finished())
        && stdout.is_finished()
        && stderr.is_finished()
}

fn terminate_group_and_finish_io(
    child: &mut std::process::Child,
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
    stdout: std::thread::JoinHandle<Vec<u8>>,
    stderr: std::thread::JoinHandle<Vec<u8>>,
    trigger: CommandCleanupTrigger,
) -> (std::io::Result<()>, CommandCleanupEvidence) {
    let started = std::time::Instant::now();
    let (kill, exit_wait) = terminate_process_group(child);
    let result = match kill {
        Ok(()) => finish_io_with_grace(writer, stdout, stderr),
        Err(error) => Err(error),
    };
    let evidence = CommandCleanupEvidence {
        trigger,
        elapsed_ms: duration_millis(started.elapsed()),
        exit_wait,
        io_drain_grace_ms: duration_millis(COMMAND_IO_DRAIN_GRACE),
        kill_attempted: true,
        verified: result.is_ok(),
        error: result.as_ref().err().map(ToString::to_string),
    };
    (result, evidence)
}

/// SIGKILL the command's process group, wait until every thread of it has
/// exited, and reap the command, the group's leader.
fn terminate_process_group(
    child: &mut std::process::Child,
) -> (
    std::io::Result<()>,
    Option<Box<crate::completion::CompletionWaitEvidence>>,
) {
    let Some(pid) = rustix::process::Pid::from_raw(child.id() as i32) else {
        return (Err(std::io::Error::other("child process id is zero")), None);
    };
    let signal = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
    if signal.is_err() {
        match child.try_wait() {
            Ok(None) => {
                if let Err(error) = child.kill() {
                    return (Err(error), None);
                }
            }
            Ok(Some(_)) => {}
            Err(error) => return (Err(error), None),
        }
    }
    let wait = crate::completion::wait(
        &mut crate::completion::process::ProcessExit {
            target: None,
            processes: crate::completion::process::Processes::Group(child.id()),
        },
        &mut |_| {},
    );
    let result = match wait.failure() {
        Some(failure) => Err(std::io::Error::new(std::io::ErrorKind::TimedOut, failure)),
        // Every thread of the group, the command included, has exited.
        None => match child.try_wait() {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err(std::io::Error::other(
                "the command's process group exited but the command did not",
            )),
            Err(error) => Err(error),
        },
    };
    (result, Some(Box::new(wait)))
}

fn finish_io_with_grace(
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
    stdout: std::thread::JoinHandle<Vec<u8>>,
    stderr: std::thread::JoinHandle<Vec<u8>>,
) -> std::io::Result<()> {
    let started = std::time::Instant::now();
    while !io_finished(&writer, &stdout, &stderr) {
        if started.elapsed() >= COMMAND_IO_DRAIN_GRACE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "command I/O did not drain within {} seconds after termination",
                    COMMAND_IO_DRAIN_GRACE.as_secs()
                ),
            ));
        }
        std::thread::sleep(INTERRUPT_POLL_INTERVAL);
    }
    // A writer blocked when the operation expires normally observes a broken
    // pipe after the process group is killed. Cleanup needs to prove that the
    // writer thread finished; its write result still belongs to the terminal
    // operation and must not reclassify verified termination as cleanup
    // failure.
    join_writer_completion(writer)?;
    let _ = join_drain(stdout).map_err(bounded_error_into_io)?;
    let _ = join_drain(stderr).map_err(bounded_error_into_io)?;
    Ok(())
}

fn join_writer_completion(
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
) -> std::io::Result<()> {
    let Some(writer) = writer else {
        return Ok(());
    };
    writer
        .join()
        .map(|_| ())
        .map_err(|_| std::io::Error::other("stdin writer thread panicked"))
}

fn bounded_error_into_io(error: BoundedError) -> std::io::Error {
    match error {
        BoundedError::Launch(error) | BoundedError::Stdin(error) | BoundedError::Wait(error) => {
            error
        }
        BoundedError::WaitCleanup { source, .. } => source,
    }
}

fn join_writer_result(
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
) -> std::io::Result<()> {
    let Some(writer) = writer else {
        return Ok(());
    };
    match writer.join() {
        Ok(result) => result,
        Err(_) => Err(std::io::Error::other("stdin writer thread panicked")),
    }
}

fn join_drain(drain: std::thread::JoinHandle<Vec<u8>>) -> Result<Vec<u8>, BoundedError> {
    drain
        .join()
        .map_err(|_| BoundedError::Wait(std::io::Error::other("output drain thread panicked")))
}

/// A container's removal on its launch machine
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): one bounded `docker rm -f` request,
/// then, when the daemon did not complete it, a completion wait until the
/// container is gone ([[RFC-0009:C-COMPLETION-WAITS]]).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerRemovalEvidence {
    pub container: String,
    pub elapsed_ms: u64,
    /// The bound of the `docker rm -f` request.
    pub attempt_bound_ms: u64,
    /// Why the request did not itself complete the removal — including the
    /// daemon giving up on its own kill wait.
    pub removal_attempt_error: Option<String>,
    /// Termination of a request client that outlived its bound.
    pub client_cleanup: Option<CommandCleanupEvidence>,
    /// The wait for the container to disappear after an incomplete request.
    pub removal_wait: Option<crate::completion::CompletionWaitEvidence>,
    pub confirmed: bool,
    pub already_absent: bool,
    pub error: Option<String>,
}

/// Remove `container` on its launch machine — locally, or over SSH to
/// `target` — and confirm it is gone. A container the daemon no longer
/// knows is the confirmation, not a failure; a request the daemon did not
/// complete is an attempt outcome, and the container disappearing within
/// the completion wait confirms the removal.
pub fn remove_container(
    target: Option<&str>,
    container: &str,
    notice: &mut dyn FnMut(&str),
) -> ContainerRemovalEvidence {
    let started = std::time::Instant::now();
    let attempt_bound = crate::completion::ATTEMPT_BOUND;
    let mut evidence = ContainerRemovalEvidence {
        container: container.to_owned(),
        elapsed_ms: 0,
        attempt_bound_ms: duration_millis(attempt_bound),
        removal_attempt_error: None,
        client_cleanup: None,
        removal_wait: None,
        confirmed: false,
        already_absent: false,
        error: None,
    };
    // The SSH transport is a host tool: strip the stack's dynamic-linker
    // overrides ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). A local docker client is
    // not in that scope.
    let (argv, env_remove): (Vec<String>, &[&str]) = match target {
        Some(target) => (
            crate::ssh::ssh_argv(
                target,
                &format!("docker rm -f {}", crate::shell::shell_quote(container)),
            ),
            crate::ssh::SSH_ENV_REMOVE,
        ),
        None => (
            vec![
                "docker".to_owned(),
                "rm".to_owned(),
                "-f".to_owned(),
                container.to_owned(),
            ],
            &[],
        ),
    };
    let attempt = OperationBound::finite(attempt_bound);
    let attempt_error = match run_cleanup_with_bound(&argv, env_remove, None, None, &attempt, None)
    {
        Ok(BoundedWait::Exited { status, stderr, .. }) => {
            let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
            if status.success() || stderr.contains("No such container") {
                evidence.confirmed = true;
                evidence.already_absent = !status.success();
                evidence.elapsed_ms = duration_millis(started.elapsed());
                return evidence;
            }
            format!("docker rm -f exited with {status}: {stderr}")
        }
        Ok(BoundedWait::Expired { cleanup, .. }) => {
            evidence.client_cleanup = cleanup;
            format!(
                "docker rm -f did not answer within its {} s attempt bound",
                attempt_bound.as_secs()
            )
        }
        Ok(BoundedWait::Interrupted { cleanup, .. }) => {
            evidence.client_cleanup = Some(cleanup);
            "docker rm -f was interrupted".to_owned()
        }
        Err(BoundedError::Launch(error)) => match target {
            Some(target) => format!("failed to launch SSH for {target:?}: {error}"),
            None => format!("docker rm failed to launch: {error}"),
        },
        Err(BoundedError::Stdin(error) | BoundedError::Wait(error)) => {
            format!("docker rm wait failed: {error}")
        }
        Err(BoundedError::WaitCleanup {
            source, cleanup, ..
        }) => {
            evidence.client_cleanup = Some(cleanup);
            format!("docker rm wait failed: {source}")
        }
    };
    evidence.removal_attempt_error = Some(attempt_error.clone());
    let wait = crate::completion::wait(
        &mut crate::completion::container::ContainerGone::new(target, container),
        notice,
    );
    evidence.confirmed = wait.reached();
    if !evidence.confirmed {
        evidence.error = Some(match wait.failure() {
            Some(failure) => format!("{failure}; removal request: {attempt_error}"),
            None => attempt_error,
        });
    }
    evidence.removal_wait = Some(wait);
    evidence.elapsed_ms = duration_millis(started.elapsed());
    evidence
}

#[cfg(test)]
mod tests {
    use super::{BoundedWait, CommandCleanupTrigger, run_with_bound, run_with_bound_mode};
    use crate::operation_bound::OperationBound;
    use std::time::Duration;

    #[test]
    fn unbounded_command_wait_terminates_when_the_process_exits() {
        let argv = vec!["sh".to_owned(), "-c".to_owned(), "exit 0".to_owned()];
        let outcome = run_with_bound(&argv, &[], None, None, &OperationBound::unbounded(), None);

        assert!(matches!(
            outcome,
            Ok(BoundedWait::Exited { status, .. }) if status.success()
        ));
    }

    #[test]
    fn pending_interruption_terminates_an_unbounded_command() {
        let argv = vec!["sh".to_owned(), "-c".to_owned(), "sleep 60".to_owned()];
        let outcome = run_with_bound_mode(
            &argv,
            &[],
            None,
            None,
            &OperationBound::unbounded(),
            None,
            true,
            || true,
        );

        let (operation_elapsed_ms, cleanup) = match outcome {
            Ok(BoundedWait::Interrupted {
                kill: Ok(()),
                operation_elapsed_ms,
                cleanup,
            }) => (operation_elapsed_ms, cleanup),
            other => {
                assert!(
                    matches!(other, Ok(BoundedWait::Interrupted { .. })),
                    "interrupted command did not return cleanup evidence"
                );
                return;
            }
        };
        assert!(operation_elapsed_ms < 1_000);
        assert!(cleanup.verified, "{cleanup:?}");
        assert_eq!(cleanup.trigger, CommandCleanupTrigger::Interruption);
        assert!(cleanup.kill_attempted);
        assert!(
            cleanup
                .exit_wait
                .as_ref()
                .is_some_and(|wait| wait.reached()),
            "{cleanup:?}"
        );
        assert_eq!(cleanup.io_drain_grace_ms, 5_000);
    }

    #[test]
    fn blocked_stdin_write_cannot_outlive_the_owner_bound() {
        let argv = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "while :; do :; done".to_owned(),
        ];
        let payload = vec![b'x'; 1024 * 1024];
        let bound = OperationBound::finite(Duration::from_millis(50));

        let outcome = run_with_bound(&argv, &[], None, Some(&payload), &bound, None);

        assert!(matches!(
            outcome,
            Ok(BoundedWait::Expired { kill: Ok(()), .. })
        ));
    }

    #[test]
    fn inherited_output_pipe_cannot_outlive_the_owner_bound() {
        let argv = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "sleep 60 & exit 0".to_owned(),
        ];
        let bound = OperationBound::finite(Duration::from_millis(50));

        let outcome = run_with_bound(&argv, &[], None, None, &bound, None);

        assert!(matches!(
            outcome,
            Ok(BoundedWait::Expired { kill: Ok(()), .. })
        ));
    }

    #[test]
    fn env_remove_strips_inherited_variables_from_the_child() {
        // `/usr/bin/env` reports the received environment verbatim; a shell
        // child would re-default an unset PATH and hide the strip.
        let argv = vec!["/usr/bin/env".to_owned()];

        let inherited = run_with_bound(&argv, &[], None, None, &OperationBound::unbounded(), None);
        assert!(matches!(
            inherited,
            Ok(BoundedWait::Exited { ref stdout, .. }) if stdout.starts_with(b"PATH=")
                || stdout.windows(6).any(|window| window == b"\nPATH=")
        ));

        let stripped = run_with_bound(
            &argv,
            &["PATH"],
            None,
            None,
            &OperationBound::unbounded(),
            None,
        );
        assert!(matches!(
            stripped,
            Ok(BoundedWait::Exited { ref stdout, .. }) if !stdout.starts_with(b"PATH=")
                && !stdout.windows(6).any(|window| window == b"\nPATH=")
        ));
    }
}
