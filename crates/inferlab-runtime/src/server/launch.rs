use super::cleanup::{
    KILL_POLL_LIMIT, REMOTE_SERVER_CLEANUP_DEADLINE, TERM_POLL_LIMIT, cleanup_error,
    cleanup_failed_local_launch, completed_cleanup, remove_server_container,
};
use super::observation::{remote_group_alive_script, run_cleanup_command};
use super::{
    CleanupTrigger, HostProcessHandle, LaunchFailure, ProcessHandle, ProcessLauncher, ProcessSpec,
    ServerLaunchError, SshProcessHandle, SystemProcessRuntime,
};
use crate::operation_bound::{OperationBound, duration_millis};
use crate::plan::{CommandPlan, LaunchFilePlan, LaunchPlan};
use crate::shell::{shell_quote, shell_quote_path};
use crate::ssh::{SSH_ENV_REMOVE, ssh_argv, ssh_output, ssh_output_with_input};
use sha2::{Digest, Sha256};
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const HANDLE_MARKER: &str = "INFERLAB_HANDLE\t";

pub(super) fn spawn_local(spec: ProcessSpec<'_>) -> Result<HostProcessHandle, LaunchFailure> {
    let fail = |message: String| LaunchFailure::before_launch(message);
    fs::create_dir_all(spec.cache_root).map_err(|source| {
        LaunchFailure::from_error(ServerLaunchError::FileIo {
            operation: "create runtime cache root",
            path: spec.cache_root.to_path_buf(),
            source,
        })
    })?;
    materialize_local_launch_files(spec.launch_files).map_err(LaunchFailure::from_error)?;
    let (program, args) = spec
        .command
        .argv
        .split_first()
        .ok_or_else(|| fail("resolved server command is empty".to_owned()))?;
    let stdout = File::create(spec.stdout).map_err(|source| {
        LaunchFailure::from_error(ServerLaunchError::FileIo {
            operation: "create server stdout log",
            path: spec.stdout.to_path_buf(),
            source,
        })
    })?;
    let stderr = File::create(spec.stderr).map_err(|source| {
        LaunchFailure::from_error(ServerLaunchError::FileIo {
            operation: "create server stderr log",
            path: spec.stderr.to_path_buf(),
            source,
        })
    })?;
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(&spec.command.cwd)
        .env_clear()
        .envs(&spec.command.env)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .process_group(0);
    // Declared pass-through values flow from the launching machine's
    // environment — here the invoking process — into the docker client,
    // which forwards each name-referenced variable into the container. On a
    // local launch the invoking environment is also composed into the
    // recorded env map (the standing unredacted-records posture); the
    // reference channel is what keeps the value out of the plan where no
    // ambient composition exists ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    for name in &spec.command.pass_env {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    let mut child = command.spawn().map_err(|source| {
        LaunchFailure::from_error(ServerLaunchError::Process {
            program: program.clone(),
            source,
        })
    })?;
    HostProcessHandle::new(child.id(), spec.container.map(str::to_owned)).map_err(|error| {
        let mut cleanup = cleanup_failed_local_launch(&mut child);
        let error = ServerLaunchError::Preparation { message: error };
        // The client may already have asked the daemon to create the
        // container, which the group kill cannot reach. The group was
        // stopped above, so this final removal races nothing; an
        // unconfirmed one means the workload may still be running, which
        // cleanup must never call verified ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        match spec.container {
            Some(container) => {
                let removal = remove_server_container(None, container);
                if !removal.confirmed {
                    cleanup.verified = false;
                    if cleanup.error.is_none() {
                        cleanup.error = removal.error.clone();
                    }
                }
                cleanup.container_removal = Some(removal.clone());
                LaunchFailure {
                    error,
                    ownership_unknown: !cleanup.verified,
                    container_removal: Some(Box::new(removal)),
                    cleanup: Some(Box::new(cleanup)),
                    cleanup_note: None,
                }
            }
            None => LaunchFailure {
                error,
                ownership_unknown: !cleanup.verified,
                container_removal: None,
                cleanup: Some(Box::new(cleanup)),
                cleanup_note: None,
            },
        }
    })
}

pub(super) fn materialize_local_launch_files(
    launch_files: &[LaunchFilePlan],
) -> Result<(), ServerLaunchError> {
    for launch_file in launch_files {
        publish_local_launch_file(launch_file)?;
    }
    Ok(())
}

fn publish_local_launch_file(launch_file: &LaunchFilePlan) -> Result<(), ServerLaunchError> {
    let target = &launch_file.resolved_path;
    let parent = target
        .parent()
        .ok_or_else(|| ServerLaunchError::Preparation {
            message: format!(
                "launch file target {} has no parent directory",
                target.display()
            ),
        })?;
    fs::create_dir_all(parent).map_err(|source| ServerLaunchError::FileIo {
        operation: "create launch file directory",
        path: parent.to_path_buf(),
        source,
    })?;
    let mut staged = tempfile::Builder::new()
        .prefix(".inferlab-launch.")
        .tempfile_in(parent)
        .map_err(|source| ServerLaunchError::FileIo {
            operation: "stage launch file",
            path: target.to_path_buf(),
            source,
        })?;
    staged
        .write_all(launch_file.text.as_bytes())
        .and_then(|()| staged.flush())
        .map_err(|source| ServerLaunchError::FileIo {
            operation: "write staged launch file",
            path: target.to_path_buf(),
            source,
        })?;
    staged
        .as_file()
        .set_permissions(fs::Permissions::from_mode(0o444))
        .and_then(|()| staged.as_file().sync_all())
        .map_err(|source| ServerLaunchError::FileIo {
            operation: "finalize staged launch file",
            path: target.to_path_buf(),
            source,
        })?;

    match staged.persist_noclobber(target) {
        Ok(_) => Ok(()),
        Err(failure) => {
            let source = failure.error;
            if source.kind() == io::ErrorKind::AlreadyExists {
                verify_existing_launch_file(target, &launch_file.sha256)
            } else {
                Err(ServerLaunchError::FileIo {
                    operation: "publish launch file",
                    path: target.to_path_buf(),
                    source,
                })
            }
        }
    }
}

fn verify_existing_launch_file(
    target: &Path,
    expected_sha256: &str,
) -> Result<(), ServerLaunchError> {
    let metadata = fs::symlink_metadata(target).map_err(|source| ServerLaunchError::FileIo {
        operation: "inspect existing launch file",
        path: target.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_file() {
        return Err(ServerLaunchError::NotRegularFile {
            path: target.to_path_buf(),
        });
    }
    let actual_sha256 = file_sha256(target)?;
    if actual_sha256 != expected_sha256 {
        return Err(ServerLaunchError::FileDigestMismatch {
            path: target.to_path_buf(),
            expected: expected_sha256.to_owned(),
            actual: actual_sha256,
        });
    }
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String, ServerLaunchError> {
    let mut file = File::open(path).map_err(|source| ServerLaunchError::FileIo {
        operation: "read launch file",
        path: path.to_path_buf(),
        source,
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| ServerLaunchError::FileIo {
                operation: "read launch file",
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// A one-line human summary of a structured removal outcome for the launch
/// failure message; the structured evidence itself rides
/// [`LaunchFailure::container_removal`].
pub(super) fn spawn_ssh(
    target: &str,
    spec: ProcessSpec<'_>,
) -> Result<SshProcessHandle, LaunchFailure> {
    let remote_stdout = spec.remote_dir.join("stdout.log");
    let remote_stderr = spec.remote_dir.join("stderr.log");
    let remote_handle = spec.remote_dir.join("launch.handle");
    let command = render_env_command(spec.command).map_err(LaunchFailure::before_launch)?;
    materialize_ssh_launch_files(target, spec.launch_files).map_err(LaunchFailure::from_error)?;
    let script = format!(
        "set -eu; mkdir -p {dir} {cache}; cd {cwd}; nohup setsid {command} >{stdout} 2>{stderr} </dev/null & pid=$!; cleanup_pending=1; cleanup_launch() {{ if [ \"$cleanup_pending\" = 1 ]; then kill -KILL -- -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true; fi; }}; trap cleanup_launch EXIT; ticks=$(awk '{{print $22}}' /proc/$pid/stat); printf '%s %s\\n' \"$pid\" \"$ticks\" > {handle}; printf 'INFERLAB_HANDLE\\t%s\\t%s\\n' \"$pid\" \"$ticks\"; cleanup_pending=0; trap - EXIT",
        dir = shell_quote_path(spec.remote_dir),
        cache = shell_quote_path(spec.cache_root),
        cwd = shell_quote_path(&spec.command.cwd),
        stdout = shell_quote_path(&remote_stdout),
        stderr = shell_quote_path(&remote_stderr),
        handle = shell_quote_path(&remote_handle),
    );
    let output = ssh_output(target, &script)
        .map_err(ServerLaunchError::from)
        .map_err(LaunchFailure::from_error)?;
    if !output.status.success() {
        return Err(failed_ssh_handle_delivery(
            target,
            &remote_handle,
            spec.container,
            ServerLaunchError::Exit {
                operation: format!("SSH launch on {target:?}"),
                status: output.status,
                diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            },
        ));
    }
    let text = match String::from_utf8(output.stdout) {
        Ok(text) => text,
        Err(error) => {
            return Err(failed_ssh_handle_delivery(
                target,
                &remote_handle,
                spec.container,
                ServerLaunchError::NonUtf8Identity {
                    target: target.to_owned(),
                    source: error,
                },
            ));
        }
    };
    parse_ssh_handle(
        target,
        remote_stdout,
        remote_stderr,
        &text,
        spec.container.map(str::to_owned),
    )
    .map_err(|error| failed_ssh_handle_delivery(target, &remote_handle, spec.container, error))
}

fn materialize_ssh_launch_files(
    target: &str,
    launch_files: &[LaunchFilePlan],
) -> Result<(), ServerLaunchError> {
    for launch_file in launch_files {
        let script = remote_launch_file_script(launch_file)?;
        let output = ssh_output_with_input(target, &script, launch_file.text.as_bytes())?;
        if !output.status.success() {
            return Err(ServerLaunchError::Exit {
                operation: format!(
                    "materializing launch file {} over SSH on {target:?}",
                    launch_file.resolved_path.display()
                ),
                status: output.status,
                diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            });
        }
    }
    Ok(())
}

pub(super) fn remote_launch_file_script(
    launch_file: &LaunchFilePlan,
) -> Result<String, ServerLaunchError> {
    let target = &launch_file.resolved_path;
    let parent = target
        .parent()
        .ok_or_else(|| ServerLaunchError::Preparation {
            message: format!(
                "launch file target {} has no parent directory",
                target.display()
            ),
        })?;
    Ok(format!(
        "# INFERLAB_LAUNCH_FILE\nset -eu\numask 077\nparent={parent}\ntarget={target}\ndigest={digest}\nmkdir -p -- \"$parent\"\nstage=$(mktemp \"$parent/.inferlab-launch.XXXXXX\")\ntrap 'rm -f -- \"$stage\"' EXIT\ncat > \"$stage\"\nactual=$(sha256sum -- \"$stage\" | awk '{{print $1}}')\nif [ \"$actual\" != \"$digest\" ]; then printf 'staged launch file digest mismatch for %s: expected %s, found %s\\n' \"$target\" \"$digest\" \"$actual\" >&2; exit 1; fi\nchmod 0444 -- \"$stage\"\nif ln -T -- \"$stage\" \"$target\" 2>/dev/null; then exit 0; fi\nif [ ! -f \"$target\" ] || [ -L \"$target\" ]; then printf 'existing launch file target %s is not a regular file\\n' \"$target\" >&2; exit 1; fi\nactual=$(sha256sum -- \"$target\" | awk '{{print $1}}')\nif [ \"$actual\" != \"$digest\" ]; then printf 'existing launch file %s does not match declared digest %s; found %s\\n' \"$target\" \"$digest\" \"$actual\" >&2; exit 1; fi",
        parent = shell_quote_path(parent),
        target = shell_quote_path(target),
        digest = shell_quote(&launch_file.sha256),
    ))
}

fn parse_ssh_handle(
    target: &str,
    remote_stdout: PathBuf,
    remote_stderr: PathBuf,
    output: &str,
    container: Option<String>,
) -> Result<SshProcessHandle, ServerLaunchError> {
    let result = output
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(HANDLE_MARKER))
        .ok_or_else(|| ServerLaunchError::MissingProcessId {
            target: target.to_owned(),
        })?
        .split_once('\t')
        .ok_or_else(|| ServerLaunchError::MissingStartTime {
            target: target.to_owned(),
        })?;
    let leader_pid =
        result
            .0
            .parse::<u32>()
            .map_err(|source| ServerLaunchError::InvalidProcessId {
                target: target.to_owned(),
                value: result.0.to_owned(),
                source,
            })?;
    let leader_start_time_ticks =
        result
            .1
            .parse::<u64>()
            .map_err(|source| ServerLaunchError::InvalidStartTime {
                target: target.to_owned(),
                value: result.1.to_owned(),
                source,
            })?;
    let handle = SshProcessHandle {
        target: target.to_owned(),
        leader_pid,
        process_group: leader_pid,
        leader_start_time_ticks,
        stdout: remote_stdout,
        stderr: remote_stderr,
        container,
    };
    handle
        .validate()
        .map_err(|details| ServerLaunchError::InvalidIdentity {
            target: target.to_owned(),
            details,
        })?;
    Ok(handle)
}

fn failed_ssh_handle_delivery(
    target: &str,
    remote_handle: &Path,
    container: Option<&str>,
    error: ServerLaunchError,
) -> LaunchFailure {
    let cleanup_started = Instant::now();
    // Order matters: stop the remote launcher first, so a docker client
    // that had not yet created the container cannot create it after an
    // early rm reported it absent, then do the final container-removal
    // confirmation against a quiescent group — the same order the local
    // path uses ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let process_cleanup = cleanup_incomplete_ssh_launch(target, remote_handle);
    let removal = container.map(|container| remove_server_container(Some(target), container));
    let removal_confirmed = removal.as_ref().is_none_or(|removal| removal.confirmed);
    let (process_confirmed, cleanup_note) = match &process_cleanup {
        Ok(()) => (
            true,
            format!(
                "cleaned the remote process using {}",
                remote_handle.display()
            ),
        ),
        Err(cleanup) => (
            false,
            format!(
                "remote launch cleanup using {} was not verified: {cleanup}",
                remote_handle.display()
            ),
        ),
    };
    let verified = removal_confirmed && process_confirmed;
    let mut cleanup = if process_confirmed {
        completed_cleanup(CleanupTrigger::StartupRollback, false, false, Vec::new())
    } else {
        cleanup_error(
            CleanupTrigger::StartupRollback,
            false,
            Vec::new(),
            process_cleanup
                .as_ref()
                .err()
                .cloned()
                .unwrap_or_else(|| "remote launch cleanup was not verified".to_owned()),
        )
    };
    cleanup.elapsed_ms = duration_millis(cleanup_started.elapsed());
    cleanup.remote_deadline_ms = Some(duration_millis(REMOTE_SERVER_CLEANUP_DEADLINE));
    cleanup.container_removal = removal.clone();
    cleanup.verified = verified;
    if !verified && cleanup.error.is_none() {
        cleanup.error = removal.as_ref().and_then(|removal| removal.error.clone());
    }
    LaunchFailure {
        error,
        ownership_unknown: !verified,
        container_removal: removal.map(Box::new),
        cleanup: Some(Box::new(cleanup)),
        cleanup_note: Some(cleanup_note),
    }
}

fn cleanup_incomplete_ssh_launch(target: &str, remote_handle: &Path) -> Result<(), String> {
    let bound = OperationBound::finite(REMOTE_SERVER_CLEANUP_DEADLINE);
    let alive = remote_group_alive_script("$pid");
    let script = format!(
        "set +e; file={file}; if [ ! -r \"$file\" ]; then exit 4; fi; read pid expected < \"$file\" || exit 4; if [ -r /proc/$pid/stat ]; then actual=$(awk '{{print $22}}' /proc/$pid/stat) || exit 4; [ \"$actual\" = \"$expected\" ] || exit 4; elif {alive}; then exit 5; else rm -f \"$file\"; exit 0; fi; if ! {alive}; then rm -f \"$file\"; exit 0; fi; kill -TERM -- -$pid; i=0; while {alive} && [ $i -lt {term_limit} ]; do sleep 0.1; i=$((i+1)); done; if {alive}; then kill -KILL -- -$pid; i=0; while {alive} && [ $i -lt {kill_limit} ]; do sleep 0.1; i=$((i+1)); done; fi; if {alive}; then exit 6; fi; rm -f \"$file\"",
        file = shell_quote_path(remote_handle),
        term_limit = TERM_POLL_LIMIT,
        kill_limit = KILL_POLL_LIMIT,
    );
    let output = run_cleanup_command(
        &ssh_argv(target, &script),
        SSH_ENV_REMOVE,
        &bound,
        "SSH launch cleanup",
    )
    .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "SSH cleanup exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn render_env_command(command: &CommandPlan) -> Result<String, String> {
    if command.argv.is_empty() {
        return Err("resolved server command is empty".to_owned());
    }
    let mut parts = vec!["env".to_owned(), "-i".to_owned()];
    parts.extend(
        command
            .env
            .iter()
            .map(|(name, value)| shell_quote(&format!("{name}={value}"))),
    );
    // Declared pass-through values flow from the launching machine's
    // environment: the remote login shell expands the reference before
    // `env -i` strips it, so the value reaches the docker client — which
    // forwards each name-referenced variable into the container — while
    // the script text carries only the reference
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). Names are load-validated bare
    // identifiers, safe to splice unquoted.
    parts.extend(
        command
            .pass_env
            .iter()
            .map(|name| format!("{name}=\"${{{name}}}\"")),
    );
    parts.extend(command.argv.iter().map(|value| shell_quote(value)));
    Ok(parts.join(" "))
}

impl ProcessLauncher for SystemProcessRuntime {
    fn spawn(&self, spec: ProcessSpec<'_>) -> Result<ProcessHandle, LaunchFailure> {
        match spec.launch {
            LaunchPlan::Local => spawn_local(spec).map(ProcessHandle::Local),
            LaunchPlan::Ssh { target } => spawn_ssh(target, spec).map(ProcessHandle::Ssh),
        }
    }
}
