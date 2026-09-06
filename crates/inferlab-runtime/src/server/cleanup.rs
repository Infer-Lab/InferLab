use super::observation::{remote_group_alive_script, run_cleanup_command};
use super::{
    CleanupEvidence, CleanupTrigger, ContainerRemovalEvidence, HostProcessHandle, ProcessCleanup,
    ProcessHandle, SshProcessHandle, SystemProcessRuntime,
};
use crate::operation_bound::{OperationBound, duration_millis};
use crate::process_group::{LocalProcessGroup, SignalEvidence, TerminationSignal, VerifiedStatus};
use crate::ssh::{SSH_ENV_REMOVE, ssh_argv};
use std::time::{Duration, Instant};
use wait_timeout::ChildExt;

const POLL_INTERVAL: Duration = Duration::from_millis(100);
const TERM_GRACE: Duration = Duration::from_secs(2);
const KILL_GRACE: Duration = Duration::from_secs(10);
const SERVER_CLEANUP_STATUS_DEADLINE: Duration = Duration::from_secs(2);
pub(super) const REMOTE_SERVER_CLEANUP_DEADLINE: Duration = Duration::from_secs(30);
const LOCAL_LAUNCH_FAILURE_REAP_GRACE: Duration = Duration::from_secs(5);
pub(super) const TERM_POLL_LIMIT: u128 = TERM_GRACE.as_millis() / POLL_INTERVAL.as_millis();
pub(super) const KILL_POLL_LIMIT: u128 = KILL_GRACE.as_millis() / POLL_INTERVAL.as_millis();
const CLEANUP_MARKER: &str = "INFERLAB_CLEANUP\t";

impl CleanupEvidence {
    pub fn unavailable(trigger: CleanupTrigger, message: String) -> Self {
        Self {
            trigger,
            elapsed_ms: 0,
            status_deadline_ms: duration_millis(SERVER_CLEANUP_STATUS_DEADLINE),
            term_grace_ms: duration_millis(TERM_GRACE),
            kill_grace_ms: duration_millis(KILL_GRACE),
            reap_grace_ms: None,
            remote_deadline_ms: None,
            verified: false,
            already_exited: false,
            forced: false,
            signals: Vec::new(),
            error: Some(message),
            container_removal: None,
            device_residuals: None,
        }
    }

    /// Cleanup evidence for a launch failure that removed (or tried to
    /// remove) the container it created. `verified` is the caller's
    /// conjunction of process cleanup AND container removal — a confirmed
    /// removal alone is not verified cleanup if the launcher stop was not
    /// confirmed — and the structured outcome names the actual container
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub fn from_launch_removal(
        trigger: CleanupTrigger,
        verified: bool,
        removal: ContainerRemovalEvidence,
        error: Option<String>,
    ) -> Self {
        Self {
            trigger,
            elapsed_ms: removal.elapsed_ms,
            status_deadline_ms: duration_millis(SERVER_CLEANUP_STATUS_DEADLINE),
            term_grace_ms: duration_millis(TERM_GRACE),
            kill_grace_ms: duration_millis(KILL_GRACE),
            reap_grace_ms: None,
            remote_deadline_ms: None,
            verified,
            already_exited: false,
            forced: false,
            signals: Vec::new(),
            error,
            container_removal: Some(removal),
            device_residuals: None,
        }
    }
}

pub(super) fn cleanup_failed_local_launch(child: &mut std::process::Child) -> CleanupEvidence {
    let started = Instant::now();
    let initial_status_error = match child.try_wait() {
        Ok(Some(_)) => {
            let mut evidence =
                completed_cleanup(CleanupTrigger::StartupRollback, true, false, Vec::new());
            evidence.elapsed_ms = duration_millis(started.elapsed());
            evidence.status_deadline_ms = 0;
            evidence.term_grace_ms = 0;
            evidence.reap_grace_ms = Some(duration_millis(LOCAL_LAUNCH_FAILURE_REAP_GRACE));
            return evidence;
        }
        Ok(None) => None,
        // The subsequent kill and bounded reap are authoritative cleanup
        // verification. Preserve this diagnostic only if that verification
        // also fails; a successful reap resolves the transient status error.
        Err(error) => Some(format!("failed to inspect failed launch child: {error}")),
    };
    let group = match LocalProcessGroup::capture_child(child) {
        Ok(group) => group,
        Err(error) => {
            let mut evidence = CleanupEvidence::unavailable(
                CleanupTrigger::StartupRollback,
                format!("failed to capture failed launch process-group identity: {error}"),
            );
            evidence.elapsed_ms = duration_millis(started.elapsed());
            evidence.status_deadline_ms = 0;
            evidence.term_grace_ms = 0;
            evidence.reap_grace_ms = Some(duration_millis(LOCAL_LAUNCH_FAILURE_REAP_GRACE));
            return evidence;
        }
    };
    let bound = OperationBound::finite(KILL_GRACE);
    let signal = group.send_signal(TerminationSignal::Kill, &bound);
    let reaped = match child.wait_timeout(LOCAL_LAUNCH_FAILURE_REAP_GRACE) {
        Ok(Some(_)) => Ok(()),
        Ok(None) => Err(format!(
            "child did not reap within {} seconds",
            LOCAL_LAUNCH_FAILURE_REAP_GRACE.as_secs()
        )),
        Err(error) => Err(format!("failed to reap failed launch child: {error}")),
    };
    let mut evidence = match reaped {
        Ok(()) => completed_cleanup(CleanupTrigger::StartupRollback, false, true, vec![signal]),
        Err(error) => {
            let error = initial_status_error
                .map(|status_error| format!("{status_error}; {error}"))
                .unwrap_or(error);
            cleanup_error(CleanupTrigger::StartupRollback, true, vec![signal], error)
        }
    };
    evidence.elapsed_ms = duration_millis(started.elapsed());
    evidence.status_deadline_ms = 0;
    evidence.term_grace_ms = 0;
    evidence.reap_grace_ms = Some(duration_millis(LOCAL_LAUNCH_FAILURE_REAP_GRACE));
    evidence
}

pub(super) fn removal_summary(removal: &ContainerRemovalEvidence) -> String {
    match (removal.confirmed, removal.already_absent, &removal.error) {
        (true, true, _) => format!("container {} was already absent", removal.container),
        (true, _, _) => format!("container {} was removed", removal.container),
        (false, _, Some(error)) => {
            format!(
                "container {} removal was not confirmed: {error}",
                removal.container
            )
        }
        (false, _, None) => format!("container {} removal was not confirmed", removal.container),
    }
}

pub(super) fn terminate_local(
    handle: &HostProcessHandle,
    trigger: CleanupTrigger,
) -> CleanupEvidence {
    let started = Instant::now();
    if let Err(error) = handle.validate() {
        let mut evidence = CleanupEvidence::unavailable(trigger, error);
        evidence.elapsed_ms = duration_millis(started.elapsed());
        return evidence;
    }
    let group = match LocalProcessGroup::new(
        handle.leader_pid,
        handle.process_group,
        handle.leader_start_time_ticks,
    ) {
        Ok(group) => group,
        Err(error) => {
            let mut evidence = CleanupEvidence::unavailable(trigger, error.to_string());
            evidence.elapsed_ms = duration_millis(started.elapsed());
            return evidence;
        }
    };
    let status_bound = OperationBound::finite(SERVER_CLEANUP_STATUS_DEADLINE);
    match group.verified_status(&status_bound) {
        Ok(VerifiedStatus::Alive) => {}
        Ok(VerifiedStatus::Exited | VerifiedStatus::Reused) => {
            let mut evidence = completed_cleanup(trigger, true, false, Vec::new());
            evidence.elapsed_ms = duration_millis(started.elapsed());
            return evidence;
        }
        Ok(VerifiedStatus::LeaderMissingWithMembers) => {
            // The leader died on its own: members forked from it at or after
            // its recorded start, so a cohort-consistent group is still ours
            // to signal; a member predating the leader means a recycled pgid.
            let violations = match group.cohort_violations(&status_bound) {
                Ok(violations) => violations,
                Err(error) => {
                    let mut evidence = CleanupEvidence::unavailable(trigger, error.to_string());
                    evidence.elapsed_ms = duration_millis(started.elapsed());
                    return evidence;
                }
            };
            if !violations.is_empty() {
                let members = violations
                    .iter()
                    .map(|(pid, ticks)| format!("member {pid} started at {ticks}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut evidence = CleanupEvidence::unavailable(
                    trigger,
                    format!(
                        "process-group {} still has members but recorded leader {} no longer exists and the cohort check failed: {} before the recorded leader start {}",
                        handle.process_group,
                        handle.leader_pid,
                        members,
                        handle.leader_start_time_ticks
                    ),
                );
                evidence.elapsed_ms = duration_millis(started.elapsed());
                return evidence;
            }
        }
        Err(error) => {
            let mut evidence = CleanupEvidence::unavailable(trigger, error.to_string());
            evidence.elapsed_ms = duration_millis(started.elapsed());
            return evidence;
        }
    }
    let term_bound = OperationBound::finite(TERM_GRACE);
    let mut signals = vec![group.send_signal(TerminationSignal::Term, &term_bound)];
    let mut evidence = match group.wait_until_stopped(None, &term_bound, POLL_INTERVAL) {
        Ok(true) => completed_cleanup(trigger, false, false, signals),
        Ok(false) => {
            let kill_bound = OperationBound::finite(KILL_GRACE);
            signals.push(group.send_signal(TerminationSignal::Kill, &kill_bound));
            match group.wait_until_stopped(None, &kill_bound, POLL_INTERVAL) {
                Ok(true) => completed_cleanup(trigger, false, true, signals),
                Ok(false) => CleanupEvidence {
                    trigger,
                    elapsed_ms: 0,
                    status_deadline_ms: duration_millis(SERVER_CLEANUP_STATUS_DEADLINE),
                    term_grace_ms: duration_millis(TERM_GRACE),
                    kill_grace_ms: duration_millis(KILL_GRACE),
                    reap_grace_ms: None,
                    remote_deadline_ms: None,
                    verified: false,
                    already_exited: false,
                    forced: true,
                    signals,
                    error: Some(format!(
                        "server process group {} did not exit after SIGKILL",
                        handle.process_group
                    )),
                    container_removal: None,
                    device_residuals: None,
                },
                Err(error) => cleanup_error(trigger, true, signals, error.to_string()),
            }
        }
        Err(error) => cleanup_error(trigger, false, signals, error.to_string()),
    };
    evidence.elapsed_ms = duration_millis(started.elapsed());
    evidence
}

pub(super) fn terminate_ssh(handle: &SshProcessHandle, trigger: CleanupTrigger) -> CleanupEvidence {
    let started = Instant::now();
    let bound = OperationBound::finite(REMOTE_SERVER_CLEANUP_DEADLINE);
    let mut evidence = terminate_ssh_under(handle, trigger, &bound);
    evidence.elapsed_ms = duration_millis(started.elapsed());
    evidence.remote_deadline_ms = Some(duration_millis(REMOTE_SERVER_CLEANUP_DEADLINE));
    evidence
}

/// The remote cleanup script: the leader-missing branch replicates the local
/// cohort check — every live member must start at or after the recorded
/// leader ticks, else the pgid reads as recycled and the script declines to
/// signal, naming the offending members in the detail field.
pub(super) fn remote_cleanup_script(handle: &SshProcessHandle) -> String {
    format!(
        "set +e; pgid={}; pid={}; expected={}; if [ -r /proc/$pid/stat ]; then actual=$(awk '{{print $22}}' /proc/$pid/stat); if [ $? -ne 0 ]; then printf '{marker}unknown\\t-\\t0\\t-\\t1\\tstat-unreadable\\n'; exit 0; fi; if [ \"$actual\" != \"$expected\" ]; then printf '{marker}stale\\t-\\t0\\t-\\t0\\t%s\\n' \"$actual\"; exit 0; fi; elif {}; then bad=\"\"; for mpid in $(ps -eo pid=,pgid=,stat= | awk -v pgid=\"$pgid\" '$2 == pgid && $3 !~ /^Z/ {{print $1}}'); do mticks=$(awk '{{print $22}}' /proc/$mpid/stat 2>/dev/null) || mticks=\"\"; if [ -n \"$mticks\" ] && [ \"$mticks\" -lt \"$expected\" ]; then bad=\"$bad $mpid:$mticks\"; fi; done; if [ -n \"$bad\" ]; then printf '{marker}unknown\\t-\\t0\\t-\\t1\\tleader-missing; cohort members%s predate recorded leader start %s\\n' \"$bad\" \"$expected\"; exit 0; fi; else printf '{marker}already\\t-\\t0\\t-\\t0\\t-\\n'; exit 0; fi; if ! {}; then printf '{marker}already\\t-\\t0\\t-\\t0\\t-\\n'; exit 0; fi; kill -TERM -- -$pgid; term_code=$?; i=0; while {} && [ $i -lt {term_limit} ]; do sleep 0.1; i=$((i+1)); done; forced=0; kill_code=-; if {}; then forced=1; kill -KILL -- -$pgid; kill_code=$?; i=0; while {} && [ $i -lt {kill_limit} ]; do sleep 0.1; i=$((i+1)); done; fi; alive=0; if {}; then alive=1; fi; printf '{marker}cleanup\\t%s\\t%s\\t%s\\t%s\\t-\\n' \"$term_code\" \"$forced\" \"$kill_code\" \"$alive\"",
        handle.process_group,
        handle.leader_pid,
        handle.leader_start_time_ticks,
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        term_limit = TERM_POLL_LIMIT,
        kill_limit = KILL_POLL_LIMIT,
        marker = CLEANUP_MARKER,
    )
}

pub(super) fn terminate_ssh_under(
    handle: &SshProcessHandle,
    trigger: CleanupTrigger,
    bound: &OperationBound,
) -> CleanupEvidence {
    let script = remote_cleanup_script(handle);
    match run_cleanup_command(
        &ssh_argv(&handle.target, &script),
        SSH_ENV_REMOVE,
        bound,
        "SSH process cleanup",
    ) {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let Some(result) = parse_cleanup_output(&stdout) else {
                return cleanup_error(
                    trigger,
                    false,
                    Vec::new(),
                    "SSH cleanup returned no cleanup result".to_owned(),
                );
            };
            match result.state {
                RemoteCleanupState::Already => {
                    return completed_cleanup(trigger, true, false, Vec::new());
                }
                RemoteCleanupState::Stale => {
                    return CleanupEvidence::unavailable(
                        trigger,
                        format!(
                            "managed SSH process {} exited and its pid was reused: observed start time {}",
                            handle.leader_pid, result.detail
                        ),
                    );
                }
                RemoteCleanupState::Unknown => {
                    return CleanupEvidence::unavailable(
                        trigger,
                        format!(
                            "SSH process-group {} ownership could not be verified: {}",
                            handle.process_group, result.detail
                        ),
                    );
                }
                RemoteCleanupState::Cleanup => {}
            }
            let Some(term_code) = result.term_code else {
                return cleanup_error(
                    trigger,
                    false,
                    Vec::new(),
                    "SSH cleanup returned no SIGTERM status".to_owned(),
                );
            };
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let mut signals = vec![remote_signal_evidence(
                TerminationSignal::Term,
                handle.process_group,
                term_code,
                &stderr,
            )];
            if let Some(kill_code) = result.kill_code {
                signals.push(remote_signal_evidence(
                    TerminationSignal::Kill,
                    handle.process_group,
                    kill_code,
                    &stderr,
                ));
            }
            if result.alive {
                cleanup_error(
                    trigger,
                    result.forced,
                    signals,
                    format!(
                        "SSH process group {} did not exit after cleanup",
                        handle.process_group
                    ),
                )
            } else {
                completed_cleanup(trigger, false, result.forced, signals)
            }
        }
        Ok(output) => cleanup_error(
            trigger,
            false,
            Vec::new(),
            format!(
                "SSH cleanup exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ),
        Err(error) => cleanup_error(trigger, false, Vec::new(), error.to_string()),
    }
}

/// Confirm a server container is gone from its launch machine
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]), mapping the shared removal outcome
/// onto this record's evidence shape.
pub(super) fn remove_server_container(
    target: Option<&str>,
    container: &str,
) -> ContainerRemovalEvidence {
    use crate::container::{Removal, RemovalFailure, remove_container};
    let started = Instant::now();
    let evidence =
        |confirmed: bool,
         already_absent: bool,
         error: Option<String>,
         operation_elapsed_ms: u64,
         client_cleanup: Option<crate::container::CommandCleanupEvidence>| {
            ContainerRemovalEvidence {
                container: container.to_owned(),
                elapsed_ms: duration_millis(started.elapsed()),
                operation_elapsed_ms,
                deadline_ms: duration_millis(crate::container::REMOVAL_TIMEOUT),
                client_cleanup,
                confirmed,
                already_absent,
                error,
            }
        };
    match remove_container(target, container) {
        Removal::Confirmed { already_absent } => evidence(
            true,
            already_absent,
            None,
            duration_millis(started.elapsed()),
            None,
        ),
        Removal::Unconfirmed(RemovalFailure::Exit { status, stderr }) => evidence(
            false,
            false,
            Some(format!(
                "docker rm -f exited with {status}: {}",
                stderr.trim()
            )),
            duration_millis(started.elapsed()),
            None,
        ),
        Removal::Unconfirmed(RemovalFailure::Deadline {
            operation_elapsed_ms,
            client_cleanup,
        }) => evidence(
            false,
            false,
            Some(format!(
                "docker rm -f {container} exceeded its {}s deadline",
                crate::container::REMOVAL_TIMEOUT.as_secs()
            )),
            operation_elapsed_ms,
            client_cleanup,
        ),
        Removal::Unconfirmed(RemovalFailure::Launch(error)) => evidence(
            false,
            false,
            Some(format!("docker rm failed to launch: {error}")),
            duration_millis(started.elapsed()),
            None,
        ),
        Removal::Unconfirmed(RemovalFailure::Wait(error)) => evidence(
            false,
            false,
            Some(format!("docker rm wait failed: {error}")),
            duration_millis(started.elapsed()),
            None,
        ),
        Removal::Unconfirmed(RemovalFailure::WaitCleanup {
            source,
            operation_elapsed_ms,
            client_cleanup,
        }) => evidence(
            false,
            false,
            Some(format!("docker rm wait failed: {source}")),
            operation_elapsed_ms,
            Some(client_cleanup),
        ),
        Removal::Unconfirmed(RemovalFailure::Ssh(error)) => evidence(
            false,
            false,
            Some(error),
            duration_millis(started.elapsed()),
            None,
        ),
    }
}

pub(super) fn completed_cleanup(
    trigger: CleanupTrigger,
    already_exited: bool,
    forced: bool,
    signals: Vec<SignalEvidence>,
) -> CleanupEvidence {
    CleanupEvidence {
        trigger,
        elapsed_ms: 0,
        status_deadline_ms: duration_millis(SERVER_CLEANUP_STATUS_DEADLINE),
        term_grace_ms: duration_millis(TERM_GRACE),
        kill_grace_ms: duration_millis(KILL_GRACE),
        reap_grace_ms: None,
        remote_deadline_ms: None,
        verified: true,
        already_exited,
        forced,
        signals,
        error: None,
        container_removal: None,
        device_residuals: None,
    }
}

pub(super) fn cleanup_error(
    trigger: CleanupTrigger,
    forced: bool,
    signals: Vec<SignalEvidence>,
    error: String,
) -> CleanupEvidence {
    CleanupEvidence {
        trigger,
        elapsed_ms: 0,
        status_deadline_ms: duration_millis(SERVER_CLEANUP_STATUS_DEADLINE),
        term_grace_ms: duration_millis(TERM_GRACE),
        kill_grace_ms: duration_millis(KILL_GRACE),
        reap_grace_ms: None,
        remote_deadline_ms: None,
        verified: false,
        already_exited: false,
        forced,
        signals,
        error: Some(error),
        container_removal: None,
        device_residuals: None,
    }
}

enum RemoteCleanupState {
    Cleanup,
    Already,
    Stale,
    Unknown,
}

struct RemoteCleanupOutput {
    state: RemoteCleanupState,
    term_code: Option<i32>,
    forced: bool,
    kill_code: Option<i32>,
    alive: bool,
    detail: String,
}

fn parse_cleanup_output(output: &str) -> Option<RemoteCleanupOutput> {
    let result = output
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(CLEANUP_MARKER))?;
    let mut fields = result.split('\t');
    let state = match fields.next()? {
        "cleanup" => RemoteCleanupState::Cleanup,
        "already" => RemoteCleanupState::Already,
        "stale" => RemoteCleanupState::Stale,
        "unknown" => RemoteCleanupState::Unknown,
        _ => return None,
    };
    let term_code = match fields.next()? {
        "-" => None,
        value => Some(value.parse().ok()?),
    };
    let forced = fields.next()? == "1";
    let kill_code = match fields.next()? {
        "-" => None,
        value => Some(value.parse().ok()?),
    };
    let alive = fields.next()? == "1";
    let detail = fields.next()?.to_owned();
    Some(RemoteCleanupOutput {
        state,
        term_code,
        forced,
        kill_code,
        alive,
        detail,
    })
}

fn remote_signal_evidence(
    signal: TerminationSignal,
    process_group: u32,
    exit_code: i32,
    stderr: &str,
) -> SignalEvidence {
    SignalEvidence {
        signal,
        process_group,
        exit_code: Some(exit_code),
        stderr: (!stderr.is_empty()).then(|| stderr.to_owned()),
        error: None,
    }
}

impl ProcessCleanup for SystemProcessRuntime {
    fn terminate(
        &self,
        handle: &ProcessHandle,
        trigger: CleanupTrigger,
        on_container_removal: &mut dyn FnMut(&str),
    ) -> CleanupEvidence {
        let mut evidence = match handle {
            ProcessHandle::Local(handle) => terminate_local(handle, trigger),
            ProcessHandle::Ssh(handle) => terminate_ssh(handle, trigger),
        };
        // The container is a daemon-owned object: the group kill reaches
        // only the docker client, so a known container must be confirmed
        // removed on its launch machine — unconditionally, because it can
        // survive every group state observed above
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let (container, target) = match handle {
            ProcessHandle::Local(handle) => (handle.container.as_deref(), None),
            ProcessHandle::Ssh(handle) => (handle.container.as_deref(), Some(&*handle.target)),
        };
        if let Some(container) = container {
            on_container_removal(container);
            let removal = remove_server_container(target, container);
            if !removal.confirmed {
                evidence.verified = false;
                if evidence.error.is_none() {
                    evidence.error = Some(format!(
                        "container {container} removal was not confirmed: {}",
                        removal.error.as_deref().unwrap_or("unknown outcome")
                    ));
                }
            }
            evidence.container_removal = Some(removal);
        }
        evidence
    }
}
