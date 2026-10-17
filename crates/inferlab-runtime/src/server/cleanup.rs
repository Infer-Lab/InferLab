use super::observation::{remote_group_alive_script, run_cleanup_command};
use super::{
    CleanupEvent, CleanupEvidence, CleanupTrigger, ContainerRemovalEvidence, HostProcessHandle,
    MarkedProcessSweepEvidence, ProcessCleanup, ProcessHandle, SERVER_PROCESS_MARKER_ENV,
    SshProcessHandle, SystemProcessRuntime,
};
use crate::completion::process::{ProcessExit, Processes};
use crate::completion::{ATTEMPT_BOUND, CompletionWaitEvidence};
use crate::operation_bound::{OperationBound, duration_millis};
use crate::process_group::{
    LocalProcessGroup, SignalEvidence, TERM_GRACE, TerminationSignal, VerifiedStatus,
};
use crate::shell::shell_quote;
use crate::ssh::{SSH_ENV_REMOVE, ssh_argv};
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_millis(100);
const SERVER_CLEANUP_STATUS_DEADLINE: Duration = Duration::from_secs(2);
pub(super) const TERM_POLL_LIMIT: u128 = TERM_GRACE.as_millis() / POLL_INTERVAL.as_millis();

/// The `sleep` operand of the remote TERM loops: the poll limit above counts
/// iterations of this same interval.
pub(super) fn remote_poll_sleep() -> String {
    format!(
        "{}.{:03}",
        POLL_INTERVAL.as_secs(),
        POLL_INTERVAL.subsec_millis()
    )
}
const CLEANUP_MARKER: &str = "INFERLAB_CLEANUP\t";
const SWEEP_MARKER: &str = "INFERLAB_SWEEP\t";

impl CleanupEvidence {
    pub fn unavailable(trigger: CleanupTrigger, message: String) -> Self {
        Self {
            trigger,
            elapsed_ms: 0,
            status_deadline_ms: duration_millis(SERVER_CLEANUP_STATUS_DEADLINE),
            term_grace_ms: duration_millis(TERM_GRACE),
            remote_attempt_bound_ms: None,
            verified: false,
            already_exited: false,
            forced: false,
            signals: Vec::new(),
            error: Some(message),
            exit_wait: None,
            container_removal: None,
            device_residuals: None,
            device_residual_wait: None,
            marked_process_sweep: None,
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
            remote_attempt_bound_ms: None,
            verified,
            already_exited: false,
            forced: false,
            signals: Vec::new(),
            error,
            exit_wait: None,
            container_removal: Some(removal),
            device_residuals: None,
            device_residual_wait: None,
            marked_process_sweep: None,
        }
    }
}

pub(super) fn cleanup_failed_local_launch(
    child: &mut std::process::Child,
    report: &mut dyn FnMut(CleanupEvent<'_>),
) -> CleanupEvidence {
    let started = Instant::now();
    let initial_status_error = match child.try_wait() {
        Ok(Some(_)) => {
            let mut evidence =
                completed_cleanup(CleanupTrigger::StartupRollback, true, false, Vec::new());
            evidence.elapsed_ms = duration_millis(started.elapsed());
            evidence.status_deadline_ms = 0;
            evidence.term_grace_ms = 0;
            return evidence;
        }
        Ok(None) => None,
        // The subsequent kill and exit wait are authoritative cleanup
        // verification. Preserve this diagnostic only if that verification
        // also fails; a reached exit resolves the transient status error.
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
            return evidence;
        }
    };
    let signal = group.send_signal(
        TerminationSignal::Kill,
        &OperationBound::finite(ATTEMPT_BOUND),
    );
    let wait = wait_for_exit(None, Processes::Group(group.process_group), report);
    // Every thread has exited, so reaping the leader returns at once.
    let reaped = if wait.reached() {
        child
            .wait()
            .map(|_| ())
            .map_err(|error| format!("failed to reap failed launch child: {error}"))
    } else {
        Err(wait.failure().unwrap_or_default())
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
    evidence.exit_wait = Some(wait);
    evidence.elapsed_ms = duration_millis(started.elapsed());
    evidence.status_deadline_ms = 0;
    evidence.term_grace_ms = 0;
    evidence
}

/// Wait until every thread of `processes` has exited on the launch machine
/// ([[RFC-0009:C-COMPLETION-WAITS]]).
pub(super) fn wait_for_exit(
    target: Option<&str>,
    processes: Processes,
    report: &mut dyn FnMut(CleanupEvent<'_>),
) -> CompletionWaitEvidence {
    crate::completion::wait(
        &mut ProcessExit {
            target: target.map(str::to_owned),
            processes,
        },
        &mut |message| report(CleanupEvent::Waiting(message)),
    )
}

/// The cleanup outcome once the group was signalled: verified exactly when
/// every thread of the group exited.
fn after_signals(
    trigger: CleanupTrigger,
    forced: bool,
    signals: Vec<SignalEvidence>,
    wait: CompletionWaitEvidence,
) -> CleanupEvidence {
    let mut evidence = match wait.failure() {
        None => completed_cleanup(trigger, false, forced, signals),
        Some(failure) => cleanup_error(trigger, forced, signals, failure),
    };
    evidence.exit_wait = Some(wait);
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
    report: &mut dyn FnMut(CleanupEvent<'_>),
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
    // The termination grace only decides escalation; whether the group has
    // exited is the exit wait's judgment, because a leader can read as gone
    // while its threads still release what they held.
    let forced = match group.wait_until_stopped(None, &term_bound, POLL_INTERVAL) {
        Ok(true) => false,
        Ok(false) => {
            signals.push(group.send_signal(
                TerminationSignal::Kill,
                &OperationBound::finite(ATTEMPT_BOUND),
            ));
            true
        }
        Err(error) => {
            let mut evidence = cleanup_error(trigger, false, signals, error.to_string());
            evidence.elapsed_ms = duration_millis(started.elapsed());
            return evidence;
        }
    };
    let wait = wait_for_exit(None, Processes::Group(handle.process_group), report);
    let mut evidence = after_signals(trigger, forced, signals, wait);
    evidence.elapsed_ms = duration_millis(started.elapsed());
    evidence
}

pub(super) fn terminate_ssh(
    handle: &SshProcessHandle,
    trigger: CleanupTrigger,
    report: &mut dyn FnMut(CleanupEvent<'_>),
) -> CleanupEvidence {
    let started = Instant::now();
    let bound = OperationBound::finite(ATTEMPT_BOUND);
    let mut evidence = terminate_ssh_under(handle, trigger, &bound, report);
    evidence.elapsed_ms = duration_millis(started.elapsed());
    evidence.remote_attempt_bound_ms = Some(duration_millis(ATTEMPT_BOUND));
    evidence
}

/// The remote termination script: it signals the group — SIGTERM, the
/// termination grace, SIGKILL to what remains — and returns; whether the
/// group exited is the exit wait's judgment. The leader-missing branch
/// replicates the local cohort check — every live member must start at or after the recorded
/// leader ticks, else the pgid reads as recycled and the script declines to
/// signal, naming the offending members in the detail field.
pub(super) fn remote_cleanup_script(handle: &SshProcessHandle) -> String {
    format!(
        "set +e; pgid={}; pid={}; expected={}; if [ -r /proc/$pid/stat ]; then actual=$(awk '{{print $22}}' /proc/$pid/stat); if [ $? -ne 0 ]; then printf '{marker}unknown\\t-\\t0\\t-\\tstat-unreadable\\n'; exit 0; fi; if [ \"$actual\" != \"$expected\" ]; then printf '{marker}stale\\t-\\t0\\t-\\t%s\\n' \"$actual\"; exit 0; fi; elif {}; then bad=\"\"; for mpid in $(ps -eo pid=,pgid=,stat= | awk -v pgid=\"$pgid\" '$2 == pgid && $3 !~ /^Z/ {{print $1}}'); do mticks=$(awk '{{print $22}}' /proc/$mpid/stat 2>/dev/null) || mticks=\"\"; if [ -n \"$mticks\" ] && [ \"$mticks\" -lt \"$expected\" ]; then bad=\"$bad $mpid:$mticks\"; fi; done; if [ -n \"$bad\" ]; then printf '{marker}unknown\\t-\\t0\\t-\\tleader-missing; cohort members%s predate recorded leader start %s\\n' \"$bad\" \"$expected\"; exit 0; fi; else printf '{marker}already\\t-\\t0\\t-\\t-\\n'; exit 0; fi; if ! {}; then printf '{marker}already\\t-\\t0\\t-\\t-\\n'; exit 0; fi; kill -TERM -- -$pgid; term_code=$?; i=0; while {} && [ $i -lt {term_limit} ]; do sleep {poll}; i=$((i+1)); done; forced=0; kill_code=-; if {}; then forced=1; kill -KILL -- -$pgid; kill_code=$?; fi; printf '{marker}cleanup\\t%s\\t%s\\t%s\\t-\\n' \"$term_code\" \"$forced\" \"$kill_code\"",
        handle.process_group,
        handle.leader_pid,
        handle.leader_start_time_ticks,
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        term_limit = TERM_POLL_LIMIT,
        poll = remote_poll_sleep(),
        marker = CLEANUP_MARKER,
    )
}

pub(super) fn terminate_ssh_under(
    handle: &SshProcessHandle,
    trigger: CleanupTrigger,
    bound: &OperationBound,
    report: &mut dyn FnMut(CleanupEvent<'_>),
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
            let wait = wait_for_exit(
                Some(&handle.target),
                Processes::Group(handle.process_group),
                report,
            );
            after_signals(trigger, result.forced, signals, wait)
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
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]], [[RFC-0009:C-COMPLETION-WAITS]]).
pub(super) fn remove_server_container(
    target: Option<&str>,
    container: &str,
    report: &mut dyn FnMut(CleanupEvent<'_>),
) -> ContainerRemovalEvidence {
    crate::container::remove_container(target, container, &mut |message| {
        report(CleanupEvent::Waiting(message));
    })
}

/// The marked-process sweep, one script for both launch paths: it lists
/// every live process whose initial environment holds the exact marker
/// entry — a zombie's environment reads empty, and another user's is
/// unreadable — with its start time, sends SIGTERM, polls through the
/// termination grace, sends SIGKILL to what still matches, and returns
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). The script's own processes inherit no
/// marker, so the listing never names the sweep itself. Whether the found
/// processes exited is the exit wait's judgment.
pub(super) fn marked_process_sweep_script(marker: &str) -> String {
    format!(
        "set +e; entry={entry}; marked() {{ grep -lsxzF -e \"$entry\" /proc/[0-9]*/environ | sed -n 's#^/proc/\\([0-9]*\\)/environ$#\\1#p' | tr '\\n' ' '; }}; found=$(marked); pairs=; for p in $found; do read -r line < /proc/$p/stat 2>/dev/null || continue; rest=${{line##*) }}; set -- $rest; pairs=\"$pairs $p:${{20}}\"; done; term=-; forced=0; kill_code=-; if [ -n \"$found\" ]; then kill -TERM $found 2>/dev/null; term=$?; i=0; left=$(marked); while [ -n \"$left\" ] && [ $i -lt {term_limit} ]; do sleep {poll}; i=$((i+1)); left=$(marked); done; if [ -n \"$left\" ]; then forced=1; kill -KILL $left 2>/dev/null; kill_code=$?; fi; fi; printf '{sweep}%s\\t%s\\t%s\\t%s\\n' \"$pairs\" \"$term\" \"$forced\" \"$kill_code\"",
        entry = shell_quote(&format!("{SERVER_PROCESS_MARKER_ENV}={marker}")),
        term_limit = TERM_POLL_LIMIT,
        poll = remote_poll_sleep(),
        sweep = SWEEP_MARKER,
    )
}

/// Run the marked-process sweep on a launch machine — locally, or over SSH
/// to `target` — then wait until every found process has exited.
pub(super) fn sweep_marked_processes(
    target: Option<&str>,
    marker: &str,
    report: &mut dyn FnMut(CleanupEvent<'_>),
) -> MarkedProcessSweepEvidence {
    let started = Instant::now();
    let script = marked_process_sweep_script(marker);
    let bound = OperationBound::finite(ATTEMPT_BOUND);
    let output = match target {
        None => run_cleanup_command(
            &["sh", "-c", script.as_str()],
            &[],
            &bound,
            "marked-process sweep",
        ),
        Some(target) => run_cleanup_command(
            &ssh_argv(target, &script),
            SSH_ENV_REMOVE,
            &bound,
            "SSH marked-process sweep",
        ),
    };
    let mut evidence = MarkedProcessSweepEvidence {
        marker: marker.to_owned(),
        elapsed_ms: 0,
        attempt_bound_ms: duration_millis(ATTEMPT_BOUND),
        found: Vec::new(),
        term_exit_code: None,
        forced: false,
        kill_exit_code: None,
        exit_wait: None,
        error: None,
    };
    match output {
        Ok(output) if output.status.success() => {
            match parse_sweep_output(&String::from_utf8_lossy(&output.stdout)) {
                Some(result) => {
                    evidence.found = result.found.iter().map(|(pid, _)| *pid).collect();
                    evidence.term_exit_code = result.term_exit_code;
                    evidence.forced = result.forced;
                    evidence.kill_exit_code = result.kill_exit_code;
                    if !result.found.is_empty() {
                        let wait = wait_for_exit(target, Processes::Exact(result.found), report);
                        if let Some(failure) = wait.failure() {
                            evidence.error = Some(format!(
                                "processes carrying {SERVER_PROCESS_MARKER_ENV}={marker}: {failure}"
                            ));
                        }
                        evidence.exit_wait = Some(wait);
                    }
                }
                None => {
                    evidence.error = Some("marked-process sweep returned no result".to_owned());
                }
            }
        }
        Ok(output) => {
            evidence.error = Some(format!(
                "marked-process sweep exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Err(error) => evidence.error = Some(error.to_string()),
    }
    evidence.elapsed_ms = duration_millis(started.elapsed());
    evidence
}

struct SweepOutput {
    found: Vec<(u32, u64)>,
    term_exit_code: Option<i32>,
    forced: bool,
    kill_exit_code: Option<i32>,
}

fn parse_sweep_output(output: &str) -> Option<SweepOutput> {
    let result = output
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(SWEEP_MARKER))?;
    let mut fields = result.split('\t');
    let found = fields
        .next()?
        .split_whitespace()
        .map(|pair| {
            let (pid, start) = pair.split_once(':')?;
            Some((pid.parse().ok()?, start.parse().ok()?))
        })
        .collect::<Option<Vec<_>>>()?;
    let code = |field: &str| match field {
        "-" => Some(None),
        value => value.parse::<i32>().ok().map(Some),
    };
    let term_exit_code = code(fields.next()?)?;
    let forced = fields.next()? == "1";
    let kill_exit_code = code(fields.next()?)?;
    Some(SweepOutput {
        found,
        term_exit_code,
        forced,
        kill_exit_code,
    })
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
        remote_attempt_bound_ms: None,
        verified: true,
        already_exited,
        forced,
        signals,
        error: None,
        exit_wait: None,
        container_removal: None,
        device_residuals: None,
        device_residual_wait: None,
        marked_process_sweep: None,
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
        remote_attempt_bound_ms: None,
        verified: false,
        already_exited: false,
        forced,
        signals,
        error: Some(error),
        exit_wait: None,
        container_removal: None,
        device_residuals: None,
        device_residual_wait: None,
        marked_process_sweep: None,
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
    let detail = fields.next()?.to_owned();
    Some(RemoteCleanupOutput {
        state,
        term_code,
        forced,
        kill_code,
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
        report: &mut dyn FnMut(CleanupEvent<'_>),
    ) -> CleanupEvidence {
        let mut evidence = match handle {
            ProcessHandle::Local(handle) => terminate_local(handle, trigger, report),
            ProcessHandle::Ssh(handle) => terminate_ssh(handle, trigger, report),
        };
        // A launch tool can move the serving tree out of the group the kill
        // above reached; every process still carrying the marker is this
        // server's, so the sweep runs even when the group had already
        // exited — the case in which nothing else can reach the tree
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let (marker, sweep_target) = match handle {
            ProcessHandle::Local(handle) => (handle.owner_marker.as_deref(), None),
            ProcessHandle::Ssh(handle) => (handle.owner_marker.as_deref(), Some(&*handle.target)),
        };
        if let Some(marker) = marker {
            let sweep = sweep_marked_processes(sweep_target, marker, report);
            if !sweep.verified() {
                evidence.verified = false;
                if evidence.error.is_none() {
                    evidence.error.clone_from(&sweep.error);
                }
            }
            evidence.marked_process_sweep = Some(sweep);
        }
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
            report(CleanupEvent::ContainerRemoval(container));
            let removal = remove_server_container(target, container, report);
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
