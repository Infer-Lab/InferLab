//! Recorded ad-hoc execution ([[RFC-0005:C-EVIDENCE]]): the run record
//! keeps what InferLab set and observed — never the command's outputs or
//! results, which stay the command's own.

use super::context::{self, ModelContext, ServeContext};
use crate::InferlabError;
use crate::record::{RECORD_FILE, RECORDS_DIR, RecordIdentity, new_record_id, now_unix_ms};
use crate::workspace::WorkspaceSnapshot;
use inferlab_runtime::operation_bound::OperationBound;
use inferlab_runtime::process_group::{
    KILL_GRACE, LocalProcessGroup, TERM_GRACE, TerminationSignal, TreeProcess, process_tree,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const SCHEMA_VERSION: u32 = 1;
/// Polling cadence while waiting for the command; not a budget
/// ([[RFC-0009:C-OPERATION-BUDGETS]]).
const POLL_INTERVAL: Duration = Duration::from_millis(100);
const CLEANUP_SCOPE: &str = "on interrupt, InferLab terminates the process group it started and the \
     process groups of the command's live descendants; processes that left that process tree \
     and containers started through a daemon are not cleaned up by InferLab";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RunStatus {
    Running,
    Succeeded,
    Failed,
    Interrupted,
}

#[derive(Debug, Deserialize, Serialize)]
struct StackEvidence {
    id: String,
    pixi_environment: String,
}

/// The variables InferLab set: the provided context and the device
/// projection. The operator's inherited environment is not InferLab's to
/// record.
#[derive(Debug, Deserialize, Serialize)]
struct ContextEvidence {
    provided: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cuda_visible_devices: Option<String>,
    note: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ServerState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    observed_alive: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ServeEvidence {
    #[serde(flatten)]
    link: ServeContext,
    state_at_start: ServerState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    state_at_exit: Option<ServerState>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ExitEvidence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signal: Option<i32>,
}

/// Termination evidence ([[RFC-0009:C-CLEANUP-GRACE]]): trigger, the shared
/// graces, the signaled groups, elapsed time, and verification.
#[derive(Debug, Deserialize, Serialize)]
struct CleanupEvidence {
    trigger: String,
    term_grace_ms: u64,
    kill_grace_ms: u64,
    process_groups: Vec<u32>,
    elapsed_ms: u64,
    term_sent: bool,
    kill_sent: bool,
    verified: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct RunRecord {
    schema_version: u32,
    kind: String,
    id: String,
    status: RunStatus,
    inferlab_version: String,
    started_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    finished_unix_ms: Option<u64>,
    stack: StackEvidence,
    workspace: WorkspaceSnapshot,
    argv: Vec<String>,
    executed_argv: Vec<String>,
    cwd: PathBuf,
    context: ContextEvidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    serve: Option<ServeEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<ModelContext>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent_record: Option<String>,
    stdout: String,
    stderr: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exit_status: Option<ExitEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cleanup: Option<CleanupEvidence>,
    cleanup_scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub(super) struct RecordedRun<'a> {
    pub root: &'a Path,
    pub snapshot: WorkspaceSnapshot,
    pub stack_id: &'a str,
    pub pixi_environment: &'a str,
    pub argv: &'a [String],
    pub executed_argv: &'a [String],
    pub strip: &'a [String],
    pub cuda_visible_devices: Option<String>,
    pub serve: Option<ServeContext>,
    pub model: Option<ModelContext>,
    pub parent_record: Option<String>,
}

/// Execute the command under a run record and return its exit code. The
/// command runs in its own process group so an interrupted run can
/// terminate it; its output streams are copied to the operator's streams
/// and to the record's logs.
pub(super) fn execute(run: RecordedRun<'_>) -> Result<i32, InferlabError> {
    let id = new_record_id(RecordIdentity::Run {
        stack: run.stack_id,
    })?;
    let records = run.root.join(RECORDS_DIR);
    let dir = records.join(&id);
    let artifacts = dir.join("artifacts");
    create_dir(&records, true)?;
    create_dir(&dir, false)?;
    create_dir(&artifacts, false)?;
    let relative = Path::new(RECORDS_DIR).join(&id);
    let provided = context::variables(
        run.root,
        Some((&id, &artifacts)),
        run.serve.as_ref(),
        run.model.as_ref(),
    );
    let serve_state = run
        .serve
        .as_ref()
        .map(|serve| server_state(crate::server::status(run.root, &serve.record_id)));
    let mut record = RunRecord {
        schema_version: SCHEMA_VERSION,
        kind: "run".to_owned(),
        id: id.clone(),
        status: RunStatus::Running,
        inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
        started_unix_ms: now_unix_ms()?,
        finished_unix_ms: None,
        stack: StackEvidence {
            id: run.stack_id.to_owned(),
            pixi_environment: run.pixi_environment.to_owned(),
        },
        workspace: run.snapshot,
        argv: run.argv.to_vec(),
        executed_argv: run.executed_argv.to_vec(),
        cwd: std::env::current_dir().map_err(|source| InferlabError::AdHocRun {
            message: format!("failed to read the working directory: {source}"),
        })?,
        context: ContextEvidence {
            provided: provided.clone(),
            cuda_visible_devices: run.cuda_visible_devices.clone(),
            note: "provided to the command; InferLab does not verify that the command used it"
                .to_owned(),
        },
        serve: run.serve.map(|link| ServeEvidence {
            link,
            state_at_start: serve_state.unwrap_or(ServerState {
                status: None,
                observed_alive: None,
                error: None,
            }),
            state_at_exit: None,
        }),
        model: run.model,
        parent_record: run.parent_record,
        stdout: relative.join("stdout.log").display().to_string(),
        stderr: relative.join("stderr.log").display().to_string(),
        exit_status: None,
        cleanup: None,
        cleanup_scope: CLEANUP_SCOPE.to_owned(),
        error: None,
    };
    let record_path = dir.join(RECORD_FILE);
    crate::record::write_json(&record_path, &record)?;
    eprintln!("inferlab: run record {id}");
    // Every fallible step that can precede the command's exit happens before
    // it starts, so a started command is always waited for and finalized.
    let logs = create_log(&dir.join("stdout.log"))
        .and_then(|stdout| Ok((stdout, create_log(&dir.join("stderr.log"))?)));
    let (stdout_log, stderr_log) = match logs {
        Ok(logs) => logs,
        Err(error) => {
            record.error = Some(error.to_string());
            finalize(&mut record, &record_path, run.root, RunStatus::Failed)?;
            return Err(error);
        }
    };

    let mut command = Command::new(&run.executed_argv[0]);
    command.args(&run.executed_argv[1..]);
    for name in run.strip {
        command.env_remove(name);
    }
    command.envs(&provided);
    if let Some(devices) = &run.cuda_visible_devices {
        command.env("CUDA_VISIBLE_DEVICES", devices);
    }
    // Its own process group keeps the terminal's job-control signals away
    // from the command, so a terminal stdin would stop it on first read: a
    // recorded run is non-interactive. Piped or redirected stdin still
    // flows through.
    let stdin = if std::io::stdin().is_terminal() {
        Stdio::null()
    } else {
        Stdio::inherit()
    };
    command
        .process_group(0)
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(source) => {
            let message = format!("failed to launch {:?}: {source}", run.executed_argv[0]);
            record.error = Some(message.clone());
            finalize(&mut record, &record_path, run.root, RunStatus::Failed)?;
            return Err(InferlabError::AdHocRun { message });
        }
    };
    let (drained_tx, drained_rx) = mpsc::channel();
    let mut copies = 0;
    if let Some(stdout) = child.stdout.take() {
        copies += 1;
        tee(
            stdout,
            std::io::stdout(),
            stdout_log,
            dir.join("stdout.log"),
            drained_tx.clone(),
        );
    }
    if let Some(stderr) = child.stderr.take() {
        copies += 1;
        tee(
            stderr,
            std::io::stderr(),
            stderr_log,
            dir.join("stderr.log"),
            drained_tx,
        );
    }
    let waited = wait(&mut child, || {
        // Terminal before cleanup: an interrupted run's record says so even
        // if an outer run's escalation ends this process mid-cleanup.
        record.status = RunStatus::Interrupted;
        let _ = crate::record::write_json(&record_path, &record);
    });
    let (status, cleanup) = match waited {
        Ok(waited) => waited,
        Err(error) => {
            record.cleanup = Some(terminate(&mut child, "wait failure"));
            record.error = Some(error.to_string());
            finalize(&mut record, &record_path, run.root, RunStatus::Failed)?;
            return Err(error);
        }
    };
    let deadline = std::time::Instant::now() + inferlab_runtime::container::COMMAND_IO_DRAIN_GRACE;
    let mut undrained = Vec::new();
    for _ in 0..copies {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        match drained_rx.recv_timeout(remaining) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => undrained.push(error),
            Err(_) => undrained
                .push("output is still held open by a process the command left running".to_owned()),
        }
    }
    if !undrained.is_empty() {
        record.error = Some(undrained.join("; "));
    }
    record.exit_status = Some(exit_evidence(status));
    let outcome = if cleanup.is_some() {
        RunStatus::Interrupted
    } else if status.success() {
        RunStatus::Succeeded
    } else {
        RunStatus::Failed
    };
    record.cleanup = cleanup;
    finalize(&mut record, &record_path, run.root, outcome)?;
    Ok(super::exit_code(status))
}

fn finalize(
    record: &mut RunRecord,
    path: &Path,
    root: &Path,
    status: RunStatus,
) -> Result<(), InferlabError> {
    // Read exactly as at start: the run has no deadline to bound it, and a
    // remote read follows the target's own SSH configuration.
    if let Some(serve) = record.serve.as_mut() {
        serve.state_at_exit = Some(server_state(crate::server::status(
            root,
            &serve.link.record_id,
        )));
    }
    record.status = status;
    record.finished_unix_ms = Some(now_unix_ms()?);
    crate::record::write_json(path, record)
}

/// Wait for the command; an interrupt delivered to InferLab terminates the
/// command's process tree instead ([[RFC-0005:C-EVIDENCE]]).
fn wait(
    child: &mut Child,
    mut on_interrupt: impl FnMut(),
) -> Result<(ExitStatus, Option<CleanupEvidence>), InferlabError> {
    let reap = |source: std::io::Error| InferlabError::AdHocRun {
        message: format!("failed to wait for the command: {source}"),
    };
    loop {
        if let Some(status) = child.try_wait().map_err(reap)? {
            return Ok((status, None));
        }
        if inferlab_runtime::interrupt::received() {
            on_interrupt();
            let cleanup = terminate(child, "interrupt");
            let status = child.wait().map_err(reap)?;
            return Ok((status, Some(cleanup)));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// SIGTERM every process group of the command's process tree, then SIGKILL
/// the tree's processes that are still alive. A nested `inferlab run` is
/// spared SIGKILL while the grace lasts: the SIGKILL of the processes it
/// started ends its command, and it then finalizes its own record
/// ([[RFC-0005:C-EVIDENCE]]). Every signal targets a process identified by
/// pid and start time, so a reused pid is never signaled.
fn terminate(child: &mut Child, trigger: &str) -> CleanupEvidence {
    let started = std::time::Instant::now();
    let leader = child.id();
    let own_executable = std::env::current_exe().ok();
    let is_nested_run =
        |process: &TreeProcess| own_executable.is_some() && process.executable() == own_executable;
    let mut errors = Vec::new();
    let mut tree: Vec<TreeProcess> = Vec::new();
    let observe =
        |tree: &mut Vec<TreeProcess>, bound: &OperationBound| match process_tree(leader, bound) {
            Ok(found) => {
                for process in found {
                    if !tree.contains(&process) {
                        tree.push(process);
                    }
                }
                None
            }
            // A query cut short by the expiring grace means "not yet stopped".
            Err(_) if bound.is_expired() => None,
            Err(error) => Some(error.to_string()),
        };

    let term = OperationBound::finite(TERM_GRACE);
    errors.extend(observe(&mut tree, &term));
    let mut groups = tree
        .iter()
        .map(|process| process.process_group)
        .collect::<Vec<_>>();
    if groups.is_empty() {
        groups.push(leader);
    }
    groups.dedup();
    let mut term_sent = true;
    for group in &groups {
        if let Some(error) = LocalProcessGroup::unverified(*group)
            .send_signal(TerminationSignal::Term, &term)
            .error
        {
            errors.push(error);
            term_sent = false;
        }
    }
    let mut verified = loop {
        let _ = child.try_wait();
        errors.extend(observe(&mut tree, &term));
        if tree.iter().all(|process| !process.is_alive()) {
            break true;
        }
        if term.is_expired() {
            break false;
        }
        std::thread::sleep(POLL_INTERVAL);
    };

    let mut kill_sent = false;
    if !verified {
        let kill = OperationBound::finite(KILL_GRACE);
        verified = loop {
            let _ = child.try_wait();
            errors.extend(observe(&mut tree, &kill));
            let alive = tree
                .iter()
                .filter(|process| process.is_alive())
                .collect::<Vec<_>>();
            if alive.is_empty() {
                break true;
            }
            for process in alive.iter().filter(|process| !is_nested_run(process)) {
                match process.kill() {
                    Ok(()) => kill_sent = true,
                    Err(error) => {
                        errors.push(format!("failed to SIGKILL {}: {error}", process.pid))
                    }
                }
            }
            if kill.is_expired() {
                break false;
            }
            std::thread::sleep(POLL_INTERVAL);
        };
        if !verified {
            for process in tree.iter().filter(|process| process.is_alive()) {
                let _ = process.kill();
                errors.push(format!(
                    "process {} was still alive when the SIGKILL grace ended",
                    process.pid
                ));
            }
        }
    }
    for process in &tree {
        if !groups.contains(&process.process_group) {
            groups.push(process.process_group);
        }
    }
    CleanupEvidence {
        trigger: trigger.to_owned(),
        term_grace_ms: inferlab_runtime::operation_bound::duration_millis(TERM_GRACE),
        kill_grace_ms: inferlab_runtime::operation_bound::duration_millis(KILL_GRACE),
        process_groups: groups,
        elapsed_ms: inferlab_runtime::operation_bound::duration_millis(started.elapsed()),
        term_sent,
        kill_sent,
        verified,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

fn server_state(report: Result<crate::server::ServerStatusReport, InferlabError>) -> ServerState {
    match report {
        Ok(report) => ServerState {
            status: Some(report.record.status.as_str().to_owned()),
            observed_alive: Some(report.observed_alive),
            error: None,
        },
        Err(error) => ServerState {
            status: None,
            observed_alive: None,
            error: Some(error.to_string()),
        },
    }
}

fn exit_evidence(status: ExitStatus) -> ExitEvidence {
    use std::os::unix::process::ExitStatusExt;
    ExitEvidence {
        code: status.code(),
        signal: status.signal(),
    }
}

fn create_log(path: &Path) -> Result<std::fs::File, InferlabError> {
    std::fs::File::create(path).map_err(|source| InferlabError::RecordIo {
        path: path.to_path_buf(),
        source,
    })
}

/// Copy one command stream to the operator's stream and its log file, then
/// report through `drained` once the stream closes.
fn tee(
    mut source: impl Read + Send + 'static,
    mut operator: impl Write + Send + 'static,
    mut file: std::fs::File,
    log: PathBuf,
    drained: mpsc::Sender<Result<(), String>>,
) {
    std::thread::spawn(move || {
        let mut buffer = [0_u8; 8192];
        let result = loop {
            match source.read(&mut buffer) {
                Ok(0) => break Ok(()),
                Ok(read) => {
                    let chunk = &buffer[..read];
                    // The operator's stream may close (a closed pager);
                    // the log keeps the whole output regardless.
                    let _ = operator.write_all(chunk).and_then(|()| operator.flush());
                    if let Err(error) = file.write_all(chunk) {
                        break Err(format!("failed to write {}: {error}", log.display()));
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => break Err(format!("failed to read command output: {error}")),
            }
        };
        let _ = drained.send(result);
    });
}

fn create_dir(path: &Path, existing_ok: bool) -> Result<(), InferlabError> {
    let result = if existing_ok {
        std::fs::create_dir_all(path)
    } else {
        std::fs::create_dir(path)
    };
    result.map_err(|source| InferlabError::RecordIo {
        path: path.to_path_buf(),
        source,
    })
}
