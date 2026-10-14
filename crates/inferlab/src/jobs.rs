//! Detached jobs ([[RFC-0012:C-ACTIONS]], [[ADR-0056]]). A launch writes the
//! job's vector into `.inferlab/runtime/jobs/<job-id>/` and starts a hidden
//! supervisor in its own session; the supervisor starts the CLI in its own
//! process group with separate stdout and stderr logs, records the CLI's
//! producer identity, waits, and records its exit status. Job files are
//! run-time conveniences, not evidence.

use crate::InferlabError;
use crate::operation::{ProducerIdentity, process_identity};
use crate::record::state_dir;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

pub(crate) const JOBS_DIR: &str = concat!(state_dir!(), "/runtime/jobs");
const JOB_FILE: &str = "job.json";
const PROCESS_FILE: &str = "process.json";
const EXIT_FILE: &str = "exit.json";
const RECORD_FILE: &str = "record.json";
/// Names a job's directory to its command, which links its first record
/// there ([[RFC-0012:C-ACTIONS]]).
const JOB_DIR_ENV: &str = "INFERLAB_JOB_DIR";
const DETACH_POLL_INTERVAL: Duration = Duration::from_millis(100);
pub(crate) const STDOUT_LOG: &str = "stdout.log";
pub(crate) const STDERR_LOG: &str = "stderr.log";
/// Version 2 adds the working directory of a CLI-detached job; readers
/// accept both, since a version-1 file is a version-2 file without it.
const SCHEMA_VERSION: u32 = 2;
/// How long a launch waits for the supervisor to confirm the CLI started.
const START_DEADLINE: Duration = Duration::from_secs(30);
const SIGINT: i32 = 2;

/// What the service records before the supervisor starts.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JobSpec {
    pub(crate) schema_version: u32,
    pub(crate) action: String,
    pub(crate) argv: Vec<String>,
    pub(crate) started_unix_ms: u64,
    /// The working directory of a CLI-detached invocation, so its relative
    /// paths resolve as they would have; console jobs run at the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cwd: Option<PathBuf>,
}

/// The first record a job's command created ([[RFC-0012:C-ACTIONS]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JobRecord {
    pub(crate) record_id: String,
    pub(crate) record_dir: PathBuf,
}

/// How the CLI process ended, as its supervisor observed it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JobExit {
    pub(crate) exit_code: Option<i32>,
    pub(crate) signal: Option<i32>,
    pub(crate) ended_unix_ms: u64,
}

/// What the page may say about a job; it never infers an outcome.
pub(crate) enum JobState {
    Running,
    Exited(JobExit),
    Unknown,
    /// Recorded on another host or boot, so this console cannot observe it.
    Elsewhere,
}

impl JobState {
    pub(crate) fn label(&self) -> String {
        match self {
            Self::Running => "running".to_owned(),
            Self::Exited(JobExit {
                exit_code: Some(code),
                ..
            }) => format!("exited {code}"),
            Self::Exited(JobExit {
                signal: Some(SIGINT),
                ..
            }) => "interrupted (signal 2)".to_owned(),
            Self::Exited(JobExit {
                signal: Some(signal),
                ..
            }) => format!("ended by signal {signal}"),
            Self::Exited(_) | Self::Unknown => "ended, status unknown".to_owned(),
            Self::Elsewhere => "not observable from this host".to_owned(),
        }
    }
}

pub(crate) struct Job {
    pub(crate) id: String,
    pub(crate) directory: PathBuf,
    pub(crate) spec: Option<JobSpec>,
    pub(crate) producer: Option<ProducerIdentity>,
    pub(crate) state: JobState,
    /// The first record the job's command created, once it exists.
    pub(crate) record: Option<JobRecord>,
}

/// Every job of a workspace, newest first, read from its job directories so
/// that a restarted service finds them again.
pub(crate) fn list(root: &Path) -> Vec<Job> {
    let Ok(entries) = fs::read_dir(root.join(JOBS_DIR)) else {
        return Vec::new();
    };
    let mut jobs = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| read(root, &entry.file_name().to_string_lossy()))
        .collect::<Vec<_>>();
    jobs.sort_by(|left, right| {
        let started = |job: &Job| job.spec.as_ref().map_or(0, |spec| spec.started_unix_ms);
        started(right)
            .cmp(&started(left))
            .then_with(|| right.id.cmp(&left.id))
    });
    jobs
}

pub(crate) fn read(root: &Path, id: &str) -> Option<Job> {
    if crate::record::validate_record_id("job", id).is_err() {
        return None;
    }
    let directory = root.join(JOBS_DIR).join(id);
    if !directory.is_dir() {
        return None;
    }
    let spec = read_json::<JobSpec>(&directory.join(JOB_FILE));
    let producer = read_json::<ProducerIdentity>(&directory.join(PROCESS_FILE));
    let record = read_json::<JobRecord>(&directory.join(RECORD_FILE));
    let state = match read_json::<JobExit>(&directory.join(EXIT_FILE)) {
        Some(exit) => JobState::Exited(exit),
        None if producer.as_ref().is_some_and(is_live) => JobState::Running,
        None if producer
            .as_ref()
            .is_some_and(|producer| !is_local(producer)) =>
        {
            JobState::Elsewhere
        }
        None => JobState::Unknown,
    };
    Some(Job {
        id: id.to_owned(),
        directory,
        spec,
        producer,
        state,
        record,
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

/// Whether the recorded process ran on this host in this boot, where its
/// liveness can be observed.
fn is_local(producer: &ProducerIdentity) -> bool {
    process_identity(std::process::id())
        .is_ok_and(|local| local.host == producer.host && local.boot_id == producer.boot_id)
}

/// Whether the recorded process is the one alive now: same host, same boot,
/// and the same start time for its process identifier.
fn is_live(producer: &ProducerIdentity) -> bool {
    process_identity(producer.pid).is_ok_and(|current| current == *producer)
}

/// Create the job directory and start its supervisor. A launch that cannot
/// start leaves no job directory and returns its error.
pub(crate) fn launch(
    root: &Path,
    action: &str,
    argv: Vec<String>,
    started_unix_ms: u64,
    cwd: Option<PathBuf>,
) -> Result<String, String> {
    let jobs = root.join(JOBS_DIR);
    fs::create_dir_all(&jobs)
        .map_err(|error| format!("could not create {}: {error}", jobs.display()))?;
    let timestamp =
        crate::record::utc_timestamp(started_unix_ms).map_err(|error| error.to_string())?;
    let (id, directory) = loop {
        let id = format!("{timestamp}-{action}-{}", random_suffix()?);
        let directory = jobs.join(&id);
        match fs::create_dir(&directory) {
            Ok(()) => break (id, directory),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(format!("could not create {}: {error}", directory.display()));
            }
        }
    };
    let spec = JobSpec {
        schema_version: SCHEMA_VERSION,
        action: action.to_owned(),
        argv,
        started_unix_ms,
        cwd,
    };
    match start(&directory, &spec) {
        Ok(()) => Ok(id),
        Err(error) => {
            let _ = fs::remove_dir_all(&directory);
            Err(error)
        }
    }
}

fn random_suffix() -> Result<String, String> {
    let mut bytes = [0_u8; 3];
    fs::File::open("/dev/urandom")
        .and_then(|mut source| source.read_exact(&mut bytes))
        .map_err(|error| format!("could not read the secure random source: {error}"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn start(directory: &Path, spec: &JobSpec) -> Result<(), String> {
    crate::atomic_json::write(&directory.join(JOB_FILE), spec)
        .map_err(|error| format!("could not write the job: {error}"))?;
    let executable = spec
        .argv
        .first()
        .ok_or_else(|| "the job has no executable".to_owned())?;
    let mut supervisor = Command::new(executable)
        .args(["__internal", "job-supervise"])
        .arg(directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("could not start the job supervisor: {error}"))?;
    let stdout = supervisor
        .stdout
        .take()
        .ok_or_else(|| "the job supervisor has no handshake channel".to_owned())?;
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = BufReader::new(stdout).read_line(&mut line);
        let _ = sender.send(line);
    });
    let handshake = receiver.recv_timeout(START_DEADLINE);
    // A slow handshake does not mean a failed start: once the job's process
    // is recorded, the job is running and its directory stays.
    let started = matches!(&handshake, Ok(line) if line.trim() == "started")
        || directory.join(PROCESS_FILE).is_file();
    if !started {
        // Not confirmed: stop the supervisor and report what it said.
        let _ = supervisor.kill();
    }
    // The supervisor outlives this request; reap it whenever it ends.
    std::thread::spawn(move || {
        let _ = supervisor.wait();
    });
    match handshake {
        _ if started => Ok(()),
        Ok(line) => Err(line.trim().strip_prefix("failed:").map_or_else(
            || "the job supervisor exited before starting the job".to_owned(),
            |message| message.trim().to_owned(),
        )),
        Err(_) => Err("the job supervisor did not confirm the start in time".to_owned()),
    }
}

/// Deliver the interrupt a terminal Ctrl+C delivers to the job's process
/// group, only when the recorded identity is the live process.
pub(crate) fn cancel(job: &Job) -> Result<(), String> {
    let Some(producer) = &job.producer else {
        return Err("the job recorded no process, so nothing was signalled".to_owned());
    };
    if matches!(job.state, JobState::Exited(_)) || !is_live(producer) {
        return Err(
            "the job's recorded process is not running on this host and boot, so nothing was signalled"
                .to_owned(),
        );
    }
    let group = i32::try_from(producer.pid)
        .ok()
        .and_then(rustix::process::Pid::from_raw)
        .ok_or_else(|| format!("invalid process identifier {}", producer.pid))?;
    rustix::process::kill_process_group(group, rustix::process::Signal::INT).map_err(|error| {
        format!(
            "could not interrupt process group {}: {error}",
            producer.pid
        )
    })
}

/// The hidden supervisor: start the job's CLI invocation detached, record
/// its identity and exit status, and survive interrupts aimed at it.
pub(crate) fn supervise(directory: &Path) -> Result<(), InferlabError> {
    let failure = |message: String| {
        let _ = writeln!(std::io::stdout(), "failed: {message}");
        InferlabError::WebJob {
            path: directory.to_path_buf(),
            message,
        }
    };
    // A new session leaves the service's terminal and process group.
    rustix::process::setsid()
        .map_err(|error| failure(format!("could not start a session: {error}")))?;
    // Catching, not ignoring, keeps the CLI's default dispositions.
    inferlab_runtime::interrupt::prepare()
        .map_err(|error| failure(format!("could not install signal handling: {error}")))?;
    let spec = read_json::<JobSpec>(&directory.join(JOB_FILE))
        .ok_or_else(|| failure("the job description is unreadable".to_owned()))?;
    let (executable, arguments) = spec
        .argv
        .split_first()
        .ok_or_else(|| failure("the job has no executable".to_owned()))?;
    let log = |name: &str| {
        fs::File::create_new(directory.join(name))
            .map_err(|error| failure(format!("could not create {name}: {error}")))
    };
    let (stdout, stderr) = (log(STDOUT_LOG)?, log(STDERR_LOG)?);
    // jobs/<id> sits under <root>/.inferlab/runtime.
    let root = directory.ancestors().nth(4).unwrap_or(directory);
    let mut child = Command::new(executable)
        .args(arguments)
        .current_dir(spec.cwd.as_deref().unwrap_or(root))
        .env(JOB_DIR_ENV, directory)
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .map_err(|error| failure(format!("could not start {executable}: {error}")))?;
    let recorded = process_identity(child.id())
        .map_err(|error| error.to_string())
        .and_then(|identity| {
            crate::atomic_json::write(&directory.join(PROCESS_FILE), &identity)
                .map_err(|error| error.to_string())
        });
    if let Err(error) = recorded {
        // Without a recorded identity the job could never be cancelled.
        let _ = child.kill();
        let _ = child.wait();
        return Err(failure(format!(
            "could not record the job's process: {error}"
        )));
    }
    let _ = writeln!(std::io::stdout(), "started");
    let _ = std::io::stdout().flush();
    let status = child.wait().map_err(|error| InferlabError::WebJob {
        path: directory.to_path_buf(),
        message: format!("could not wait for the job: {error}"),
    })?;
    let exit = JobExit {
        exit_code: status.code(),
        signal: status.signal(),
        ended_unix_ms: crate::record::now_unix_ms().unwrap_or_default(),
    };
    crate::atomic_json::write(&directory.join(EXIT_FILE), &exit).map_err(|error| {
        InferlabError::WebJob {
            path: directory.to_path_buf(),
            message: format!("could not record the exit status: {error}"),
        }
    })
}

/// Link the record this process created into its job directory when it runs
/// as a job. Only the first record links — a recipe's, before the server and
/// Bench records it creates — and a failure leaves the run itself unaffected.
pub(crate) fn link_record(record_id: &str, record_dir: &Path) {
    let Some(directory) = std::env::var_os(JOB_DIR_ENV).map(PathBuf::from) else {
        return;
    };
    let target = directory.join(RECORD_FILE);
    if target.exists() {
        return;
    }
    let staged = directory.join(format!("{RECORD_FILE}.{}", std::process::id()));
    let record = JobRecord {
        record_id: record_id.to_owned(),
        record_dir: record_dir.to_path_buf(),
    };
    if crate::atomic_json::write(&staged, &record).is_ok() {
        // A hard link publishes atomically and refuses an existing target,
        // so the first record wins.
        let _ = fs::hard_link(&staged, &target);
    }
    let _ = fs::remove_file(&staged);
}

/// What a detached invocation reports once its job's first record exists.
#[derive(Debug, Serialize)]
pub(crate) struct DetachedJob {
    pub(crate) job_id: String,
    pub(crate) job_dir: PathBuf,
    pub(crate) stdout: PathBuf,
    pub(crate) stderr: PathBuf,
    pub(crate) record_id: String,
    pub(crate) record_dir: PathBuf,
}

/// Launch this invocation without `--detach` as a job and wait until its
/// command links its first record or exits without one
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). The wait is unbounded and ends on the
/// job's exit or an operator interrupt, which never signals the job.
pub(crate) fn detach(
    root: &Path,
    action: &str,
    argv: Vec<String>,
    cwd: PathBuf,
) -> Result<DetachedJob, InferlabError> {
    inferlab_runtime::interrupt::prepare()
        .map_err(|source| InferlabError::ServerInterrupt { source })?;
    let started = crate::record::now_unix_ms()?;
    let failure = |job_id: &str, message: String| InferlabError::DetachedJob {
        job_id: job_id.to_owned(),
        message,
    };
    let job_id = launch(root, action, argv, started, Some(cwd))
        .map_err(|message| failure("not started", message))?;
    let job_dir = root.join(JOBS_DIR).join(&job_id);
    // Diagnostics name the job at once, so an interrupted wait never leaves
    // a running job without a name.
    eprintln!(
        "detached job {job_id} started in {}; waiting for its first record",
        job_dir.display()
    );
    loop {
        if let Some(record) = read_json::<JobRecord>(&job_dir.join(RECORD_FILE)) {
            return Ok(DetachedJob {
                stdout: job_dir.join(STDOUT_LOG),
                stderr: job_dir.join(STDERR_LOG),
                job_id,
                job_dir,
                record_id: record.record_id,
                record_dir: record.record_dir,
            });
        }
        if let Some(exit) = read_json::<JobExit>(&job_dir.join(EXIT_FILE)) {
            let stderr = fs::read_to_string(job_dir.join(STDERR_LOG)).unwrap_or_default();
            let status = match (exit.exit_code, exit.signal) {
                (Some(code), _) => format!("exit status {code}"),
                (None, Some(signal)) => format!("signal {signal}"),
                (None, None) => "an unknown status".to_owned(),
            };
            return Err(failure(
                &job_id,
                format!(
                    "the job in {} ended with {status} before creating a record: {}",
                    job_dir.display(),
                    stderr.trim()
                ),
            ));
        }
        if inferlab_runtime::interrupt::received() {
            return Err(failure(
                &job_id,
                format!(
                    "stopped waiting for its first record; the job continues in {}",
                    job_dir.display()
                ),
            ));
        }
        std::thread::sleep(DETACH_POLL_INTERVAL);
    }
}
