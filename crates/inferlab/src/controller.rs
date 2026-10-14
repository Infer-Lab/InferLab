//! Record controllers and lock waits ([[RFC-0003:C-RUNTIME-WORKFLOWS]],
//! [[RFC-0002:C-LOCAL-PLACEMENT]]). The process that drives a recipe — or a
//! server while it starts — holds an advisory lock beside the record for as
//! long as it drives it, so the lock's state is the controller's liveness:
//! the kernel releases it when the process exits or is killed. The holder's
//! identity sits in a separate, closed file, so another host sharing the
//! workspace reads it. A stop request is written into a file the controller
//! creates empty when it takes control; the controller's watcher rereads it
//! and turns a request into the same interruption Ctrl-C raises. Rereading
//! an existing file, rather than looking for a new one, keeps the request
//! visible within one poll across hosts sharing a network file system,
//! whose cached lookups may miss a newly created file for far longer.

use crate::InferlabError;
use crate::progress::{Phase, Progress};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const CONTROLLER_LOCK: &str = "controller.lock";
const CONTROLLER_HOLDER: &str = "controller.json";
const STOP_REQUEST: &str = "stop.request";

/// How often a waiter retries a held lock and a controller looks for a stop
/// request: a polling cadence, not an operation budget
/// ([[RFC-0009:C-OPERATION-BUDGETS]]).
pub(crate) const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Who holds a lock, as its holder recorded it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LockHolder {
    pub host: String,
    pub pid: u32,
    pub started_unix_ms: u64,
    pub command: String,
}

impl LockHolder {
    fn current() -> Self {
        Self {
            host: rustix::system::uname()
                .nodename()
                .to_string_lossy()
                .into_owned(),
            pid: std::process::id(),
            started_unix_ms: crate::record::now_unix_ms().unwrap_or_default(),
            command: std::env::args().collect::<Vec<_>>().join(" "),
        }
    }
}

impl std::fmt::Display for LockHolder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let since = crate::record::utc_timestamp(self.started_unix_ms)
            .unwrap_or_else(|_| self.started_unix_ms.to_string());
        write!(
            formatter,
            "pid {} on {} since {since}, running `{}`",
            self.pid, self.host, self.command
        )
    }
}

fn open_lock(path: &Path) -> Result<File, InferlabError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| InferlabError::RecordIo {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|source| InferlabError::RecordIo {
            path: path.to_path_buf(),
            source,
        })
}

/// Try to take a lock without waiting: `Some` holds it, `None` means another
/// process holds it.
fn try_lock(path: &Path) -> Result<Option<File>, InferlabError> {
    let file = open_lock(path)?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(Some(file)),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
        Err(source) => Err(InferlabError::RecordIo {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn read_holder(path: &Path) -> Option<LockHolder> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

/// Take a lock, waiting while another process holds it. The wait reports the
/// holder and ends on an operator interrupt, because a blocking lock call
/// would not return to observe one. The taker then records itself as holder.
pub(crate) fn lock_waiting(
    lock: &Path,
    holder: &Path,
    waiting_for: &str,
    progress: &Progress,
) -> Result<File, InferlabError> {
    let mut reported = None;
    loop {
        if let Some(file) = try_lock(lock)? {
            crate::record::write_json(holder, &LockHolder::current())?;
            return Ok(file);
        }
        let current = read_holder(holder).map_or_else(
            || "a holder that recorded no identity".to_owned(),
            |holder| holder.to_string(),
        );
        // The wait reports its holder once, and again only when it changes.
        if reported.as_ref() != Some(&current) {
            progress
                .phase(Phase::named(format!("waiting for {waiting_for}")).current_item(&current))?;
            reported = Some(current.clone());
        }
        if inferlab_runtime::interrupt::received() {
            return Err(InferlabError::LockWaitInterrupted {
                waiting_for: waiting_for.to_owned(),
                holder: current,
            });
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// The record a controller drives, held for as long as the guard lives; its
/// watcher turns a stop request into an interruption.
pub(crate) struct ControllerGuard {
    _lock: File,
    done: Arc<AtomicBool>,
}

impl ControllerGuard {
    pub(crate) fn acquire(record_dir: &Path) -> Result<Self, InferlabError> {
        let lock_path = record_dir.join(CONTROLLER_LOCK);
        let Some(lock) = try_lock(&lock_path)? else {
            return Err(InferlabError::ServerLifecycle {
                message: format!("record {} already has a controller", record_dir.display()),
            });
        };
        let holder = record_dir.join(CONTROLLER_HOLDER);
        crate::record::write_json(&holder, &LockHolder::current())?;
        let request = record_dir.join(STOP_REQUEST);
        fs::write(&request, b"").map_err(|source| InferlabError::RecordIo {
            path: request.clone(),
            source,
        })?;
        let done = Arc::new(AtomicBool::new(false));
        let watching = Arc::clone(&done);
        // The request stands until the controller exits, so it is raised
        // again each poll: a step that rearms interruption cannot lose it.
        std::thread::spawn(move || {
            while !watching.load(Ordering::SeqCst) {
                if fs::read(&request).is_ok_and(|bytes| !bytes.is_empty()) {
                    inferlab_runtime::interrupt::request();
                }
                std::thread::sleep(POLL_INTERVAL);
            }
        });
        Ok(Self { _lock: lock, done })
    }
}

impl Drop for ControllerGuard {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
    }
}

/// Whether a record's controller is still driving it.
pub(crate) enum Controller {
    /// The controller is gone; the stopper now holds its lock.
    Gone(File),
    /// A controller holds the lock.
    Running(Option<LockHolder>),
}

pub(crate) fn inspect(record_dir: &Path) -> Result<Controller, InferlabError> {
    Ok(match try_lock(&record_dir.join(CONTROLLER_LOCK))? {
        Some(lock) => {
            // The stopper now holds the record; a concurrent stop names it.
            crate::record::write_json(&record_dir.join(CONTROLLER_HOLDER), &LockHolder::current())?;
            Controller::Gone(lock)
        }
        None => Controller::Running(read_holder(&record_dir.join(CONTROLLER_HOLDER))),
    })
}

/// Ask a running controller to stop; the request stands until it acts.
pub(crate) fn request_stop(record_dir: &Path) -> Result<(), InferlabError> {
    let path: PathBuf = record_dir.join(STOP_REQUEST);
    fs::write(&path, b"stop\n").map_err(|source| InferlabError::RecordIo { path, source })
}

/// Wait for a stopped controller to release its record. Only a stop command
/// waits here, so it arms operator interruption for the wait itself.
pub(crate) fn wait_for_release(
    record_dir: &Path,
    progress: &Progress,
) -> Result<File, InferlabError> {
    inferlab_runtime::interrupt::prepare()
        .map_err(|source| InferlabError::ServerInterrupt { source })?;
    lock_waiting(
        &record_dir.join(CONTROLLER_LOCK),
        &record_dir.join(CONTROLLER_HOLDER),
        "the record's controller to stop",
        progress,
    )
}

/// Describe a controller for an operator.
pub(crate) fn describe(holder: Option<&LockHolder>) -> String {
    holder.map_or_else(
        || "a controller that recorded no identity".to_owned(),
        LockHolder::to_string,
    )
}
