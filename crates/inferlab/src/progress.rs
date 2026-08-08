use crate::InferlabError;
use crate::operation::{OperationPosition, OperationProgress, OperationPublisher};
use std::fmt;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tracing_subscriber::fmt::time::UtcTime;

const FIRST_REPORT_AFTER: Duration = Duration::from_secs(10);
const HEARTBEAT_EVERY: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Mode {
    Immediate,
    Delayed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Phase {
    name: String,
    item: Option<String>,
    position: Option<(usize, usize)>,
    lock: Option<String>,
    readiness_failure: Option<String>,
    record: Option<String>,
    record_dir: Option<String>,
    log: Option<String>,
}

impl Phase {
    pub(crate) fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            item: None,
            position: None,
            lock: None,
            readiness_failure: None,
            record: None,
            record_dir: None,
            log: None,
        }
    }

    pub(crate) fn item(mut self, item: impl Into<String>, index: usize, total: usize) -> Self {
        self.item = Some(item.into());
        self.position = Some((index, total));
        self
    }

    pub(crate) fn current_item(mut self, item: impl Into<String>) -> Self {
        self.item = Some(item.into());
        self
    }

    pub(crate) fn lock(mut self, path: &Path) -> Self {
        self.lock = Some(path.display().to_string());
        self
    }

    pub(crate) fn log(mut self, path: impl AsRef<Path>) -> Self {
        self.log = Some(path.as_ref().display().to_string());
        self
    }

    pub(crate) fn record(mut self, id: impl Into<String>, dir: impl AsRef<Path>) -> Self {
        self.record = Some(id.into());
        self.record_dir = Some(dir.as_ref().display().to_string());
        self
    }
}

#[derive(Debug)]
struct ActivePhase {
    phase: Phase,
    started: Instant,
    reported_at: Option<Instant>,
    next_report: Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReportKind {
    Phase,
    Heartbeat,
}

#[derive(Clone, Copy)]
struct ItemPosition {
    index: usize,
    total: usize,
}

impl fmt::Display for ItemPosition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.index, self.total)
    }
}

#[derive(Clone, Copy)]
struct ElapsedSeconds(u64);

impl fmt::Display for ElapsedSeconds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}s", self.0)
    }
}

#[derive(Debug)]
struct ProgressState {
    command: String,
    mode: Mode,
    delayed_deadline: Option<Instant>,
    visible: bool,
    active: Option<ActivePhase>,
    observed_record: Option<String>,
    observed_record_dir: Option<String>,
}

impl ProgressState {
    fn new(command: impl Into<String>, mode: Mode) -> Self {
        Self {
            command: command.into(),
            mode,
            delayed_deadline: None,
            visible: false,
            active: None,
            observed_record: None,
            observed_record_dir: None,
        }
    }

    fn begin(&mut self, now: Instant, phase: Phase) -> Option<ReportKind> {
        if self.observed_record.is_none()
            && let Some(record) = &phase.record
        {
            self.observed_record = Some(record.clone());
            self.observed_record_dir = phase.record_dir.clone();
        }
        let deadline = *self
            .delayed_deadline
            .get_or_insert(now + FIRST_REPORT_AFTER);
        let immediate = self.mode == Mode::Immediate || self.visible || now >= deadline;
        self.visible |= immediate;
        self.active = Some(ActivePhase {
            phase,
            started: now,
            reported_at: immediate.then_some(now),
            next_report: if immediate {
                now + FIRST_REPORT_AFTER
            } else {
                deadline
            },
        });
        immediate.then_some(ReportKind::Phase)
    }

    fn due(&mut self, now: Instant) -> Option<ReportKind> {
        let active = self.active.as_mut()?;
        if now < active.next_report {
            return None;
        }
        let heartbeat = active.reported_at.is_some();
        active.reported_at = Some(now);
        self.visible = true;
        active.next_report = now
            + if heartbeat {
                HEARTBEAT_EVERY
            } else {
                FIRST_REPORT_AFTER
            };
        Some(if heartbeat {
            ReportKind::Heartbeat
        } else {
            ReportKind::Phase
        })
    }

    fn set_readiness_failure(&mut self, failure: impl Into<String>) {
        if let Some(active) = self.active.as_mut() {
            active.phase.readiness_failure = Some(failure.into());
        }
    }

    fn observation(&self) -> Option<OperationProgress> {
        self.active.as_ref().map(|active| OperationProgress {
            phase: active.phase.name.clone(),
            item: active.phase.item.clone(),
            position: active
                .phase
                .position
                .map(|(index, total)| OperationPosition { index, total }),
            lock: active.phase.lock.clone(),
            readiness_failure: active.phase.readiness_failure.clone(),
            record_ref: self.observed_record.clone(),
            record_dir: self.observed_record_dir.clone(),
            log_ref: active.phase.log.clone(),
        })
    }

    fn wait_duration(&self, now: Instant) -> Duration {
        self.active.as_ref().map_or(HEARTBEAT_EVERY, |active| {
            active.next_report.saturating_duration_since(now)
        })
    }
}

fn emit_report(
    dispatcher: &tracing::Dispatch,
    progress: &ProgressState,
    report: ReportKind,
    now: Instant,
) {
    let Some(active) = progress.active.as_ref() else {
        return;
    };
    let message = match report {
        ReportKind::Phase => format!("[{}] {}", progress.command, active.phase.name),
        ReportKind::Heartbeat => {
            format!(
                "[{}] {}: still running",
                progress.command, active.phase.name
            )
        }
    };
    let position = active
        .phase
        .position
        .map(|(index, total)| tracing::field::display(ItemPosition { index, total }));
    let elapsed = (report == ReportKind::Heartbeat).then(|| {
        tracing::field::display(ElapsedSeconds(
            now.saturating_duration_since(active.started).as_secs(),
        ))
    });

    tracing::dispatcher::with_default(dispatcher, || {
        tracing::info!(
            item = active.phase.item.as_deref(),
            position,
            elapsed,
            lock = active.phase.lock.as_deref(),
            readiness_failure = active.phase.readiness_failure.as_deref(),
            record = active.phase.record.as_deref(),
            record_dir = active.phase.record_dir.as_deref(),
            log = active.phase.log.as_deref(),
            "{message}"
        );
    });
}

fn progress_dispatch(writer: Box<dyn Write + Send>) -> tracing::Dispatch {
    let subscriber = tracing_subscriber::fmt()
        .with_writer(Mutex::new(writer))
        .with_timer(UtcTime::rfc_3339())
        .with_ansi(false)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .log_internal_errors(false)
        .finish();
    tracing::Dispatch::new(subscriber)
}

struct SharedState {
    progress: ProgressState,
    stopped: bool,
    observer: Option<OperationPublisher>,
}

struct Shared {
    state: Mutex<SharedState>,
    changed: Condvar,
    dispatcher: tracing::Dispatch,
}

/// One command-scoped diagnostic progress stream. Phase changes are written
/// synchronously before the caller starts the corresponding operation; a
/// small worker owns only elapsed-time heartbeats
/// ([[RFC-0001:C-OPERATOR-PROGRESS]]).
pub(crate) struct Progress {
    shared: Option<Arc<Shared>>,
    worker: Option<JoinHandle<()>>,
}

impl Progress {
    pub(crate) fn stderr(command: &str, mode: Mode) -> Result<Self, InferlabError> {
        Self::with_writer(command, mode, Box::new(io::stderr()), None)
    }

    pub(crate) fn stderr_observed(
        command: &str,
        mode: Mode,
        observer: OperationPublisher,
    ) -> Result<Self, InferlabError> {
        Self::with_writer(command, mode, Box::new(io::stderr()), Some(observer))
    }

    pub(crate) fn silent() -> Self {
        Self {
            shared: None,
            worker: None,
        }
    }

    fn with_writer(
        command: &str,
        mode: Mode,
        writer: Box<dyn Write + Send>,
        observer: Option<OperationPublisher>,
    ) -> Result<Self, InferlabError> {
        let shared = Arc::new(Shared {
            state: Mutex::new(SharedState {
                progress: ProgressState::new(command, mode),
                stopped: false,
                observer,
            }),
            changed: Condvar::new(),
            dispatcher: progress_dispatch(writer),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("inferlab-progress".to_owned())
            .spawn(move || heartbeat_loop(&worker_shared))
            .map_err(|source| InferlabError::WriteOutput { source })?;
        Ok(Self {
            shared: Some(shared),
            worker: Some(worker),
        })
    }

    pub(crate) fn phase(&self, phase: Phase) -> Result<(), InferlabError> {
        let Some(shared) = &self.shared else {
            return Ok(());
        };
        let mut state = lock_state(shared);
        let now = Instant::now();
        if let Some(report) = state.progress.begin(now, phase) {
            emit_report(&shared.dispatcher, &state.progress, report, now);
        }
        publish_observation(&state)?;
        drop(state);
        shared.changed.notify_all();
        Ok(())
    }

    pub(crate) fn readiness_failure(&self, failure: &str) {
        let Some(shared) = &self.shared else {
            return;
        };
        let mut state = lock_state(shared);
        state.progress.set_readiness_failure(failure);
        let _ = publish_observation(&state);
    }

    pub(crate) fn finish(mut self) -> Result<(), InferlabError> {
        self.stop_worker();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.shared
            .as_ref()
            .and_then(|shared| lock_state(shared).observer.clone())
            .map_or(Ok(()), |observer| observer.health())
    }

    fn stop_worker(&self) {
        let Some(shared) = &self.shared else {
            return;
        };
        let mut state = lock_state(shared);
        state.stopped = true;
        drop(state);
        shared.changed.notify_all();
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.stop_worker();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn heartbeat_loop(shared: &Arc<Shared>) {
    let mut state = lock_state(shared);
    loop {
        if state.stopped {
            return;
        }
        let now = Instant::now();
        if let Some(report) = state.progress.due(now) {
            emit_report(&shared.dispatcher, &state.progress, report, now);
            let _ = publish_observation(&state);
        }
        let wait = state.progress.wait_duration(Instant::now());
        state = match shared.changed.wait_timeout(state, wait) {
            Ok((state, _)) => state,
            Err(poisoned) => poisoned.into_inner().0,
        };
    }
}

fn lock_state(shared: &Shared) -> MutexGuard<'_, SharedState> {
    match shared.state.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn publish_observation(state: &SharedState) -> Result<(), InferlabError> {
    if let (Some(observer), Some(progress)) = (&state.observer, state.progress.observation()) {
        observer.publish(progress)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Mode, Phase, Progress, ProgressState, ReportKind, emit_report, progress_dispatch};
    use std::io::{self, Write};
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed stderr"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct RecordingWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for RecordingWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            let mut bytes = self
                .bytes
                .lock()
                .map_err(|_| io::Error::other("recording writer lock poisoned"))?;
            bytes.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn report_line(state: &ProgressState, report: ReportKind, now: Instant) -> String {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let dispatcher = progress_dispatch(Box::new(RecordingWriter {
            bytes: Arc::clone(&bytes),
        }));
        emit_report(&dispatcher, state, report, now);
        recorded_text(&bytes)
    }

    fn recorded_text(bytes: &Mutex<Vec<u8>>) -> String {
        let bytes = match bytes.lock() {
            Ok(bytes) => bytes,
            Err(poisoned) => poisoned.into_inner(),
        };
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn begin_line(state: &mut ProgressState, now: Instant, phase: Phase) -> Option<String> {
        let report = state.begin(now, phase)?;
        Some(report_line(state, report, now))
    }

    fn due_line(state: &mut ProgressState, now: Instant) -> Option<String> {
        let report = state.due(now)?;
        Some(report_line(state, report, now))
    }

    #[test]
    fn immediate_phase_reports_then_heartbeats_at_bounded_intervals() {
        let start = Instant::now();
        let mut state = ProgressState::new("toolchain install", Mode::Immediate);

        let phase = begin_line(&mut state, start, Phase::named("Pixi installation"));
        assert!(phase.is_some_and(|line| {
            line.contains(" INFO [toolchain install] Pixi installation")
                && line.ends_with('\n')
                && !line.contains("command=")
                && !line.contains("progress:")
        }));

        assert_eq!(state.due(start + Duration::from_secs(9)), None);
        let heartbeat = due_line(&mut state, start + Duration::from_secs(10));
        assert!(heartbeat.is_some_and(|line| {
            line.contains(" INFO [toolchain install] Pixi installation: still running")
                && line.contains("elapsed=10s")
        }));
        assert_eq!(state.due(start + Duration::from_secs(39)), None);
        let heartbeat = due_line(&mut state, start + Duration::from_secs(40));
        assert!(heartbeat.is_some_and(|line| line.contains("elapsed=40s")));
    }

    #[test]
    fn emitted_progress_is_one_complete_stderr_line() -> Result<(), crate::InferlabError> {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let progress = Progress::with_writer(
            "toolchain install",
            Mode::Immediate,
            Box::new(RecordingWriter {
                bytes: Arc::clone(&bytes),
            }),
            None,
        )?;

        progress.phase(Phase::named("Pixi installation"))?;
        progress.finish()?;

        let output = recorded_text(&bytes);
        assert!(output.ends_with('\n'));
        assert_eq!(output.lines().count(), 1);
        assert!(output.contains(" INFO [toolchain install] Pixi installation"));
        assert!(!output.contains("command="));
        assert!(!output.contains("progress:"));
        Ok(())
    }

    #[test]
    fn ephemeral_progress_write_failure_does_not_replace_the_command_result() {
        let progress = Progress::with_writer(
            "toolchain install",
            Mode::Immediate,
            Box::new(FailingWriter),
            None,
        );
        assert!(progress.is_ok());
        let progress = progress.ok();
        assert!(progress.is_some_and(|progress| {
            progress.phase(Phase::named("Pixi installation")).is_ok() && progress.finish().is_ok()
        }));
    }

    #[test]
    fn delayed_phase_stays_quiet_until_ten_seconds_then_starts_heartbeat_clock() {
        let start = Instant::now();
        let mut state = ProgressState::new("stack status", Mode::Delayed);

        assert_eq!(
            state.begin(start, Phase::named("realization inspection")),
            None
        );
        assert_eq!(state.due(start + Duration::from_secs(9)), None);
        let phase = due_line(&mut state, start + Duration::from_secs(10));
        assert!(phase.is_some_and(|line| {
            line.contains(" INFO [stack status] realization inspection")
                && !line.contains("still running")
        }));
        assert_eq!(state.due(start + Duration::from_secs(19)), None);
        let heartbeat = due_line(&mut state, start + Duration::from_secs(20));
        assert!(heartbeat.is_some_and(|line| {
            line.contains("[stack status] realization inspection: still running")
                && line.contains("elapsed=20s")
        }));
    }

    #[test]
    fn delayed_phase_changes_do_not_restart_the_command_quiet_period() {
        let start = Instant::now();
        let mut state = ProgressState::new("image build", Mode::Delayed);

        assert_eq!(state.begin(start, Phase::named("resolution")), None);
        assert_eq!(
            state.begin(
                start + Duration::from_secs(9),
                Phase::named("package-build")
            ),
            None
        );
        let phase = due_line(&mut state, start + Duration::from_secs(10));
        assert!(phase.is_some_and(|line| line.contains(" INFO [image build] package-build")));
        let next_phase = begin_line(
            &mut state,
            start + Duration::from_secs(11),
            Phase::named("assembly"),
        );
        assert!(next_phase.is_some_and(|line| line.contains(" INFO [image build] assembly")));
    }

    #[test]
    fn heartbeat_uses_latest_phase_context() {
        let start = Instant::now();
        let mut state = ProgressState::new("serve start", Mode::Immediate);
        let phase = Phase::named("readiness")
            .item("worker-0", 1, 2)
            .lock(Path::new("/tmp/serve.lock"))
            .record("serve-1", "/records/serve-1")
            .log("/records/server.stderr.log");
        let initial = begin_line(&mut state, start, phase);
        assert!(initial.is_some_and(|line| {
            line.contains("item=\"worker-0\"")
                && line.contains("position=1/2")
                && line.contains("lock=\"/tmp/serve.lock\"")
                && line.contains("record=\"serve-1\"")
                && line.contains("record_dir=\"/records/serve-1\"")
                && line.contains("log=\"/records/server.stderr.log\"")
        }));

        state.set_readiness_failure("connection refused\nretrying");
        let heartbeat = due_line(&mut state, start + Duration::from_secs(10));
        assert!(heartbeat.is_some_and(|line| {
            line.contains("readiness_failure=\"connection refused\\nretrying\"")
                && line.contains("elapsed=10s")
        }));
    }

    #[test]
    fn a_new_phase_resets_elapsed_time_and_escapes_fields() {
        let start = Instant::now();
        let mut state = ProgressState::new("recipe run", Mode::Immediate);
        let _ = state.begin(start, Phase::named("server startup"));
        let line = begin_line(
            &mut state,
            start + Duration::from_secs(8),
            Phase::named("Eval").item("quote \" and slash \\", 2, 3),
        );
        assert!(line.is_some_and(|line| {
            line.contains("item=\"quote \\\"")
                && line.contains("slash \\\\")
                && !line.contains("elapsed=")
        }));
        assert_eq!(state.due(start + Duration::from_secs(17)), None);
        assert!(
            due_line(&mut state, start + Duration::from_secs(18))
                .is_some_and(|line| line.contains("elapsed=10s"))
        );
    }

    #[test]
    fn operation_projection_retains_record_after_progress_moves_to_a_later_phase() {
        let start = Instant::now();
        let mut state = ProgressState::new("recipe run", Mode::Immediate);
        let _ = state.begin(
            start,
            Phase::named("record created").record("recipe-1", "/records/recipe-1"),
        );
        let _ = state.begin(
            start + Duration::from_secs(1),
            Phase::named("server startup"),
        );
        let observation = state.observation();
        assert!(observation.is_some_and(|observation| {
            observation.phase == "server startup"
                && observation.record_ref.as_deref() == Some("recipe-1")
                && observation.record_dir.as_deref() == Some("/records/recipe-1")
        }));
    }

    #[test]
    fn nested_workflow_records_do_not_replace_the_invocation_record() {
        let start = Instant::now();
        let mut state = ProgressState::new("recipe run", Mode::Immediate);
        let _ = state.begin(
            start,
            Phase::named("recipe record created").record("recipe-1", "/records/recipe-1"),
        );
        let _ = state.begin(
            start + Duration::from_secs(1),
            Phase::named("server record created").record("server-1", "/records/server-1"),
        );

        let observation = state.observation();

        assert!(observation.is_some_and(|observation| {
            observation.record_ref.as_deref() == Some("recipe-1")
                && observation.record_dir.as_deref() == Some("/records/recipe-1")
        }));
    }
}
