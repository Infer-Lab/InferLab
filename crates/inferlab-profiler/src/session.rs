use crate::error::ProfilerError;
use crate::finalization;
use crate::plan::{CaptureSelection, ProfilerFinalization, compile_plan};
use crate::record::{
    CaptureActionRecord, CapturePlanRecord, CaptureRangeEndRecord, CaptureRecord,
    CaptureReportRecord, CaptureStatus, CaptureWindowRecord, MEASUREMENT_FINALIZATION_START,
    ProfilerTargetRecord,
};
use crate::transport;
use inferlab_protocol::CaptureMechanism;
use inferlab_runtime::operation_bound::OperationBound;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

pub struct CaptureSession {
    targets: Vec<ProfilerTargetRecord>,
    plan: CapturePlanRecord,
    record: CaptureRecord,
    stop_failure: Option<String>,
    /// Trace-directory file snapshots taken when the engine-trace targets
    /// were armed, before any window opened; keyed by replica identity.
    engine_trace_baselines: BTreeMap<String, BTreeSet<PathBuf>>,
    /// The one global finalization budget
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]): started by the first engine-trace
    /// window-close dispatch, or by finalization itself when no engine-trace
    /// target ever dispatched a close.
    finalization: Option<OperationBound>,
}

impl CaptureSession {
    pub fn open(
        server_record_id: &str,
        workload_id: &str,
        window_ids: &[String],
        selection: CaptureSelection,
    ) -> Result<Self, Box<CaptureRecord>> {
        let plan = compile_plan(
            server_record_id,
            workload_id,
            window_ids,
            &selection.targets,
            selection.deadlines,
        )
        .map_err(|error| Box::new(CaptureRecord::failed(format!("profiling failed: {error}"))))?;
        let mut session = Self {
            targets: selection.targets,
            record: CaptureRecord {
                status: CaptureStatus::Running,
                plan: Some(plan.clone()),
                arm: Vec::new(),
                windows: Vec::new(),
                finalization: Vec::new(),
                reports: Vec::new(),
                engine_trace: Vec::new(),
                error: None,
            },
            plan,
            stop_failure: None,
            engine_trace_baselines: BTreeMap::new(),
            finalization: None,
        };
        let arm_bound = OperationBound::finite(Duration::from_secs(
            session.plan.deadlines.capture_arm_deadline_seconds,
        ));
        if let Err(message) = session.arm_targets(&arm_bound) {
            session.fail(message);
            let finalization_bound = session.finalization_budget();
            session.finalize_collections(&finalization_bound);
            return Err(Box::new(session.record));
        }
        Ok(session)
    }

    fn arm_targets(&mut self, bound: &OperationBound) -> Result<(), String> {
        let mut armed_engine_trace_replicas = BTreeSet::new();
        for index in 0..self.targets.len() {
            // The per-target helpers mutate the record, so the shared target
            // and plan facts are cloned out of the borrow first.
            let target = self.targets[index].clone();
            let plan = self.plan.targets[index].clone();
            if plan.mechanism == CaptureMechanism::EngineTrace {
                if armed_engine_trace_replicas.insert(plan.replica_id.clone()) {
                    self.arm_engine_trace_replica(&target, &plan, bound)?;
                }
                continue;
            }
            self.arm_managed_target(&target, &plan, bound)?;
        }
        Ok(())
    }

    fn arm_managed_target(
        &mut self,
        target: &ProfilerTargetRecord,
        plan: &crate::record::CaptureTargetPlan,
        bound: &OperationBound,
    ) -> Result<(), String> {
        let parent = plan
            .output_base
            .parent()
            .ok_or_else(|| format!("capture output {:?} has no parent", plan.output_base))?;
        let mkdir = transport::prepare_output(target, parent, bound);
        let mkdir_ok = mkdir.succeeded();
        let mkdir_error = mkdir.error();
        self.record.arm.push(mkdir);
        if !mkdir_ok {
            return Err(mkdir_error.unwrap_or_else(|| {
                format!("failed to prepare profiler target {:?}", target.process_id)
            }));
        }
        let count = plan.expected_range_count.ok_or_else(|| {
            format!(
                "profiler target {:?} has no static range count",
                target.process_id
            )
        })?;
        let start = transport::arm_range_collection(target, &plan.output_base, count, bound);
        let start_ok = start.succeeded();
        let start_error = start.error();
        self.record.arm.push(start);
        if !start_ok {
            return Err(start_error.unwrap_or_else(|| {
                format!("failed to arm profiler target {:?}", target.process_id)
            }));
        }
        Ok(())
    }

    /// Arming an engine-trace replica is creating its record-owned trace
    /// directory and snapshotting its contents as the window baseline; the
    /// engine's internal profiler needs no managed collection start
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    fn arm_engine_trace_replica(
        &mut self,
        target: &ProfilerTargetRecord,
        plan: &crate::record::CaptureTargetPlan,
        bound: &OperationBound,
    ) -> Result<(), String> {
        let mkdir = transport::prepare_output(target, &plan.output_base, bound);
        let mkdir_ok = mkdir.succeeded();
        let mkdir_error = mkdir.error();
        self.record.arm.push(mkdir);
        if !mkdir_ok {
            return Err(mkdir_error.unwrap_or_else(|| {
                format!(
                    "failed to prepare engine-trace storage for replica {:?}",
                    plan.replica_id
                )
            }));
        }
        let baseline = finalization::snapshot_trace_files(&plan.output_base)?;
        self.engine_trace_baselines
            .insert(plan.replica_id.clone(), baseline);
        Ok(())
    }

    pub fn run_window<T, E>(
        &mut self,
        id: &str,
        run_client: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<ProfilerError> + ToString,
    {
        let range_index = self.window_range_index(id).map_err(E::from)?;
        let mut start = self.start_window();
        if let Some(action) = start.iter().find(|action| !action.succeeded()) {
            let message = action
                .error()
                .unwrap_or_else(|| format!("failed to open capture window {id:?}"));
            let stop = self.stop_window(&start);
            self.record.windows.push(CaptureWindowRecord {
                id: id.to_owned(),
                range_index,
                start,
                stop,
                client_succeeded: false,
                succeeded: false,
                error: Some(message.clone()),
            });
            self.fail(message.clone());
            return Err(ProfilerError::WindowStartFailed {
                window_id: id.to_owned(),
                message,
            }
            .into());
        }
        let client = run_client();
        let stop = self.stop_window(&start);
        if self.stop_failure.is_none()
            && let Some(action) = stop.iter().find(|action| !action.succeeded())
        {
            self.stop_failure = Some(
                action
                    .error()
                    .unwrap_or_else(|| format!("failed to close capture window {id:?}")),
            );
        }
        let client_succeeded = client.is_ok();
        let error = client.as_ref().err().map(ToString::to_string);
        self.record.windows.push(CaptureWindowRecord {
            id: id.to_owned(),
            range_index,
            start: std::mem::take(&mut start),
            stop,
            client_succeeded,
            succeeded: client_succeeded,
            error: error.clone(),
        });
        if let Some(message) = error {
            self.fail(message);
        }
        client
    }

    pub fn record_unopened_window(
        &mut self,
        id: &str,
        client_succeeded: bool,
        message: String,
    ) -> Result<(), ProfilerError> {
        let range_index = self.window_range_index(id)?;
        self.record.windows.push(CaptureWindowRecord {
            id: id.to_owned(),
            range_index,
            start: Vec::new(),
            stop: Vec::new(),
            client_succeeded,
            succeeded: false,
            error: Some(message.clone()),
        });
        self.fail(message);
        Ok(())
    }

    fn window_range_index(&self, id: &str) -> Result<Option<usize>, ProfilerError> {
        self.plan
            .windows
            .iter()
            .find(|window| window.id == id)
            .map(|window| window.range_index)
            .ok_or_else(|| ProfilerError::UnknownWindow {
                window_id: id.to_owned(),
            })
    }

    fn start_window(&self) -> Vec<CaptureActionRecord> {
        transport::start_windows(
            &self.targets.iter().collect::<Vec<_>>(),
            self.plan.deadlines.capture_control_deadline_seconds,
        )
    }

    /// Window closing splits by mechanism
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]): managed-collection targets keep
    /// the per-action control budget, while engine-trace targets dispatch the
    /// close into the one global finalization budget, where a delivery
    /// failure is window-closing control failure evidence and a slow or
    /// absent response is neutral flush-pending evidence.
    fn stop_window(&mut self, start: &[CaptureActionRecord]) -> Vec<CaptureActionRecord> {
        let started = start
            .iter()
            .filter(|action| action.succeeded())
            .filter_map(|action| match action {
                CaptureActionRecord::Http { process_id, .. } => Some(process_id.as_str()),
                CaptureActionRecord::Command { .. }
                | CaptureActionRecord::CollectionFinalization { .. }
                | CaptureActionRecord::EngineTraceFlush { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        let mut actions = {
            let managed = self
                .targets
                .iter()
                .filter(|target| target.mechanism == CaptureMechanism::ManagedCollection)
                .collect::<Vec<_>>();
            transport::stop_windows(
                &managed,
                &started,
                self.plan.deadlines.capture_control_deadline_seconds,
            )
        };
        // Cloned out of the session borrow so the finalization budget can
        // start under a mutable borrow.
        let engine_trace = self
            .targets
            .iter()
            .filter(|target| target.mechanism == CaptureMechanism::EngineTrace)
            .cloned()
            .collect::<Vec<_>>();
        if !engine_trace.is_empty() {
            let bound = self.start_finalization_budget();
            actions.extend(transport::close_engine_trace_windows(
                &engine_trace.iter().collect::<Vec<_>>(),
                &started,
                bound,
            ));
        }
        actions
    }

    /// Start the one global finalization budget at the first engine-trace
    /// window-close dispatch ([[RFC-0004:C-WORKLOAD-PROFILING]]); later
    /// dispatches, response consumption, flush waiting, and coverage
    /// verification draw the same budget without restarting it.
    fn start_finalization_budget(&mut self) -> &OperationBound {
        self.finalization.get_or_insert_with(|| {
            OperationBound::finite(Duration::from_secs(
                self.plan.deadlines.capture_finalization_deadline_seconds,
            ))
        })
    }

    #[must_use]
    pub fn finish(mut self) -> CaptureRecord {
        let finalization_bound = self.finalization_budget();
        self.finalize_collections(&finalization_bound);
        self.verify_reports(&finalization_bound);
        self.verify_engine_trace_coverage(&finalization_bound);
        if self.record.error.is_none()
            && self.record.windows.iter().all(|window| window.succeeded)
            && self.record.reports.iter().all(|report| report.verified)
            && self
                .record
                .engine_trace
                .iter()
                .all(|coverage| coverage.verified)
        {
            self.record.status = CaptureStatus::Succeeded;
        } else {
            self.record.status = CaptureStatus::Failed;
        }
        self.record
    }

    /// Take the one global finalization budget, starting it now when no
    /// engine-trace window-close dispatch already started it: stop-response
    /// consumption, flush waiting, and coverage verification draw the same
    /// budget without restarting it ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    fn finalization_budget(&mut self) -> OperationBound {
        self.finalization.take().unwrap_or_else(|| {
            OperationBound::finite(Duration::from_secs(
                self.plan.deadlines.capture_finalization_deadline_seconds,
            ))
        })
    }

    fn finalize_collections(&mut self, bound: &OperationBound) {
        let mut failure = None;
        for target in &self.targets {
            let action = finalization::finalize_target(
                target,
                self.range_end_for(target),
                self.engine_trace_close_confirmed(target),
                bound,
                MEASUREMENT_FINALIZATION_START,
            );
            if !action.succeeded() && failure.is_none() {
                failure = Some(action.error().unwrap_or_else(|| {
                    format!("failed to finalize target {:?}", target.process_id)
                }));
            }
            self.record.finalization.push(action);
        }
        if let Some(message) = failure {
            self.fail(message);
        }
    }

    /// The window-closing control receipt for an engine-trace target:
    /// confirmed when every window this target's control process opened also
    /// recorded a successful closing response for it. Receipt acknowledges
    /// the close only; artifact flush completion is judged by coverage
    /// verification alone, and a flush-pending close is neutral evidence
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    fn engine_trace_close_confirmed(&self, target: &ProfilerTargetRecord) -> bool {
        if target.finalization != ProfilerFinalization::EngineTraceFlush {
            return false;
        }
        let crate::plan::ProfilerControl::Http { process_id, .. } = &target.control;
        self.record
            .windows
            .iter()
            .filter(|window| {
                window.start.iter().any(|action| {
                    matches!(
                        action,
                        CaptureActionRecord::Http { process_id: pid, .. }
                            if pid == process_id && action.succeeded()
                    )
                })
            })
            .all(|window| {
                window.stop.iter().any(|action| {
                    matches!(
                        action,
                        CaptureActionRecord::Http {
                            process_id: pid,
                            succeeded: true,
                            flush_pending: false,
                            ..
                        } if pid == process_id
                    )
                })
            })
    }

    fn range_end_for(&self, target: &ProfilerTargetRecord) -> Option<CaptureRangeEndRecord> {
        let expected_range_count = self
            .plan
            .targets
            .iter()
            .find(|plan| plan.process_id == target.process_id)?
            .expected_range_count?;
        let final_window = self.plan.windows.last()?;
        if final_window.range_index != Some(expected_range_count) {
            return None;
        }
        let recorded = self
            .record
            .windows
            .iter()
            .find(|window| window.id == final_window.id)?;
        let control_process_id = match &target.control {
            crate::plan::ProfilerControl::Http { process_id, .. } => process_id,
        };
        recorded
            .stop
            .iter()
            .any(|action| {
                matches!(
                    action,
                    CaptureActionRecord::Http { process_id, .. }
                        if process_id == control_process_id && action.succeeded()
                )
            })
            .then(|| CaptureRangeEndRecord {
                window_id: final_window.id.clone(),
                range_index: expected_range_count,
                expected_range_count,
            })
    }

    fn verify_reports(&mut self, bound: &OperationBound) {
        let mut failure = None;
        for (target, target_plan) in self.targets.iter().zip(&self.plan.targets) {
            let control_process_id = match &target.control {
                crate::plan::ProfilerControl::Http { process_id, .. } => process_id,
            };
            for (window, path) in self.plan.windows.iter().zip(&target_plan.reports) {
                let wait_for_completion = self
                    .record
                    .windows
                    .iter()
                    .find(|record| record.id == window.id)
                    .is_some_and(|record| {
                        record.start.iter().any(|action| {
                            matches!(
                                action,
                                CaptureActionRecord::Http { process_id, .. }
                                    if process_id == control_process_id && action.succeeded()
                            )
                        })
                    });
                let verification = if wait_for_completion {
                    transport::verify_report(target, path, bound, MEASUREMENT_FINALIZATION_START)
                } else {
                    transport::check_report(target, path, bound, MEASUREMENT_FINALIZATION_START)
                };
                let verified = verification.succeeded();
                if !verified && failure.is_none() {
                    let mut message = format!(
                        "missing Nsight Systems report for target {:?}, window {:?}: {}",
                        target.process_id,
                        window.id,
                        path.display()
                    );
                    if let Some(stop_failure) = &self.stop_failure {
                        message.push_str(&format!(
                            "; a window-closing control action had failed: {stop_failure}"
                        ));
                    }
                    failure = Some(message);
                }
                self.record.reports.push(CaptureReportRecord {
                    process_id: target.process_id.clone(),
                    role_id: target.role_id.clone(),
                    window_id: window.id.clone(),
                    range_index: window.range_index,
                    path: path.clone(),
                    verified,
                    verification,
                });
            }
        }
        if let Some(message) = failure {
            self.fail(message);
        }
    }

    fn verify_engine_trace_coverage(&mut self, bound: &OperationBound) {
        let mut verified_replicas = BTreeSet::new();
        let mut failure = None;
        for plan in &self.plan.targets {
            if plan.mechanism != CaptureMechanism::EngineTrace
                || !verified_replicas.insert(plan.replica_id.clone())
            {
                continue;
            }
            let baseline = self
                .engine_trace_baselines
                .get(&plan.replica_id)
                .cloned()
                .unwrap_or_default();
            let coverage = finalization::verify_engine_trace_coverage(
                &plan.replica_id,
                &plan.output_base,
                plan.device_count,
                &baseline,
                bound,
            );
            if !coverage.verified && failure.is_none() {
                let mut message = format!(
                    "engine-trace replica {:?} produced {} new trace artifacts in {:?} for a \
                     {}-device replica; every device must produce one",
                    plan.replica_id,
                    coverage.new_files.len(),
                    plan.output_base,
                    plan.device_count,
                );
                if let Some(error) = &coverage.error {
                    message.push_str(&format!("; trace-directory snapshot failed: {error}"));
                }
                if let Some(stop_failure) = &self.stop_failure {
                    message.push_str(&format!(
                        "; a window-closing control action had failed: {stop_failure}"
                    ));
                }
                failure = Some(message);
            }
            self.record.engine_trace.push(coverage);
        }
        if let Some(message) = failure {
            self.fail(message);
        }
    }

    fn fail(&mut self, message: String) {
        self.record.status = CaptureStatus::Failed;
        if self.record.error.is_none() {
            self.record.error = Some(message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CaptureSession;
    use crate::plan::{
        CaptureDeadlines, CaptureSelection, CaptureWindowActionPlan,
        CaptureWindowControlEndpointPlan, CaptureWindowHttpMethodPlan, NsysEscapes,
        ProfilerControl, ProfilerFinalization, ProfilerLaunch, WindowControlKind, compile_plan,
    };
    use crate::record::{CaptureActionRecord, CaptureRecord, CaptureStatus, ProfilerTargetRecord};
    use inferlab_protocol::{CaptureMechanism, EndpointAssignment};
    use inferlab_runtime::operation_bound::{
        OperationBound, OperationBudgetEvidence, OperationTerminalCause,
    };
    use std::error::Error;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::time::Duration;

    #[test]
    fn missing_range_report_is_capture_failure_evidence() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let target = ProfilerTargetRecord {
            process_id: "serve".to_owned(),
            role_id: "serve".to_owned(),
            replica_id: "serve".to_owned(),
            replica_index: 0,
            rank: 0,
            rank_count: 1,
            device_count: 1,
            mechanism: inferlab_protocol::CaptureMechanism::ManagedCollection,
            trace_storage: None,
            session: "inferlab-fixture".to_owned(),
            executable: "true".to_owned(),
            launch: ProfilerLaunch::Local,
            finalization: ProfilerFinalization::NsysStop,
            control: ProfilerControl::Http {
                window_control_endpoint: CaptureWindowControlEndpointPlan::ReplicaEntry,
                process_id: "serve".to_owned(),
                endpoint: EndpointAssignment {
                    host: "127.0.0.1".to_owned(),
                    port: 1,
                },
                start: CaptureWindowActionPlan {
                    method: CaptureWindowHttpMethodPlan::Post,
                    path: "/start_profile".to_owned(),
                    body: None,
                    effective_url: "http://127.0.0.1:1/start_profile".to_owned(),
                },
                stop: CaptureWindowActionPlan {
                    method: CaptureWindowHttpMethodPlan::Post,
                    path: "/stop_profile".to_owned(),
                    body: None,
                    effective_url: "http://127.0.0.1:1/stop_profile".to_owned(),
                },
            },
            supported_window_controls: vec![WindowControlKind::FrameworkRange],
            command_cwd: temp.path().to_path_buf(),
            runtime_root: temp.path().join("profiles"),
            launch_prefix: Vec::new(),
            escapes: NsysEscapes::default(),
        };
        let plan = compile_plan(
            "serve",
            "bench",
            &["c1".to_owned()],
            std::slice::from_ref(&target),
            CaptureDeadlines {
                capture_arm_deadline_seconds: 60,
                capture_control_deadline_seconds: 60,
                capture_finalization_deadline_seconds: 1,
            },
        )?;
        let mut capture = CaptureSession {
            targets: vec![target],
            record: CaptureRecord {
                status: CaptureStatus::Running,
                plan: Some(plan.clone()),
                arm: Vec::new(),
                windows: Vec::new(),
                finalization: Vec::new(),
                reports: Vec::new(),
                engine_trace: Vec::new(),
                error: None,
            },
            plan,
            stop_failure: None,
            engine_trace_baselines: std::collections::BTreeMap::new(),
            finalization: None,
        };

        let bound = OperationBound::finite(Duration::from_millis(50));
        capture.verify_reports(&bound);

        assert_eq!(capture.record.status, CaptureStatus::Failed);
        assert!(!capture.record.reports[0].verified);
        assert!(
            capture
                .record
                .error
                .as_deref()
                .is_some_and(|error| error.contains("missing"))
        );
        Ok(())
    }

    /// An engine-trace window close dispatches when the measured phase ends
    /// and draws the one global finalization budget, not the per-action
    /// control budget; a response that outlasts the budget is neutral
    /// flush-pending evidence, so the capture fails only through coverage
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn engine_trace_close_dispatch_consumes_the_finalization_budget() -> Result<(), Box<dyn Error>>
    {
        let temp = tempfile::tempdir()?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            // start_profile answers promptly; stop_profile hangs past the
            // one-second finalization budget.
            for hang in [false, true] {
                let (mut stream, _) = listener.accept()?;
                let mut request = [0_u8; 1_024];
                let _ = stream.read(&mut request)?;
                if hang {
                    std::thread::sleep(Duration::from_millis(1_500));
                }
                // The close client is gone by then; a broken pipe is fine.
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
            }
            Ok(())
        });
        let action = |path: &str| CaptureWindowActionPlan {
            method: CaptureWindowHttpMethodPlan::Post,
            path: path.to_owned(),
            body: None,
            effective_url: format!("http://{address}{path}"),
        };
        let target = ProfilerTargetRecord {
            process_id: "serve".to_owned(),
            role_id: "serve".to_owned(),
            replica_id: "serve".to_owned(),
            replica_index: 0,
            rank: 0,
            rank_count: 1,
            device_count: 1,
            mechanism: CaptureMechanism::EngineTrace,
            trace_storage: Some(temp.path().join("trace")),
            session: String::new(),
            executable: String::new(),
            launch: ProfilerLaunch::Local,
            finalization: ProfilerFinalization::EngineTraceFlush,
            control: ProfilerControl::Http {
                window_control_endpoint: CaptureWindowControlEndpointPlan::ReplicaEntry,
                process_id: "serve".to_owned(),
                endpoint: EndpointAssignment {
                    host: "127.0.0.1".to_owned(),
                    port: address.port(),
                },
                start: action("/start_profile"),
                stop: action("/stop_profile"),
            },
            supported_window_controls: vec![WindowControlKind::FrameworkRange],
            command_cwd: temp.path().to_path_buf(),
            runtime_root: temp.path().join("profiles"),
            launch_prefix: Vec::new(),
            escapes: NsysEscapes::default(),
        };
        let mut capture = CaptureSession::open(
            "serve",
            "bench",
            &["w1".to_owned()],
            CaptureSelection {
                targets: vec![target],
                deadlines: CaptureDeadlines {
                    capture_arm_deadline_seconds: 60,
                    capture_control_deadline_seconds: 60,
                    capture_finalization_deadline_seconds: 1,
                },
            },
        )
        .map_err(|record| format!("capture arming failed: {record:?}"))?;

        capture.run_window("w1", || Ok::<(), crate::error::ProfilerError>(()))?;
        let record = capture.finish();

        let stop = record.windows[0]
            .stop
            .first()
            .ok_or("window close recorded no action")?;
        let CaptureActionRecord::Http {
            succeeded,
            flush_pending,
            failure_kind,
            timing,
            ..
        } = stop
        else {
            return Err("window close recorded non-HTTP evidence".into());
        };
        assert!(succeeded);
        assert!(flush_pending);
        assert_eq!(failure_kind, &None);
        assert_eq!(
            timing.budget,
            OperationBudgetEvidence::Finite {
                configured_ms: 1_000
            },
            "the close drew the finalization budget, not the 60 s control budget"
        );
        assert_eq!(timing.terminal_cause, OperationTerminalCause::TimedOut);
        assert_eq!(record.status, CaptureStatus::Failed);
        assert!(!record.engine_trace[0].verified);
        let error = record.error.as_deref().ok_or("missing capture error")?;
        assert!(
            error.contains("every device must produce one"),
            "coverage is the failing verdict: {error}"
        );
        assert!(
            !error.contains("window-closing control action had failed"),
            "flush-pending evidence must not read as a control failure: {error}"
        );
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }
}
