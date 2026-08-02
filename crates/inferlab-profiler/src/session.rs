use crate::error::ProfilerError;
use crate::finalization;
use crate::plan::{CapturePlanRecord, ProfilerTargetRecord, compile_plan};
use crate::record::{
    CaptureActionRecord, CaptureRangeEndRecord, CaptureRecord, CaptureReportRecord, CaptureStatus,
    CaptureWindowRecord,
};
use crate::transport;
use std::collections::BTreeSet;

pub struct CaptureSession {
    targets: Vec<ProfilerTargetRecord>,
    plan: CapturePlanRecord,
    record: CaptureRecord,
    stop_failure: Option<String>,
}

impl CaptureSession {
    pub fn open(
        server_record_id: &str,
        workload_id: &str,
        window_ids: &[String],
        targets: Vec<ProfilerTargetRecord>,
    ) -> Result<Self, Box<CaptureRecord>> {
        let plan =
            compile_plan(server_record_id, workload_id, window_ids, &targets).map_err(|error| {
                Box::new(CaptureRecord::failed(format!("profiling failed: {error}")))
            })?;
        let mut session = Self {
            targets,
            record: CaptureRecord {
                status: CaptureStatus::Running,
                plan: Some(plan.clone()),
                arm: Vec::new(),
                windows: Vec::new(),
                finalization: Vec::new(),
                reports: Vec::new(),
                error: None,
            },
            plan,
            stop_failure: None,
        };
        if let Err(message) = session.arm_range_collection() {
            session.fail(message);
            session.finalize_collections();
            return Err(Box::new(session.record));
        }
        Ok(session)
    }

    fn arm_range_collection(&mut self) -> Result<(), String> {
        for (target, plan) in self.targets.iter().zip(&self.plan.targets) {
            let parent = plan
                .output_base
                .parent()
                .ok_or_else(|| format!("capture output {:?} has no parent", plan.output_base))?;
            let mkdir = transport::prepare_output(target, parent);
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
            let start = transport::arm_range_collection(target, &plan.output_base, count);
            let start_ok = start.succeeded();
            let start_error = start.error();
            self.record.arm.push(start);
            if !start_ok {
                return Err(start_error.unwrap_or_else(|| {
                    format!("failed to arm profiler target {:?}", target.process_id)
                }));
            }
        }
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
        transport::start_windows(&self.targets)
    }

    fn stop_window(&self, start: &[CaptureActionRecord]) -> Vec<CaptureActionRecord> {
        let started = start
            .iter()
            .filter(|action| action.succeeded())
            .filter_map(|action| match action {
                CaptureActionRecord::Http { process_id, .. } => Some(process_id.as_str()),
                CaptureActionRecord::Command { .. }
                | CaptureActionRecord::CollectionFinalization { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        transport::stop_windows(&self.targets, &started)
    }

    #[must_use]
    pub fn finish(mut self) -> CaptureRecord {
        self.finalize_collections();
        self.verify_reports();
        if self.record.error.is_none()
            && self.record.windows.iter().all(|window| window.succeeded)
            && self.record.reports.iter().all(|report| report.verified)
        {
            self.record.status = CaptureStatus::Succeeded;
        } else {
            self.record.status = CaptureStatus::Failed;
        }
        self.record
    }

    fn finalize_collections(&mut self) {
        let mut failure = None;
        for target in &self.targets {
            let action = finalization::finalize_target(target, self.range_end_for(target));
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
                        if process_id == control_process_id
                )
            })
            .then(|| CaptureRangeEndRecord {
                window_id: final_window.id.clone(),
                range_index: expected_range_count,
                expected_range_count,
            })
    }

    fn verify_reports(&mut self) {
        let mut failure = None;
        for (target, target_plan) in self.targets.iter().zip(&self.plan.targets) {
            for (window, path) in self.plan.windows.iter().zip(&target_plan.reports) {
                let verification = finalization::verify_report(target, path);
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
        CaptureWindowActionPlan, CaptureWindowControlEndpointPlan, CaptureWindowHttpMethodPlan,
        NsysEscapes, ProfilerControl, ProfilerFinalization, ProfilerLaunch, ProfilerTargetRecord,
        WindowControlKind, compile_plan,
    };
    use crate::record::{CaptureRecord, CaptureStatus};
    use inferlab_protocol::EndpointAssignment;
    use std::error::Error;

    #[test]
    fn missing_range_report_is_capture_failure_evidence() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let target = ProfilerTargetRecord {
            process_id: "serve".to_owned(),
            role_id: "serve".to_owned(),
            replica_id: "serve".to_owned(),
            replica_index: 0,
            rank: 0,
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
                deadline_seconds: 60,
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
                error: None,
            },
            plan,
            stop_failure: None,
        };

        capture.verify_reports();

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
}
