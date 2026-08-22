use crate::plan::{
    CaptureDeadlines, CaptureWindowHttpMethodPlan, CaptureWindowPlan, NsysEscapes, ProfilerControl,
    ProfilerFinalization, ProfilerLaunch, WindowControlKind, default_one,
};
use inferlab_protocol::{CaptureMechanism, SettingValue};
use inferlab_runtime::operation_bound::OperationTimingEvidence;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The `start_boundary` values recorded in operation timing evidence: each
/// names the point in the capture lifecycle at which the owning operation's
/// clock started ([[RFC-0004:C-WORKLOAD-PROFILING]]).
pub const ARM_START_BOUNDARY: &str = "before_profiler_arm";
pub const CONTROL_START_BOUNDARY: &str = "before_profiler_window_control_request";
pub const MEASUREMENT_FINALIZATION_START: &str =
    "after_measurement_business_terminal_before_profiler_finalization";
pub const SERVER_FINALIZATION_START: &str = "before_server_profiler_finalization";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureStatus {
    Running,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureRecord {
    pub status: CaptureStatus,
    pub plan: Option<CapturePlanRecord>,
    pub arm: Vec<CaptureActionRecord>,
    pub windows: Vec<CaptureWindowRecord>,
    pub finalization: Vec<CaptureActionRecord>,
    pub reports: Vec<CaptureReportRecord>,
    /// Per-replica trace-storage coverage verification for engine-trace
    /// targets ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[serde(default)]
    pub engine_trace: Vec<EngineTraceCoverageRecord>,
    pub error: Option<String>,
}

impl CaptureRecord {
    #[must_use]
    pub fn failed(message: String) -> Self {
        Self {
            status: CaptureStatus::Failed,
            plan: None,
            arm: Vec::new(),
            windows: Vec::new(),
            finalization: Vec::new(),
            reports: Vec::new(),
            engine_trace: Vec::new(),
            error: Some(message),
        }
    }

    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.status == CaptureStatus::Succeeded
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureWindowRecord {
    pub id: String,
    pub range_index: Option<usize>,
    pub start: Vec<CaptureActionRecord>,
    pub stop: Vec<CaptureActionRecord>,
    /// The measured workload client's verdict for this window.
    pub client_succeeded: bool,
    /// The window-level verdict the capture's terminal status consumes: the
    /// client verdict for an opened window, and always `false` for a window
    /// that never opened, whatever the client did. A failed window-closing
    /// control action does not flip this field; it surfaces through report
    /// and coverage verification, whose evidence lives on `start` and `stop`.
    pub succeeded: bool,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectionFinalizationOutcome {
    RangeEnd,
    Inactive,
    Stopped,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureRangeEndRecord {
    pub window_id: String,
    pub range_index: usize,
    pub expected_range_count: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CaptureActionRecord {
    Command {
        target_id: String,
        operation: String,
        argv: Vec<String>,
        exit_code: Option<i32>,
        stdout: String,
        stderr: String,
        succeeded: bool,
        timing: OperationTimingEvidence,
        cleanup: Option<inferlab_runtime::container::CommandCleanupEvidence>,
    },
    Http {
        process_id: String,
        operation: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        method: Option<CaptureWindowHttpMethodPlan>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        body: Option<BTreeMap<String, SettingValue>>,
        status: Option<u16>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        failure_kind: Option<CaptureHttpFailureKind>,
        /// Engine-trace window closing only: the close request was dispatched
        /// but no response was consumed before the shared finalization budget
        /// expired. Neutral flush-pending evidence, never a failure by itself;
        /// coverage verification is the sole completion verdict
        /// ([[RFC-0004:C-WORKLOAD-PROFILING]]). Records written before
        /// workload schema version 19 predate the field.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        flush_pending: bool,
        error: Option<String>,
        succeeded: bool,
        timing: OperationTimingEvidence,
    },
    CollectionFinalization {
        target_id: String,
        operation: String,
        session: String,
        outcome: CollectionFinalizationOutcome,
        observed_state: Option<String>,
        range_end: Option<CaptureRangeEndRecord>,
        inspection: Box<CaptureActionRecord>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        inspection_error: Option<String>,
        stop: Option<Box<CaptureActionRecord>>,
        succeeded: bool,
        error: Option<String>,
    },
    /// Engine-trace flush adjudication: coverage verification of the
    /// trace-storage delta is the sole artifact-flush completion verdict
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); the window-closing control
    /// response only acknowledges receipt, so this evidence never fails the
    /// capture by itself. Records written before workload schema version 19
    /// named the receipt field `flush_confirmed` and treated a successful
    /// window-closing control response as flush completion.
    EngineTraceFlush {
        target_id: String,
        operation: String,
        trace_dir: PathBuf,
        /// Every window this target's control process opened also recorded a
        /// successful window-closing control response for it.
        close_confirmed: bool,
        succeeded: bool,
        error: Option<String>,
    },
}

impl CaptureActionRecord {
    #[must_use]
    pub fn succeeded(&self) -> bool {
        match self {
            Self::Command { succeeded, .. }
            | Self::Http { succeeded, .. }
            | Self::CollectionFinalization { succeeded, .. }
            | Self::EngineTraceFlush { succeeded, .. } => *succeeded,
        }
    }

    #[must_use]
    pub fn error(&self) -> Option<String> {
        match self {
            Self::Command { stderr, .. } if !stderr.trim().is_empty() => {
                Some(stderr.trim().to_owned())
            }
            Self::Http { error, .. } => error.clone(),
            Self::CollectionFinalization { error, .. } => error.clone(),
            Self::EngineTraceFlush { error, .. } => error.clone(),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureHttpFailureKind {
    Serialization,
    Transport,
    Deadline,
    InvalidResponse,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureReportRecord {
    pub process_id: String,
    pub role_id: String,
    pub window_id: String,
    pub range_index: Option<usize>,
    pub path: PathBuf,
    pub verified: bool,
    pub verification: CaptureActionRecord,
}

/// Storage-delta coverage verification of one engine-trace replica: the
/// file snapshot of the dedicated trace directory taken at collection arming
/// and the new files observed at finalization. Coverage holds when the delta
/// contains at least one new trace artifact per device of the replica's
/// whole-replica device count, counted without framework-specific file
/// naming ([[RFC-0004:C-WORKLOAD-PROFILING]]); extra files (for example a
/// frontend trace) are allowed.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineTraceCoverageRecord {
    pub replica_id: String,
    pub trace_dir: PathBuf,
    /// The verification baseline: the target replica's whole-replica device
    /// count. Engine-internal profilers write one artifact per engine worker
    /// process, which the device count bounds, while the control-plane rank
    /// model counts entry processes ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    /// Records written before workload schema version 18 named this field
    /// `rank_count` and used the entry-process rank count as the baseline.
    pub expected_artifacts: u32,
    pub baseline_files: Vec<PathBuf>,
    pub new_files: Vec<PathBuf>,
    pub verified: bool,
    /// A trace-directory snapshot failure, when one ended verification early.
    /// Records written before workload schema version 17 predate the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// The capture plan as recorded plan evidence
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]): the deadlines, window identities, and
/// per-target expectations the capture session was executed against.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturePlanRecord {
    pub server_record_id: String,
    pub workload_id: String,
    pub deadlines: CaptureDeadlines,
    pub control: WindowControlKind,
    pub windows: Vec<CaptureWindowPlan>,
    pub targets: Vec<CaptureTargetPlan>,
}

/// One capture target as recorded plan evidence. The `role_id`,
/// `replica_index`, `rank`, `rank_count`, and `session` identity fields are
/// not read back by the control plane; they are recorded deliberately so a
/// record reader can map each capture target to its role, replica, rank, and
/// Nsight Systems session without joining the server record.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureTargetPlan {
    pub process_id: String,
    pub role_id: String,
    pub replica_id: String,
    pub replica_index: u32,
    pub rank: u32,
    #[serde(default = "default_one")]
    pub rank_count: u32,
    /// The replica's declared whole-replica device count: engine-trace
    /// coverage verification expects one new trace artifact per device
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]). Older records predate the field
    /// and default to 1.
    #[serde(default = "default_one")]
    pub device_count: u32,
    /// Records written before workload schema version 16 predate the field
    /// and are managed collection.
    #[serde(default)]
    pub mechanism: CaptureMechanism,
    pub session: String,
    pub expected_range_count: Option<usize>,
    pub output_base: PathBuf,
    pub reports: Vec<PathBuf>,
}

/// One profiled target process as recorded in the server record's per-process
/// evidence: the effective launch, control, and finalization facts the
/// profiler acted on ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilerTargetRecord {
    pub process_id: String,
    pub role_id: String,
    pub replica_id: String,
    pub replica_index: u32,
    pub rank: u32,
    #[serde(default = "default_one")]
    pub rank_count: u32,
    /// The target replica's declared whole-replica device count
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); engine-trace coverage verification
    /// expects one new trace artifact per device. Older records predate the
    /// field and default to 1.
    #[serde(default = "default_one")]
    pub device_count: u32,
    /// The effective capture mechanism of this target
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); server records written before
    /// schema version 7 predate the field and are managed collection.
    #[serde(default)]
    pub mechanism: CaptureMechanism,
    /// The record-owned trace directory assigned to an engine-trace target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_storage: Option<PathBuf>,
    pub session: String,
    pub executable: String,
    pub launch: ProfilerLaunch,
    pub finalization: ProfilerFinalization,
    pub control: ProfilerControl,
    pub supported_window_controls: Vec<WindowControlKind>,
    pub command_cwd: PathBuf,
    pub runtime_root: PathBuf,
    pub launch_prefix: Vec<String>,
    #[serde(default, skip_serializing_if = "NsysEscapes::is_empty")]
    pub escapes: NsysEscapes,
}
