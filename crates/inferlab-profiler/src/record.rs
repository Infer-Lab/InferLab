use crate::plan::{CapturePlanRecord, CaptureWindowHttpMethodPlan};
use inferlab_protocol::SettingValue;
use inferlab_runtime::operation_bound::OperationTimingEvidence;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

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
    pub client_succeeded: bool,
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
}

impl CaptureActionRecord {
    #[must_use]
    pub fn succeeded(&self) -> bool {
        match self {
            Self::Command { succeeded, .. }
            | Self::Http { succeeded, .. }
            | Self::CollectionFinalization { succeeded, .. } => *succeeded,
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
