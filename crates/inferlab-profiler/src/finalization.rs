use crate::plan::{ProfilerFinalization, ProfilerTargetRecord};
use crate::record::CaptureActionRecord;
use crate::transport;
use std::path::Path;
use std::time::Duration;

const PROFILER_FINALIZATION_DEADLINE: Duration = Duration::from_secs(300);

#[must_use]
pub fn finalize_target(target: &ProfilerTargetRecord) -> CaptureActionRecord {
    match target.finalization {
        ProfilerFinalization::NsysStop => {
            transport::finalize_collection(target, PROFILER_FINALIZATION_DEADLINE)
        }
    }
}

#[must_use]
pub fn finalization_succeeded(action: &CaptureActionRecord) -> bool {
    action.succeeded() || collection_already_finalized(action)
}

pub(crate) fn verify_report(target: &ProfilerTargetRecord, path: &Path) -> CaptureActionRecord {
    transport::verify_report(target, path)
}

fn collection_already_finalized(action: &CaptureActionRecord) -> bool {
    matches!(
        action,
        CaptureActionRecord::Command { stderr, .. }
            if stderr.contains("Collection stop is not allowed in this state.")
    )
}
