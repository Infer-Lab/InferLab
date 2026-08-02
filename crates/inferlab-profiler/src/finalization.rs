use crate::plan::{ProfilerFinalization, ProfilerTargetRecord};
use crate::record::{CaptureActionRecord, CaptureRangeEndRecord, CollectionFinalizationOutcome};
use crate::transport;
use serde::Deserialize;
use std::path::Path;
use std::time::Duration;

const PROFILER_FINALIZATION_DEADLINE: Duration = Duration::from_secs(300);
const NSYS_INACTIVE_SESSION_STATE: &str = "Launched";

#[derive(Debug, Deserialize)]
struct NsysSessionRecord {
    name: String,
    state: String,
}

#[derive(Debug, thiserror::Error)]
enum SessionInspectionError {
    #[error("Nsight Systems session inspection failed: {message}")]
    Command { message: String },
    #[error("Nsight Systems returned malformed session JSON: {source}")]
    Json {
        #[source]
        source: serde_json::Error,
    },
}

#[must_use]
pub fn finalize_target(
    target: &ProfilerTargetRecord,
    range_end: Option<CaptureRangeEndRecord>,
) -> CaptureActionRecord {
    match target.finalization {
        ProfilerFinalization::NsysStop => finalize_nsys_session(target, range_end),
    }
}

fn finalize_nsys_session(
    target: &ProfilerTargetRecord,
    range_end: Option<CaptureRangeEndRecord>,
) -> CaptureActionRecord {
    let inspection = transport::inspect_collection_state(target, PROFILER_FINALIZATION_DEADLINE);
    let observed = observed_session_state(&inspection, &target.session);
    let (observed_state, inspection_error) = match observed {
        Ok(state) => (state, None),
        Err(error) => (None, Some(error.to_string())),
    };

    if inspection_error.is_none()
        && observed_state
            .as_deref()
            .is_none_or(|state| state == NSYS_INACTIVE_SESSION_STATE)
    {
        let outcome = if range_end.is_some() {
            CollectionFinalizationOutcome::RangeEnd
        } else {
            CollectionFinalizationOutcome::Inactive
        };
        return CaptureActionRecord::CollectionFinalization {
            target_id: target.process_id.clone(),
            operation: "finalize-collection".to_owned(),
            session: target.session.clone(),
            outcome,
            observed_state,
            range_end,
            inspection: Box::new(inspection),
            inspection_error: None,
            stop: None,
            succeeded: true,
            error: None,
        };
    }

    let stop = transport::stop_collection(target, PROFILER_FINALIZATION_DEADLINE);
    let succeeded = stop.succeeded();
    let error = (!succeeded).then(|| {
        let stop_error = stop
            .error()
            .unwrap_or_else(|| "Nsight Systems collection stop failed".to_owned());
        match &inspection_error {
            Some(inspection_error) => format!("{inspection_error}; fallback {stop_error}"),
            None => stop_error,
        }
    });
    CaptureActionRecord::CollectionFinalization {
        target_id: target.process_id.clone(),
        operation: "finalize-collection".to_owned(),
        session: target.session.clone(),
        outcome: if succeeded {
            CollectionFinalizationOutcome::Stopped
        } else {
            CollectionFinalizationOutcome::Failed
        },
        observed_state,
        range_end: None,
        inspection: Box::new(inspection),
        inspection_error,
        stop: Some(Box::new(stop)),
        succeeded,
        error,
    }
}

pub(crate) fn verify_report(target: &ProfilerTargetRecord, path: &Path) -> CaptureActionRecord {
    transport::verify_report(target, path)
}

fn observed_session_state(
    action: &CaptureActionRecord,
    session: &str,
) -> Result<Option<String>, SessionInspectionError> {
    let CaptureActionRecord::Command {
        stdout, succeeded, ..
    } = action
    else {
        return Err(SessionInspectionError::Command {
            message: "session inspection produced non-command evidence".to_owned(),
        });
    };
    if !succeeded {
        return Err(SessionInspectionError::Command {
            message: action
                .error()
                .unwrap_or_else(|| "command exited unsuccessfully".to_owned()),
        });
    }
    if stdout.trim().is_empty() {
        return Ok(None);
    }
    let sessions: Vec<NsysSessionRecord> =
        serde_json::from_str(stdout).map_err(|source| SessionInspectionError::Json { source })?;
    Ok(sessions
        .into_iter()
        .find(|candidate| candidate.name == session)
        .map(|candidate| candidate.state))
}
