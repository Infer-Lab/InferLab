use crate::plan::ProfilerFinalization;
use crate::poll::{Poll, poll_until};
use crate::record::{
    CaptureActionRecord, CaptureRangeEndRecord, CollectionFinalizationOutcome,
    EngineTraceCoverageRecord, ProfilerTargetRecord,
};
use crate::transport;
use inferlab_runtime::operation_bound::OperationBound;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

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
    close_confirmed: bool,
    bound: &OperationBound,
    start_boundary: &str,
) -> CaptureActionRecord {
    match target.finalization {
        ProfilerFinalization::NsysStop => {
            finalize_nsys_session(target, range_end, bound, start_boundary)
        }
        ProfilerFinalization::EngineTraceFlush => match target.trace_storage.clone() {
            Some(trace_dir) => CaptureActionRecord::EngineTraceFlush {
                target_id: target.process_id.clone(),
                operation: "finalize-engine-trace".to_owned(),
                trace_dir,
                close_confirmed,
                // An unconfirmed close response — whether failed delivery or
                // a still-pending flush — is adjudicated by coverage
                // verification ([[RFC-0004:C-WORKLOAD-PROFILING]]), so this
                // evidence record never fails the capture by itself.
                succeeded: true,
                error: None,
            },
            // A target that reached finalization without its assigned trace
            // directory cannot evidence a flush; that absence is a capture
            // failure, not an empty path recorded as success.
            None => CaptureActionRecord::EngineTraceFlush {
                target_id: target.process_id.clone(),
                operation: "finalize-engine-trace".to_owned(),
                trace_dir: PathBuf::new(),
                close_confirmed,
                succeeded: false,
                error: Some(format!(
                    "engine-trace target {:?} has no assigned trace directory",
                    target.process_id
                )),
            },
        },
    }
}

fn finalize_nsys_session(
    target: &ProfilerTargetRecord,
    range_end: Option<CaptureRangeEndRecord>,
    bound: &OperationBound,
    start_boundary: &str,
) -> CaptureActionRecord {
    let inspection = transport::inspect_collection_state(target, bound, start_boundary);
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

    let stop = transport::stop_collection(target, bound, start_boundary);
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

/// The files under an engine-trace trace directory, relative to it. The
/// window baseline and the finalization delta use the same listing so the
/// delta is naming-scheme agnostic ([[RFC-0004:C-WORKLOAD-PROFILING]]).
/// Symlinks are never followed: a linked directory cannot turn this listing
/// into an unbounded traversal of storage outside the trace directory, and a
/// link itself is recorded as a plain entry.
pub(crate) fn snapshot_trace_files(trace_dir: &Path) -> Result<BTreeSet<PathBuf>, String> {
    let mut files = BTreeSet::new();
    let mut pending = vec![trace_dir.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .map_err(|error| format!("failed to list trace directory {directory:?}: {error}"))?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!("failed to list trace directory {directory:?}: {error}")
            })?;
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                format!("failed to stat trace directory entry {path:?}: {error}")
            })?;
            if metadata.is_dir() {
                pending.push(path);
            } else {
                files.insert(
                    path.strip_prefix(trace_dir)
                        .map_err(|error| {
                            format!("trace directory entry {path:?} escaped its root: {error}")
                        })?
                        .to_path_buf(),
                );
            }
        }
    }
    Ok(files)
}

const INITIAL_TRACE_POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_TRACE_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Verify one engine-trace replica's coverage: the dedicated-directory
/// storage delta since collection arming must contain at least one new trace
/// artifact per device of the replica's whole-replica device count
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]). Extra files are allowed. The check
/// polls inside the remaining finalization budget so a still-flushing engine
/// can land its artifacts. A snapshot failure is evidence in itself: it ends
/// polling immediately with an unverified record instead of burning the
/// shared budget on repeated failing listings.
pub(crate) fn verify_engine_trace_coverage(
    replica_id: &str,
    trace_dir: &Path,
    expected_artifacts: u32,
    baseline: &BTreeSet<PathBuf>,
    bound: &OperationBound,
) -> EngineTraceCoverageRecord {
    let record = |new_files: Vec<PathBuf>, verified: bool, error: Option<String>| {
        EngineTraceCoverageRecord {
            replica_id: replica_id.to_owned(),
            trace_dir: trace_dir.to_path_buf(),
            expected_artifacts,
            baseline_files: baseline.iter().cloned().collect(),
            new_files,
            verified,
            error,
        }
    };
    poll_until(
        bound,
        INITIAL_TRACE_POLL_INTERVAL,
        MAX_TRACE_POLL_INTERVAL,
        || {
            let current = match snapshot_trace_files(trace_dir) {
                Ok(current) => current,
                Err(error) => return Poll::Done(record(Vec::new(), false, Some(error))),
            };
            let new_files = current.difference(baseline).cloned().collect::<Vec<_>>();
            if new_files.len() >= expected_artifacts as usize {
                Poll::Done(record(new_files, true, None))
            } else {
                Poll::Pending(record(new_files, false, None))
            }
        },
    )
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
