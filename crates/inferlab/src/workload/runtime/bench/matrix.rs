//! Deterministic matrix-case scheduling and per-execution identity checks.

use super::BenchRunOutcome;
use super::case::run_case_on_live_server;
use super::session::duplicate_runtime_session_identity;
use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::workload::record::{WorkloadRecordSession, WorkloadStatus};
use crate::workload::{BenchCasePlan, BenchPlan};
use inferlab_runtime::interrupt;

pub(super) fn run_matrix_cases(
    server_record_id: &str,
    plan: &BenchPlan,
    cases: &[BenchCasePlan],
    session: &mut WorkloadRecordSession,
    mut capture: Option<&mut inferlab_profiler::session::CaptureSession>,
    progress: &Progress,
) -> Result<BenchRunOutcome, InferlabError> {
    let mut measurement_succeeded = true;
    let mut passed = true;
    for (index, case) in cases.iter().enumerate() {
        if interrupt::received() {
            measurement_succeeded = false;
            passed = false;
            session.record_mut().skip_reason =
                Some("remaining Bench cases skipped because recipe was interrupted".to_owned());
            break;
        }
        let paths = session.case_paths(&case.id)?;
        progress.phase(
            Phase::named("Bench case")
                .item(&case.id, index + 1, cases.len())
                .log(session.absolute(&paths.stderr)),
        )?;
        let (mut record, server_lost) = run_case_on_live_server(
            server_record_id,
            index > 0,
            plan,
            case,
            session,
            capture.as_deref_mut(),
        )?;
        if let Some(evidence) = record.evidence.session.as_ref()
            && let Some(duplicate) = duplicate_runtime_session_identity(
                session
                    .bench_cases()?
                    .iter()
                    .filter_map(|case| case.evidence.session.as_ref()),
                evidence,
            )
        {
            let identity_error = format!(
                "runtime session identity {duplicate:?} is not unique within the Bench execution"
            );
            record.status = WorkloadStatus::Failed;
            record.error = Some(record.error.take().map_or(identity_error.clone(), |error| {
                format!("{error}; {identity_error}")
            }));
        }
        let case_succeeded = record.status == WorkloadStatus::Succeeded;
        measurement_succeeded &= case_succeeded;
        passed &= case_succeeded
            && record
                .evidence
                .slo
                .as_ref()
                .is_none_or(|evaluation| evaluation.passed);
        session.push_bench_case(record)?;
        session.rewrite()?;
        if server_lost {
            session.record_mut().skip_reason =
                Some("remaining Bench cases skipped because the server lost a process".to_owned());
            break;
        }
    }
    Ok(BenchRunOutcome {
        measurement_succeeded,
        passed,
    })
}
