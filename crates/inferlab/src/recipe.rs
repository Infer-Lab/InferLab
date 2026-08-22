use crate::InferlabError;
use crate::execution::ResolvedExecution;
use crate::progress::{Phase, Progress};
use crate::record::{RECORD_FILE, RECORDS_DIR, RecordIdentity, now_unix_ms, record_id};
use crate::server::{self, ServerRecord, ServerStatus};
use crate::workload::{
    self, DataAssetConsumerKind, DataAssetPreparationAttempt, WorkloadDataAssetEvidence,
    WorkloadStatus, attempt_id_for, attempts_from_plans, prepare_data_assets,
};
use inferlab_runtime::interrupt;
use inferlab_runtime::operation_bound::{
    OperationBound, OperationTerminalCause, OperationTimingEvidence,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RecipeStatus {
    Running,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServerRecordRef {
    pub id: String,
    pub status: Option<ServerStatus>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkloadRecordRef {
    pub definition_id: String,
    pub id: String,
    pub status: WorkloadStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecipeCleanupEvidence {
    pub server_record_id: String,
    pub status: Option<ServerStatus>,
    pub verified: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecipeRecord {
    pub schema_version: u32,
    pub inferlab_version: String,
    pub id: String,
    pub status: RecipeStatus,
    pub started_unix_ms: u64,
    pub finished_unix_ms: Option<u64>,
    pub resolved: ResolvedExecution,
    pub server: ServerRecordRef,
    pub evals: Vec<WorkloadRecordRef>,
    pub benches: Vec<WorkloadRecordRef>,
    pub interrupted: bool,
    pub errors: Vec<String>,
    pub cleanup: Option<RecipeCleanupEvidence>,
    pub data_assets: Vec<DataAssetPreparationAttempt>,
    pub source_preparation_completed: bool,
    pub serving_launch_attempted: bool,
    pub source_preparation_timing: Option<OperationTimingEvidence>,
}

impl RecipeRecord {
    /// Bumped with `ServerRecord::SCHEMA_VERSION` on the protocol-v7 to v8
    /// hard cut: version-3 records (products 0.10 and 0.11) embed a
    /// pre-cut resolved execution and record references.
    const SCHEMA_VERSION: u32 = 4;
}

pub(crate) fn run(
    root: &Path,
    resolved: ResolvedExecution,
    progress: &Progress,
) -> Result<RecipeRecord, InferlabError> {
    interrupt::prepare().map_err(|source| InferlabError::ServerInterrupt { source })?;
    let measurements =
        resolved
            .measurements
            .as_ref()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: "closed-loop recipe has no resolved measurements".to_owned(),
            })?;
    let mut session = RecipeRecordSession::begin(root, &resolved)?;
    progress.phase(Phase::named("record created").record(
        session.record().id.clone(),
        root.join(RECORDS_DIR).join(&session.record().id),
    ))?;
    let server_id = session.record().server.id.clone();
    let mut server_started = false;

    let mut data_assets = session.record().data_assets.clone();
    let owner_record_id = session.record().id.clone();
    let source_preparation_bound = (!data_assets.is_empty()).then(OperationBound::unbounded);
    let source_preparation = prepare_data_assets(
        root,
        &owner_record_id,
        &measurements.data_assets,
        &mut data_assets,
        progress,
        |updates| {
            for update in updates {
                session.update_data_asset_attempt(update)?;
            }
            Ok(())
        },
    );
    let source_preparation_succeeded = match source_preparation {
        Ok(()) => {
            session.record_mut().source_preparation_completed = true;
            session.rewrite()?;
            true
        }
        Err(error) => {
            session.record_mut().errors.push(format!(
                "measurement source preparation failed before serving launch: {error}"
            ));
            session.rewrite()?;
            false
        }
    };
    if let Some(bound) = source_preparation_bound {
        session.record_mut().source_preparation_timing = Some(bound.timing(
            "before_first_source_preparation_effect",
            if source_preparation_succeeded {
                OperationTerminalCause::Succeeded
            } else if interrupt::received() {
                OperationTerminalCause::Interrupted
            } else {
                OperationTerminalCause::Failed
            },
        ));
        session.rewrite()?;
    }

    if source_preparation_succeeded {
        session.record_mut().serving_launch_attempted = true;
        session.rewrite()?;
        progress.phase(Phase::named("server startup"))?;
    }
    match source_preparation_succeeded
        .then(|| server::start_for_recipe(root, resolved.clone(), &server_id, progress))
    {
        Some(Ok(_)) => {
            server_started = true;
            session.record_mut().server.status = Some(ServerStatus::Running);
            session.rewrite()?;
        }
        Some(Err(error)) => {
            session.record_mut().server.status = Some(ServerStatus::Failed);
            session
                .record_mut()
                .errors
                .push(format!("server start failed: {error}"));
        }
        None => {}
    }

    let mut gate_succeeded = measurements.gate.is_none();
    let eval_total = measurements.evals.len();
    for (index, plan) in measurements.evals.iter().enumerate() {
        progress.phase(Phase::named("Eval").item(&plan.id, index + 1, eval_total))?;
        let id = format!("{}-eval-{index:03}-{}", session.record().id, plan.id);
        let data_assets = recipe_workload_data_assets(
            session.record(),
            measurements,
            DataAssetConsumerKind::Eval,
            &plan.id,
        )?;
        let outcome = if !server_started {
            workload::skip(
                root,
                &id,
                workload::ResolvedWorkloadPlan::Eval(Box::new(plan.clone())),
                "server did not start",
                progress,
                data_assets.clone(),
            )
        } else if interrupt::received() {
            workload::skip(
                root,
                &id,
                workload::ResolvedWorkloadPlan::Eval(Box::new(plan.clone())),
                "recipe interrupted",
                progress,
                data_assets.clone(),
            )
        } else {
            workload::run_eval(root, &id, plan, &server_id, progress, data_assets)
        };
        match outcome {
            Ok(record) => {
                if measurements.gate.as_deref() == Some(plan.id.as_str()) {
                    gate_succeeded =
                        record.status == WorkloadStatus::Succeeded && record.passed == Some(true);
                }
                session.record_mut().evals.push(WorkloadRecordRef {
                    definition_id: plan.id.clone(),
                    id: record.id,
                    status: record.status,
                });
            }
            Err(error) => {
                if measurements.gate.as_deref() == Some(plan.id.as_str()) {
                    gate_succeeded = false;
                }
                session.record_mut().evals.push(WorkloadRecordRef {
                    definition_id: plan.id.clone(),
                    id,
                    status: WorkloadStatus::Failed,
                });
                session
                    .record_mut()
                    .errors
                    .push(format!("Eval {:?} failed: {error}", plan.id));
            }
        }
        session.rewrite()?;
    }

    let bench_total = measurements.benches.len();
    for (index, plan) in measurements.benches.iter().enumerate() {
        progress.phase(Phase::named("Bench").item(&plan.id, index + 1, bench_total))?;
        let id = format!("{}-bench-{index:03}-{}", session.record().id, plan.id);
        let data_assets = recipe_workload_data_assets(
            session.record(),
            measurements,
            DataAssetConsumerKind::Bench,
            &plan.id,
        )?;
        let outcome = if !server_started {
            workload::skip(
                root,
                &id,
                workload::ResolvedWorkloadPlan::Bench(Box::new(plan.clone())),
                "server did not start",
                progress,
                data_assets.clone(),
            )
        } else if interrupt::received() {
            workload::skip(
                root,
                &id,
                workload::ResolvedWorkloadPlan::Bench(Box::new(plan.clone())),
                "recipe interrupted",
                progress,
                data_assets.clone(),
            )
        } else if !gate_succeeded {
            workload::skip(
                root,
                &id,
                workload::ResolvedWorkloadPlan::Bench(Box::new(plan.clone())),
                "eval gate did not succeed",
                progress,
                data_assets.clone(),
            )
        } else {
            workload::run_bench(
                root,
                &id,
                plan,
                workload::WorkloadServerAccess::RecipeOwned {
                    record_id: &server_id,
                },
                workload::ResolvedWorkloadPlan::Bench(Box::new(plan.clone())),
                progress,
                data_assets,
            )
        };
        match outcome {
            Ok(record) => session.record_mut().benches.push(WorkloadRecordRef {
                definition_id: plan.id.clone(),
                id: record.id,
                status: record.status,
            }),
            Err(error) => {
                session.record_mut().benches.push(WorkloadRecordRef {
                    definition_id: plan.id.clone(),
                    id,
                    status: WorkloadStatus::Failed,
                });
                session
                    .record_mut()
                    .errors
                    .push(format!("Bench {:?} failed: {error}", plan.id));
            }
        }
        session.rewrite()?;
    }

    if server_started {
        progress.phase(Phase::named("server cleanup"))?;
        match server::stop(root, &server_id, progress) {
            Ok(record) => {
                let (verified, cleanup_error) = server_cleanup_summary(&record);
                session.record_mut().server.status = Some(record.status);
                session.record_mut().cleanup = Some(RecipeCleanupEvidence {
                    server_record_id: server_id,
                    status: Some(record.status),
                    verified,
                    error: cleanup_error,
                });
            }
            Err(error) => {
                session.record_mut().cleanup = Some(RecipeCleanupEvidence {
                    server_record_id: server_id,
                    status: Some(ServerStatus::Failed),
                    verified: false,
                    error: Some(error.to_string()),
                });
                session
                    .record_mut()
                    .errors
                    .push(format!("server cleanup failed: {error}"));
            }
        }
    } else if session.record().serving_launch_attempted {
        match server::status(root, &server_id) {
            Ok(report) => {
                let (verified, cleanup_error) = server_cleanup_summary(&report.record);
                session.record_mut().server.status = Some(report.record.status);
                session.record_mut().cleanup = Some(RecipeCleanupEvidence {
                    server_record_id: server_id,
                    status: Some(report.record.status),
                    verified,
                    error: cleanup_error,
                });
            }
            Err(error) => {
                session.record_mut().cleanup = Some(RecipeCleanupEvidence {
                    server_record_id: server_id,
                    status: Some(ServerStatus::Failed),
                    verified: false,
                    error: Some(error.to_string()),
                });
            }
        }
    }

    session.record_mut().interrupted = interrupt::received();
    let succeeded = server_started
        && !session.record().interrupted
        && session.record().errors.is_empty()
        && session
            .record()
            .evals
            .iter()
            .all(|child| child.status == WorkloadStatus::Succeeded)
        && session
            .record()
            .benches
            .iter()
            .all(|child| child.status == WorkloadStatus::Succeeded)
        && session
            .record()
            .cleanup
            .as_ref()
            .is_some_and(|cleanup| cleanup.verified);
    session.finish(if succeeded {
        RecipeStatus::Succeeded
    } else {
        RecipeStatus::Failed
    })?;
    Ok(session.into_record())
}

fn recipe_workload_data_assets(
    record: &RecipeRecord,
    measurements: &crate::workload::MeasurementPlan,
    kind: DataAssetConsumerKind,
    definition_id: &str,
) -> Result<WorkloadDataAssetEvidence, InferlabError> {
    let Some(attempt_id) = attempt_id_for(&measurements.data_assets, kind, definition_id)? else {
        return Ok(WorkloadDataAssetEvidence::None);
    };
    let prepared_source = record
        .data_assets
        .iter()
        .find(|attempt| attempt.attempt_id == attempt_id)
        .and_then(DataAssetPreparationAttempt::eval_binding);
    Ok(WorkloadDataAssetEvidence::Recipe {
        recipe_record_id: record.id.clone(),
        attempt_ids: vec![attempt_id],
        prepared_source,
    })
}

fn server_cleanup_summary(record: &ServerRecord) -> (bool, Option<String>) {
    let verified = record.process_evidence.values().all(|process| {
        process
            .cleanup
            .last()
            .map_or(process.handle.is_none(), |cleanup| cleanup.verified)
    });
    let error = record
        .process_evidence
        .values()
        .filter_map(|process| process.cleanup.last())
        .find_map(|cleanup| cleanup.error.clone());
    (verified, error)
}

struct RecipeRecordSession {
    root: PathBuf,
    record: RecipeRecord,
}

impl RecipeRecordSession {
    fn begin(root: &Path, resolved: &ResolvedExecution) -> Result<Self, InferlabError> {
        let records_dir = root.join(RECORDS_DIR);
        fs::create_dir_all(&records_dir).map_err(|source| InferlabError::RecordIo {
            path: records_dir.clone(),
            source,
        })?;
        let started_unix_ms = now_unix_ms()?;
        let recipe = resolved
            .recipe
            .as_ref()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: "recipe execution resolved without a recipe identity".to_owned(),
            })?;
        let case = resolved.server.case.as_ref().map(|case| case.id.as_str());
        let id = record_id(
            RecordIdentity::Recipe {
                recipe: &recipe.id,
                case,
            },
            started_unix_ms,
        )?;
        let record_dir = records_dir.join(&id);
        fs::create_dir(&record_dir).map_err(|source| InferlabError::RecordIo {
            path: record_dir,
            source,
        })?;
        let server_record_id = record_id(
            RecordIdentity::Serve {
                server: &resolved.server.id,
                case,
            },
            started_unix_ms,
        )?;
        let data_assets = resolved
            .measurements
            .as_ref()
            .map_or_else(Vec::new, |plan| attempts_from_plans(&plan.data_assets));
        let record = RecipeRecord {
            schema_version: RecipeRecord::SCHEMA_VERSION,
            inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
            server: ServerRecordRef {
                id: server_record_id,
                status: None,
            },
            id,
            status: RecipeStatus::Running,
            started_unix_ms,
            finished_unix_ms: None,
            resolved: resolved.clone(),
            evals: Vec::new(),
            benches: Vec::new(),
            interrupted: false,
            errors: Vec::new(),
            cleanup: None,
            data_assets,
            source_preparation_completed: false,
            serving_launch_attempted: false,
            source_preparation_timing: None,
        };
        let session = Self {
            root: root.to_path_buf(),
            record,
        };
        session.rewrite()?;
        Ok(session)
    }

    fn record(&self) -> &RecipeRecord {
        &self.record
    }

    fn record_mut(&mut self) -> &mut RecipeRecord {
        &mut self.record
    }

    fn update_data_asset_attempt(
        &mut self,
        update: &DataAssetPreparationAttempt,
    ) -> Result<(), InferlabError> {
        let slot = self
            .record
            .data_assets
            .iter_mut()
            .find(|attempt| attempt.attempt_id == update.attempt_id)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown recipe data-asset attempt {:?}", update.attempt_id),
            })?;
        *slot = update.clone();
        self.rewrite()
    }

    fn rewrite(&self) -> Result<(), InferlabError> {
        write_record(&self.root, &self.record)
    }

    fn finish(&mut self, status: RecipeStatus) -> Result<(), InferlabError> {
        self.record.status = status;
        self.record.finished_unix_ms = Some(now_unix_ms()?);
        self.rewrite()
    }

    fn into_record(self) -> RecipeRecord {
        self.record
    }
}

fn write_record(root: &Path, record: &RecipeRecord) -> Result<(), InferlabError> {
    let path = root.join(RECORDS_DIR).join(&record.id).join(RECORD_FILE);
    crate::record::write_json(&path, record)
}
