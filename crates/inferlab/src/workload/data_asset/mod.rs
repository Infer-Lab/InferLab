//! Measurement data-source preparation shared by recipe and standalone Bench
//! owners. Dataset semantics remain with lm-eval, AIPerf, or the release catalog.

mod model;
mod planning;
mod runtime;

pub(crate) use model::{
    DataAssetConsumerKind, DataAssetPlan, DataAssetPreparationAttempt, PreparedEvalSource,
    WorkloadDataAssetEvidence,
};
pub(crate) use planning::{attempt_id_for, attempts_from_plans, plan_measurement_data_assets};
pub(crate) use runtime::{observe_data_asset_dry_run, prepare_data_assets};

/// Version 18 renames the engine-trace coverage baseline from the
/// entry-process `rank_count` to the whole-replica device count
/// (`expected_artifacts`, [[RFC-0004:C-WORKLOAD-PROFILING]]).
///
/// Version 19 moves the engine-trace window close into the global
/// finalization budget ([[RFC-0004:C-WORKLOAD-PROFILING]], ADR-0039): the
/// window-closing HTTP action gains the neutral `flush_pending` evidence
/// field, and the flush-adjudication record renames `flush_confirmed` to
/// `close_confirmed` because the control response no longer attests flush
/// completion.
pub(crate) const EVIDENCE_WORKLOAD_SCHEMA_VERSION: u32 = 19;
