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

pub(crate) const EVIDENCE_WORKLOAD_SCHEMA_VERSION: u32 = 14;
