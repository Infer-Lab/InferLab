//! Frozen Eval and Bench plans shared by dry-run, manual execution, recipes,
//! and record production.

use super::domain::{
    BenchPopulation, MeasurementModel, ResolvedBenchDefinition, ResolvedBenchSloPolicy,
    WorkloadEndpoint, WorkloadHttpAction,
};
use crate::execution::ResolvedExecution;
use crate::toolchain::{BenchToolchainIdentity, BundledEvalTask, EvalToolchainIdentity};
use crate::workspace::{
    BenchDefinition, BenchTpotApplicability, EvalDefinition, RequestRate, WorkspaceSnapshot,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct MeasurementPlan {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate: Option<String>,
    pub evals: Vec<EvalPlan>,
    pub benches: Vec<BenchPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct EvalPlan {
    pub id: String,
    pub capture: bool,
    pub declared_definition: EvalDefinition,
    pub definition: EvalDefinition,
    pub overrides: Vec<MeasurementOverridePlan>,
    pub endpoint: WorkloadEndpoint,
    pub model: MeasurementModel,
    pub workspace_source_exclusions: Vec<PathBuf>,
    pub execution: EvalExecutionPlan,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BenchPlan {
    pub id: String,
    pub capture: bool,
    pub declared_definition: BenchDefinition,
    pub definition: BenchDefinition,
    pub overrides: Vec<MeasurementOverridePlan>,
    pub execution: BenchExecutionPlan,
    pub client: BenchClientPlan,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct MeasurementOverridePlan {
    pub invocation_index: usize,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum ResolvedWorkloadPlan {
    Eval(Box<EvalPlan>),
    Bench(Box<BenchPlan>),
    ManualBench(Box<ManualBenchPlan>),
}

impl From<EvalPlan> for ResolvedWorkloadPlan {
    fn from(plan: EvalPlan) -> Self {
        Self::Eval(Box::new(plan))
    }
}

impl From<BenchPlan> for ResolvedWorkloadPlan {
    fn from(plan: BenchPlan) -> Self {
        Self::Bench(Box::new(plan))
    }
}

pub(crate) enum WorkloadServerAccess<'a> {
    RecipeOwned { record_id: &'a str },
    ManagedServer { record_id: &'a str },
}

impl WorkloadServerAccess<'_> {
    pub(crate) fn record_id(&self) -> &str {
        match self {
            Self::RecipeOwned { record_id } | Self::ManagedServer { record_id } => record_id,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ManualBenchTarget {
    pub server_record_id: String,
    pub producing_inferlab_version: String,
    pub serving_snapshot: ResolvedExecution,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ManualBenchPlan {
    pub invoking_inferlab_version: String,
    pub target: ManualBenchTarget,
    pub measurement_workspace: WorkspaceSnapshot,
    pub overrides: Vec<String>,
    pub bench: BenchPlan,
}

#[derive(Debug, Serialize)]
pub(crate) struct ManualBenchDryRun<'a> {
    pub dry_run: bool,
    pub invoking_inferlab_version: &'a str,
    pub target: &'a ManualBenchTarget,
    pub measurement_workspace: &'a WorkspaceSnapshot,
    pub overrides: &'a [String],
    pub bench: &'a BenchPlan,
}

impl ManualBenchPlan {
    pub(crate) fn dry_run_plan(&self) -> ManualBenchDryRun<'_> {
        ManualBenchDryRun {
            dry_run: true,
            invoking_inferlab_version: &self.invoking_inferlab_version,
            target: &self.target,
            measurement_workspace: &self.measurement_workspace,
            overrides: &self.overrides,
            bench: &self.bench,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum EvalExecutionPlan {
    #[serde(rename = "native_openai_smoke")]
    NativeOpenAiSmoke,
    LmEval {
        toolchain: Box<EvalToolchainIdentity>,
        #[serde(skip_serializing_if = "Option::is_none")]
        bundled_task: Option<Box<BundledEvalTask>>,
        command: ClientCommandPlan,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BenchClientPlan {
    pub toolchain: BenchToolchainIdentity,
    pub tokenizer_backend: String,
    pub endpoint: WorkloadEndpoint,
    pub model: MeasurementModel,
    pub effective_definition: ResolvedBenchDefinition,
    pub tpot_applicability: BenchTpotApplicability,
    pub slo: ResolvedBenchSloPolicy,
    pub required_population_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population: Option<BenchPopulation>,
    pub command: ClientCommandPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<WorkloadHttpAction>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ClientCommandPlan {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub cwd: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub(crate) enum BenchExecutionPlan {
    Matrix {
        cases: Vec<BenchCasePlan>,
    },
    Adaptive {
        policy: String,
        initial_request_rates: Vec<f64>,
        max_search_steps: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        min_rate_resolution: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        request_count: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_seconds: Option<u64>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BenchCasePlan {
    pub id: String,
    pub load_shape: LoadShape,
    pub request_count: u32,
    pub warmup_request_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_session_count: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SessionPopulationLayout {
    pub profiling_start: u32,
    pub required_entries: u32,
}

pub(super) fn session_population_layout(
    warmup_sessions: u32,
    profiling_sessions: u32,
) -> Option<SessionPopulationLayout> {
    let profiling_start = warmup_sessions.checked_add(u32::from(warmup_sessions > 0))?;
    let required_entries = profiling_start.checked_add(profiling_sessions)?;
    Some(SessionPopulationLayout {
        profiling_start,
        required_entries,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub(crate) enum LoadShape {
    ConcurrencyLimited {
        concurrency: u32,
    },
    RequestRateLimited {
        request_rate: RequestRate,
        #[serde(skip_serializing_if = "Option::is_none")]
        burstiness: Option<f64>,
    },
}

pub(crate) struct MeasurementResolveContext<'a> {
    pub workspace_root: &'a Path,
    pub workspace_source_exclusions: &'a [PathBuf],
    pub endpoint: WorkloadEndpoint,
    pub model: MeasurementModel,
    pub prefix_cache_reset: Option<WorkloadHttpAction>,
    pub capture_ids: &'a [String],
    pub command_env: &'a BTreeMap<String, String>,
    pub command_cwd: &'a Path,
}

#[cfg(test)]
mod tests {
    use super::{SessionPopulationLayout, session_population_layout};

    #[test]
    fn positive_warmup_reserves_the_native_terminal_prefetch_entry() {
        assert_eq!(
            session_population_layout(0, 6),
            Some(SessionPopulationLayout {
                profiling_start: 0,
                required_entries: 6,
            })
        );
        assert_eq!(
            session_population_layout(2, 6),
            Some(SessionPopulationLayout {
                profiling_start: 3,
                required_entries: 9,
            })
        );
    }
}
