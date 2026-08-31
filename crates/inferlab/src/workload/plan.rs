//! Frozen Eval and Bench plans shared by dry-run, manual execution, recipes,
//! and record production.

use super::domain::{
    BenchPopulation, MeasurementModel, ResolvedBenchDefinition, ResolvedBenchPrompt,
    ResolvedBenchSloPolicy, WorkloadEndpoint, WorkloadHttpAction,
};
use crate::execution::ResolvedExecution;
use crate::toolchain::{BenchToolchainIdentity, BundledEvalTask, EvalToolchainIdentity};
use crate::workspace::{
    BenchDefinition, BenchTpotApplicability, EvalDefinition, JsonValue, RequestRate,
    WorkspaceSnapshot,
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
    pub data_assets: Vec<super::data_asset::DataAssetPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct EvalPlan {
    pub id: String,
    pub capture: bool,
    pub declared_definition: EvalDefinition,
    pub definition: EvalDefinition,
    /// The prompt authority the workspace definition actually declared. The
    /// definitions above serialize their effective authority, so this is the
    /// only place a defaulted authority is distinguishable from a declared one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_prompt: Option<crate::workspace::EvalPrompt>,
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

impl ResolvedWorkloadPlan {
    pub(crate) fn kind(&self) -> super::record::WorkloadKind {
        match self {
            Self::Eval(_) => super::record::WorkloadKind::Eval,
            Self::Bench(_) | Self::ManualBench(_) => super::record::WorkloadKind::Bench,
        }
    }

    pub(crate) fn definition_id(&self) -> &str {
        match self {
            Self::Eval(plan) => &plan.id,
            Self::Bench(plan) => &plan.id,
            Self::ManualBench(plan) => &plan.bench.id,
        }
    }
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
    pub data_assets: Vec<super::data_asset::DataAssetPlan>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ManualBenchDryRun<'a> {
    pub dry_run: bool,
    pub invoking_inferlab_version: &'a str,
    pub target: &'a ManualBenchTarget,
    pub measurement_workspace: &'a WorkspaceSnapshot,
    pub overrides: &'a [String],
    pub bench: &'a BenchPlan,
    pub data_assets: &'a [super::data_asset::DataAssetPlan],
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
            data_assets: &self.data_assets,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_conditioning: Option<BenchPrefixCacheConditioningPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BenchPrefixCacheConditioningPlan {
    pub route: String,
    pub model: String,
    pub prompt: ResolvedBenchPrompt,
    pub request_body: BTreeMap<String, JsonValue>,
    /// Resolved maximum shared-prefix token count. Synthetic sources compute
    /// it at plan time; a replay ratio declaration resolves it from the
    /// replayed population during preparation.
    pub maximum_shared_prefix_tokens: Option<u32>,
    pub output_tokens: u32,
    pub consumes_population_entry: bool,
    /// Effective attention data-parallel size of the public serving role:
    /// conditioning issues one rank-pinned request per rank when greater
    /// than one ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    pub attention_data_parallel_size: u32,
    /// The route belongs to a Gateway frontend conditioning fan-out
    /// capability: the control plane issues one request and the frontend
    /// covers every prefill replica and data-parallel rank.
    pub frontend_fanout: bool,
}

/// The public-serving shape prefix-cache conditioning plans against: whether
/// the public workload endpoint belongs to a Gateway, the effective attention
/// data-parallel size the conditioning loop must cover, and the number of
/// cache-owning conditioning targets behind the endpoint
/// ([[RFC-0004:C-BENCH-CACHE-STATE]]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConditioningServingShape {
    pub gateway_frontend: bool,
    /// Effective attention data-parallel size of the cache-owning (prefill
    /// side) roles: behind a Gateway this is the prefill-side maximum; on a
    /// direct endpoint it is the public serving role's size.
    pub attention_data_parallel_size: u32,
    /// Cache-owning targets behind a Gateway: prefill-side replicas times
    /// their attention data-parallel sizes. Decode-side cache state is
    /// incidental to conditioning and does not count. A single target cannot
    /// be missed by ordinary frontend routing, so the declared fan-out
    /// capability is required only when this exceeds one.
    pub conditioning_targets: u32,
}

impl ConditioningServingShape {
    /// `roles` yields `(serves_public_endpoint, kind, effective replica count,
    /// effective attention data-parallel size)` per model-serving role. A
    /// Gateway frontend makes every prefill-side role relevant because its
    /// targets force the fan-out capability requirement; a direct endpoint
    /// conditions the role that serves it.
    pub(crate) fn resolve(
        gateway_frontend: bool,
        roles: impl IntoIterator<Item = (bool, inferlab_protocol::ServeRoleKind, u32, u32)>,
    ) -> Self {
        let mut attention_data_parallel_size = 1u32;
        let mut conditioning_targets = 0u32;
        for (serves_public, kind, replicas, data_parallel_size) in roles {
            if !gateway_frontend && !serves_public {
                continue;
            }
            if gateway_frontend && kind == inferlab_protocol::ServeRoleKind::Decode {
                continue;
            }
            attention_data_parallel_size = attention_data_parallel_size.max(data_parallel_size);
            conditioning_targets += replicas * data_parallel_size;
        }
        Self {
            gateway_frontend,
            attention_data_parallel_size,
            conditioning_targets,
        }
    }
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
        #[serde(default)]
        preparation_order: Vec<BenchPreparationStep>,
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
    #[serde(default)]
    pub preparation_order: Vec<BenchPreparationStep>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_session_count: Option<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchPreparationStep {
    WarmupDrain,
    CacheReset,
    CacheConditioning,
    ProfilingRelease,
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
    /// The Gateway frontend's conditioning fan-out action, when the selected
    /// frontend backend declares one ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    pub prefix_cache_conditioning: Option<WorkloadHttpAction>,
    pub conditioning_serving: ConditioningServingShape,
    /// Whether the bound server's resolved configuration carries synthetic
    /// acceptance ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]); every Eval kind
    /// fails measurement planning against it while Benches stay plannable.
    pub synthetic_acceptance: bool,
    pub capture_ids: &'a [String],
    pub command_env: &'a BTreeMap<String, String>,
    pub command_cwd: &'a Path,
}

#[cfg(test)]
mod tests {
    use super::{ConditioningServingShape, SessionPopulationLayout, session_population_layout};
    use inferlab_protocol::ServeRoleKind;

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

    // [[RFC-0004:C-BENCH-CACHE-STATE]] 0.30.2: the fan-out capability is
    // required only when more than one cache-owning (prefill-side) target
    // sits behind the Gateway; decode roles are incidental and never count.
    #[test]
    fn conditioning_targets_count_prefill_side_replicas_times_attention_dp() {
        let serve = |public, kind, replicas, dp| (public, kind, replicas, dp);

        // Direct single endpoint: only the public role is relevant.
        let shape =
            ConditioningServingShape::resolve(false, [serve(true, ServeRoleKind::Serve, 1, 1)]);
        assert!(!shape.gateway_frontend);
        assert_eq!(shape.attention_data_parallel_size, 1);
        assert_eq!(shape.conditioning_targets, 1);

        // Gateway-fronted single topology, one replica, DP1: the 0.12.0
        // regression shape (cuda-oxide decode-primed-8k) is a single target.
        let shape =
            ConditioningServingShape::resolve(true, [serve(true, ServeRoleKind::Serve, 1, 1)]);
        assert!(shape.gateway_frontend);
        assert_eq!(shape.conditioning_targets, 1);

        // Gateway-fronted P/D with one prefill and one decode replica, DP1:
        // decode is incidental, so this is still a single-target shape.
        let shape = ConditioningServingShape::resolve(
            true,
            [
                serve(false, ServeRoleKind::Prefill, 1, 1),
                serve(false, ServeRoleKind::Decode, 1, 1),
            ],
        );
        assert_eq!(shape.conditioning_targets, 1);
        assert_eq!(shape.attention_data_parallel_size, 1);

        // Attention DP on the single role multiplies targets.
        let shape =
            ConditioningServingShape::resolve(true, [serve(true, ServeRoleKind::Serve, 1, 2)]);
        assert_eq!(shape.conditioning_targets, 2);
        assert_eq!(shape.attention_data_parallel_size, 2);

        // Two prefill replicas force the fan-out requirement even at DP1.
        let shape = ConditioningServingShape::resolve(
            true,
            [
                serve(false, ServeRoleKind::Prefill, 2, 1),
                serve(false, ServeRoleKind::Decode, 2, 1),
            ],
        );
        assert_eq!(shape.conditioning_targets, 2);
    }
}
