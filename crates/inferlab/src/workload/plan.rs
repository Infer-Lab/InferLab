//! Frozen Eval and Bench plans shared by dry-run, manual execution, recipes,
//! and record production.

use super::domain::{
    BenchPopulation, MeasurementModel, ResolvedBenchDefinition, ResolvedBenchPrompt,
    ResolvedBenchSloPolicy, WorkloadEndpoint, WorkloadHttpAction, WorkloadReplicaReset,
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
    /// The model whose `model_thresholds` entry selected the effective
    /// threshold, when one did ([[RFC-0005:C-LM-EVAL-EVIDENCE]]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_model: Option<String>,
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
    /// The nonce drawn when the execution was resolved, from which each
    /// case's warmup and profiling namespaces derive; present exactly when the
    /// Bench separates reuse by namespace instead of resetting the server
    /// ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_namespace_nonce: Option<String>,
    /// Each case's own population, keyed by case id, when the Bench
    /// namespaces its population.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub case_populations: BTreeMap<String, BenchPopulation>,
    pub command: ClientCommandPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<WorkloadHttpAction>,
    /// The per-target reset of every model-serving replica, used instead of
    /// a public reset action ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub replica_prefix_cache_resets: Vec<WorkloadReplicaReset>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_conditioning: Option<BenchPrefixCacheConditioningPlan>,
    /// The store's declared publication bound on a shared prefix cache: how
    /// long after a writer's first output token its prefix counts as
    /// available to later requests ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    #[serde(default, skip_serializing_if = "crate::execution::is_zero_u64")]
    pub prefix_publication_ms: u64,
    /// The server reports a cache-owning rank count or its store sharing only
    /// at readiness: preparation applies the reported facts to this plan and
    /// verifies a primed start's targets first ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub serving_facts_observed: bool,
    /// Probed at every profiling window's start and end; a case whose store
    /// is not engaged at either boundary fails ([[RFC-0003:C-SERVE-KV-STORE]]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serve_state_probe: Option<WorkloadServeStateProbe>,
}

impl BenchClientPlan {
    /// The population a case reads: its own when the Bench namespaces its
    /// population, else the one every case shares.
    pub(crate) fn population_for(&self, case_id: &str) -> Option<&BenchPopulation> {
        self.case_populations
            .get(case_id)
            .or(self.population.as_ref())
    }
}

/// A namespace from which one case phase draws its population content:
/// distinct for every Bench execution, case, and phase, and shared with
/// nothing else in the run ([[RFC-0004:C-BENCH-CACHE-STATE]]).
pub(crate) fn population_namespace(nonce: &str, seed: u64, case_id: &str, phase: &str) -> String {
    format!("{nonce}:{seed}:{case_id}:{phase}")
}

/// A nonce for one Bench execution. It only has to differ between
/// executions; nothing relies on it being unpredictable.
pub(crate) fn draw_population_namespace_nonce() -> String {
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicU64, Ordering};
    static DRAWN: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let digest = Sha256::digest(format!(
        "{nanos}:{}:{}",
        std::process::id(),
        DRAWN.fetch_add(1, Ordering::Relaxed)
    ));
    base16ct::lower::encode_string(&digest[..8])
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BenchPrefixCacheConditioningPlan {
    pub route: String,
    pub model: String,
    pub prompt: ResolvedBenchPrompt,
    pub request_body: BTreeMap<String, JsonValue>,
    /// Maximum shared-prefix token count the frozen population resolved. A
    /// token declaration, or a ratio over one input length, is known at plan
    /// time; a ratio over variable input lengths or a replayed population
    /// resolves it during preparation.
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
    /// Waited after a successful conditioning request before profiling, for
    /// the store to publish the shared prefix ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    #[serde(default, skip_serializing_if = "crate::execution::is_zero_u64")]
    pub publication_wait_ms: u64,
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
    /// How long a primed start waits after conditioning a shared prefix
    /// cache for the store to publish it ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    pub publication_wait_ms: u64,
    /// A cache-owning role's data-parallel rank count is observed only once
    /// the server is ready, so the target count above assumes one rank for
    /// it and a primed start must verify the observed count before
    /// preparation ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    pub data_parallel_pending: bool,
    /// Whether the store's roles share one prefix cache, and its publication
    /// bound, are reported only at readiness; until then the target count
    /// above may be one shared target ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    pub sharing_pending: bool,
}

/// Where a role's attention data-parallel rank count comes from
/// ([[RFC-0003:C-SERVE-PARALLELISM]]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RoleDataParallel {
    /// The role's effective parallelism shape declares it.
    Declared(u32),
    /// Each replica's Engine reported it at readiness.
    Observed(Vec<u32>),
    /// The role returned no shape and the server has not reported yet.
    Pending,
}

impl RoleDataParallel {
    /// The data-parallel source a role's effective shape states: declared by
    /// the shape, or pending when the role returned none.
    pub(crate) fn from_shape(shape: &inferlab_protocol::Parallelism) -> Self {
        if *shape == inferlab_protocol::Parallelism::default() {
            return Self::Pending;
        }
        Self::Declared(
            shape
                .attention
                .as_ref()
                .and_then(|attention| attention.data_parallel_size)
                .unwrap_or(1),
        )
    }

    /// A role's data-parallel source: its declared shape, else the counts its
    /// replicas' Engines reported at readiness, else pending.
    pub(crate) fn of(
        role: &crate::execution::RolePlan,
        observed: Option<&BTreeMap<String, u32>>,
    ) -> Self {
        let declared = Self::from_shape(&role.effective_parallelism);
        if !matches!(declared, Self::Pending) {
            return declared;
        }
        observed
            .and_then(|observed| {
                role.replicas
                    .iter()
                    .map(|replica| observed.get(&replica.entry_process).copied())
                    .collect::<Option<Vec<_>>>()
            })
            .map_or(Self::Pending, Self::Observed)
    }
}

impl ConditioningServingShape {
    /// `roles` yields `(serves_public_endpoint, kind, effective replica count,
    /// data-parallel source)` per model-serving role. A
    /// Gateway frontend makes every prefill-side role relevant because its
    /// targets force the fan-out capability requirement; a direct endpoint
    /// conditions the role that serves it.
    pub(crate) fn resolve(
        gateway_frontend: bool,
        roles: impl IntoIterator<
            Item = (
                bool,
                inferlab_protocol::ServeRoleKind,
                u32,
                RoleDataParallel,
            ),
        >,
    ) -> Self {
        let mut attention_data_parallel_size = 1u32;
        let mut conditioning_targets = 0u32;
        let mut data_parallel_pending = false;
        for (serves_public, kind, replicas, data_parallel) in roles {
            if !gateway_frontend && !serves_public {
                continue;
            }
            if gateway_frontend && kind == inferlab_protocol::ServeRoleKind::Decode {
                continue;
            }
            match data_parallel {
                RoleDataParallel::Declared(size) => {
                    attention_data_parallel_size = attention_data_parallel_size.max(size);
                    conditioning_targets += replicas * size;
                }
                RoleDataParallel::Observed(sizes) => {
                    attention_data_parallel_size = sizes
                        .iter()
                        .copied()
                        .fold(attention_data_parallel_size, u32::max);
                    conditioning_targets += sizes.iter().sum::<u32>();
                }
                RoleDataParallel::Pending => {
                    data_parallel_pending = true;
                    conditioning_targets += replicas;
                }
            }
        }
        Self {
            gateway_frontend,
            attention_data_parallel_size,
            conditioning_targets,
            publication_wait_ms: 0,
            data_parallel_pending,
            sharing_pending: false,
        }
    }

    /// The store's sharing is observed at readiness rather than declared
    /// ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    #[must_use]
    pub(crate) const fn with_sharing_pending(self) -> Self {
        Self {
            sharing_pending: true,
            ..self
        }
    }

    /// A shared prefix cache is one cache-owning target: one unpinned
    /// conditioning request covers the server ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    #[must_use]
    pub(crate) const fn shared(self, shared_prefix_cache: bool, publication_ms: u64) -> Self {
        if shared_prefix_cache {
            Self {
                attention_data_parallel_size: 1,
                conditioning_targets: 1,
                publication_wait_ms: publication_ms,
                data_parallel_pending: false,
                sharing_pending: false,
                ..self
            }
        } else {
            self
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

/// Everything a Bench case needs to ask the integration for the bound
/// server's serve state at its profiling window boundaries
/// ([[RFC-0003:C-SERVE-KV-STORE]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WorkloadServeStateProbe {
    pub integration: String,
    pub pixi_environment: String,
    pub adapter_timeout_seconds: u64,
    pub input: inferlab_protocol::ProbeServeInput,
}

impl WorkloadServeStateProbe {
    pub(crate) fn new(
        integration: &str,
        pixi_environment: &str,
        model: inferlab_protocol::ServeModelInput,
        topology: inferlab_protocol::ServeTopology,
        plan: &crate::execution::ServeStateProbePlan,
    ) -> Self {
        Self {
            integration: integration.to_owned(),
            pixi_environment: pixi_environment.to_owned(),
            adapter_timeout_seconds: plan.adapter_timeout_seconds,
            input: inferlab_protocol::ProbeServeInput {
                model,
                topology,
                attempt_bound_seconds: plan.attempt_bound_seconds,
                processes: plan.processes.clone(),
                kv_store: plan.kv_store.clone(),
            },
        }
    }
}

pub(crate) struct MeasurementResolveContext<'a> {
    pub workspace_root: &'a Path,
    pub workspace_source_exclusions: &'a [PathBuf],
    pub endpoint: WorkloadEndpoint,
    pub model: MeasurementModel,
    /// The workspace model the bound server serves, which selects a
    /// model-keyed Eval threshold ([[RFC-0004:C-LM-EVAL]]).
    pub model_id: &'a str,
    pub prefix_cache_reset: Option<WorkloadHttpAction>,
    /// The per-target reset of every model-serving replica
    /// ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    pub replica_prefix_cache_resets: Vec<WorkloadReplicaReset>,
    /// The Gateway frontend's conditioning fan-out action, when the selected
    /// frontend backend declares one ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    pub prefix_cache_conditioning: Option<WorkloadHttpAction>,
    pub conditioning_serving: ConditioningServingShape,
    /// The bound server's serve-state probe, when its integration declares
    /// one ([[RFC-0003:C-SERVE-KV-STORE]]).
    pub serve_state_probe: Option<WorkloadServeStateProbe>,
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
    use super::{
        ConditioningServingShape, RoleDataParallel, SessionPopulationLayout,
        session_population_layout,
    };
    use inferlab_protocol::ServeRoleKind;
    use std::collections::BTreeMap;

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
    /// A shared prefix cache collapses every replica and rank into one
    /// unpinned target ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    /// An Engine that owns its layout reports its rank count only at
    /// readiness: planning counts one target per replica and marks the shape
    /// pending, and the reported counts then decide the target set
    /// ([[RFC-0004:C-BENCH-CACHE-STATE]]).
    #[test]
    fn an_observed_rank_count_decides_the_targets_a_pending_one_defers() {
        let pending = ConditioningServingShape::resolve(
            true,
            [(false, ServeRoleKind::Serve, 1, RoleDataParallel::Pending)],
        );
        assert_eq!(pending.conditioning_targets, 1);
        assert!(pending.data_parallel_pending);
        assert!(!pending.shared(true, 0).data_parallel_pending);

        let observed = ConditioningServingShape::resolve(
            true,
            [(
                false,
                ServeRoleKind::Serve,
                2,
                RoleDataParallel::Observed(vec![4, 4]),
            )],
        );
        assert_eq!(observed.conditioning_targets, 8);
        assert_eq!(observed.attention_data_parallel_size, 4);
        assert!(!observed.data_parallel_pending);
    }

    /// Each replica's entry process carries its reported count; a replica
    /// that reported none leaves the role pending.
    #[test]
    fn a_shapeless_role_takes_each_replica_reported_count() -> Result<(), serde_json::Error> {
        let replica = |id: &str| {
            serde_json::json!({
                "id": id, "index": 0, "device_count": 4, "ports": [], "primary_ports": [],
                "primary_readiness": {"kind": "process_alive"},
                "worker_readiness": {"kind": "process_alive"},
                "entry_process": id, "ranks": []
            })
        };
        let role: crate::execution::RolePlan = serde_json::from_value(serde_json::json!({
            "id": "serve", "kind": "serve", "declared_replica_count": 2,
            "effective_replica_count": 2, "declared_parallelism": {},
            "effective_parallelism": {}, "declared_settings": {}, "effective_settings": {},
            "replicas": [replica("server-0"), replica("server-1")]
        }))?;
        let reported = BTreeMap::from([("server-0".to_owned(), 4), ("server-1".to_owned(), 4)]);
        let partial = BTreeMap::from([("server-0".to_owned(), 4)]);

        assert_eq!(
            RoleDataParallel::of(&role, Some(&reported)),
            RoleDataParallel::Observed(vec![4, 4])
        );
        assert_eq!(
            RoleDataParallel::of(&role, Some(&partial)),
            RoleDataParallel::Pending
        );
        Ok(())
    }

    #[test]
    fn a_shared_prefix_cache_is_one_unpinned_target() {
        let shape = ConditioningServingShape::resolve(
            true,
            [(true, ServeRoleKind::Serve, 2, RoleDataParallel::Declared(8))],
        );
        assert_eq!(shape.conditioning_targets, 16);

        let shared = shape.shared(true, 0);
        assert_eq!(shared.conditioning_targets, 1);
        assert_eq!(shared.attention_data_parallel_size, 1);
        assert_eq!(shape.shared(false, 0).conditioning_targets, 16);
    }

    #[test]
    fn conditioning_targets_count_prefill_side_replicas_times_attention_dp() {
        let serve =
            |public, kind, replicas, dp| (public, kind, replicas, RoleDataParallel::Declared(dp));

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
