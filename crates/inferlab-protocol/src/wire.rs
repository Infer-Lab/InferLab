//! The versioned wire types used by framework integrations
//! ([[RFC-0006:C-INTEGRATIONS]]) and by the release-owned Eval and Bench
//! measurement clients ([[RFC-0004:C-MEASUREMENTS]]). [`AdapterProtocol`] and
//! [`MeasurementProtocol`] are separate schema roots over this one Rust source
//! authority.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The shared protocol version used by framework integrations and release-owned
/// measurement clients. The only accepted value is `7` (serialized as the
/// string `"7"`); a mismatch is rejected before lowering.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
pub enum ProtocolVersion {
    /// Protocol version 7.
    #[serde(rename = "7")]
    V7,
}

/// The one JSON request an integration reads from stdin, tagged by the
/// requested operation.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdapterRequest {
    /// Plan a serve topology: declare roles, per-replica requirements, links,
    /// and endpoint requirements from the requested shape.
    PlanServe {
        protocol_version: ProtocolVersion,
        input: PlanServeInput,
    },
    /// Render final process invocations for a planned topology, given the
    /// control plane's concrete process allocations.
    RenderServe {
        protocol_version: ProtocolVersion,
        input: RenderServeInput,
    },
}

impl AdapterRequest {
    /// The protocol version carried by this request, regardless of operation.
    #[must_use]
    pub const fn protocol_version(&self) -> ProtocolVersion {
        match self {
            Self::PlanServe {
                protocol_version, ..
            }
            | Self::RenderServe {
                protocol_version, ..
            } => *protocol_version,
        }
    }
}

/// The requested serve shape a `PlanServe` operation lowers into a topology.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlanServeInput {
    pub model: ServeModelInput,
    pub topology: ServeTopology,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gateway_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pd_router_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer: Option<KvTransferMechanism>,
    pub roles: Vec<ServeRoleInput>,
    #[serde(default)]
    pub profiling: bool,
}

/// The planned topology plus the control plane's concrete allocations that a
/// `RenderServe` operation turns into final process invocations.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RenderServeInput {
    pub model: ServeModelInput,
    pub topology: ServeTopology,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gateway_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pd_router_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer: Option<KvTransferMechanism>,
    pub allocations: Vec<ServeProcessAllocation>,
    #[serde(default)]
    pub profiling: bool,
}

/// The serving deployment topology.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServeTopology {
    /// One aggregated serving role.
    Single,
    /// Disaggregated prefill and decode roles.
    PrefillDecode,
}

/// The logical role a replica plays in a serve topology.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServeRoleKind {
    /// Aggregated serving role of a single topology.
    Serve,
    /// Prefill role of a disaggregated topology.
    Prefill,
    /// Decode role of a disaggregated topology.
    Decode,
}

/// The KV-transfer mechanism connecting prefill and decode.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvTransferMechanism {
    /// Mooncake KV-cache transfer.
    Mooncake,
    /// NIXL KV-cache transfer.
    Nixl,
}

/// A requested serving role: its identity, kind, replica cardinality, and
/// declared (not-yet-completed) parallelism and settings.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeRoleInput {
    pub id: String,
    pub kind: ServeRoleKind,
    pub replica_count: u32,
    pub parallelism: Parallelism,
    pub settings: BTreeMap<String, SettingValue>,
}

/// A role as the integration resolved it: preserved identity and cardinality
/// with the complete effective settings and parallelism.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeRoleResult {
    pub id: String,
    pub kind: ServeRoleKind,
    pub declared_replica_count: u32,
    pub effective_replica_count: u32,
    pub effective_settings: BTreeMap<String, SettingValue>,
    pub effective_parallelism: Parallelism,
    /// The public endpoint contract for a direct `single` Engine. Gateway-
    /// backed shapes leave this absent and carry the contract on Gateway.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_endpoint: Option<EndpointRequirement>,
    #[serde(default)]
    pub render_inputs: Vec<RenderInputDeclaration>,
}

/// Framework-neutral component-aware parallelism ([[RFC-0003:C-SERVE-PARALLELISM]]).
/// Every component is optional so an operator can override one without
/// repeating the rest; omitted components are filled by the integration into a
/// complete effective shape.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parallelism {
    /// Outer deployment parallelism shared by attention and experts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer: Option<ParallelismOuter>,
    /// Attention-block parallelism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention: Option<ParallelismAttention>,
    /// MoE expert parallelism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experts: Option<ParallelismExperts>,
}

impl Parallelism {
    /// Overlay the components present in `other` onto `self`, leaving
    /// components `other` omits untouched (the per-component precedence merge).
    pub fn merge_from(&mut self, other: &Self) {
        if let Some(other) = &other.outer {
            self.outer.get_or_insert_default().merge_from(other);
        }
        if let Some(other) = &other.attention {
            self.attention.get_or_insert_default().merge_from(other);
        }
        if let Some(other) = &other.experts {
            self.experts.get_or_insert_default().merge_from(other);
        }
    }
}

/// Outer deployment parallelism: tensor and pipeline degrees.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParallelismOuter {
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_parallel_size: Option<u32>,
}

impl ParallelismOuter {
    fn merge_from(&mut self, other: &Self) {
        if other.tensor_parallel_size.is_some() {
            self.tensor_parallel_size = other.tensor_parallel_size;
        }
        if other.pipeline_parallel_size.is_some() {
            self.pipeline_parallel_size = other.pipeline_parallel_size;
        }
    }
}

/// Attention-block parallelism: tensor, data, and context degrees.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParallelismAttention {
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_parallel_size: Option<u32>,
}

impl ParallelismAttention {
    fn merge_from(&mut self, other: &Self) {
        if other.tensor_parallel_size.is_some() {
            self.tensor_parallel_size = other.tensor_parallel_size;
        }
        if other.data_parallel_size.is_some() {
            self.data_parallel_size = other.data_parallel_size;
        }
        if other.context_parallel_size.is_some() {
            self.context_parallel_size = other.context_parallel_size;
        }
    }
}

/// MoE expert parallelism: tensor, data, expert, and dense-tensor degrees.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParallelismExperts {
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_tensor_parallel_size: Option<u32>,
}

impl ParallelismExperts {
    fn merge_from(&mut self, other: &Self) {
        if other.tensor_parallel_size.is_some() {
            self.tensor_parallel_size = other.tensor_parallel_size;
        }
        if other.data_parallel_size.is_some() {
            self.data_parallel_size = other.data_parallel_size;
        }
        if other.expert_parallel_size.is_some() {
            self.expert_parallel_size = other.expert_parallel_size;
        }
        if other.dense_tensor_parallel_size.is_some() {
            self.dense_tensor_parallel_size = other.dense_tensor_parallel_size;
        }
    }
}

/// The logical model supplied during serving planning and rendering.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeModelInput {
    pub id: String,
    pub served_name: String,
}

/// The model identity used by measurement clients. Unlike integration
/// planning, a benchmark client may need a controller-visible tokenizer
/// locator.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementModelInput {
    pub locator: String,
    pub served_name: String,
}

/// A concrete host/port endpoint the control plane allocated for a process.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointAssignment {
    pub host: String,
    pub port: u16,
}

/// The public workload endpoint an Eval or Bench client connects to.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientEndpointInput {
    pub protocol: EndpointProtocol,
    pub host: String,
    pub port: u16,
    pub completions_path: String,
    pub chat_completions_path: String,
    #[serde(default)]
    pub server_metrics: Option<ServerMetricsEndpointInput>,
}

/// The resolved server-metrics endpoint supplied to a measurement client.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerMetricsEndpointInput {
    pub path: String,
    #[serde(default)]
    pub port_name: Option<String>,
    pub url: String,
}

/// The measurement an Eval client runs against the workload endpoint.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvalDefinitionInput {
    /// A single-prompt liveness check.
    #[serde(rename = "openai_smoke")]
    OpenAiSmoke {
        prompt: String,
        max_tokens: u32,
        timeout_seconds: u64,
    },
    /// An lm-eval task run with a pass threshold on the chosen metric.
    LmEval {
        task: Box<EvalTaskSourceInput>,
        #[serde(default)]
        request_body: BTreeMap<String, SettingValue>,
        limit: Option<u32>,
        few_shot: Option<u32>,
        seed: Option<u64>,
        trials: u32,
        max_tokens: Option<u32>,
        concurrency: Option<u32>,
        metric: String,
        #[serde(default)]
        metric_filter: Option<String>,
        threshold: f64,
        timeout_seconds: u64,
    },
}

/// The resolved lm-eval task source consumed by the release-owned Eval runner.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvalTaskSourceInput {
    /// One individual task shipped by the pinned lm-eval runtime.
    BuiltIn { name: String },
    /// One task closure shipped and identity-bound by the Inferlab release.
    Bundled {
        name: String,
        task_identity: String,
        path: PathBuf,
        task_closure_sha256: String,
        task_definition_sha256: String,
        prompt_asset_sha256: String,
        dataset_asset_sha256: String,
        scorer_sha256: String,
    },
    /// One validated workspace-local lm-eval YAML file.
    WorkspaceYaml { path: PathBuf },
}

/// The workload shape a Bench client drives, shared across its load cases.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchDefinitionInput {
    #[serde(default)]
    pub request_source: Option<BenchRequestSourceInput>,
    #[serde(default)]
    pub session_source: Option<BenchSessionSourceInput>,
    #[serde(default)]
    pub server_metrics: bool,
    pub seed: u64,
    #[serde(default)]
    pub request_body: BTreeMap<String, SettingValue>,
    #[serde(default)]
    pub request_slo: Option<BenchRequestSloInput>,
    pub timeout_seconds: u64,
    #[serde(default)]
    pub reset_prefix_cache: bool,
}

/// One closed request origin lowered by Inferlab for the Bench runtime.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BenchRequestSourceInput {
    /// AIPerf generates exact token-shape prompts from the release-pinned
    /// synthetic generator.
    Random {
        input_tokens: BenchTokenSelectorInput,
        output_tokens: BenchTokenSelectorInput,
        #[serde(default)]
        prefix_sharing: Option<BenchPrefixSharingInput>,
    },
    /// AIPerf samples exact token-shape pairs from one seeded categorical
    /// distribution.
    RandomMixture {
        shapes: Vec<BenchRandomShapeInput>,
        total_weight: u64,
    },
    /// Inferlab materializes a release-catalog conversation snapshot before
    /// AIPerf starts.
    Dataset {
        dataset: String,
        #[serde(default)]
        profile: Option<String>,
        max_input_tokens: u32,
        output_tokens: Option<u32>,
        catalog: Box<BenchDatasetCatalogInput>,
    },
}

/// One release-qualified population of dependent linear session templates.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSessionSourceInput {
    pub dataset: String,
    #[serde(default)]
    pub profile: Option<String>,
    pub max_input_tokens: u32,
    pub output_tokens: Option<u32>,
    pub inter_turn_delay_scale: f64,
    pub max_inter_turn_delay_seconds: Option<f64>,
    pub catalog: Box<BenchSessionDatasetCatalogInput>,
}

/// One fixed or bounded inclusive-uniform token-length selector.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(untagged)]
pub enum BenchTokenSelectorInput {
    Fixed(u32),
    InclusiveUniform(BenchInclusiveUniformInput),
}

/// The one bounded distribution currently supported for a token selector.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchInclusiveUniformInput {
    pub kind: BenchTokenDistributionKindInput,
    pub min: u32,
    pub max: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchTokenDistributionKindInput {
    InclusiveUniform,
}

/// The effective split between one shared system-message prefix and each
/// request's independently generated user suffix.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPrefixSharingInput {
    pub shared_prefix_ratio: f64,
    pub shared_prefix_tokens: u32,
    pub unique_suffix_tokens: u32,
}

/// One exact ISL/OSL pair and its relative categorical sampling weight.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRandomShapeInput {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub weight: u32,
}

/// Immutable release-catalog facts resolved by the Rust control plane.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchDatasetCatalogInput {
    pub dataset: String,
    #[serde(default)]
    pub profile: Option<String>,
    pub source: String,
    pub upstream_identity: String,
    pub url: String,
    pub sha256: String,
    pub source_format: String,
    pub aiperf_format: String,
    #[serde(default)]
    pub configuration: Option<String>,
    #[serde(default)]
    pub split: Option<String>,
    #[serde(default)]
    pub filter: Option<BenchDatasetFilterInput>,
    pub license: String,
    pub cache_path: PathBuf,
    pub cache_state: BenchDatasetCacheState,
    pub materialization_identity: String,
    pub provides_output_targets: bool,
}

/// Immutable release-catalog facts for a linear-session materializer. AIPerf
/// loader names are deliberately absent from this source boundary.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSessionDatasetCatalogInput {
    pub dataset: String,
    #[serde(default)]
    pub profile: Option<String>,
    pub source: String,
    pub upstream_identity: String,
    pub url: String,
    pub sha256: String,
    pub source_format: String,
    #[serde(default)]
    pub configuration: Option<String>,
    #[serde(default)]
    pub split: Option<String>,
    #[serde(default)]
    pub filter: Option<BenchDatasetFilterInput>,
    pub license: String,
    pub cache_path: PathBuf,
    pub cache_state: BenchDatasetCacheState,
    pub materialization_identity: String,
    pub provides_output_targets: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchDatasetFilterInput {
    pub field: String,
    pub value: String,
}

/// Read-only cache state observed while resolving the Bench plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchDatasetCacheState {
    Missing,
    Present,
}

/// One frozen dataset population consumed sequentially by every Bench case.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPopulationInput {
    pub path: PathBuf,
    pub sha256: String,
    pub entries: u32,
    pub tpot_applicable: bool,
    #[serde(default)]
    pub session_templates: Vec<BenchSessionTemplateInput>,
}

/// One frozen linear-template summary needed to assign case slices without
/// duplicating the content-bearing population evidence artifact.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSessionTemplateInput {
    pub template_identity: String,
    pub turn_count: u32,
}

/// The bounded tokenizer-backed operation that freezes one request population
/// before any Bench case starts.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPopulationPreparationRequest {
    pub protocol_version: ProtocolVersion,
    pub model: MeasurementModelInput,
    pub tokenizer_backend: String,
    pub transformers_version: String,
    #[serde(default)]
    pub request_source: Option<BenchRequestSourceInput>,
    #[serde(default)]
    pub session_source: Option<BenchSessionSourceInput>,
    #[serde(default)]
    pub source_path: Option<PathBuf>,
    pub required_entries: u32,
    pub seed: u64,
    #[serde(default)]
    pub request_body: BTreeMap<String, SettingValue>,
    pub artifact_dir: PathBuf,
}

/// Summary of the realized token counts. Exact per-entry values remain in the
/// population evidence artifact.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchTokenCountSummary {
    pub minimum: u32,
    pub maximum: u32,
    pub mean: f64,
}

/// Where the concrete template used for local prompt-length projection came
/// from. The model server remains authoritative for transport rendering.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchPromptTemplateSource {
    RequestBody,
    TokenizerDefault,
}

/// The concrete template used only for local complete-prompt projection.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPromptTemplateProjection {
    pub source: BenchPromptTemplateSource,
    pub content: String,
    pub sha256: String,
}

/// Summary of best-effort complete-prompt targeting for a synthetic population.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPromptTokenTargetingSummary {
    pub selected_prompt_tokens: BenchTokenCountSummary,
    pub pre_template_content_tokens: BenchTokenCountSummary,
    #[serde(default)]
    pub projection_template: Option<BenchPromptTemplateProjection>,
    pub exact_entries: u32,
    pub fallback_entries: u32,
    #[serde(default)]
    pub fallback_reasons: BTreeMap<String, u32>,
}

/// Terminal result of request-population materialization.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPopulationPreparationResult {
    pub schema_version: u32,
    pub status: ClientStatus,
    pub materialization_identity: String,
    pub requested_entries: u32,
    pub candidate_entries: u64,
    pub admitted_entries: u64,
    pub ineligible_entries: u64,
    #[serde(default)]
    pub ineligible_reasons: BTreeMap<String, u64>,
    pub population: Option<BenchPopulationInput>,
    pub input_tokens: Option<BenchTokenCountSummary>,
    pub output_tokens: Option<BenchTokenCountSummary>,
    #[serde(default)]
    pub prompt_token_targeting: Option<BenchPromptTokenTargetingSummary>,
    pub evidence_path: Option<PathBuf>,
    pub error: Option<String>,
}

/// Per-request latency bounds lowered to the release-owned Bench runner.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRequestSloInput {
    #[serde(default)]
    pub request_latency_ms: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub tpot_ms: Option<f64>,
    pub minimum_good_request_ratio: f64,
}

/// A framework-specific server setting value carried as structured JSON data
/// (never a pre-rendered shell fragment) across the integration boundary.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SettingValue {
    /// A JSON boolean.
    Bool(bool),
    /// A JSON integer.
    Integer(i64),
    /// A JSON floating-point number.
    Float(f64),
    /// A JSON string.
    String(String),
    /// A JSON array of setting values.
    Array(Vec<SettingValue>),
    /// A JSON object of named setting values.
    Object(BTreeMap<String, SettingValue>),
}

/// The one JSON response an integration writes to stdout, tagged by outcome.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdapterResponse {
    /// The operation succeeded and carries its result.
    Ok {
        protocol_version: ProtocolVersion,
        result: Box<AdapterResult>,
    },
    /// The operation was rejected with a structured error.
    Error {
        protocol_version: ProtocolVersion,
        error: AdapterError,
    },
}

impl AdapterResponse {
    /// The protocol version carried by this response, regardless of outcome.
    #[must_use]
    pub const fn protocol_version(&self) -> ProtocolVersion {
        match self {
            Self::Ok {
                protocol_version, ..
            }
            | Self::Error {
                protocol_version, ..
            } => *protocol_version,
        }
    }
}

/// The successful result of an [`AdapterRequest`], tagged by the operation it
/// answers.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdapterResult {
    /// The planned topology from a `PlanServe` request.
    PlanServe { output: Box<PlanServeResult> },
    /// The rendered process invocations from a `RenderServe` request.
    RenderServe { output: Box<RenderServeResult> },
}

/// The lowered topology returned by a `PlanServe`: effective Engine roles,
/// whole-replica requirements, logical links, and separate optional Gateway
/// and P/D Router component plans.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlanServeResult {
    pub integration: IntegrationIdentity,
    pub roles: Vec<ServeRoleResult>,
    pub replicas: Vec<ServeReplicaRequirement>,
    pub links: Vec<ServeRoleLink>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gateway: Option<GatewayPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pd_router: Option<PdRouterPlan>,
}

/// Which bounded implementation renders an accepted concrete frontend
/// allocation. This is a lowering boundary only; the control plane always
/// owns placement, lifecycle, cleanup, endpoints, and records.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderSource {
    ControlPlane,
    Integration,
}

/// The one canonical process role available to a protocol-v7 frontend.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontendProcessRole {
    Gateway,
}

/// The fixed co-rendering requirement shared by compatible frontend plans.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendCoRendering {
    pub process_role: FrontendProcessRole,
}

/// The literal first member of every closed frontend component binding.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontendGatewayComponent {
    Gateway,
}

/// The literal second member of the fused P/D frontend component binding.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontendPdRouterComponent {
    PdRouter,
}

/// The stable schema branch for a Gateway-only frontend binding.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
pub struct GatewayFrontendBinding(pub [FrontendGatewayComponent; 1]);

/// The stable schema branch for a fused Gateway and P/D Router binding.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
pub struct GatewayPdRouterFrontendBinding(
    pub (FrontendGatewayComponent, FrontendPdRouterComponent),
);

/// The only two frontend bindings protocol v7 accepts. Tuple representation
/// deliberately serializes as the closed ordered arrays `["gateway"]` and
/// `["gateway", "pd_router"]` rather than as an open component list.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FrontendComponents {
    Gateway(GatewayFrontendBinding),
    GatewayPdRouter(GatewayPdRouterFrontendBinding),
}

impl FrontendComponents {
    #[must_use]
    pub const fn gateway() -> Self {
        Self::Gateway(GatewayFrontendBinding([FrontendGatewayComponent::Gateway]))
    }

    #[must_use]
    pub const fn gateway_pd_router() -> Self {
        Self::GatewayPdRouter(GatewayPdRouterFrontendBinding((
            FrontendGatewayComponent::Gateway,
            FrontendPdRouterComponent::PdRouter,
        )))
    }

    #[must_use]
    pub const fn includes_pd_router(&self) -> bool {
        matches!(self, Self::GatewayPdRouter(_))
    }
}

/// The target a Gateway forwards accepted public requests to.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GatewayTarget {
    /// A routed-single Gateway targets the sole Engine role entry point.
    Engine { role: String },
    /// A P/D Gateway hands requests to its co-rendered P/D Router component.
    PdRouter,
}

/// The public frontend component plan returned independently from Engine and
/// P/D Router planning.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayPlan {
    pub backend: String,
    pub implementation: String,
    pub implementation_version: String,
    pub effective_settings: BTreeMap<String, SettingValue>,
    pub endpoint: EndpointRequirement,
    pub readiness: ReadinessProbe,
    #[serde(default)]
    pub ports: Vec<String>,
    pub targets: Vec<GatewayTarget>,
    #[serde(default)]
    pub render_inputs: Vec<RenderInputDeclaration>,
    pub render_source: RenderSource,
    pub co_rendering: FrontendCoRendering,
}

/// Independent policies for choosing prefill and decode targets.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PdRoutingPolicies {
    pub prefill: String,
    pub decode: String,
}

/// The currently demonstrated Gateway-to-P/D Router handoff is internal to
/// one fused frontend process.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontendHandoff {
    InProcess,
}

/// The P/D orchestration component plan, kept separate even when the same
/// process and provider also realize Gateway.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PdRouterPlan {
    pub backend: String,
    pub implementation: String,
    pub implementation_version: String,
    pub effective_settings: BTreeMap<String, SettingValue>,
    pub policies: PdRoutingPolicies,
    pub prefill_role: String,
    pub decode_role: String,
    pub target_scheme: TargetEndpointScheme,
    #[serde(default)]
    pub ports: Vec<String>,
    pub readiness: ReadinessProbe,
    pub handoff: FrontendHandoff,
    #[serde(default)]
    pub render_inputs: Vec<RenderInputDeclaration>,
    pub render_source: RenderSource,
    pub co_rendering: FrontendCoRendering,
}

/// One workspace-authored UTF-8 source file an integration declares during
/// planning for the control plane to supply during final rendering.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RenderInputDeclaration {
    pub source_path: String,
}

/// The original declared path plus the exact UTF-8 contents and digest the
/// control plane supplies to an integration during final rendering.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuppliedRenderInput {
    pub source_path: String,
    pub text: String,
    pub sha256: String,
}

/// A whole-replica resource and readiness requirement the integration declares
/// without choosing placement, ranks, or concrete endpoints.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeReplicaRequirement {
    pub id: String,
    pub role_id: String,
    pub replica_index: u32,
    pub device_count: u32,
    pub ports: Vec<String>,
    pub primary_ports: Vec<String>,
    pub primary_readiness: ReadinessProbe,
    pub worker_readiness: ReadinessProbe,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_target: Option<CaptureTargetRequirement>,
}

/// One concrete process allocation supplied to `RenderServe`. Model-rank and
/// process-only frontend identities are distinct so a frontend cannot acquire
/// model coordinates or a model locator by construction.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ServeProcessAllocation {
    ModelRank {
        process: String,
        role: String,
        role_kind: ServeRoleKind,
        replica: u32,
        rank: u32,
        rank_count: u32,
        machine: String,
        devices: Vec<u32>,
        model_locator: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        endpoint: Option<EndpointAssignment>,
        ports: BTreeMap<String, EndpointAssignment>,
        cache: String,
        launch: AllocationLaunch,
        effective_settings: BTreeMap<String, SettingValue>,
        effective_parallelism: Parallelism,
        #[serde(default)]
        links: Vec<ServeRoleLink>,
        #[serde(default)]
        dependencies: Vec<String>,
        #[serde(default)]
        render_inputs: Vec<SuppliedRenderInput>,
    },
    Frontend {
        process: String,
        process_role: FrontendProcessRole,
        components: FrontendComponents,
        machine: String,
        devices: Vec<u32>,
        endpoint: EndpointAssignment,
        ports: BTreeMap<String, EndpointAssignment>,
        cache: String,
        launch: AllocationLaunch,
        gateway: Box<GatewayPlan>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pd_router: Option<Box<PdRouterPlan>>,
        #[serde(default)]
        links: Vec<ServeRoleLink>,
        #[serde(default)]
        dependencies: Vec<String>,
        #[serde(default)]
        render_inputs: Vec<SuppliedRenderInput>,
    },
}

/// The machine-local launch channel selected by the control plane.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AllocationLaunch {
    Local,
    Ssh { target: String },
}

/// A directed link between serve roles the integration declares as part of the
/// topology.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ServeRoleLink {
    /// The source role routes requests to the target roles.
    RequestRouting {
        source: String,
        targets: Vec<String>,
    },
    /// KV cache is transferred from source to target over `mechanism`.
    KvTransfer {
        source: String,
        target: String,
        mechanism: KvTransferMechanism,
    },
    /// The source discovers the target through a bootstrap port.
    Bootstrap {
        source: String,
        target: String,
        port: String,
    },
    /// The source and target exchange out-of-band data over a side-channel port.
    SideChannel {
        source: String,
        target: String,
        port: String,
    },
}

/// Marks a replica as a profiling capture target and carries its window
/// control ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureTargetRequirement {
    pub window_control: CaptureWindowControlRequirement,
}

/// The logical workload endpoint and typed actions that open and close a
/// capture window.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureWindowControlRequirement {
    pub endpoint: CaptureWindowControlEndpoint,
    pub start: CaptureWindowHttpActionSpec,
    pub stop: CaptureWindowHttpActionSpec,
}

/// The logical workload endpoint exposing a capture target's window-control
/// actions.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureWindowControlEndpoint {
    /// The entry process of the capture target's Engine replica.
    ReplicaEntry,
    /// The separately planned Gateway process.
    Gateway,
}

/// The final process invocations returned by a `RenderServe`, one per supplied
/// allocation.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RenderServeResult {
    pub integration: IntegrationIdentity,
    pub processes: Vec<RenderedServeProcess>,
}

/// A rendered process bound to the model-rank or frontend allocation identity
/// it was produced for.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RenderedServeProcess {
    ModelRank {
        process: String,
        role: String,
        replica: u32,
        rank: u32,
        rank_count: u32,
        launch_files: Vec<LaunchFileDeclaration>,
        command: ProcessSpec,
    },
    Frontend {
        process: String,
        process_role: FrontendProcessRole,
        components: FrontendComponents,
        launch_files: Vec<LaunchFileDeclaration>,
        command: ProcessSpec,
    },
}

/// One immutable text input a rendered process requires before it can launch.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchFileDeclaration {
    pub relative_path: String,
    pub text: String,
    pub sha256: String,
}

/// A capture-window HTTP action invoked against a logical serving endpoint.
/// The integration owns any framework-specific JSON body; the control plane
/// owns execution and evidence.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureWindowHttpActionSpec {
    pub method: HttpMethod,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<BTreeMap<String, SettingValue>>,
}

/// An HTTP action invoked against a logical serving endpoint.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HttpActionSpec {
    pub method: HttpMethod,
    pub path: String,
}

/// The HTTP method of an action specification.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpMethod {
    /// HTTP POST.
    Post,
}

/// The integration's identity recorded on its results: adapter id, adapter
/// version, and the framework it lowers to.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrationIdentity {
    pub adapter_id: String,
    pub adapter_version: String,
    pub framework: String,
    pub framework_version: String,
}

/// A launchable process: its argument vector and environment.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessSpec {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// How the control plane decides a process is ready.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadinessProbe {
    /// Ready when an HTTP GET of `path` succeeds.
    Http { path: String },
    /// Ready when the public endpoint succeeds and its HTTP target registry
    /// contains every control-plane-derived serving target.
    HttpTargetRegistry(Box<HttpTargetRegistryReadiness>),
    /// Ready as soon as the process is alive.
    ProcessAlive,
}

/// The integration-owned HTTP registry contract for target-aware readiness.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HttpTargetRegistryReadiness {
    pub target_scheme: TargetEndpointScheme,
    pub readiness_path: String,
    pub registry_path: String,
    pub targets_field: String,
    pub target_url_field: String,
    pub target_role_field: String,
    pub target_healthy_field: String,
    pub target_bootstrap_port_field: String,
    pub prefill_role_value: String,
    pub decode_role_value: String,
    pub prefill_bootstrap_port: String,
}

/// The application protocol used to identify serving targets in a registry.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetEndpointScheme {
    /// HTTP serving endpoint.
    Http,
    /// gRPC serving endpoint.
    Grpc,
}

/// The workload endpoint's protocol and named OpenAI paths, plus an optional
/// prefix-cache-reset action a Bench case can invoke between runs.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointRequirement {
    pub protocol: EndpointProtocol,
    pub completions_path: String,
    pub chat_completions_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ServerMetricsEndpointRequirement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<HttpActionSpec>,
}

/// An integration-owned logical server-metrics endpoint.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerMetricsEndpointRequirement {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<String>,
}

/// The application protocol a workload endpoint speaks.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointProtocol {
    /// HTTP.
    Http,
}

/// A structured rejection an integration returns in an [`AdapterResponse::Error`].
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterError {
    pub code: AdapterErrorCode,
    pub message: String,
}

/// Machine-readable failure category an adapter reports in an [`AdapterError`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterErrorCode {
    /// The request was malformed or missing required fields.
    InvalidRequest,
    /// The request's protocol version is not accepted.
    UnsupportedProtocolVersion,
    /// A framework setting was unknown or invalid.
    InvalidSettings,
    /// An unexpected internal failure occurred in the integration.
    Internal,
    /// The requested operation is not supported by this integration.
    UnsupportedOperation,
}

/// The request the Eval measurement runtime passes to its client: the endpoint
/// to hit, the model, the eval definition, and where to write artifacts.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalClientRequest {
    pub protocol_version: ProtocolVersion,
    pub workspace_root: PathBuf,
    pub workspace_source_exclusions: Vec<PathBuf>,
    pub endpoint: ClientEndpointInput,
    pub model: MeasurementModelInput,
    pub definition: EvalDefinitionInput,
    /// Remaining control-plane case budget when the client is released.
    pub case_budget_seconds: f64,
    pub artifact_dir: PathBuf,
}

/// The request the Bench measurement runtime passes to its client: the
/// endpoint, model, bench definition, the load case to run, and the artifact
/// directory.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchClientRequest {
    pub protocol_version: ProtocolVersion,
    pub endpoint: ClientEndpointInput,
    pub model: MeasurementModelInput,
    pub definition: BenchDefinitionInput,
    #[serde(default)]
    pub population: Option<BenchPopulationInput>,
    pub case: BenchCaseInput,
    /// Remaining control-plane case budget when the client is released.
    pub case_budget_seconds: f64,
    pub artifact_dir: PathBuf,
}

/// A single Bench case: its load shape and the number of requests to send.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchCaseInput {
    pub load_shape: BenchLoadInput,
    pub request_count: u32,
    #[serde(default)]
    pub warmup_request_count: u32,
    #[serde(default)]
    pub session_count: Option<u32>,
    #[serde(default)]
    pub warmup_session_count: Option<u32>,
}

/// How a Bench case paces its requests.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BenchLoadInput {
    /// A fixed number of in-flight requests.
    ConcurrencyLimited { concurrency: u32 },
    /// A target arrival rate, optionally shaped by a burstiness factor.
    RequestRateLimited {
        request_rate: f64,
        burstiness: Option<f64>,
    },
    /// All requests issued as fast as possible.
    UnboundedRequestRate,
}

/// The terminal outcome a measurement client reports.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientStatus {
    /// The client completed its measurement successfully.
    Succeeded,
    /// The client did not complete successfully.
    Failed,
}

/// A typed Eval failure category preserved across the client boundary.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalFailureKind {
    TaskResolution,
    ProbeTokenizer,
    ProbeTransport,
    ProbeHttp,
    ProbeMalformedResponse,
    ProbeGeneratedOnlyLogprobs,
    ProbeTokenizerAlignment,
    MetricNormalization,
}

/// The threshold comparison selected from lm-eval's scoring direction.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalMetricComparison {
    AtLeast,
    AtMost,
}

/// The terminal conclusion of an lm-eval metric threshold gate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalMetricGateConclusion {
    Passed,
    Failed,
}

/// One finite lm-eval metric with its exact native provenance.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalNormalizedMetric {
    pub source_identity: String,
    pub metric: String,
    pub filter: Option<String>,
    pub native_metric_key: String,
    pub value: f64,
    pub higher_is_better: bool,
}

/// The effective threshold comparison for the configured primary metric.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalMetricGate {
    pub metric: EvalNormalizedMetric,
    pub threshold: f64,
    pub comparison: EvalMetricComparison,
    pub conclusion: EvalMetricGateConclusion,
}

/// Reconstructible aggregate counts for fixed outer Eval trials.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalTrialSummary {
    pub requested_trials: u32,
    pub issued_trials: u32,
    pub unissued_trials: u32,
    pub completed_trials: u32,
    pub request_failure_trials: u32,
    pub passed_trials: u32,
    pub pass_rate: Option<f64>,
    pub per_trial_metric: String,
    pub per_trial_filter: Option<String>,
    pub higher_is_better: bool,
}

/// The result an Eval client writes for the measurement runtime to consume.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalClientResult {
    /// Result envelope version; clients write `1`. The measurement runtime
    /// rejects an eval result whose version is not `1`.
    pub schema_version: u32,
    pub status: ClientStatus,
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub normalized_metrics: BTreeMap<String, EvalNormalizedMetric>,
    #[serde(default)]
    pub gate: Option<EvalMetricGate>,
    #[serde(default)]
    pub trial_summary: Option<EvalTrialSummary>,
    pub native_command: Vec<String>,
    #[serde(default)]
    pub native_exit_code: Option<i32>,
    #[serde(default)]
    pub native_timed_out: bool,
    pub raw_artifacts: Vec<RawArtifact>,
    pub failure_kind: Option<EvalFailureKind>,
    pub error: Option<String>,
}

/// The result a Bench client writes for the measurement runtime to consume.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchClientResult {
    /// Result envelope version; clients write `1`. The measurement runtime
    /// rejects a bench result whose version is not `1`.
    pub schema_version: u32,
    pub status: ClientStatus,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub normalization_schema: String,
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub request_slo: Option<BenchRequestSloResult>,
    #[serde(default)]
    pub session_evidence: Option<BenchSessionResultEvidence>,
    pub native_command: Vec<String>,
    pub native_exit_code: Option<i32>,
    #[serde(default)]
    pub report_invocations: Vec<BenchNativeInvocation>,
    pub raw_artifacts: Vec<RawArtifact>,
    pub error: Option<String>,
}

/// Reconciled native evidence for one session-bounded Bench phase.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSessionPhaseSummary {
    pub planned_sessions: u32,
    pub started_sessions: u32,
    pub succeeded_sessions: u32,
    pub failed_sessions: u32,
    pub planned_requests: u32,
    pub attempted_requests: u32,
    pub completed_requests: u32,
    pub failed_requests: u32,
    pub reconciled: bool,
}

/// Terminal evidence for one admitted runtime session.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRuntimeSessionResult {
    pub phase: String,
    pub runtime_session_id: String,
    pub template_identity: String,
    pub planned_turns: u32,
    pub attempted_turns: u32,
    pub status: ClientStatus,
    #[serde(default)]
    pub failure_classification: Option<String>,
    #[serde(default)]
    pub diagnostic: Option<String>,
    #[serde(default)]
    pub failing_turn: Option<u32>,
    pub suppressed_later_turns: u32,
}

/// One native transport request reconciled to a linear-session turn.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSessionTurnResult {
    pub phase: String,
    pub runtime_session_id: String,
    pub turn_index: u32,
    pub pre_template_content_tokens: u32,
    #[serde(default)]
    pub observed_prompt_tokens: Option<u32>,
    pub native_session_num: u64,
    #[serde(default)]
    pub preceding_native_session_num: Option<u64>,
    #[serde(default)]
    pub preceding_terminal_response_receipt_ns: Option<u64>,
    #[serde(default)]
    pub effective_inter_turn_delay_seconds: Option<f64>,
    pub request_start_ns: u64,
    #[serde(default)]
    pub inter_turn_delay_reconciled: Option<bool>,
    #[serde(default)]
    pub post_failure_continuation: bool,
    pub native_artifact_name: String,
}

/// Session and transport reconciliation returned by the Bench client.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSessionResultEvidence {
    pub warmup: BenchSessionPhaseSummary,
    pub profiling: BenchSessionPhaseSummary,
    pub sessions: Vec<BenchRuntimeSessionResult>,
    pub turns: Vec<BenchSessionTurnResult>,
    pub population_slice_reconciled: bool,
    pub sessions_reconciled: bool,
    pub turn_order_reconciled: bool,
    pub inter_turn_delays_reconciled: bool,
    pub native_requests_reconciled: bool,
    pub counts_reconciled: bool,
}

/// One bounded native post-processing command and its terminal outcome.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchNativeInvocation {
    pub purpose: String,
    pub command: Vec<String>,
    pub exit_code: Option<i32>,
    pub interrupted: bool,
    pub timed_out: bool,
}

/// File-bound request-SLO evidence derived from AIPerf profiling records.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRequestSloResult {
    pub good_requests: u64,
    pub good_request_ratio: f64,
    pub goodput: f64,
    pub profiling_duration_seconds: f64,
    pub profiling_duration_source: String,
    pub request_count_reconciled: bool,
    #[serde(default)]
    pub native_aggregate_good_request_count: Option<u64>,
    #[serde(default)]
    pub native_aggregate_good_request_count_consistent: Option<bool>,
}

/// A raw output file a client produced, retained as workload evidence.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawArtifact {
    pub name: String,
    pub kind: String,
    pub path: PathBuf,
}

/// The schema root for the independently released framework adapter SDK.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterProtocol {
    pub request: AdapterRequest,
    pub response: AdapterResponse,
}

/// The schema root aggregating the request and result surfaces used by the
/// product-owned Eval and Bench measurement clients. Its optional fields are
/// code-generation anchors and are never all populated in one message.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementProtocol {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_client_request: Option<EvalClientRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_client_result: Option<EvalClientResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_client_request: Option<BenchClientRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_client_result: Option<BenchClientResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_population_preparation_request: Option<BenchPopulationPreparationRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_population_preparation_result: Option<BenchPopulationPreparationResult>,
}
