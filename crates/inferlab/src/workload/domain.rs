use crate::bench_metric::BenchMetric;
use crate::workspace::{
    BenchArtifactLevel, BenchCacheStart, BenchPrefixSharing, BenchPrompt, BenchPromptSelection,
    BenchSharedSystemContent, BenchTokenSelector, BenchTpotApplicability, JsonValue, RequestSlo,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkloadEndpointProtocol {
    Http,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct WorkloadEndpoint {
    pub protocol: WorkloadEndpointProtocol,
    pub host: String,
    pub port: u16,
    pub completions_path: String,
    pub chat_completions_path: String,
    pub server_metrics: Option<WorkloadServerMetricsEndpoint>,
    pub prompt_cache_read_zero_representation:
        Option<inferlab_protocol::PromptCacheReadZeroRepresentation>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct WorkloadServerMetricsEndpoint {
    pub path: String,
    pub port_name: Option<String>,
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct MeasurementModel {
    pub locator: String,
    pub served_name: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkloadHttpMethod {
    Post,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct WorkloadHttpAction {
    pub method: WorkloadHttpMethod,
    pub path: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DatasetCacheState {
    Missing,
    Present,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BenchDatasetCatalog {
    pub dataset: String,
    pub profile: Option<String>,
    pub source: String,
    pub upstream_identity: String,
    pub url: String,
    pub sha256: String,
    pub source_format: String,
    pub aiperf_format: String,
    pub configuration: Option<String>,
    pub split: Option<String>,
    pub filter: Option<BenchDatasetFilter>,
    pub license: String,
    pub cache_path: PathBuf,
    pub cache_state: DatasetCacheState,
    pub materialization_identity: String,
    pub provides_output_targets: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BenchSessionDatasetCatalog {
    pub dataset: String,
    pub profile: Option<String>,
    pub source: String,
    pub upstream_identity: String,
    pub url: String,
    pub sha256: String,
    pub source_format: String,
    pub configuration: Option<String>,
    pub split: Option<String>,
    pub filter: Option<BenchDatasetFilter>,
    pub license: String,
    pub cache_path: PathBuf,
    pub cache_state: DatasetCacheState,
    pub materialization_identity: String,
    pub provides_output_targets: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct BenchAgenticCatalog {
    pub repository: String,
    pub revision: String,
    pub filename: String,
    pub sha256: String,
    pub cache_path: Option<PathBuf>,
    pub cache_state: Option<DatasetCacheState>,
    pub trace_count: u32,
    pub approximate_bytes: u64,
    pub license: String,
    pub source_format: String,
    pub aiperf_loader: String,
    pub materialization_identity: String,
    pub scenario: String,
    pub concurrency_semantics: String,
    pub replay_semantics: String,
    pub cache_bust: String,
    pub trajectory_start_min: f64,
    pub trajectory_start_max: f64,
    pub global_idle_gap_cap_seconds: f64,
    pub cache_warmup_seconds: u64,
    pub warmup_grace_seconds: u64,
    pub dataset_configuration_timeout_seconds: u64,
    pub service_profile_configuration_timeout_seconds: u64,
    pub default_duration_seconds: u64,
    pub minimum_duration_seconds: u64,
    pub failure_threshold: f64,
    pub dataset_entries: u32,
    pub streaming: bool,
    pub ignore_eos: bool,
    pub use_server_token_count: bool,
    pub gpu_telemetry: bool,
    pub server_metric_slice_seconds: u64,
    pub required_artifacts: Vec<String>,
    pub unavailable_dimensions: Vec<String>,
    pub inferencex_repository: String,
    pub inferencex_revision: String,
    pub inferencex_reference: String,
    pub aiperf_revision: String,
    pub aiperf_version: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BenchDatasetFilter {
    pub field: String,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct ResolvedBenchRandomShape {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub weight: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ResolvedBenchPrompt {
    pub declared: Option<BenchPrompt>,
    #[serde(flatten)]
    pub definition: BenchPrompt,
    pub request_representation: BenchRequestRepresentation,
    pub route: BenchPromptRoute,
    pub rendering_authority: BenchRenderingAuthority,
}

impl ResolvedBenchPrompt {
    pub(crate) fn from_declared_and_effective(
        declared: Option<&BenchPromptSelection>,
        effective: &BenchPromptSelection,
    ) -> Self {
        Self::resolve(
            declared.and_then(BenchPromptSelection::declared).cloned(),
            effective.effective().clone(),
        )
    }

    pub(crate) fn from_definition(definition: &BenchPrompt) -> Self {
        Self::resolve(None, definition.clone())
    }

    fn resolve(declared: Option<BenchPrompt>, definition: BenchPrompt) -> Self {
        let (request_representation, route, rendering_authority) = match &definition {
            BenchPrompt::Flat => (
                BenchRequestRepresentation::FlatPrompt,
                BenchPromptRoute::Completions,
                BenchRenderingAuthority::LocalFlat,
            ),
            BenchPrompt::RenderedChat { .. } => (
                BenchRequestRepresentation::FlatPrompt,
                BenchPromptRoute::Completions,
                BenchRenderingAuthority::LocalTemplate,
            ),
            BenchPrompt::ServerChat => (
                BenchRequestRepresentation::StructuredMessages,
                BenchPromptRoute::ChatCompletions,
                BenchRenderingAuthority::Server,
            ),
        };
        Self {
            declared,
            definition,
            request_representation,
            route,
            rendering_authority,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchRequestRepresentation {
    FlatPrompt,
    StructuredMessages,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchPromptRoute {
    Completions,
    ChatCompletions,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchRenderingAuthority {
    LocalFlat,
    LocalTemplate,
    Server,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ResolvedBenchRequestSource {
    Random {
        input_tokens: BenchTokenSelector,
        output_tokens: BenchTokenSelector,
        #[serde(default)]
        prefix_sharing: Option<BenchPrefixSharing>,
        #[serde(default)]
        shared_system_content: Option<BenchSharedSystemContent>,
        #[serde(default)]
        corpus: Option<ResolvedBenchCorpus>,
    },
    RandomMixture {
        shapes: Vec<ResolvedBenchRandomShape>,
        total_weight: u64,
        #[serde(default)]
        prefix_sharing: Option<BenchPrefixSharing>,
    },
    Dataset {
        dataset: String,
        profile: Option<String>,
        max_input_tokens: u32,
        output_tokens: Option<u32>,
        catalog: Box<BenchDatasetCatalog>,
    },
    Replay {
        /// Workspace-relative population path as declared.
        path: String,
        expected_sha256: Option<String>,
        #[serde(default)]
        prefix_sharing: Option<BenchPrefixSharing>,
        /// Absolute resolution of `path` against the workspace root.
        resolved_path: PathBuf,
        /// File facts observed while resolving the plan; absent when the file
        /// was unreadable or malformed, never fabricated.
        observed_sha256: Option<String>,
        observed_entries: Option<u32>,
        observed_tpot_applicability: Option<BenchTpotApplicability>,
    },
}

/// One resolved corpus binding on a random request source.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct ResolvedBenchCorpus {
    /// Workspace-relative corpus path as declared.
    pub path: String,
    pub expected_sha256: Option<String>,
    /// Absolute resolution of `path` against the workspace root.
    pub resolved_path: PathBuf,
    /// Content digest observed while resolving the plan; absent when the file
    /// was unreadable, never fabricated. The corpus token length stays
    /// unresolved here because tokenization is runner-owned.
    pub observed_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ResolvedBenchSessionSource {
    pub dataset: String,
    pub profile: Option<String>,
    pub max_input_tokens: u32,
    pub output_tokens: Option<u32>,
    pub inter_turn_delay_scale: f64,
    pub max_inter_turn_delay_seconds: Option<f64>,
    pub catalog: Box<BenchSessionDatasetCatalog>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ResolvedBenchAgenticSource {
    pub dataset: String,
    pub profile: String,
    pub catalog: Box<BenchAgenticCatalog>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum ResolvedBenchSource {
    Requests {
        request_source: ResolvedBenchRequestSource,
    },
    Sessions {
        session_source: ResolvedBenchSessionSource,
    },
    Agentic {
        agentic_source: ResolvedBenchAgenticSource,
    },
}

impl ResolvedBenchSource {
    pub(crate) fn request_source(&self) -> Option<&ResolvedBenchRequestSource> {
        match self {
            Self::Requests { request_source } => Some(request_source),
            Self::Sessions { .. } | Self::Agentic { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ResolvedBenchDefinition {
    #[serde(flatten)]
    pub source: ResolvedBenchSource,
    pub prompt: ResolvedBenchPrompt,
    pub server_metrics: bool,
    pub artifact_level: BenchArtifactLevel,
    pub seed: u64,
    pub request_body: BTreeMap<String, JsonValue>,
    pub request_slo: Option<RequestSlo>,
    pub timeout_seconds: u64,
    pub cache_start: BenchCacheStart,
}

impl ResolvedBenchDefinition {
    /// A primed start or declared prefix geometry requires backend-reported
    /// prompt cache-read usage for every successful profiling request
    /// ([[RFC-0004:C-BENCH-CACHE-STATE]]). Planning rejects benches with this
    /// requirement against endpoints that expose no cache-read capability;
    /// runtime normalization re-checks the same predicate.
    pub(crate) fn requires_prompt_cache_evidence(&self) -> bool {
        if self.cache_start == BenchCacheStart::Primed {
            return true;
        }
        matches!(
            self.source.request_source(),
            Some(
                ResolvedBenchRequestSource::Random {
                    prefix_sharing: Some(_),
                    ..
                } | ResolvedBenchRequestSource::Random {
                    shared_system_content: Some(_),
                    ..
                } | ResolvedBenchRequestSource::RandomMixture {
                    prefix_sharing: Some(_),
                    ..
                } | ResolvedBenchRequestSource::Replay {
                    prefix_sharing: Some(_),
                    ..
                }
            )
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BenchPopulation {
    pub path: PathBuf,
    pub evidence_path: PathBuf,
    pub sha256: String,
    pub entries: u32,
    pub tpot_applicable: bool,
    pub prefix_conditioning: Option<inferlab_protocol::BenchPrefixConditioningInput>,
    pub session_templates: Vec<BenchSessionTemplate>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct BenchSessionTemplate {
    pub template_identity: String,
    pub turn_count: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "direction", content = "value", rename_all = "snake_case")]
pub(crate) enum AggregateSloBound {
    AtMost(f64),
    AtLeast(f64),
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ResolvedAggregateSlo {
    pub metric: BenchMetric,
    pub bound: AggregateSloBound,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct ResolvedBenchSloPolicy {
    pub aggregate: Vec<ResolvedAggregateSlo>,
    pub request: Option<RequestSlo>,
}
