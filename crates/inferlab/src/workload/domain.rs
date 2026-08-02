use crate::bench_metric::BenchMetric;
use crate::workspace::{BenchTokenSelector, JsonValue, RequestSlo};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadEndpointProtocol {
    Http,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkloadEndpoint {
    pub protocol: WorkloadEndpointProtocol,
    pub host: String,
    pub port: u16,
    pub completions_path: String,
    pub chat_completions_path: String,
    pub server_metrics: Option<WorkloadServerMetricsEndpoint>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkloadServerMetricsEndpoint {
    pub path: String,
    pub port_name: Option<String>,
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeasurementModel {
    pub locator: String,
    pub served_name: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadHttpMethod {
    Post,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkloadHttpAction {
    pub method: WorkloadHttpMethod,
    pub path: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetCacheState {
    Missing,
    Present,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchDatasetCatalog {
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
pub struct BenchSessionDatasetCatalog {
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchDatasetFilter {
    pub field: String,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResolvedBenchPrefixSharing {
    pub shared_prefix_ratio: f64,
    pub shared_prefix_tokens: u32,
    pub unique_suffix_tokens: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResolvedBenchRandomShape {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub weight: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedBenchRequestSource {
    Random {
        input_tokens: BenchTokenSelector,
        output_tokens: BenchTokenSelector,
        #[serde(default)]
        prefix_sharing: Option<ResolvedBenchPrefixSharing>,
    },
    RandomMixture {
        shapes: Vec<ResolvedBenchRandomShape>,
        total_weight: u64,
    },
    Dataset {
        dataset: String,
        profile: Option<String>,
        max_input_tokens: u32,
        output_tokens: Option<u32>,
        catalog: Box<BenchDatasetCatalog>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResolvedBenchSessionSource {
    pub dataset: String,
    pub profile: Option<String>,
    pub max_input_tokens: u32,
    pub output_tokens: Option<u32>,
    pub inter_turn_delay_scale: f64,
    pub max_inter_turn_delay_seconds: Option<f64>,
    pub catalog: Box<BenchSessionDatasetCatalog>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ResolvedBenchSource {
    Requests {
        request_source: ResolvedBenchRequestSource,
    },
    Sessions {
        session_source: ResolvedBenchSessionSource,
    },
}

impl ResolvedBenchSource {
    pub fn request_source(&self) -> Option<&ResolvedBenchRequestSource> {
        match self {
            Self::Requests { request_source } => Some(request_source),
            Self::Sessions { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolvedBenchDefinition {
    #[serde(flatten)]
    pub source: ResolvedBenchSource,
    pub server_metrics: bool,
    pub seed: u64,
    pub request_body: BTreeMap<String, JsonValue>,
    pub request_slo: Option<RequestSlo>,
    pub timeout_seconds: u64,
    pub reset_prefix_cache: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchPopulation {
    pub path: PathBuf,
    pub sha256: String,
    pub entries: u32,
    pub tpot_applicable: bool,
    pub session_templates: Vec<BenchSessionTemplate>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchSessionTemplate {
    pub template_identity: String,
    pub turn_count: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "direction", content = "value", rename_all = "snake_case")]
pub enum AggregateSloBound {
    AtMost(f64),
    AtLeast(f64),
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResolvedAggregateSlo {
    pub metric: BenchMetric,
    pub bound: AggregateSloBound,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ResolvedBenchSloPolicy {
    pub aggregate: Vec<ResolvedAggregateSlo>,
    pub request: Option<RequestSlo>,
}
