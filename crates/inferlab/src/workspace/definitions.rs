//! Portable workspace definitions rooted at the single `WorkspaceConfig`
//! serde authority.

use crate::bench_metric::BenchMetric;
use inferlab_profiler::plan::NsysEscapes;
use inferlab_protocol::{CaptureMechanism, KvTransferMechanism, Parallelism, ServeTopology};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

pub(super) const WORKSPACE_FILE: &str = ".inferlab/workspace.toml";
pub(super) const WORKSPACE_FRAGMENT_DIR: &str = ".inferlab/workspace.d";
pub(super) const DEFAULT_LOCAL_FILE: &str = ".inferlab/local.toml";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkspaceConfig {
    pub schema_version: u32,
    // Every identifier-keyed section defaults to empty so a section may be
    // supplied entirely by workspace.d fragments; the root file need not
    // declare it ([[RFC-0002:C-WORKSPACE-AUTHORITY]]). Referential integrity
    // is still enforced after composition by validate_workspace, so an
    // accidentally undeclared definition surfaces as an unresolved reference.
    #[serde(default)]
    pub models: BTreeMap<String, ModelDefinition>,
    #[serde(default)]
    pub stacks: BTreeMap<String, StackDefinition>,
    #[serde(default)]
    pub servers: BTreeMap<String, ServerDefinition>,
    #[serde(default)]
    pub evals: BTreeMap<String, EvalDefinition>,
    #[serde(default, deserialize_with = "deserialize_defaulted_bench_definitions")]
    pub benches: BTreeMap<String, BenchDefinition>,
    #[serde(default)]
    pub workload_suites: BTreeMap<String, WorkloadSuiteDefinition>,
    #[serde(default)]
    pub recipes: BTreeMap<String, RecipeDefinition>,
    #[serde(default)]
    pub images: BTreeMap<String, ImageDefinition>,
    #[serde(default)]
    pub external_images: BTreeMap<String, ExternalImageDefinition>,
}
/// A digest-pinned serving image this workspace did not build
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): official releases, colleagues'
/// builds, older baselines. The declaration claims the integration the
/// image's serving stack answers; nothing else about the image is assumed
/// or qualified.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExternalImageDefinition {
    /// A registry reference carrying its immutable digest,
    /// `repository[:tag]@sha256:<64 hex>`.
    pub reference: String,
    pub integration: String,
}

/// A named runtime-image production unit ([[RFC-0007:C-IMAGE-BUILD]]): the
/// stack selection, base image, target platform batch, and
/// recipe-referenced model validation coordinates.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImageDefinition {
    pub stack: String,
    pub base_image: String,
    pub platforms: Vec<String>,
    /// Stack source paths built into wheels for the image. Omit to build every
    /// stack source path. Paths consumed only at wheel-build time through the
    /// activation environment (for example DeepGEMM, compiled into the vLLM
    /// wheel) are excluded by declaring the subset.
    #[serde(default)]
    pub packages: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub validations: Vec<ImageValidationCoordinate>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImageValidationCoordinate {
    pub recipe: String,
    #[serde(default)]
    pub server_case: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelDefinition {
    pub served_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StackDefinition {
    pub integration: String,
    pub pixi_environment: String,
    #[serde(default)]
    pub source_paths: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<EnvironmentCheckDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_postprocess: Vec<EnvironmentScriptDefinition>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServerDefinition {
    pub stack: String,
    pub model: String,
    pub topology: ServeTopology,
    pub readiness_timeout_seconds: u64,
    #[serde(default)]
    pub readiness_attempt_timeout_seconds: Option<u64>,
    /// Shared deadline for profiler output preparation and collection arming
    /// across every selected target ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[serde(default)]
    pub capture_arm_deadline_seconds: Option<u64>,
    /// Complete response deadline for each framework window-control action;
    /// the readiness timeout does not apply to capture-armed serving, but a
    /// lost window start silently shifts range identities.
    #[serde(default)]
    pub capture_control_deadline_seconds: Option<u64>,
    /// Shared deadline for collection finalization and report verification
    /// across every selected target.
    #[serde(default)]
    pub capture_finalization_deadline_seconds: Option<u64>,
    #[serde(default)]
    pub gateway_backend: Option<String>,
    #[serde(default)]
    pub pd_router_backend: Option<String>,
    #[serde(default)]
    pub kv_transfer: Option<KvTransferMechanism>,
    #[serde(default)]
    pub profiling: Option<bool>,
    /// Operator escape inputs onto the managed profiler commands
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[serde(default, skip_serializing_if = "ProfilerEscapes::is_empty")]
    pub profiler: ProfilerEscapes,
    #[serde(default)]
    pub parallelism: Parallelism,
    #[serde(default)]
    pub settings: BTreeMap<String, JsonValue>,
    #[serde(default)]
    pub roles: BTreeMap<String, ServeRoleDefinition>,
    #[serde(default)]
    pub cases: BTreeMap<String, ServerCaseDefinition>,
    #[serde(default)]
    pub default_case: Option<String>,
}

pub(crate) const DEFAULT_READINESS_ATTEMPT_TIMEOUT_SECONDS: u64 = 30;
pub(crate) const DEFAULT_CAPTURE_ARM_DEADLINE_SECONDS: u64 = 60;
pub(crate) const DEFAULT_CAPTURE_CONTROL_DEADLINE_SECONDS: u64 = 60;
pub(crate) const DEFAULT_CAPTURE_FINALIZATION_DEADLINE_SECONDS: u64 = 300;
/// The engine-trace finalization default ([[RFC-0003:C-RESOLUTION]]): the
/// close dispatch, its response consumption, and the artifact flush wait
/// share this one budget, and real flushes have run tens of minutes
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
pub(crate) const DEFAULT_ENGINE_TRACE_CAPTURE_FINALIZATION_DEADLINE_SECONDS: u64 = 3600;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeRoleDefinition {
    #[serde(default)]
    pub replicas: Option<u32>,
    #[serde(default)]
    pub parallelism: Parallelism,
    /// Role escapes merge into the server's common inputs
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[serde(default, skip_serializing_if = "ProfilerEscapes::is_empty")]
    pub profiler: ProfilerEscapes,
    #[serde(default)]
    pub settings: BTreeMap<String, JsonValue>,
}

/// Operator escape inputs onto the managed Nsight Systems commands plus the
/// capture-mechanism selection ([[RFC-0004:C-WORKLOAD-PROFILING]]): option
/// lists splice ahead of the managed argv tails so managed values win on
/// collision, dedicated fields replace their managed defaults, and the
/// mechanism is a scalar that replaces under composition.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ProfilerEscapes {
    /// The requested capture mechanism; omission resolves to
    /// `managed_collection` at resolution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mechanism: Option<CaptureMechanism>,
    pub nsys: NsysEscapes,
}

impl ProfilerEscapes {
    pub(crate) fn is_empty(&self) -> bool {
        self.mechanism.is_none() && self.nsys.is_empty()
    }

    /// Role declarations merge into the server's common inputs: the nsys
    /// escapes follow [`NsysEscapes::merged_with`] and the mechanism is a
    /// scalar the role replaces ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[must_use]
    pub(crate) fn merged_with(&self, role: &Self) -> Self {
        Self {
            mechanism: role.mechanism.or(self.mechanism),
            nsys: self.nsys.merged_with(&role.nsys),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ServeRoleOverride {
    pub replicas: Option<u32>,
    pub parallelism: Parallelism,
    pub settings: BTreeMap<String, JsonValue>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EnvironmentCheckDefinition {
    pub id: String,
    /// Workspace-relative Python script; exit status zero is the sole pass
    /// signal, and output reports facts, not remedies.
    pub script: PathBuf,
    /// Operator remedy shown only on local-realization failure; an image
    /// failure means a systematic input defect, not drift.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_hint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EnvironmentScriptDefinition {
    pub id: String,
    pub script: PathBuf,
}

/// One JSON-compatible value declared in framework settings or an inference
/// request body.
///
/// A dedicated visitor keeps TOML date/time values from being coerced to
/// strings: the workspace definition is an exact JSON body fragment, not a
/// general TOML value tree.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum JsonValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl<'de> Deserialize<'de> for JsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct JsonValueVisitor;

        impl<'de> Visitor<'de> for JsonValueVisitor {
            type Value = JsonValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(
                    "a JSON-compatible value (boolean, integer, finite float, string, array, or table)",
                )
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(JsonValue::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(JsonValue::Integer(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i64::try_from(value)
                    .map(JsonValue::Integer)
                    .map_err(|_| E::custom("request body integer exceeds the supported range"))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E> {
                Ok(JsonValue::Float(value))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(JsonValue::String(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(JsonValue::String(value))
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(JsonValue::Array(values))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = map.next_entry()? {
                    values.insert(key, value);
                }
                if values.contains_key("$__toml_private_datetime") {
                    return Err(serde::de::Error::custom(
                        "JSON-compatible values do not support TOML date or time values",
                    ));
                }
                Ok(JsonValue::Object(values))
            }
        }

        deserializer.deserialize_any(JsonValueVisitor)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum EvalDefinition {
    #[serde(rename = "openai-smoke")]
    OpenAiSmoke {
        #[serde(default = "default_openai_smoke_prompt")]
        prompt: String,
        #[serde(default = "default_openai_smoke_max_tokens")]
        max_tokens: u32,
        #[serde(default = "default_openai_smoke_timeout_seconds")]
        timeout_seconds: u64,
    },
    LmEval {
        task: EvalTaskSource,
        #[serde(default)]
        prompt: EvalPromptSelection,
        #[serde(default)]
        request_body: BTreeMap<String, JsonValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        few_shot: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
        #[serde(default = "default_eval_trials")]
        trials: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_tokens: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        concurrency: Option<u32>,
        metric: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metric_filter: Option<String>,
        threshold: f64,
        timeout_seconds: u64,
    },
}

fn default_openai_smoke_prompt() -> String {
    "Hello".to_owned()
}

const fn default_openai_smoke_max_tokens() -> u32 {
    16
}

const fn default_openai_smoke_timeout_seconds() -> u64 {
    60
}

const fn default_eval_trials() -> u32 {
    1
}

/// The prompt rendering authority a generative lm-eval definition may declare.
///
/// The resolved task output type bounds the legal domain, and that bound is
/// enforced where the task is loaded rather than here.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum EvalPrompt {
    #[default]
    Flat,
    ServerChat,
}

/// An lm-eval prompt authority separated into what the definition declared and
/// what resolution selected, so records can tell a declaration from a default.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EvalPromptSelection {
    declared: Option<EvalPrompt>,
    effective: EvalPrompt,
}

impl EvalPromptSelection {
    pub(crate) fn explicit(prompt: EvalPrompt) -> Self {
        Self {
            declared: Some(prompt.clone()),
            effective: prompt,
        }
    }

    pub(crate) fn declared(&self) -> Option<&EvalPrompt> {
        self.declared.as_ref()
    }

    pub(crate) fn effective(&self) -> &EvalPrompt {
        &self.effective
    }
}

impl Default for EvalPromptSelection {
    fn default() -> Self {
        Self {
            declared: None,
            effective: EvalPrompt::Flat,
        }
    }
}

impl<'de> Deserialize<'de> for EvalPromptSelection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        EvalPrompt::deserialize(deserializer).map(Self::explicit)
    }
}

impl Serialize for EvalPromptSelection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.effective.serialize(serializer)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum EvalTaskSource {
    BuiltIn(String),
    Bundled { bundled: String },
    WorkspaceYaml { yaml: PathBuf },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AggregateSlo {
    pub metric: BenchMetric,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_most: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_least: Option<f64>,
}

fn deserialize_nonempty_aggregate_slos<'de, D>(
    deserializer: D,
) -> Result<Vec<AggregateSlo>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let constraints = Vec::<AggregateSlo>::deserialize(deserializer)?;
    if constraints.is_empty() {
        return Err(serde::de::Error::custom(
            "aggregate_slos must be non-empty when declared",
        ));
    }
    Ok(constraints)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RequestSlo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tpot_ms: Option<f64>,
    pub minimum_good_request_ratio: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum BenchDefinition {
    Serving {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_source: Option<BenchRequestSource>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_source: Option<BenchSessionSource>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agentic_source: Option<BenchAgenticSource>,
        #[serde(default)]
        seed: u64,
        #[serde(default)]
        server_metrics: bool,
        #[serde(default)]
        artifact_level: BenchArtifactLevel,
        #[serde(default)]
        request_body: BTreeMap<String, JsonValue>,
        #[serde(
            default,
            deserialize_with = "deserialize_nonempty_aggregate_slos",
            skip_serializing_if = "Vec::is_empty"
        )]
        aggregate_slos: Vec<AggregateSlo>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_slo: Option<RequestSlo>,
        #[serde(default)]
        concurrency: Vec<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompts_per_concurrency: Option<u32>,
        #[serde(default)]
        warmup_prompts_per_concurrency: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sessions_per_concurrency: Option<u32>,
        #[serde(default)]
        warmup_sessions_per_concurrency: u32,
        #[serde(default)]
        request_rates: Vec<RequestRate>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_count: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_seconds: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        burstiness: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<BenchCachePolicy>,
        timeout_seconds: u64,
    },
    AdaptiveServing {
        request_source: BenchRequestSource,
        #[serde(default)]
        seed: u64,
        #[serde(default)]
        server_metrics: bool,
        #[serde(default)]
        artifact_level: BenchArtifactLevel,
        #[serde(default)]
        request_body: BTreeMap<String, JsonValue>,
        #[serde(
            default,
            deserialize_with = "deserialize_nonempty_aggregate_slos",
            skip_serializing_if = "Vec::is_empty"
        )]
        aggregate_slos: Vec<AggregateSlo>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_slo: Option<RequestSlo>,
        initial_request_rates: Vec<f64>,
        max_search_steps: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min_rate_resolution: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_count: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_seconds: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        burstiness: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<BenchCachePolicy>,
        timeout_seconds: u64,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchCachePolicy {
    pub start: BenchCacheStart,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchCacheStart {
    #[default]
    Uncontrolled,
    Cold,
    Primed,
}

/// The Bench artifact level: `diagnostic` retains the full raw
/// request/response export; `performance` keeps normalized per-request
/// records and the summary only ([[RFC-0004:C-BENCH-ARTIFACT-LEVEL]]).
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchArtifactLevel {
    Performance,
    #[default]
    Diagnostic,
}

pub(super) fn deserialize_defaulted_bench_definitions<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, BenchDefinition>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut definitions = BTreeMap::<String, serde_value::Value>::deserialize(deserializer)?;
    for definition in definitions.values_mut() {
        let serde_value::Value::Map(fields) = definition else {
            return Err(de::Error::custom("a Bench definition must be a table"));
        };
        // Default only the absent discriminator, then delegate every field and
        // error to the ordinary tagged definition deserializer. Explicit kinds
        // and malformed definitions therefore retain the same one authority.
        fields
            .entry(serde_value::Value::String("kind".to_owned()))
            .or_insert_with(|| serde_value::Value::String("serving".to_owned()));
    }
    definitions
        .into_iter()
        .map(|(id, definition)| {
            let definition = BenchDefinition::deserialize(definition).map_err(|error| {
                let error = error.into_error::<de::value::Error>();
                de::Error::custom(format!("invalid Bench {id:?}: {error}"))
            })?;
            Ok((id, definition))
        })
        .collect()
}

/// One release-qualified agentic trace corpus and replay policy.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchAgenticSource {
    pub dataset: String,
    pub profile: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchSessionSource {
    pub dataset: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    pub max_input_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(default = "default_inter_turn_delay_scale")]
    pub inter_turn_delay_scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_inter_turn_delay_seconds: Option<f64>,
}

const fn default_inter_turn_delay_scale() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum BenchRequestSource {
    Random {
        #[serde(default)]
        prompt: BenchPromptSelection,
        input_tokens: BenchTokenSelector,
        output_tokens: BenchTokenSelector,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefix_sharing: Option<BenchPrefixSharing>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shared_system_content: Option<BenchSharedSystemContent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        corpus: Option<BenchCorpusDeclaration>,
    },
    RandomMixture {
        #[serde(default)]
        prompt: BenchPromptSelection,
        shapes: Vec<BenchRandomShape>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefix_sharing: Option<BenchPrefixSharing>,
    },
    Dataset {
        dataset: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        profile: Option<String>,
        max_input_tokens: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u32>,
    },
    Replay {
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected_sha256: Option<String>,
        #[serde(default)]
        prompt: BenchPromptSelection,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefix_sharing: Option<BenchPrefixSharing>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum BenchPrompt {
    Flat,
    RenderedChat {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chat_template: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        chat_template_kwargs: BTreeMap<String, JsonValue>,
    },
    ServerChat,
}

impl Default for BenchPrompt {
    fn default() -> Self {
        Self::Flat
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BenchPromptSelection {
    declared: Option<BenchPrompt>,
    effective: BenchPrompt,
}

impl BenchPromptSelection {
    pub(crate) fn explicit(prompt: BenchPrompt) -> Self {
        Self {
            declared: Some(prompt.clone()),
            effective: prompt,
        }
    }

    pub(crate) fn declared(&self) -> Option<&BenchPrompt> {
        self.declared.as_ref()
    }

    pub(crate) fn effective(&self) -> &BenchPrompt {
        &self.effective
    }
}

impl Default for BenchPromptSelection {
    fn default() -> Self {
        Self {
            declared: None,
            effective: BenchPrompt::Flat,
        }
    }
}

impl<'de> Deserialize<'de> for BenchPromptSelection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        BenchPrompt::deserialize(deserializer).map(Self::explicit)
    }
}

impl Serialize for BenchPromptSelection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.effective.serialize(serializer)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BenchTokenSelector {
    Fixed(u32),
    InclusiveUniform { min: u32, max: u32 },
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum BenchTokenSelectorWire {
    Fixed(u32),
    Distribution(BenchTokenDistributionWire),
    InclusiveUniform(BenchInclusiveUniformWire),
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum BenchTokenDistributionWire {
    InclusiveUniform { min: u32, max: u32 },
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BenchInclusiveUniformWire {
    min: u32,
    max: u32,
}

impl<'de> Deserialize<'de> for BenchTokenSelector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match BenchTokenSelectorWire::deserialize(deserializer)? {
            BenchTokenSelectorWire::Fixed(value) => Ok(Self::Fixed(value)),
            BenchTokenSelectorWire::Distribution(
                BenchTokenDistributionWire::InclusiveUniform { min, max },
            ) => Ok(Self::InclusiveUniform { min, max }),
            BenchTokenSelectorWire::InclusiveUniform(BenchInclusiveUniformWire { min, max }) => {
                Ok(Self::InclusiveUniform { min, max })
            }
        }
    }
}

impl Serialize for BenchTokenSelector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let wire = match self {
            Self::Fixed(value) => BenchTokenSelectorWire::Fixed(*value),
            Self::InclusiveUniform { min, max } => {
                BenchTokenSelectorWire::Distribution(BenchTokenDistributionWire::InclusiveUniform {
                    min: *min,
                    max: *max,
                })
            }
        };
        wire.serialize(serializer)
    }
}

impl BenchTokenSelector {
    pub(crate) const fn minimum(&self) -> u32 {
        match self {
            Self::Fixed(value) => *value,
            Self::InclusiveUniform { min, .. } => *min,
        }
    }

    const fn tpot_applicability(&self) -> BenchTpotApplicability {
        match self {
            Self::Fixed(value) => BenchTpotApplicability::from_output_tokens(*value),
            Self::InclusiveUniform { min, .. } if *min >= 2 => BenchTpotApplicability::Applicable,
            Self::InclusiveUniform { .. } => BenchTpotApplicability::Inapplicable,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum BenchPrefixSharing {
    Tokens { shared_prefix_tokens: u32 },
    Ratio { shared_prefix_ratio: f64 },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum BenchSharedSystemContent {
    Tokens { tokens: u32 },
    Ratio { ratio: f64 },
}

/// One operator-supplied text corpus binding on the random request source
/// ([[RFC-0004:C-BENCH-REQUEST-SOURCES]]).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchCorpusDeclaration {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchRandomShape {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub weight: u32,
}

impl BenchRequestSource {
    pub(crate) fn tpot_applicability(&self) -> BenchTpotApplicability {
        match self {
            Self::Random { output_tokens, .. } => output_tokens.tpot_applicability(),
            Self::RandomMixture { shapes, .. } => shapes
                .first()
                .map_or(BenchTpotApplicability::Inapplicable, |shape| {
                    BenchTpotApplicability::from_output_tokens(shape.output_tokens)
                }),
            Self::Dataset { output_tokens, .. } => output_tokens.map_or(
                BenchTpotApplicability::Applicable,
                BenchTpotApplicability::from_output_tokens,
            ),
            // Replay entry output targets live in the population file, so the
            // definition-level answer is optimistic; planning refines it from
            // the observed file entries.
            Self::Replay { .. } => BenchTpotApplicability::Applicable,
        }
    }
}

impl BenchSessionSource {
    pub(crate) fn tpot_applicability(&self) -> BenchTpotApplicability {
        self.output_tokens.map_or(
            BenchTpotApplicability::Applicable,
            BenchTpotApplicability::from_output_tokens,
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchTpotApplicability {
    Applicable,
    Inapplicable,
}

impl BenchTpotApplicability {
    pub(crate) const fn from_output_tokens(output_tokens: u32) -> Self {
        if output_tokens >= 2 {
            Self::Applicable
        } else {
            Self::Inapplicable
        }
    }

    pub(crate) const fn is_applicable(self) -> bool {
        matches!(self, Self::Applicable)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RequestRate {
    Finite(f64),
    Unbounded,
}

impl RequestRate {
    pub(crate) const fn finite(&self) -> Option<f64> {
        match self {
            Self::Finite(value) => Some(*value),
            Self::Unbounded => None,
        }
    }
}

impl Serialize for RequestRate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Finite(value) => serializer.serialize_f64(*value),
            Self::Unbounded => serializer.serialize_str("inf"),
        }
    }
}

impl<'de> Deserialize<'de> for RequestRate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RequestRateVisitor;

        impl Visitor<'_> for RequestRateVisitor {
            type Value = RequestRate;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a positive request rate or the string \"inf\"")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequestRate::Finite(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequestRate::Finite(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequestRate::Finite(value as f64))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value {
                    "inf" | "unbounded" => Ok(RequestRate::Unbounded),
                    _ => Err(E::custom("request rate string must be \"inf\"")),
                }
            }
        }

        deserializer.deserialize_any(RequestRateVisitor)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkloadSuiteDefinition {
    #[serde(default)]
    pub evals: Vec<String>,
    #[serde(default)]
    pub gate: Option<String>,
    #[serde(default)]
    pub benches: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecipeDefinition {
    pub server: String,
    pub workload_suite: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ServerCaseDefinition {
    pub readiness_timeout_seconds: Option<u64>,
    pub readiness_attempt_timeout_seconds: Option<u64>,
    pub capture_arm_deadline_seconds: Option<u64>,
    pub capture_control_deadline_seconds: Option<u64>,
    pub capture_finalization_deadline_seconds: Option<u64>,
    pub gateway_backend: Option<String>,
    pub pd_router_backend: Option<String>,
    pub kv_transfer: Option<KvTransferMechanism>,
    pub profiling: Option<bool>,
    /// Case-scoped capture-mechanism declaration; the mechanism is the only
    /// profiler field a case may carry ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[serde(skip_serializing_if = "ProfilerEscapes::is_empty")]
    pub profiler: ProfilerEscapes,
    pub parallelism: Parallelism,
    pub settings: BTreeMap<String, JsonValue>,
    pub roles: BTreeMap<String, ServeRoleOverride>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicitly_declared_aggregate_slos_must_be_nonempty()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = toml::from_str::<WorkspaceConfig>(
            r#"
schema_version = 2
[benches.invalid]
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
aggregate_slos = []
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        );
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "an explicitly empty aggregate_slos declaration must be rejected",
            )
            .into());
        };

        assert!(error.to_string().contains("non-empty"), "{error}");
        Ok(())
    }

    #[test]
    fn ordinary_serving_authoring_resolves_to_the_explicit_canonical_definition()
    -> Result<(), Box<dyn std::error::Error>> {
        let explicit = toml::from_str::<WorkspaceConfig>(
            r#"
schema_version = 2
[benches.ordinary]
kind = "serving"
request_source = { kind = "random", prompt = { kind = "flat" }, input_tokens = { kind = "inclusive_uniform", min = 64, max = 128 }, output_tokens = 32 }
concurrency = [1, 8]
prompts_per_concurrency = 4
timeout_seconds = 900
"#,
        )?;
        let ordinary = toml::from_str::<WorkspaceConfig>(
            r#"
schema_version = 2
[benches.ordinary]
request_source = { kind = "random", input_tokens = { min = 64, max = 128 }, output_tokens = 32 }
concurrency = [1, 8]
prompts_per_concurrency = 4
timeout_seconds = 900
"#,
        )?;

        assert_eq!(
            serde_json::to_value(&ordinary.benches["ordinary"])?,
            serde_json::to_value(&explicit.benches["ordinary"])?,
            "authoring defaults must disappear into one canonical definition"
        );
        Ok(())
    }

    #[test]
    fn openai_smoke_omission_resolves_to_stable_effective_values()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<EvalDefinition>(r#"kind = "openai-smoke""#)?;
        let value = serde_json::to_value(&definition)?;

        assert_eq!(value["kind"], "openai-smoke");
        assert_eq!(value["prompt"], "Hello");
        assert_eq!(value["max_tokens"], 16);
        assert_eq!(value["timeout_seconds"], 60);
        Ok(())
    }

    #[test]
    fn serving_bench_rejects_a_public_chat_template_field() -> Result<(), Box<dyn std::error::Error>>
    {
        let Err(error) = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", prompt = { kind = "server_chat" }, input_tokens = 128, output_tokens = 32 }
chat_template = "templates/qwen.jinja"
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        ) else {
            return Err(std::io::Error::other("chat_template must not be a Bench field").into());
        };
        assert!(error.to_string().contains("unknown field `chat_template`"));
        Ok(())
    }

    #[test]
    fn weighted_random_mixture_prompt_omission_resolves_to_flat()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random_mixture", shapes = [
  { input_tokens = 1024, output_tokens = 128, weight = 7 },
  { input_tokens = 8192, output_tokens = 1024, weight = 3 },
] }
concurrency = [1]
prompts_per_concurrency = 2
timeout_seconds = 60
"#,
        )?;

        let BenchDefinition::Serving { request_source, .. } = definition else {
            return Err(std::io::Error::other("expected a serving Bench").into());
        };
        assert!(matches!(
            request_source,
            Some(BenchRequestSource::RandomMixture {
                prompt,
                ..
            }) if prompt.declared().is_none() && prompt.effective() == &BenchPrompt::Flat
        ));
        Ok(())
    }

    #[test]
    fn legacy_flat_token_shape_is_not_a_second_bench_authority() {
        let result = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
input_tokens = 128
output_tokens = 32
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        );

        assert!(result.is_err_and(|error| error.to_string().contains("input_tokens")));
    }

    #[test]
    fn aggregate_slo_metric_deserializes_directly_into_the_closed_vocabulary()
    -> Result<(), Box<dyn std::error::Error>> {
        let constraint: AggregateSlo = toml::from_str("metric = \"p95_ttft_ms\"\nat_most = 800.0")?;
        let unknown =
            toml::from_str::<AggregateSlo>("metric = \"aiperf_private_latency\"\nat_most = 800.0");

        assert_eq!(constraint.metric.name(), "p95_ttft_ms");
        assert!(unknown.is_err_and(|error| error.to_string().contains("unknown Bench metric")));
        Ok(())
    }

    #[test]
    fn bundled_eval_task_uses_the_named_release_catalog_entry()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = { bundled = "estonia" }
metric = "estonia_pass"
metric_filter = "strict-terminal-answer"
threshold = 0.5
timeout_seconds = 3600
"#,
        )?;

        let EvalDefinition::LmEval { task, .. } = definition else {
            return Err(std::io::Error::other("fixture should be lm-eval").into());
        };
        assert!(matches!(
            task,
            EvalTaskSource::Bundled { bundled } if bundled == "estonia"
        ));
        Ok(())
    }

    #[test]
    fn an_omitted_eval_prompt_resolves_to_flat_without_claiming_a_declaration()
    -> Result<(), Box<dyn std::error::Error>> {
        let omitted: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300
"#,
        )?;
        let EvalDefinition::LmEval { prompt, .. } = omitted else {
            return Err(std::io::Error::other("fixture should be lm-eval").into());
        };
        assert!(prompt.declared().is_none());
        assert_eq!(prompt.effective(), &EvalPrompt::Flat);

        let declared: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
prompt = { kind = "server_chat" }
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300
"#,
        )?;
        let EvalDefinition::LmEval { prompt, .. } = declared else {
            return Err(std::io::Error::other("fixture should be lm-eval").into());
        };
        assert_eq!(prompt.declared(), Some(&EvalPrompt::ServerChat));
        assert_eq!(prompt.effective(), &EvalPrompt::ServerChat);
        Ok(())
    }

    #[test]
    fn inference_request_body_preserves_nested_toml_json_types()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300

[request_body]
temperature = 1.0
logprobs = true
stop_token_ids = [1, 2]

[request_body.chat_template_kwargs]
enable_thinking = false
"#,
        )?;

        let EvalDefinition::LmEval { request_body, .. } = definition else {
            return Err(std::io::Error::other("fixture should be lm-eval").into());
        };
        assert!(matches!(
            request_body.get("temperature"),
            Some(JsonValue::Float(value)) if *value == 1.0
        ));
        assert!(matches!(
            request_body.get("logprobs"),
            Some(JsonValue::Bool(true))
        ));
        assert!(matches!(
            request_body.get("stop_token_ids"),
            Some(JsonValue::Array(values)) if values.len() == 2
        ));
        assert!(matches!(
            request_body.get("chat_template_kwargs"),
            Some(JsonValue::Object(values))
                if values.get("enable_thinking") == Some(&JsonValue::Bool(false))
        ));
        Ok(())
    }

    #[test]
    fn role_escapes_merge_into_common_server_escapes() {
        let common = NsysEscapes {
            executable: Some("nsys".to_owned()),
            launch_options: vec!["--cuda-graph-trace=node".to_owned()],
            start_options: vec!["--nic-metrics=true".to_owned()],
            trace: vec!["cuda".to_owned()],
            sampling: Some("cpu".to_owned()),
            context_switch: None,
            env: BTreeMap::from([
                ("NSYS_SHARED".to_owned(), "common".to_owned()),
                ("NSYS_COMMON_ONLY".to_owned(), "1".to_owned()),
            ]),
        };
        let role = NsysEscapes {
            executable: None,
            launch_options: vec!["--nvtx-domain-include=prefill".to_owned()],
            start_options: Vec::new(),
            trace: vec!["cuda".to_owned(), "nvtx".to_owned()],
            sampling: Some("process-tree".to_owned()),
            context_switch: Some("system-wide".to_owned()),
            env: BTreeMap::from([("NSYS_SHARED".to_owned(), "role".to_owned())]),
        };
        let merged = common.merged_with(&role);
        assert_eq!(merged.executable.as_deref(), Some("nsys"));
        assert_eq!(
            merged.launch_options,
            ["--cuda-graph-trace=node", "--nvtx-domain-include=prefill"]
        );
        assert_eq!(merged.start_options, ["--nic-metrics=true"]);
        assert_eq!(merged.trace, ["cuda", "nvtx"]);
        assert_eq!(merged.sampling.as_deref(), Some("process-tree"));
        assert_eq!(merged.context_switch.as_deref(), Some("system-wide"));
        assert_eq!(
            merged.env,
            BTreeMap::from([
                ("NSYS_COMMON_ONLY".to_owned(), "1".to_owned()),
                ("NSYS_SHARED".to_owned(), "role".to_owned()),
            ])
        );
    }
}
