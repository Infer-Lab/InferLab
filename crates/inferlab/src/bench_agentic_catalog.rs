use crate::InferlabError;
use serde::Deserialize;
use std::collections::BTreeMap;

const RELEASE_CATALOG: &str = include_str!("../resources/bench-agentic-sources.toml");
const CATALOG_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleaseCatalog {
    schema_version: u32,
    qualification: Qualification,
    datasets: BTreeMap<String, DatasetEntry>,
    profiles: BTreeMap<String, ProfileEntry>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Qualification {
    pub inferencex_repository: String,
    pub inferencex_revision: String,
    pub inferencex_reference: String,
    pub aiperf_revision: String,
    pub aiperf_version: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatasetEntry {
    pub repository: String,
    pub revision: String,
    pub filename: String,
    pub sha256: String,
    pub trace_count: u32,
    pub approximate_bytes: u64,
    pub license: String,
    pub source_format: String,
    pub aiperf_loader: String,
    pub materialization_identity: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProfileEntry {
    pub scenario: String,
    pub concurrency_semantics: String,
    pub replay_semantics: String,
    pub cache_bust: String,
    pub trajectory_start_min: f64,
    pub trajectory_start_max: f64,
    pub global_idle_gap_cap_seconds: f64,
    pub trace_idle_gap_cap_seconds: f64,
    pub cache_warmup_requests_per_lane: u64,
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
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedAgenticCatalogEntry {
    pub dataset: String,
    pub profile: String,
    pub source: DatasetEntry,
    pub policy: ProfileEntry,
    pub qualification: Qualification,
}

pub(crate) fn resolve(
    dataset: &str,
    profile: &str,
) -> Result<ResolvedAgenticCatalogEntry, InferlabError> {
    let catalog = parse_catalog()?;
    let source =
        catalog
            .datasets
            .get(dataset)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "agentic dataset {dataset:?} is not in this InferLab release catalog"
                ),
            })?;
    let policy =
        catalog
            .profiles
            .get(profile)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "agentic profile {profile:?} is not in this InferLab release catalog"
                ),
            })?;
    Ok(ResolvedAgenticCatalogEntry {
        dataset: dataset.to_owned(),
        profile: profile.to_owned(),
        source,
        policy,
        qualification: catalog.qualification,
    })
}

fn parse_catalog() -> Result<ReleaseCatalog, InferlabError> {
    let catalog = toml::from_str::<ReleaseCatalog>(RELEASE_CATALOG).map_err(|error| {
        InferlabError::InvalidConfig {
            message: format!("release agentic catalog is invalid: {error}"),
        }
    })?;
    if catalog.schema_version != CATALOG_SCHEMA_VERSION {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "release agentic catalog schema version {} is unsupported; expected {CATALOG_SCHEMA_VERSION}",
                catalog.schema_version
            ),
        });
    }
    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::resolve;

    #[test]
    fn semianalysis_profiles_are_release_data() -> Result<(), Box<dyn std::error::Error>> {
        let full = resolve("semianalysis_agentx_062126", "inferencex")?;
        let limited = resolve("semianalysis_agentx_062126_256k", "inferencex")?;

        assert_eq!(full.source.trace_count, 393);
        assert_eq!(limited.source.revision.len(), 40);
        assert_eq!(limited.policy.minimum_duration_seconds, 900);
        assert_eq!(limited.policy.default_duration_seconds, 1800);
        assert_eq!(limited.policy.trajectory_start_min, 0.25);
        assert_eq!(limited.policy.trajectory_start_max, 0.75);
        assert_eq!(limited.policy.dataset_configuration_timeout_seconds, 1800);
        assert_eq!(
            limited.policy.service_profile_configuration_timeout_seconds,
            1800
        );
        assert_eq!(limited.qualification.aiperf_version, "0.12.0");
        assert_eq!(
            limited.qualification.inferencex_repository,
            "SemiAnalysisAI/InferenceX"
        );
        assert_eq!(
            limited.source.source_format,
            "weka_kv_cache_tester_agentic_trace_v7_jsonl"
        );
        Ok(())
    }
}
