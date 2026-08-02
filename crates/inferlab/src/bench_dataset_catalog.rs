use crate::InferlabError;
use serde::Deserialize;
use std::collections::BTreeMap;

const RELEASE_CATALOG: &str = include_str!("../resources/bench-datasets.toml");
const CATALOG_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleaseCatalog {
    schema_version: u32,
    datasets: BTreeMap<String, DatasetEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetEntry {
    #[serde(default)]
    default_source: Option<String>,
    license: String,
    materialization_identity: String,
    #[serde(default)]
    linear_session_materialization_identity: Option<String>,
    provides_output_targets: bool,
    sources: BTreeMap<String, SourceEntry>,
    #[serde(default)]
    profiles: BTreeMap<String, ProfileEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEntry {
    upstream_identity: String,
    url: String,
    sha256: String,
    source_format: String,
    #[serde(default)]
    aiperf_format: Option<String>,
    #[serde(default)]
    aiperf_format_prefix: Option<String>,
    #[serde(default)]
    configuration: Option<String>,
    #[serde(default)]
    split: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileEntry {
    source: String,
    #[serde(default)]
    filter: Option<ProfileFilter>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProfileFilter {
    pub field: String,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedDatasetCatalogEntry {
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
    pub filter: Option<ProfileFilter>,
    pub license: String,
    pub materialization_identity: String,
    pub provides_output_targets: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedSessionDatasetCatalogEntry {
    pub dataset: String,
    pub profile: Option<String>,
    pub source: String,
    pub upstream_identity: String,
    pub url: String,
    pub sha256: String,
    pub source_format: String,
    pub configuration: Option<String>,
    pub split: Option<String>,
    pub filter: Option<ProfileFilter>,
    pub license: String,
    pub materialization_identity: String,
    pub provides_output_targets: bool,
}

pub(crate) fn resolve(
    dataset: &str,
    profile: Option<&str>,
) -> Result<ResolvedDatasetCatalogEntry, InferlabError> {
    let catalog = parse_catalog()?;
    let entry = catalog
        .datasets
        .get(dataset)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("dataset {dataset:?} is not in this InferLab release catalog"),
        })?;

    let (source_name, filter) = selected_source(dataset, profile, entry)?;
    let source = entry.sources.get(source_name).ok_or_else(|| {
        InferlabError::InvalidConfig {
            message: format!(
                "release dataset catalog entry {dataset:?} references missing source {source_name:?}"
            ),
        }
    })?;
    let aiperf_format = match (&source.aiperf_format, &source.aiperf_format_prefix, &filter) {
        (Some(format), None, _) => format.clone(),
        (None, Some(prefix), Some(filter)) => format!("{prefix}{}", filter.value),
        _ => {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "release dataset catalog source {dataset:?}.{source_name:?} has no unambiguous AIPerf format"
                ),
            });
        }
    };

    Ok(ResolvedDatasetCatalogEntry {
        dataset: dataset.to_owned(),
        profile: profile.map(str::to_owned),
        source: source_name.to_owned(),
        upstream_identity: source.upstream_identity.clone(),
        url: source.url.clone(),
        sha256: source.sha256.clone(),
        source_format: source.source_format.clone(),
        aiperf_format,
        configuration: source.configuration.clone(),
        split: source.split.clone(),
        filter,
        license: entry.license.clone(),
        materialization_identity: entry.materialization_identity.clone(),
        provides_output_targets: entry.provides_output_targets,
    })
}

pub(crate) fn resolve_session(
    dataset: &str,
    profile: Option<&str>,
) -> Result<ResolvedSessionDatasetCatalogEntry, InferlabError> {
    let catalog = parse_catalog()?;
    let entry = catalog
        .datasets
        .get(dataset)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("dataset {dataset:?} is not in this InferLab release catalog"),
        })?;
    let materialization_identity = entry
        .linear_session_materialization_identity
        .clone()
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!(
                "dataset {dataset:?} is not qualified for linear sessions in this InferLab release catalog"
            ),
        })?;
    let (source_name, filter) = selected_source(dataset, profile, entry)?;
    let source = entry.sources.get(source_name).ok_or_else(|| {
        InferlabError::InvalidConfig {
            message: format!(
                "release dataset catalog entry {dataset:?} references missing source {source_name:?}"
            ),
        }
    })?;
    Ok(ResolvedSessionDatasetCatalogEntry {
        dataset: dataset.to_owned(),
        profile: profile.map(str::to_owned),
        source: source_name.to_owned(),
        upstream_identity: source.upstream_identity.clone(),
        url: source.url.clone(),
        sha256: source.sha256.clone(),
        source_format: source.source_format.clone(),
        configuration: source.configuration.clone(),
        split: source.split.clone(),
        filter,
        license: entry.license.clone(),
        materialization_identity,
        provides_output_targets: entry.provides_output_targets,
    })
}

fn selected_source<'a>(
    dataset: &str,
    profile: Option<&str>,
    entry: &'a DatasetEntry,
) -> Result<(&'a str, Option<ProfileFilter>), InferlabError> {
    match profile {
        Some(profile) => {
            let profile_entry = entry.profiles.get(profile).ok_or_else(|| {
                InferlabError::InvalidConfig {
                    message: format!(
                        "dataset {dataset:?} profile {profile:?} is not in this InferLab release catalog"
                    ),
                }
            })?;
            Ok((profile_entry.source.as_str(), profile_entry.filter.clone()))
        }
        None => Ok((
            entry
                .default_source
                .as_deref()
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "dataset {dataset:?} requires a profile from this InferLab release catalog"
                    ),
                })?,
            None,
        )),
    }
}

fn parse_catalog() -> Result<ReleaseCatalog, InferlabError> {
    let catalog = toml::from_str::<ReleaseCatalog>(RELEASE_CATALOG).map_err(|error| {
        InferlabError::InvalidConfig {
            message: format!("release dataset catalog is invalid: {error}"),
        }
    })?;
    if catalog.schema_version != CATALOG_SCHEMA_VERSION {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "release dataset catalog schema version {} is unsupported; expected {CATALOG_SCHEMA_VERSION}",
                catalog.schema_version
            ),
        });
    }
    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_profiles_are_data_driven_catalog_entries() -> Result<(), Box<dyn std::error::Error>> {
        let resolved = resolve("speed_bench", Some("throughput_8k_mixed"))?;

        assert_eq!(resolved.dataset, "speed_bench");
        assert_eq!(resolved.source, "throughput_8k");
        assert_eq!(resolved.configuration.as_deref(), Some("throughput_8k"));
        assert_eq!(resolved.split.as_deref(), Some("test"));
        assert_eq!(
            resolved.filter,
            Some(ProfileFilter {
                field: "category".to_owned(),
                value: "mixed".to_owned(),
            })
        );
        assert!(!resolved.provides_output_targets);
        Ok(())
    }

    #[test]
    fn catalog_rejects_unknown_profiles_without_a_rust_profile_enum()
    -> Result<(), Box<dyn std::error::Error>> {
        let error = resolve("speed_bench", Some("not_a_profile"))
            .err()
            .ok_or("unknown profile unexpectedly resolved")?;
        assert!(error.to_string().contains("not_a_profile"), "{error}");
        Ok(())
    }
}
