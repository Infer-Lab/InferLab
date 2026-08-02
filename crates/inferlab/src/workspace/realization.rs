//! Validation of the Pixi realization selected by portable stack intent.

use super::definitions::WorkspaceConfig;
use super::invalid;
use crate::InferlabError;
use std::fs;
use std::path::Path;

pub(super) fn validate_pixi(root: &Path, config: &WorkspaceConfig) -> Result<(), InferlabError> {
    let manifest_path = root.join("pixi.toml");
    let manifest_text =
        fs::read_to_string(&manifest_path).map_err(|source| InferlabError::Read {
            path: manifest_path.clone(),
            source,
        })?;
    let manifest: toml::Value =
        toml::from_str(&manifest_text).map_err(|source| InferlabError::ParseToml {
            path: manifest_path,
            source,
        })?;
    let declared_environments = manifest.get("environments").and_then(toml::Value::as_table);
    for (id, stack) in &config.stacks {
        let exists = stack.pixi_environment == "default"
            || declared_environments
                .is_some_and(|environments| environments.contains_key(&stack.pixi_environment));
        if !exists {
            return invalid(format!(
                "stack {id:?} references unknown Pixi environment {:?}",
                stack.pixi_environment
            ));
        }
        let package = format!("inferlab-integration-{}", stack.integration);
        if !pixi_environment_selects_dependency(&manifest, &stack.pixi_environment, &package) {
            return invalid(format!(
                "stack {id:?} integration {:?} is not selected by Pixi environment {:?} as package {package:?}",
                stack.integration, stack.pixi_environment,
            ));
        }
    }

    // A selected integration absent from the workspace's committed dependency
    // set can never lower, since the adapter packages come from that set now
    // ([[RFC-0006:C-INTEGRATIONS]]): reject the external image at load naming
    // the missing package. Any pypi-dependencies declaration in any feature or
    // workspace table counts — an exact pin or a path source both lower.
    for (id, external) in &config.external_images {
        let package = format!("inferlab-integration-{}", external.integration);
        if !manifest_declares_pypi_dependency(&manifest, &package) {
            return invalid(format!(
                "external image {id:?} claims integration {:?}, but the workspace's committed \
                 dependency set declares no package {package:?}",
                external.integration
            ));
        }
    }

    let lock_path = root.join("pixi.lock");
    let lock_text = fs::read_to_string(&lock_path).map_err(|source| InferlabError::Read {
        path: lock_path.clone(),
        source,
    })?;
    let lock: yaml_serde::Value =
        yaml_serde::from_str(&lock_text).map_err(|source| InferlabError::ParseYaml {
            path: lock_path,
            source,
        })?;
    let locked_environments = lock
        .get("environments")
        .and_then(yaml_serde::Value::as_mapping);
    for (id, stack) in &config.stacks {
        let key = yaml_serde::Value::String(stack.pixi_environment.clone());
        if !locked_environments.is_some_and(|environments| environments.contains_key(&key)) {
            return invalid(format!(
                "stack {id:?} Pixi environment {:?} is absent from pixi.lock",
                stack.pixi_environment
            ));
        }
    }
    Ok(())
}

pub(super) fn pixi_environment_selects_dependency(
    manifest: &toml::Value,
    environment: &str,
    package: &str,
) -> bool {
    let Some(root) = manifest.as_table() else {
        return false;
    };
    if dependency_tables_contain(root, package) {
        return true;
    }
    let Some(environment_value) = root
        .get("environments")
        .and_then(toml::Value::as_table)
        .and_then(|environments| environments.get(environment))
    else {
        return false;
    };
    let features: Vec<&str> = match environment_value {
        toml::Value::Array(features) => features.iter().filter_map(toml::Value::as_str).collect(),
        toml::Value::Table(environment) => environment
            .get("features")
            .and_then(toml::Value::as_array)
            .map(|features| features.iter().filter_map(toml::Value::as_str).collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    let feature_tables = root.get("feature").and_then(toml::Value::as_table);
    features.iter().any(|feature| {
        feature_tables
            .and_then(|tables| tables.get(*feature))
            .and_then(toml::Value::as_table)
            .is_some_and(|table| dependency_tables_contain(table, package))
    })
}

pub(super) fn dependency_tables_contain(table: &toml::Table, package: &str) -> bool {
    [
        "dependencies",
        "pypi-dependencies",
        "host-dependencies",
        "build-dependencies",
    ]
    .iter()
    .any(|key| {
        table
            .get(*key)
            .and_then(toml::Value::as_table)
            .is_some_and(|dependencies| dependencies.contains_key(package))
    })
}

/// Whether the manifest declares `package` as a pypi dependency in any table,
/// scanning the whole tree so a workspace-table, feature, or nested
/// declaration all count ([[RFC-0006:C-INTEGRATIONS]]).
pub(super) fn manifest_declares_pypi_dependency(manifest: &toml::Value, package: &str) -> bool {
    let Some(table) = manifest.as_table() else {
        return false;
    };
    if table
        .get("pypi-dependencies")
        .and_then(toml::Value::as_table)
        .is_some_and(|dependencies| dependencies.contains_key(package))
    {
        return true;
    }
    table
        .values()
        .any(|child| manifest_declares_pypi_dependency(child, package))
}
