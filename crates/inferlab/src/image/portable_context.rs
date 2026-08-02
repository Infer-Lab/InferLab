//! Portable-output policy for generated image context text.

use crate::InferlabError;
use crate::workspace::{LaunchBinding, LoadedWorkspace};

/// Portable-output guard ([[RFC-0007:C-IMAGE-BUILD]]): rendered context text
/// must not carry machine-private facts. Records keep exact values locally;
/// artifacts that leave the machine must exclude them by construction.
pub fn guard_portable_text(
    label: &str,
    text: &str,
    workspace: &LoadedWorkspace,
) -> Result<(), InferlabError> {
    let mut forbidden: Vec<(String, String)> = Vec::new();
    forbidden.push((
        "workspace root path".to_owned(),
        workspace.root.display().to_string(),
    ));
    for (id, weight) in &workspace.local.model_weights {
        if let Some(locator) = &weight.locator {
            forbidden.push((format!("model weight locator {id:?}"), locator.clone()));
        }
        for locator in weight.machine_locators.values() {
            forbidden.push((format!("model weight locator {id:?}"), locator.clone()));
        }
    }
    for (id, machine) in &workspace.local.machines {
        if machine.host != "127.0.0.1" && machine.host != "localhost" {
            forbidden.push((format!("machine host {id:?}"), machine.host.clone()));
        }
        if let LaunchBinding::Ssh { target } = &machine.launch {
            forbidden.push((format!("machine SSH target {id:?}"), target.clone()));
        }
    }
    for (name, value) in forbidden {
        if !value.is_empty() && text.contains(&value) {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "{label} would carry machine-private fact ({name}); \
                     portable image outputs must exclude local facts by construction"
                ),
            });
        }
    }
    if text.contains("/home/") {
        return Err(InferlabError::ImageBuild {
            message: format!(
                "{label} would carry a host home-directory path; \
                 portable image outputs must exclude local facts by construction"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::{
        AdapterBinding, LocalBindings, MachineBinding, ModelWeightBinding, WorkspaceConfig,
        WorkspaceSnapshot,
    };
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn guarded_workspace() -> LoadedWorkspace {
        let mut model_weights = BTreeMap::new();
        model_weights.insert(
            "dsv4".to_owned(),
            ModelWeightBinding {
                locator: Some("/secret/weights/dsv4".to_owned()),
                machine_locators: BTreeMap::new(),
            },
        );
        let mut machines = BTreeMap::new();
        machines.insert(
            "remote".to_owned(),
            MachineBinding {
                host: "gpu-node-7".to_owned(),
                devices: vec![0],
                ports: vec![8000],
                workspace: None,
                cache_root: None,
                container: None,
                launch: LaunchBinding::Ssh {
                    target: "operator@gpu-node-7".to_owned(),
                },
            },
        );
        LoadedWorkspace {
            root: PathBuf::from("/work/dsv4-workspace"),
            config: WorkspaceConfig {
                external_images: BTreeMap::new(),
                schema_version: 2,
                models: BTreeMap::new(),
                stacks: BTreeMap::new(),
                servers: BTreeMap::new(),
                evals: BTreeMap::new(),
                benches: BTreeMap::new(),
                workload_suites: BTreeMap::new(),
                recipes: BTreeMap::new(),
                images: BTreeMap::new(),
            },
            local: LocalBindings {
                adapter: AdapterBinding::default(),
                default_placement: Some("local".to_owned()),
                model_weights,
                machines,
                placements: BTreeMap::new(),
                builders: BTreeMap::new(),
            },
            snapshot: WorkspaceSnapshot {
                revision: "0000".to_owned(),
                dirty: false,
                source_digest: "0000".to_owned(),
                source_exclusions: Vec::new(),
                revision_reproducible: true,
                pixi_manifest_sha256: "0000".to_owned(),
                pixi_lock_sha256: "0000".to_owned(),
            },
        }
    }

    #[test]
    fn guard_rejects_machine_private_facts_in_portable_text() {
        let workspace = guarded_workspace();
        for leak in [
            "FROM base\nCOPY /secret/weights/dsv4 /weights\n",
            "LABEL host=gpu-node-7\n",
            "ENV TARGET=operator@gpu-node-7\n",
            "WORKDIR /work/dsv4-workspace\n",
            "ENV STRAY=/home/operator/data\n",
        ] {
            assert!(
                guard_portable_text("test text", leak, &workspace).is_err(),
                "guard must reject {leak:?}"
            );
        }
    }

    #[test]
    fn guard_accepts_portable_text() -> Result<(), InferlabError> {
        let workspace = guarded_workspace();
        guard_portable_text(
            "test text",
            "FROM base@sha256:0000\nRUN micromamba create -y -p /opt/inferlab-env\n",
            &workspace,
        )
    }
}
