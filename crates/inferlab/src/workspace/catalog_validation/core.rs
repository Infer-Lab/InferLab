//! Stack, model, and workspace-owned script validation.

use super::{invalid, require_id, require_nonempty};
use crate::InferlabError;
use crate::workspace::definitions::WorkspaceConfig;
use crate::workspace::source::{is_safe_relative, reject_symlink_components};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub(super) fn validate(root: &Path, config: &WorkspaceConfig) -> Result<(), InferlabError> {
    for (id, stack) in &config.stacks {
        require_id("stack", id)?;
        require_nonempty("integration", id, &stack.integration)?;
        require_nonempty("Pixi environment", id, &stack.pixi_environment)?;
        for path in &stack.source_paths {
            if !is_safe_relative(path) {
                return invalid(format!(
                    "stack {id:?} source path {} must be workspace-relative without parent traversal",
                    path.display()
                ));
            }
            reject_symlink_components(root, id, path)?;
            if !root.join(path).exists() {
                return invalid(format!(
                    "stack {id:?} source path {} does not exist",
                    path.display()
                ));
            }
        }
        let mut seen_checks = BTreeSet::new();
        for check in &stack.checks {
            require_id("stack check", &check.id)?;
            if !seen_checks.insert(&check.id) {
                return invalid(format!(
                    "stack {id:?} declares duplicate check id {:?}",
                    check.id
                ));
            }
            validate_environment_script(root, id, "check", &check.id, &check.script)?;
        }
        let mut seen_postprocess = BTreeSet::new();
        for step in &stack.image_postprocess {
            require_id("stack postprocess step", &step.id)?;
            if !seen_postprocess.insert(&step.id) {
                return invalid(format!(
                    "stack {id:?} declares duplicate image postprocess id {:?}",
                    step.id
                ));
            }
            validate_environment_script(
                root,
                id,
                "image postprocess step",
                &step.id,
                &step.script,
            )?;
        }
    }
    for (id, model) in &config.models {
        require_id("model", id)?;
        require_nonempty("served model name", id, &model.served_name)?;
    }
    Ok(())
}

fn validate_environment_script(
    root: &Path,
    environment: &str,
    label: &str,
    id: &str,
    script: &Path,
) -> Result<(), InferlabError> {
    if !is_safe_relative(script) {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} must be workspace-relative \
             without parent traversal",
            script.display()
        ));
    }
    let target = root.join(script);
    if !target.is_file() {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} does not exist",
            script.display()
        ));
    }
    // A lexically relative path can still resolve outside the workspace
    // through a symlink; scripts are workspace content, so the canonical
    // target must stay inside the (already canonical) root.
    let canonical = fs::canonicalize(&target).map_err(|source| InferlabError::Read {
        path: target,
        source,
    })?;
    if !canonical.starts_with(root) {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} resolves outside the workspace",
            script.display()
        ));
    }
    Ok(())
}
