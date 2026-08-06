//! Eval definition and workspace task-source validation.

use super::{
    invalid, require_nonempty, require_optional_positive, require_positive, validate_request_body,
};
use crate::InferlabError;
use crate::workspace::definitions::{EvalDefinition, EvalTaskSource};
use crate::workspace::source::{is_safe_relative, reject_symlink_components};
use std::ffi::OsStr;
use std::path::Path;

pub(crate) fn validate_eval(id: &str, definition: &EvalDefinition) -> Result<(), InferlabError> {
    match definition {
        EvalDefinition::OpenAiSmoke {
            prompt,
            max_tokens,
            timeout_seconds,
        } => {
            require_nonempty("eval prompt", id, prompt)?;
            require_positive("max_tokens", id, u64::from(*max_tokens))?;
            require_positive("timeout_seconds", id, *timeout_seconds)
        }
        EvalDefinition::LmEval {
            task,
            request_body,
            limit,
            seed,
            trials,
            max_tokens,
            concurrency,
            metric,
            metric_filter,
            threshold,
            timeout_seconds,
            ..
        } => {
            match task {
                EvalTaskSource::BuiltIn(task) => require_nonempty("lm-eval task", id, task)?,
                EvalTaskSource::Bundled { bundled } => {
                    require_nonempty("lm-eval bundled task", id, bundled)?
                }
                EvalTaskSource::WorkspaceYaml { .. } => {}
            }
            validate_request_body("eval", id, request_body, &["seed"])?;
            require_nonempty("lm-eval metric", id, metric)?;
            if let Some(metric_filter) = metric_filter {
                require_nonempty("lm-eval metric_filter", id, metric_filter)?;
            }
            require_optional_positive("limit", id, limit.map(u64::from))?;
            require_positive("trials", id, u64::from(*trials))?;
            let base_seed = seed.unwrap_or(1234);
            if base_seed.checked_add(u64::from(*trials - 1)).is_none() {
                return invalid(format!(
                    "eval {id:?} seed schedule exceeds the supported unsigned integer range"
                ));
            }
            require_optional_positive("max_tokens", id, max_tokens.map(u64::from))?;
            require_optional_positive("concurrency", id, concurrency.map(u64::from))?;
            if !threshold.is_finite() {
                return invalid(format!("eval {id:?} threshold must be finite"));
            }
            if *trials > 1 && !(0.0..=1.0).contains(threshold) {
                return invalid(format!(
                    "eval {id:?} threshold must be between zero and one for repeated trials"
                ));
            }
            require_positive("timeout_seconds", id, *timeout_seconds)
        }
    }
}

pub(crate) fn validate_eval_task_source(
    root: &Path,
    id: &str,
    definition: &EvalDefinition,
) -> Result<(), InferlabError> {
    let EvalDefinition::LmEval { task, .. } = definition else {
        return Ok(());
    };
    let EvalTaskSource::WorkspaceYaml { yaml } = task else {
        return Ok(());
    };
    if !is_safe_relative(yaml) {
        return invalid(format!(
            "lm-eval {id:?} task YAML {} must be workspace-relative without parent traversal",
            yaml.display()
        ));
    }
    if !matches!(
        yaml.extension(),
        Some(extension) if extension == OsStr::new("yaml") || extension == OsStr::new("yml")
    ) {
        return invalid(format!(
            "lm-eval {id:?} task YAML {} must use a .yaml or .yml extension supported by the pinned lm-eval runtime",
            yaml.display()
        ));
    }
    reject_symlink_components(root, id, yaml)?;
    let path = root.join(yaml);
    if !path.is_file() {
        return invalid(format!(
            "lm-eval {id:?} task YAML {} is not a regular workspace file",
            yaml.display()
        ));
    }
    Ok(())
}
