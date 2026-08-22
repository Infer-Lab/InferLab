//! Eval definition and workspace task-source validation.

use super::{
    invalid, require_nonempty, require_optional_positive, require_positive, validate_request_body,
};
use crate::InferlabError;
use crate::workspace::definitions::{EvalDefinition, EvalPrompt, EvalTaskSource};
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
            prompt,
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
            // Server-side template controls belong to the server only when the
            // resolved authority assigns rendering to it.
            let reserved: &[&str] = match prompt.effective() {
                EvalPrompt::Flat => &["seed", "chat_template", "chat_template_kwargs"],
                EvalPrompt::ServerChat => &["seed"],
            };
            validate_request_body("eval", id, request_body, reserved)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_eval_rejects_a_server_owned_chat_template_control()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300

[request_body.chat_template_kwargs]
enable_thinking = true
"#,
        )?;
        let Err(error) = validate_eval("gsm8k", &definition) else {
            return Err(std::io::Error::other(
                "a flat Eval must reject a server-owned template control",
            )
            .into());
        };
        assert!(
            error.to_string().contains("chat_template_kwargs"),
            "{error}"
        );

        let server_chat: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
prompt = { kind = "server_chat" }
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300

[request_body.chat_template_kwargs]
enable_thinking = true
"#,
        )?;
        validate_eval("gsm8k", &server_chat)?;
        Ok(())
    }

    #[test]
    fn inference_request_body_rejects_owned_members_and_toml_dates()
    -> Result<(), Box<dyn std::error::Error>> {
        let reserved: EvalDefinition = toml::from_str(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300
request_body = { messages = [] }
"#,
        )?;
        let Err(error) = validate_eval("gsm8k", &reserved) else {
            return Err(std::io::Error::other(
                "messages should be owned by the measurement runtime",
            )
            .into());
        };
        let error = error.to_string();
        assert!(error.contains("request_body.messages"), "{error}");

        let Err(date) = toml::from_str::<EvalDefinition>(
            r#"
kind = "lm-eval"
task = "gsm8k"
metric = "exact_match"
threshold = 0.9
timeout_seconds = 300
request_body = { vendor_date = 2026-07-15 }
"#,
        ) else {
            return Err(
                std::io::Error::other("TOML dates should have no exact JSON projection").into(),
            );
        };
        let date = date.to_string();
        assert!(date.contains("JSON-compatible value"), "{date}");
        Ok(())
    }
}
