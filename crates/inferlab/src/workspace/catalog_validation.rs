//! Domain validation for the portable workspace catalog.

mod bench;
mod core;
mod eval;
mod image;
mod serve;

use super::definitions::{JsonValue, WorkspaceConfig};
use super::invalid;
use crate::InferlabError;
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) use bench::validate_bench;
pub(crate) use eval::{validate_eval, validate_eval_task_source};

pub(super) fn validate_workspace(
    root: &Path,
    config: &WorkspaceConfig,
) -> Result<(), InferlabError> {
    if config.schema_version != 2 {
        return invalid(format!(
            "unsupported workspace schema version {}; expected 2",
            config.schema_version
        ));
    }
    core::validate(root, config)?;
    serve::validate(config)?;
    for (id, bench) in &config.benches {
        require_id("bench", id)?;
        validate_bench(id, bench)?;
    }
    for (id, eval) in &config.evals {
        require_id("eval", id)?;
        validate_eval(id, eval)?;
        validate_eval_task_source(root, id, eval)?;
    }

    for (id, suite) in &config.workload_suites {
        require_id("workload suite", id)?;
        if suite.evals.is_empty() && suite.benches.is_empty() {
            return invalid(format!(
                "workload suite {id:?} must select at least one measurement"
            ));
        }
        for eval in &suite.evals {
            require_reference("eval", eval, &config.evals)?;
        }
        for bench in &suite.benches {
            require_reference("bench", bench, &config.benches)?;
        }
        if let Some(gate) = &suite.gate {
            require_reference("eval gate", gate, &config.evals)?;
            if !suite.evals.contains(gate) {
                return invalid(format!(
                    "workload suite {id:?} gate {gate:?} is not in its eval list"
                ));
            }
        }
    }

    for (id, recipe) in &config.recipes {
        require_id("recipe", id)?;
        require_reference("server", &recipe.server, &config.servers)?;
        require_reference(
            "workload suite",
            &recipe.workload_suite,
            &config.workload_suites,
        )?;
    }

    image::validate(config)?;
    Ok(())
}

fn validate_request_body(
    kind: &str,
    id: &str,
    request_body: &BTreeMap<String, JsonValue>,
    additional_reserved: &[&str],
) -> Result<(), InferlabError> {
    const RESERVED: [&str; 8] = [
        "model",
        "prompt",
        "messages",
        "stream",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "stop",
    ];
    if let Some(member) = RESERVED
        .iter()
        .chain(additional_reserved)
        .find(|member| request_body.contains_key(**member))
    {
        return invalid(format!(
            "{kind} {id:?} request_body.{member} conflicts with a measurement-runtime-owned request member"
        ));
    }
    for (member, value) in request_body {
        validate_request_body_value(kind, id, &format!("request_body.{member}"), value)?;
    }
    Ok(())
}

fn validate_request_body_value(
    kind: &str,
    id: &str,
    path: &str,
    value: &JsonValue,
) -> Result<(), InferlabError> {
    match value {
        JsonValue::Float(value) if !value.is_finite() => {
            invalid(format!("{kind} {id:?} {path} must be a finite JSON number"))
        }
        JsonValue::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                validate_request_body_value(kind, id, &format!("{path}[{index}]"), value)?;
            }
            Ok(())
        }
        JsonValue::Object(values) => {
            for (member, value) in values {
                validate_request_body_value(kind, id, &format!("{path}.{member}"), value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn require_positive(field: &str, id: &str, value: u64) -> Result<(), InferlabError> {
    if value == 0 {
        invalid(format!("definition {id:?} {field} must be positive"))
    } else {
        Ok(())
    }
}

fn require_optional_positive(
    field: &str,
    id: &str,
    value: Option<u64>,
) -> Result<(), InferlabError> {
    value.map_or(Ok(()), |value| require_positive(field, id, value))
}

fn require_reference<T>(
    label: &str,
    id: &str,
    definitions: &BTreeMap<String, T>,
) -> Result<(), InferlabError> {
    if definitions.contains_key(id) {
        Ok(())
    } else {
        invalid(format!("unknown {label} {id:?}"))
    }
}

pub(super) fn require_id(label: &str, id: &str) -> Result<(), InferlabError> {
    if !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        invalid(format!("invalid {label} identifier {id:?}"))
    }
}

pub(super) fn require_nonempty(label: &str, id: &str, value: &str) -> Result<(), InferlabError> {
    if value.is_empty() {
        invalid(format!("{label} for {id:?} must not be empty"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_manifest(manifest: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let config = toml::from_str::<WorkspaceConfig>(manifest)?;
        validate_workspace(root.path(), &config)?;
        Ok(())
    }

    #[test]
    fn prefill_decode_requires_both_frontend_components_on_the_server_base()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = validate_manifest(
            r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "prefill_decode"
readiness_timeout_seconds = 60

[servers.server.cases.add-frontend]
gateway_backend = "gateway"
pd_router_backend = "router"
"#,
        );
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "a P/D case must not add frontend components absent from the server base",
            )
            .into());
        };

        assert!(
            error
                .to_string()
                .contains("prefill_decode server \"server\" must declare both gateway_backend and pd_router_backend"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn single_case_cannot_add_a_gateway_absent_from_the_server_base()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = validate_manifest(
            r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "single"
readiness_timeout_seconds = 60

[servers.server.cases.add-gateway]
gateway_backend = "gateway"
"#,
        );
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "a single case must not add a Gateway absent from the server base",
            )
            .into());
        };

        assert!(
            error.to_string().contains(
                "cannot add gateway_backend because the server base does not declare a Gateway"
            ),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn extra_args_segmentation_validates_at_workspace_load_across_layers()
    -> Result<(), Box<dyn std::error::Error>> {
        const HEADER: &str = r#"
schema_version = 2

[models.model]
served_name = "model"

[stacks.stack]
integration = "fixture"
pixi_environment = "fixture"

[servers.server]
stack = "stack"
model = "model"
topology = "single"
readiness_timeout_seconds = 60
"#;
        // A value token preceding any flag fails load validation at every
        // settings layer, even when no overriding layer touches the array
        // ([[RFC-0003:C-RESOLUTION]]).
        for (layer, path) in [
            (
                "[servers.server.settings]\nextra_args = [\"stray\", \"--a\"]",
                "server \"server\" settings.extra_args",
            ),
            (
                "[servers.server.roles.serve.settings]\nextra_args = [\"stray\"]",
                "server \"server\" role \"serve\" settings.extra_args",
            ),
            (
                "[servers.server.cases.c.settings]\nextra_args = [\"stray\"]",
                "server case \"c\" settings.extra_args",
            ),
            (
                "[servers.server.cases.c.roles.serve.settings]\nextra_args = [\"stray\"]",
                "server case \"c\" role \"serve\" settings.extra_args",
            ),
        ] {
            let Err(error) = validate_manifest(&format!("{HEADER}\n{layer}\n")) else {
                return Err(
                    std::io::Error::other(format!("{layer} must fail load validation")).into(),
                );
            };
            assert!(error.to_string().contains("precedes any flag"), "{error}");
            assert!(error.to_string().contains(path), "{error}");
        }

        // A second bare `--` is malformed even without any patch layer.
        let result = validate_manifest(&format!(
            "{HEADER}\n[servers.server.settings]\nextra_args = [\"--\", \"--a\", \"--\", \"--b\"]\n"
        ));
        let Err(error) = result else {
            return Err(std::io::Error::other(
                "a second verbatim sentinel must fail load validation",
            )
            .into());
        };
        assert!(error.to_string().contains("second bare `--`"), "{error}");

        // A well-segmented declaration, including one verbatim block, loads.
        validate_manifest(&format!(
            "{HEADER}\n[servers.server.settings]\nextra_args = [\"--max-num-seqs\", \"256\", \"--\", \"--block-size\", \"32\"]\n"
        ))?;
        Ok(())
    }
}
