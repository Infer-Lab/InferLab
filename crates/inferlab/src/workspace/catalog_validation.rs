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
#[cfg(test)]
pub(super) use bench::validate_bench_slos;
pub(crate) use eval::{validate_eval, validate_eval_task_source};
#[cfg(test)]
pub(super) use serve::validate_profiler_escapes;

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
