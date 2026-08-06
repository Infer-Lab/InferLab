//! Shared measurement selection and invocation-override projection.

use crate::InferlabError;
use crate::toml_override::InvocationOverride;
use crate::workload::plan::MeasurementOverridePlan;
use crate::workspace::{BenchDefinition, EvalDefinition, WorkloadSuiteDefinition};
use std::collections::BTreeMap;

pub(super) fn apply_definition_override(
    definition: &mut toml::Value,
    item: &InvocationOverride,
) -> Result<(), InferlabError> {
    let assignment = item.assignment()?;
    if assignment.root_key() == "kind" {
        return Err(InferlabError::InvalidOverride {
            value: item.raw().to_owned(),
            message: "measurement kind cannot be overridden".to_owned(),
        });
    }
    assignment.apply_to(definition, item.raw())
}

pub(super) fn recipe_measurement_overrides(
    section: &str,
    id: &str,
    overrides: &[InvocationOverride],
) -> Vec<InvocationOverride> {
    let prefix = format!("{section}.{id}.");
    overrides
        .iter()
        .filter_map(|item| item.under(&prefix))
        .collect()
}

pub(super) fn validate_recipe_measurement_overrides(
    suite: &WorkloadSuiteDefinition,
    evals: &BTreeMap<String, EvalDefinition>,
    benches: &BTreeMap<String, BenchDefinition>,
    overrides: &[InvocationOverride],
) -> Result<(), InferlabError> {
    for item in overrides {
        let path = item.path();
        if path.starts_with("server.") {
            continue;
        }
        let (section, remaining, selected) = if let Some(remaining) = path.strip_prefix("evals.") {
            ("evals", remaining, &suite.evals)
        } else if let Some(remaining) = path.strip_prefix("benches.") {
            ("benches", remaining, &suite.benches)
        } else {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: "recipe override must be under server., evals.<id>., or benches.<id>."
                    .to_owned(),
            });
        };
        let Some((id, field)) = remaining.split_once('.') else {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: format!("expected {section}.<id>.<field>=<TOML-value>"),
            });
        };
        let declared = match section {
            "evals" => evals.contains_key(id),
            "benches" => benches.contains_key(id),
            _ => false,
        };
        if id.is_empty()
            || field.is_empty()
            || !declared
            || !selected.iter().any(|selected| selected == id)
        {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: format!(
                    "{section} override must name a definition selected by the recipe's workload suite"
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn override_plan(overrides: &[InvocationOverride]) -> Vec<MeasurementOverridePlan> {
    overrides
        .iter()
        .map(|item| MeasurementOverridePlan {
            invocation_index: item.index(),
            value: item.raw().to_owned(),
        })
        .collect()
}
