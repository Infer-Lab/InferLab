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

/// The span of segments an override path addresses: the longest segment
/// prefix matching an id in `ids`, leaving a non-empty field path. Validation
/// and collection share this one rule so a path addresses exactly one
/// definition even when an id is a dotted prefix of another selected id.
fn addressed_span(segments: &[String], ids: &[String]) -> Option<usize> {
    (1..segments.len())
        .rev()
        .find(|&span| ids.iter().any(|id| id == &segments[1..=span].join(".")))
}

pub(super) fn recipe_measurement_overrides(
    section: &str,
    selected: &[String],
    id: &str,
    overrides: &[InvocationOverride],
) -> Vec<InvocationOverride> {
    overrides
        .iter()
        .filter_map(|item| {
            let segments = item.path_segments().ok()?;
            if segments.first().map(String::as_str) != Some(section) {
                return None;
            }
            let span = addressed_span(&segments, selected)?;
            if segments[1..=span].join(".") != id {
                return None;
            }
            item.under_segments(1 + span)
        })
        .collect()
}

pub(super) fn validate_recipe_measurement_overrides(
    suite: &WorkloadSuiteDefinition,
    evals: &BTreeMap<String, EvalDefinition>,
    benches: &BTreeMap<String, BenchDefinition>,
    overrides: &[InvocationOverride],
) -> Result<(), InferlabError> {
    for item in overrides {
        let segments = item.path_segments()?;
        if segments.first().map(String::as_str) == Some("server") {
            continue;
        }
        let (section, selected): (&str, &[String]) = match segments.first().map(String::as_str) {
            Some("evals") => ("evals", &suite.evals),
            Some("benches") => ("benches", &suite.benches),
            _ => {
                return Err(InferlabError::InvalidOverride {
                    value: item.raw().to_owned(),
                    message: "recipe override must be under server., evals.<id>., or benches.<id>."
                        .to_owned(),
                });
            }
        };
        let declared = |id: &str| match section {
            "evals" => evals.contains_key(id),
            _ => benches.contains_key(id),
        };
        if segments.len() < 3 {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: format!("expected {section}.<id>.<field>=<TOML-value>"),
            });
        }
        // The id may itself contain dots, quoted or not; match the longest
        // declared-and-selected segment prefix, leaving a non-empty field
        // path.
        let candidate = |span: usize| segments[1..=span].join(".");
        let eligible = selected
            .iter()
            .filter(|id| declared(id))
            .cloned()
            .collect::<Vec<_>>();
        if addressed_span(&segments, &eligible).is_none() {
            let unknown = (1..segments.len() - 1)
                .rev()
                .find(|&span| declared(&candidate(span)))
                .map_or_else(|| segments[1].clone(), candidate);
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: format!(
                    "{section} override names {unknown:?}, which is not a definition selected by the recipe's workload suite"
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
