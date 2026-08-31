//! Eval definition lowering and client planning.

use super::overrides::{apply_definition_override, override_plan};
use crate::InferlabError;
use crate::toml_override::InvocationOverride;
use crate::toolchain::InstalledEvalToolchain;
use crate::workload::plan::{
    ClientCommandPlan, EvalExecutionPlan, EvalPlan, MeasurementOverridePlan,
    MeasurementResolveContext,
};
use crate::workspace::{EvalDefinition, validate_eval};
use std::collections::BTreeMap;

fn apply_eval_overrides(
    id: &str,
    definition: EvalDefinition,
    overrides: &[InvocationOverride],
) -> Result<(EvalDefinition, Vec<MeasurementOverridePlan>), InferlabError> {
    let mut value =
        toml::Value::try_from(definition).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare eval {id:?} for overrides: {error}"),
        })?;
    for item in overrides {
        apply_definition_override(&mut value, item)?;
    }
    let definition = value
        .try_into()
        .map_err(|error| InferlabError::InvalidOverride {
            value: overrides
                .iter()
                .map(InvocationOverride::raw)
                .collect::<Vec<_>>()
                .join(", "),
            message: format!("invalid effective Eval definition: {error}"),
        })?;
    validate_eval(id, &definition)?;
    Ok((definition, override_plan(overrides)))
}

pub(super) fn resolve_eval(
    id: &str,
    definitions: &BTreeMap<String, EvalDefinition>,
    overrides: &[InvocationOverride],
    context: &MeasurementResolveContext<'_>,
    toolchain: Option<&InstalledEvalToolchain>,
) -> Result<EvalPlan, InferlabError> {
    let declared_definition =
        definitions
            .get(id)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown selected eval definition {id:?}"),
            })?;
    let (mut definition, override_plan) =
        apply_eval_overrides(id, declared_definition.clone(), overrides)?;
    // Synthetic acceptance bypasses real draft-model verification, so an Eval
    // of any kind bound to such a server is not plannable
    // ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]).
    if context.synthetic_acceptance {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "eval {id:?} is bound to a server whose resolved configuration carries synthetic \
                 acceptance, which bypasses real draft-model verification; evals are not \
                 plannable against it"
            ),
        });
    }
    crate::workspace::validate_eval_task_source(context.workspace_root, id, &definition)?;
    if let EvalDefinition::LmEval {
        task: crate::workspace::EvalTaskSource::WorkspaceYaml { yaml },
        ..
    } = &mut definition
    {
        *yaml = context.workspace_root.join(&*yaml);
    }
    let execution = match &definition {
        EvalDefinition::OpenAiSmoke { .. } => EvalExecutionPlan::NativeOpenAiSmoke,
        EvalDefinition::LmEval { .. } => {
            let toolchain = toolchain.ok_or_else(|| InferlabError::InvalidConfig {
                message: "lm-eval toolchain was not resolved".to_owned(),
            })?;
            let bundled_task = match &definition {
                EvalDefinition::LmEval {
                    task: crate::workspace::EvalTaskSource::Bundled { bundled },
                    ..
                } => Some(Box::new(toolchain.bundled_task(bundled)?)),
                _ => None,
            };
            let mut env = context.command_env.clone();
            env.insert(
                "PYTHONPATH".to_owned(),
                toolchain.python_path.to_string_lossy().into_owned(),
            );
            env.insert("PYTHONNOUSERSITE".to_owned(), "1".to_owned());
            EvalExecutionPlan::LmEval {
                toolchain: Box::new(toolchain.identity.clone()),
                bundled_task,
                command: ClientCommandPlan {
                    argv: vec![
                        toolchain.python.to_string_lossy().into_owned(),
                        toolchain.runner.to_string_lossy().into_owned(),
                    ],
                    env,
                    cwd: context.command_cwd.to_path_buf(),
                },
            }
        }
    };
    let declared_prompt = match &declared_definition {
        EvalDefinition::LmEval { prompt, .. } => prompt.declared().cloned(),
        EvalDefinition::OpenAiSmoke { .. } => None,
    };
    Ok(EvalPlan {
        id: id.to_owned(),
        capture: context.capture_ids.iter().any(|capture| capture == id),
        declared_definition,
        definition,
        declared_prompt,
        overrides: override_plan,
        endpoint: context.endpoint.clone(),
        model: context.model.clone(),
        workspace_source_exclusions: context.workspace_source_exclusions.to_vec(),
        execution,
    })
}

pub(super) fn definitions_are_lm_eval(
    definitions: &BTreeMap<String, EvalDefinition>,
    id: &str,
) -> bool {
    matches!(definitions.get(id), Some(EvalDefinition::LmEval { .. }))
}
