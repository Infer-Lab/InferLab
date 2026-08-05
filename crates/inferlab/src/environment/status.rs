//! Read-only local stack readiness reporting.

use super::{
    CheckOutcome, CheckRealization, CompletedLocalCheck, EnvironmentCheck, LocalCheckConclusion,
    PlannedEnvironmentCheck, check_environment, run_local_checks,
};
use crate::InferlabError;
use crate::progress::{Phase, Progress};
use serde::Serialize;
use std::path::Path;

/// One declared stack's local realization state, reported by `stack status`
/// ([[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]).
#[derive(Debug, Serialize)]
pub struct EnvironmentStatusReport {
    pub stack: String,
    pub pixi_environment: String,
    pub status: EnvironmentStatusKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub install_command: Option<String>,
    pub checks: StackStatusChecks,
    pub ready: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EnvironmentStatusKind {
    Confirmed,
    NeverInstalled,
    NotUsable,
}

pub struct StackStatusRequest {
    pub stack: String,
    pub pixi_environment: String,
    pub checks: Vec<PlannedEnvironmentCheck>,
}

#[derive(Debug, Serialize)]
pub struct StackStatusChecks {
    pub state: StackStatusCheckState,
    pub evidence: Vec<StackStatusCheckEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<StackStatusCheckError>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StackStatusCheckState {
    NotDeclared,
    Skipped,
    Passed,
    Failed,
    Error,
}

#[derive(Debug, Serialize)]
pub struct StackStatusCheckEvidence {
    pub id: String,
    pub realization: CheckRealization,
    pub outcome: CheckOutcome,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repair_hint: Option<String>,
}

impl From<CompletedLocalCheck> for StackStatusCheckEvidence {
    fn from(check: CompletedLocalCheck) -> Self {
        Self {
            id: check.id,
            realization: CheckRealization::LocalWorkspace,
            outcome: check.outcome,
            output: check.output,
            repair_hint: check.repair_hint,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct StackStatusCheckError {
    pub id: String,
    pub diagnostics: String,
}

/// Report each named stack's confirmation and declared-check readiness without
/// installing packages or updating the manifest or lock.
pub fn status_with_progress(
    root: &Path,
    stacks: &[StackStatusRequest],
    progress: &Progress,
) -> Result<Vec<EnvironmentStatusReport>, InferlabError> {
    stacks
        .iter()
        .enumerate()
        .map(|(index, request)| {
            progress.phase(Phase::named("environment realization inspection").item(
                &request.stack,
                index + 1,
                stacks.len(),
            ))?;
            let pixi_environment = &request.pixi_environment;
            let install_command = format!("pixi install --locked --environment {pixi_environment}");
            let (status, diagnostics, install_command) =
                match check_environment(root, pixi_environment) {
                    Ok(EnvironmentCheck::Confirmed) => {
                        (EnvironmentStatusKind::Confirmed, None, None)
                    }
                    Ok(EnvironmentCheck::NeverInstalled) => (
                        EnvironmentStatusKind::NeverInstalled,
                        None,
                        Some(install_command),
                    ),
                    Ok(EnvironmentCheck::NotUsable(diagnostics)) => (
                        EnvironmentStatusKind::NotUsable,
                        Some(diagnostics),
                        Some(install_command),
                    ),
                    Err(error) => (
                        EnvironmentStatusKind::NotUsable,
                        Some(error.to_string()),
                        Some(install_command),
                    ),
                };
            let checks = status_checks(root, request, status, progress)?;
            let ready = status == EnvironmentStatusKind::Confirmed
                && matches!(
                    checks.state,
                    StackStatusCheckState::NotDeclared | StackStatusCheckState::Passed
                );
            Ok(EnvironmentStatusReport {
                stack: request.stack.clone(),
                pixi_environment: pixi_environment.clone(),
                status,
                diagnostics,
                install_command,
                checks,
                ready,
            })
        })
        .collect()
}

fn status_checks(
    root: &Path,
    request: &StackStatusRequest,
    environment_status: EnvironmentStatusKind,
    progress: &Progress,
) -> Result<StackStatusChecks, InferlabError> {
    if request.checks.is_empty() {
        return Ok(StackStatusChecks {
            state: StackStatusCheckState::NotDeclared,
            evidence: Vec::new(),
            error: None,
        });
    }
    if environment_status != EnvironmentStatusKind::Confirmed {
        return Ok(StackStatusChecks {
            state: StackStatusCheckState::Skipped,
            evidence: Vec::new(),
            error: None,
        });
    }

    let run = run_local_checks(
        root,
        &request.pixi_environment,
        &request.checks,
        progress,
        "stack realization checks",
    )?;
    let evidence = run.completed.into_iter().map(Into::into).collect();
    let (state, error) = match run.conclusion {
        LocalCheckConclusion::Passed => (StackStatusCheckState::Passed, None),
        LocalCheckConclusion::Failed(_) => (StackStatusCheckState::Failed, None),
        LocalCheckConclusion::ExecutionError(failure) => {
            let diagnostics = failure.diagnostics();
            (
                StackStatusCheckState::Error,
                Some(StackStatusCheckError {
                    id: failure.id().to_owned(),
                    diagnostics,
                }),
            )
        }
    };
    Ok(StackStatusChecks {
        state,
        evidence,
        error,
    })
}
