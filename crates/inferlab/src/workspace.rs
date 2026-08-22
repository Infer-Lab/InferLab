//! Workspace aggregate facade. Portable definitions, machine-local bindings,
//! composition, realization checks, and source identity keep separate owners.

mod catalog_validation;
mod composition;
mod definitions;
mod local;
mod realization;
mod source;
mod state;

use crate::InferlabError;

pub(crate) use catalog_validation::{validate_bench, validate_eval, validate_eval_task_source};
pub(crate) use composition::{
    discover_workspace, load_workspace, load_workspace_config, workspace_summary,
};
pub(crate) use composition::{snapshot_workspace, workspace_identity};
#[cfg(test)]
pub(crate) use definitions::BenchRandomShape;
pub(crate) use definitions::{
    AggregateSlo, BenchAgenticSource, BenchArtifactLevel, BenchCacheStart, BenchDefinition,
    BenchRequestSource, BenchSessionSource, BenchTokenSelector, BenchTpotApplicability,
    EvalDefinition, EvalPrompt, EvalTaskSource, JsonValue, ModelDefinition, RecipeDefinition,
    RequestRate, RequestSlo, ServerCaseDefinition, ServerDefinition, StackDefinition,
    WorkloadSuiteDefinition, WorkspaceConfig,
};
pub(crate) use definitions::{
    BenchPrefixSharing, BenchPrompt, BenchPromptSelection, BenchSharedSystemContent,
};
pub(crate) use definitions::{
    DEFAULT_CAPTURE_ARM_DEADLINE_SECONDS, DEFAULT_CAPTURE_CONTROL_DEADLINE_SECONDS,
    DEFAULT_CAPTURE_FINALIZATION_DEADLINE_SECONDS,
    DEFAULT_ENGINE_TRACE_CAPTURE_FINALIZATION_DEADLINE_SECONDS,
    DEFAULT_READINESS_ATTEMPT_TIMEOUT_SECONDS,
};
#[cfg(test)]
pub(crate) use local::LocalBindings;
pub(crate) use local::MANAGED_CONTAINER_ENV;
pub(crate) use local::{
    AdapterBinding, BuilderKind, LaunchBinding, MachineBinding, ModelWeightBinding,
    PlacementBinding, PlacementRoleBinding,
};
pub(crate) use source::{
    git_status_flags, source_digest_script, source_pathspecs, workspace_mutations,
};
pub(crate) use state::{LoadedWorkspace, WorkspaceSnapshot};

fn invalid<T>(message: String) -> Result<T, InferlabError> {
    Err(InferlabError::InvalidConfig { message })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // The script text feeds recorded evidence and remote execution; a byte
    // drift here must fail the suite, not surface later as a digest change.
    #[test]
    fn source_digest_script_text_is_pinned() {
        insta::assert_snapshot!(source_digest_script(&[PathBuf::from(".inferlab")]));
    }
}
