//! Frozen workspace identity and loaded aggregate projections consumed by
//! resolution and records.

use super::definitions::WorkspaceConfig;
use super::local::LocalBindings;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub(crate) struct LoadedWorkspace {
    pub root: PathBuf,
    pub config: WorkspaceConfig,
    pub local: LocalBindings,
    pub snapshot: WorkspaceSnapshot,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WorkspaceSnapshot {
    pub revision: String,
    pub dirty: bool,
    pub source_digest: String,
    #[serde(skip)]
    pub source_exclusions: Vec<PathBuf>,
    pub revision_reproducible: bool,
    pub pixi_manifest_sha256: String,
    pub pixi_lock_sha256: String,
    /// The git-ignored local definition files composed into this run,
    /// outside source identity ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub local_definitions: Vec<LocalDefinitionFile>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LocalDefinitionFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub(crate) struct WorkspaceIdentity {
    pub revision: String,
    pub dirty: bool,
}
