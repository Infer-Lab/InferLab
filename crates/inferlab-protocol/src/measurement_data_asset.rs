//! Wire declarations owned by measurement data-asset preparation.

use crate::wire::{BenchAgenticSourceInput, ClientStatus, EvalDefinitionInput, ProtocolVersion};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// One narrowly scoped source-preparation request issued before measurement
/// materialization. The selected measurement definition remains authoritative
/// for dataset semantics.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementDataAssetPreparationRequest {
    pub protocol_version: ProtocolVersion,
    pub phase: MeasurementDataAssetPreparationPhase,
    pub source: MeasurementDataAssetSourceInput,
    pub artifact_dir: PathBuf,
}

/// One externally observable source-preparation phase. Separating resolution
/// from acquisition lets the control plane durably commit each result.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeasurementDataAssetPreparationPhase {
    Resolve,
    SnapshotLocal,
    Acquire {
        resolved_revision: String,
        cache_state_before: MeasurementDataAssetCacheOutcome,
    },
}

/// The next separately durable preparation phase selected by the owning
/// runtime after source resolution.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementDataAssetPreparationNextPhase {
    SnapshotLocal,
}

/// The measurement-owned source whose bytes or task selection are prepared.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeasurementDataAssetSourceInput {
    Eval {
        workspace_root: PathBuf,
        #[serde(default)]
        workspace_source_exclusions: Vec<PathBuf>,
        definition: Box<EvalDefinitionInput>,
    },
    Agentic {
        source: Box<BenchAgenticSourceInput>,
    },
}

/// A normalized mapping from dataset split names to selected file patterns.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
pub struct MeasurementDataFiles(pub BTreeMap<String, Vec<String>>);

/// The effective selection reported by the owning measurement runtime.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeasurementDataAssetEffectiveSelection {
    Eval {
        task_identity: String,
        #[serde(default)]
        dataset_path: Option<String>,
        #[serde(default)]
        dataset_name: Option<String>,
        #[serde(default)]
        evaluation_split: Option<String>,
        #[serde(default)]
        fewshot_split: Option<String>,
        #[serde(default)]
        data_files: Option<MeasurementDataFiles>,
    },
    Agentic {
        repository: String,
        requested_revision: String,
        #[serde(default)]
        observed_revision: Option<String>,
        filename: String,
    },
}

/// One member of a complete, ordered content closure.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementDataAssetContentEntry {
    pub relative_path: String,
    pub sha256: String,
}

/// Expected-versus-observed integrity evidence for qualified source content.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementDataAssetVerification {
    pub subject: String,
    pub expected: String,
    #[serde(default)]
    pub observed: Option<String>,
    pub matched: bool,
}

/// The complete immutable source identity established by preparation.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeasurementDataAssetAcquiredSource {
    ReleaseQualified {
        identity: String,
        closure: Vec<MeasurementDataAssetContentEntry>,
    },
    LocalFileClosure {
        source_root: PathBuf,
        files: Vec<MeasurementDataAssetContentEntry>,
    },
}

/// The immutable workspace-task snapshot consumed by one Eval client.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalPreparedSourceBinding {
    pub workspace_root: PathBuf,
    pub task_path: PathBuf,
}

/// Whether preparation established a closed source or an explicitly opaque
/// non-reproducible source.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeasurementDataAssetReadiness {
    Closed {
        acquired_source: Box<MeasurementDataAssetAcquiredSource>,
        #[serde(default)]
        verification: Vec<MeasurementDataAssetVerification>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        eval_binding: Option<Box<EvalPreparedSourceBinding>>,
    },
    Opaque {
        reason: String,
        #[serde(default)]
        unresolved_path: Option<PathBuf>,
        deferred_source_access: bool,
    },
}

/// Read-only or preparation-time outcome for one independently owned cache.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementDataAssetCacheOutcome {
    FullHit,
    Miss,
    PartialReuse,
    Unavailable,
}

/// One effective physical cache store as reported by its owning runtime.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementDataAssetCacheStore {
    pub authority: String,
    pub purpose: String,
    #[serde(default)]
    pub path: Option<PathBuf>,
    pub outcome: MeasurementDataAssetCacheOutcome,
}

/// Whether preparation contacted a remote metadata authority.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementDataAssetRemoteMetadataOutcome {
    NotAccessed,
    Accessed,
    Unavailable,
}

/// Whether immutable source bytes were reused or downloaded.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementDataAssetSourceBytesOutcome {
    NotAccessed,
    Reused,
    Downloaded,
    Unavailable,
}

/// Terminal result returned by the owning measurement runtime.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementDataAssetPreparationResult {
    pub schema_version: u32,
    pub status: ClientStatus,
    #[serde(default)]
    pub effective_selection: Option<MeasurementDataAssetEffectiveSelection>,
    #[serde(default)]
    pub readiness: Option<MeasurementDataAssetReadiness>,
    #[serde(default)]
    pub next_phase: Option<MeasurementDataAssetPreparationNextPhase>,
    #[serde(default)]
    pub cache_stores: Vec<MeasurementDataAssetCacheStore>,
    pub remote_metadata: MeasurementDataAssetRemoteMetadataOutcome,
    pub source_bytes: MeasurementDataAssetSourceBytesOutcome,
    #[serde(default)]
    pub error: Option<String>,
}
