//! Runtime facade. Eval and Bench own domain adjudication, preparation owns
//! frozen request populations, and client supervision owns process lifecycle.

mod bench;
mod client;
mod eval;
mod preparation;

use super::domain::{
    BenchPopulation, BenchSessionTemplate, ResolvedBenchRequestSource, ResolvedBenchSource,
    WorkloadEndpointProtocol,
};
use super::record::{
    BenchAgenticSourceEvidence, BenchCorpusSourceEvidence, BenchDatasetRequestSourceEvidence,
    BenchPopulationPreparationEvidence, BenchRequestSourceEvidence, BenchSessionSourceEvidence,
    ClientCasePaths, ClientProcessEvidence, ClientTerminationEvidence, ClientTerminationTrigger,
    DataAssetMaterializationEvidence, DatasetAcquisitionEvidence, DatasetAcquisitionOutcome,
    EvalCaseEvidence, EvalCaseRecord, WorkloadKind, WorkloadRecord, WorkloadRecordSession,
    WorkloadStatus, write_json,
};
use super::{
    BenchExecutionPlan, BenchPlan, ClientCommandPlan, EvalExecutionPlan, EvalPlan,
    ResolvedWorkloadPlan,
};
use inferlab_runtime::operation_bound::OperationTimingEvidence;
use inferlab_runtime::process_group::LocalProcessGroup;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Child;
use std::time::Duration;

const CLIENT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const CLIENT_TERM_GRACE: Duration = Duration::from_secs(2);
const CLIENT_KILL_GRACE: Duration = Duration::from_secs(2);
const CLIENT_CLEANUP_STATUS_DEADLINE: Duration = Duration::from_secs(2);
const SYNTHETIC_MATERIALIZATION_IDENTITY: &str = "inferlab-synthetic-prompt-authority-v4";
const REPLAY_MATERIALIZATION_IDENTITY: &str = "inferlab-replay-population-v1";
const CORPUS_MATERIALIZATION_IDENTITY: &str = "inferlab-corpus-slice-v1";
struct AdjudicatedClient<T> {
    accepted: AcceptedClient<T>,
    succeeded: bool,
    error: Option<String>,
}

struct ClientRun {
    process: Option<ClientProcessEvidence>,
    error: Option<String>,
    pending_cleanup: Option<PendingClientCleanup>,
    /// Frozen before an early terminal path starts process cleanup. Ordinary
    /// exits leave this empty because result decoding and acceptance still
    /// belong to the measurement-case operation.
    terminal_timing: Option<OperationTimingEvidence>,
}

struct PendingClientCleanup {
    child: Child,
    group: LocalProcessGroup,
    handle_path: PathBuf,
}

/// The lenient result-envelope header: only the version, no field policy, so
/// an evolved envelope still reads far enough to be rejected by version
/// rather than dying in the strict v1 parse ([[RFC-0004:C-MEASUREMENTS]]).
#[derive(Deserialize)]
struct ClientResultEnvelope {
    schema_version: u32,
}

struct AcceptedClient<T> {
    run: ClientRun,
    result: Option<T>,
    decode_error: Option<String>,
    timing: OperationTimingEvidence,
    terminal_timing_frozen: bool,
}

pub(crate) const CLIENT_HANDLE_FILE: &str = "client-handle.json";
const SWEEP_WALK_DEPTH: usize = 6;

/// Durable client process-group handle, recorded at launch so a later run
/// can terminate survivors of an unclean exit by leader start-time
/// identity ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). The owner identity makes
/// "unclean exit" observable: a live handle belongs to a live concurrent
/// run exactly while the owning Inferlab process's identity still matches.
/// Unknown fields are tolerated so an older binary's sweep can still read
/// a newer handle instead of clearing it unparsed.
#[derive(Debug, Deserialize, Serialize)]
struct ClientGroupHandle {
    #[serde(flatten)]
    group: LocalProcessGroup,
    owner_pid: u32,
    owner_start_time_ticks: u64,
}

pub(crate) use bench::run_bench;
pub(crate) use bench::skip;
pub(crate) use client::{ClientProcessPaths, run_unbounded_client};
pub(crate) use eval::run_eval;
pub(crate) use preparation::acquire_dataset_snapshot;
