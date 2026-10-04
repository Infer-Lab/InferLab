//! The workspace read model shared by the view-only TUI ([[RFC-0010]]) and the
//! web console ([[RFC-0012]], [[ADR-0056]]): snapshot collection, record and
//! definition projection, presentation into the four views, metrics, and
//! search. Neither surface reconstructs these facts on its own.

pub(crate) mod bench_detail;
pub(crate) mod collector;
pub(crate) mod metrics;
pub(crate) mod presentation;
pub(crate) mod records;
pub(crate) mod search;
pub(crate) mod views;

use std::collections::BTreeMap;
use std::path::PathBuf;

const LOG_TAIL_BYTES: u64 = 64 * 1024;

/// The four top-level views both surfaces present.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum View {
    #[default]
    Overview,
    Operations,
    Records,
    Workspace,
}

impl View {
    pub(crate) const ALL: [Self; 4] = [
        Self::Overview,
        Self::Operations,
        Self::Records,
        Self::Workspace,
    ];

    pub(crate) const fn title(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Operations => "Operations",
            Self::Records => "Records",
            Self::Workspace => "Workspace",
        }
    }
}

#[derive(Clone)]
pub(crate) struct Snapshot {
    pub(crate) root: PathBuf,
    pub(crate) observed_unix_ms: u64,
    pub(crate) workspace: ObjectState<WorkspaceView>,
    pub(crate) operations: Vec<OperationView>,
    pub(crate) records: Vec<RecordView>,
    pub(crate) child_records: Vec<RecordView>,
    pub(crate) definitions: Vec<DefinitionView>,
    pub(crate) journal: Vec<JournalView>,
    pub(crate) operations_error: Option<String>,
    pub(crate) records_error: Option<String>,
    pub(crate) definitions_error: Option<String>,
    pub(crate) journal_error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum State {
    Live,
    Stale,
    Unavailable,
    Incompatible,
}

impl State {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Stale => "stale",
            Self::Unavailable => "unavailable",
            Self::Incompatible => "incompatible",
        }
    }
}

#[derive(Clone)]
pub(crate) struct ObjectState<T> {
    pub(crate) state: State,
    pub(crate) value: Option<T>,
    pub(crate) reason: Option<String>,
    pub(crate) observed_unix_ms: u64,
    pub(crate) last_success_unix_ms: Option<u64>,
}

#[derive(Clone)]
pub(crate) struct WorkspaceView {
    pub(crate) revision: String,
    pub(crate) dirty: bool,
}

#[derive(Clone)]
pub(crate) struct OperationView {
    pub(crate) key: String,
    pub(crate) state: State,
    pub(crate) reason: Option<String>,
    pub(crate) command: Option<String>,
    pub(crate) phase: Option<String>,
    pub(crate) item: Option<String>,
    pub(crate) record_ref: Option<String>,
    pub(crate) log_ref: Option<String>,
    pub(crate) started_unix_ms: Option<u64>,
    pub(crate) updated_unix_ms: Option<u64>,
    pub(crate) observed_unix_ms: u64,
    pub(crate) last_success_unix_ms: Option<u64>,
    pub(crate) schema_version: Option<u32>,
    pub(crate) producer: Option<crate::operation::ProducerIdentity>,
    pub(crate) position: Option<crate::operation::OperationPosition>,
    pub(crate) lock: Option<String>,
    pub(crate) readiness_failure: Option<String>,
}

#[derive(Clone)]
pub(crate) struct RecordView {
    pub(crate) path: PathBuf,
    pub(crate) state: State,
    pub(crate) reason: Option<String>,
    pub(crate) id: Option<String>,
    pub(crate) kind: String,
    pub(crate) status: Option<String>,
    pub(crate) definition_ids: Vec<String>,
    pub(crate) case: Option<String>,
    pub(crate) workflow: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) started_unix_ms: Option<u64>,
    pub(crate) finished_unix_ms: Option<u64>,
    pub(crate) log_refs: Vec<String>,
    pub(crate) observed_unix_ms: u64,
    pub(crate) last_success_unix_ms: Option<u64>,
    pub(crate) child_refs: Vec<String>,
    pub(crate) topology: Option<String>,
    pub(crate) cases: Vec<CaseView>,
    pub(crate) outcome_facts: Vec<(String, String)>,
    pub(crate) bench_details: Vec<FactSection>,
    pub(crate) artifact_refs: Vec<String>,
    pub(crate) process_observation: Option<ObjectState<bool>>,
}

#[derive(Clone)]
pub(crate) struct CaseView {
    pub(crate) id: Option<String>,
    pub(crate) load: CaseLoad,
    pub(crate) status: Option<String>,
    pub(crate) stdout: Option<String>,
    pub(crate) stderr: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CaseLoad {
    Concurrency(u32),
    RequestRate(f64),
    UnboundedRequestRate,
    Unknown,
}

#[derive(Clone)]
pub(crate) struct DefinitionView {
    pub(crate) kind: String,
    pub(crate) id: String,
    pub(crate) relationship: String,
    pub(crate) fact_sections: Vec<FactSection>,
    pub(crate) state: State,
    pub(crate) observed_unix_ms: u64,
    pub(crate) last_success_unix_ms: u64,
    pub(crate) reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FactSection {
    pub(crate) title: &'static str,
    pub(crate) rows: Vec<(String, String)>,
}

#[derive(Clone)]
pub(crate) struct JournalView {
    pub(crate) timestamp: String,
    pub(crate) topic: Option<String>,
    pub(crate) author: String,
    pub(crate) text: String,
    pub(crate) records: Vec<String>,
    pub(crate) state: State,
    pub(crate) observed_unix_ms: u64,
    pub(crate) last_success_unix_ms: u64,
    pub(crate) reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum EntryKind {
    Workspace,
    Operation,
    Record,
    Definition,
    Journal,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum Authority {
    Declared,
    Recorded,
    Ephemeral,
    Observed,
}

impl Authority {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Recorded => "recorded",
            Self::Ephemeral => "ephemeral",
            Self::Observed => "observed",
        }
    }

    pub(crate) const fn badge(self) -> &'static str {
        match self {
            Self::Declared => "DECL",
            Self::Recorded => "REC",
            Self::Ephemeral => "EPH",
            Self::Observed => "OBS",
        }
    }
}

impl EntryKind {
    pub(crate) const fn group_label(self) -> &'static str {
        match self {
            Self::Workspace => "WORKSPACE",
            Self::Operation => "OPERATIONS",
            Self::Record => "RECORDS",
            Self::Definition => "DEFINITIONS",
            Self::Journal => "SCRATCHPAD",
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum OverviewSection {
    Attention,
    Active,
    Recent,
    Workspace,
}

impl OverviewSection {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Attention => "ATTENTION",
            Self::Active => "NOW",
            Self::Recent => "RECENT",
            Self::Workspace => "WORKSPACE",
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct OverviewSummary {
    pub(crate) ephemeral_active: usize,
    pub(crate) ephemeral_attention: usize,
    pub(crate) recorded_active: usize,
    pub(crate) recorded_attention: usize,
    pub(crate) recorded_recent: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum DisplayTone {
    Normal,
    Success,
    Active,
    Warning,
    Critical,
}

#[derive(Clone)]
pub(crate) struct DetailSection {
    pub(crate) title: &'static str,
    pub(crate) rows: Vec<(String, DetailValue)>,
    pub(crate) body: Vec<String>,
}

#[derive(Clone)]
pub(crate) enum DetailValue {
    Text(String),
    Age(Option<u64>),
    TimestampWithAge(Option<u64>),
    Elapsed {
        start: Option<u64>,
        finish: Option<u64>,
        advances: bool,
    },
}

#[derive(Clone)]
pub(crate) struct DisplayEntry {
    pub(crate) kind: EntryKind,
    pub(crate) key: String,
    pub(crate) record_ref: Option<String>,
    pub(crate) title: String,
    pub(crate) summary: String,
    pub(crate) authority: Authority,
    pub(crate) state: State,
    pub(crate) lifecycle: Option<String>,
    pub(crate) tone: DisplayTone,
    pub(crate) details: Vec<DetailSection>,
    pub(crate) search_fields: Vec<String>,
    pub(crate) log_refs: Vec<String>,
    /// When the object last changed, for the list row's age.
    pub(crate) moment_unix_ms: Option<u64>,
    /// An active operation's reported item position.
    pub(crate) progress: Option<(usize, usize)>,
}

impl DisplayTone {
    /// The status glyph both surfaces show: ✓ succeeded, × failed, ◆
    /// warning, ● live or running, · declared or neutral ([[ADR-0055]]).
    pub(crate) const fn glyph(self) -> &'static str {
        match self {
            Self::Normal => "·",
            Self::Success => "✓",
            Self::Active => "●",
            Self::Warning => "◆",
            Self::Critical => "×",
        }
    }
}

impl DisplayEntry {
    /// The row's kind and name: a `kind / name` title splits; other entries
    /// take a short label of their entry kind.
    pub(crate) fn kind_and_name(&self) -> (&str, &str) {
        if let Some((kind, name)) = self.title.split_once(" / ") {
            let kind = match kind {
                "workload-suite" => "suite",
                "external-image" => "ext-img",
                other => other,
            };
            return (kind, name);
        }
        let kind = match self.kind {
            EntryKind::Operation => "run",
            EntryKind::Workspace => "ws",
            EntryKind::Journal => "note",
            EntryKind::Record | EntryKind::Definition => "",
        };
        (kind, self.title.as_str())
    }
}

/// The visible status filter of Overview and Records ([[RFC-0010:C-NAVIGATION]]).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum StatusFilter {
    #[default]
    All,
    /// Failed, stale, or unavailable objects.
    Attention,
    Running,
}

impl StatusFilter {
    pub(crate) const ALL: [Self; 3] = [Self::All, Self::Attention, Self::Running];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Attention => "issues",
            Self::Running => "running",
        }
    }

    pub(crate) const fn next(self) -> Self {
        match self {
            Self::All => Self::Attention,
            Self::Attention => Self::Running,
            Self::Running => Self::All,
        }
    }

    pub(crate) fn admits(self, entry: &DisplayEntry) -> bool {
        match self {
            Self::All => true,
            Self::Attention => {
                entry.tone == DisplayTone::Critical
                    || matches!(
                        entry.state,
                        State::Stale | State::Unavailable | State::Incompatible
                    )
            }
            Self::Running => entry.tone == DisplayTone::Active,
        }
    }
}
