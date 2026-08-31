use super::super::FactSection;
use super::definition::{
    fact, optional_text, prefix_summary, shared_system_summary, token_selector, yes_no,
};
use crate::workspace::{BenchPrefixSharing, BenchSharedSystemContent, BenchTokenSelector};
use inferlab_protocol::{
    BenchAgenticResultEvidence, BenchPromptTokenReconciliation, BenchSessionPhaseSummary,
    BenchSessionResultEvidence, ClientStatus, RawArtifact,
};
use serde::Deserialize;

#[derive(Deserialize)]
pub(in crate::tui) struct CaptureProjection {
    status: String,
    #[serde(default)]
    windows: Vec<serde_json::Value>,
    #[serde(default)]
    reports: Vec<serde_json::Value>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(in crate::tui) enum RequestSourceProjection {
    Random {
        input_tokens: BenchTokenSelector,
        output_tokens: BenchTokenSelector,
        #[serde(default)]
        prefix_sharing: Option<BenchPrefixSharing>,
        #[serde(default)]
        shared_system_content: Option<BenchSharedSystemContent>,
        #[serde(default)]
        preparation: Option<PopulationPreparationProjection>,
    },
    RandomMixture {
        #[serde(default)]
        shapes: Vec<serde_json::Value>,
        total_weight: u64,
        #[serde(default)]
        prefix_sharing: Option<BenchPrefixSharing>,
        #[serde(default)]
        preparation: Option<PopulationPreparationProjection>,
    },
    Dataset(Box<DatasetRequestSourceProjection>),
}

#[derive(Deserialize)]
pub(in crate::tui) struct PopulationPreparationProjection {
    #[serde(default)]
    result: Option<PopulationPreparationResultProjection>,
}

#[derive(Deserialize)]
struct PopulationPreparationResultProjection {
    admitted_entries: u64,
}

#[derive(Deserialize)]
pub(in crate::tui) struct DatasetRequestSourceProjection {
    catalog: DatasetCatalogProjection,
    #[serde(default)]
    acquisition: Option<AcquisitionProjection>,
    #[serde(default)]
    preparation_attempt_id: Option<String>,
    #[serde(default)]
    preparation: Option<PopulationPreparationResultProjection>,
}

#[derive(Deserialize)]
pub(in crate::tui) struct SessionSourceProjection {
    catalog: DatasetCatalogProjection,
    #[serde(default)]
    acquisition: Option<AcquisitionProjection>,
    #[serde(default)]
    preparation_attempt_id: Option<String>,
    #[serde(default)]
    preparation: Option<PopulationPreparationResultProjection>,
}

#[derive(Deserialize)]
pub(in crate::tui) struct AgenticSourceProjection {
    #[serde(default)]
    preparation_attempt_id: Option<String>,
    dataset: String,
    profile: String,
    catalog: AgenticCatalogProjection,
}

#[derive(Deserialize)]
struct DatasetCatalogProjection {
    dataset: String,
    #[serde(default)]
    profile: Option<String>,
    upstream_identity: String,
    sha256: String,
}

#[derive(Deserialize)]
struct AgenticCatalogProjection {
    repository: String,
    revision: String,
    filename: String,
    sha256: String,
    scenario: String,
    trace_count: u32,
}

#[derive(Deserialize)]
pub(in crate::tui) struct AcquisitionProjection {
    outcome: AcquisitionOutcomeProjection,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AcquisitionOutcomeProjection {
    Reused,
    Downloaded,
    Failed,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(in crate::tui) enum PopulationSliceProjection {
    Requests {
        population_sha256: String,
        warmup_count: u32,
        profiling_count: u32,
    },
    Sessions {
        population_sha256: String,
        warmup_session_count: u32,
        warmup_request_count: u32,
        profiling_session_count: u32,
        profiling_request_count: u32,
    },
}

#[derive(Deserialize)]
pub(in crate::tui) struct PrefixCacheResetProjection {
    succeeded: bool,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
pub(in crate::tui) struct PrefixCacheConditioningRankProjection {
    rank: u32,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    http_status: Option<u16>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
pub(in crate::tui) struct PrefixCacheConditioningProjection {
    succeeded: bool,
    prompt_tokens: u32,
    #[serde(default)]
    ranks: Vec<PrefixCacheConditioningRankProjection>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
pub(in crate::tui) struct CachePreparationProjection {
    start: String,
    reset: PrefixCacheResetProjection,
    #[serde(default)]
    conditioning: Option<PrefixCacheConditioningProjection>,
}

#[derive(Deserialize)]
pub(in crate::tui) struct CaseSloProjection {
    #[serde(default)]
    aggregate_slos: Vec<serde_json::Value>,
    passed: bool,
}

pub(in crate::tui) struct CaseEvidence<'a> {
    pub(in crate::tui) id: Option<&'a str>,
    pub(in crate::tui) cache_preparation: Option<&'a CachePreparationProjection>,
    pub(in crate::tui) slo: Option<&'a CaseSloProjection>,
    pub(in crate::tui) population_slice: Option<&'a PopulationSliceProjection>,
    pub(in crate::tui) completed_requests: Option<u64>,
    pub(in crate::tui) failed_requests: Option<u64>,
    pub(in crate::tui) normalization_schema: Option<&'a str>,
    pub(in crate::tui) request_unavailable: bool,
    pub(in crate::tui) session: Option<&'a BenchSessionResultEvidence>,
    pub(in crate::tui) session_unavailable: bool,
    pub(in crate::tui) agentic: Option<&'a BenchAgenticResultEvidence>,
    pub(in crate::tui) agentic_unavailable: bool,
    pub(in crate::tui) prompt_token_reconciliation: &'a [BenchPromptTokenReconciliation],
    pub(in crate::tui) raw_artifacts: &'a [RawArtifact],
}

pub(in crate::tui) fn record_source(
    schema_version: Option<u32>,
    request: Option<&RequestSourceProjection>,
    session: Option<&SessionSourceProjection>,
    agentic: Option<&AgenticSourceProjection>,
) -> FactSection {
    if let Some(source) = request {
        return request_record_source(schema_version, source);
    }
    if let Some(source) = session {
        return FactSection {
            title: "RECORDED SOURCE · LINEAR SESSION",
            rows: vec![
                fact("Dataset", source.catalog.dataset.clone()),
                fact("Profile", optional_text(source.catalog.profile.as_deref())),
                fact(
                    "Upstream identity",
                    source.catalog.upstream_identity.clone(),
                ),
                fact("Population digest", source.catalog.sha256.clone()),
                fact(
                    "Source preparation",
                    source_preparation(
                        schema_version,
                        source.preparation_attempt_id.as_deref(),
                        source.acquisition.as_ref(),
                    ),
                ),
                fact(
                    "Preparation",
                    source.preparation.as_ref().map_or("—".to_owned(), |value| {
                        format!("{} admitted", value.admitted_entries)
                    }),
                ),
            ],
        };
    }
    if let Some(source) = agentic {
        return FactSection {
            title: "RECORDED SOURCE · AGENTIC REPLAY",
            rows: vec![
                fact("Dataset", source.dataset.clone()),
                fact("Profile", source.profile.clone()),
                fact("Repository", source.catalog.repository.clone()),
                fact("Revision", source.catalog.revision.clone()),
                fact("Filename", source.catalog.filename.clone()),
                fact("Source digest", source.catalog.sha256.clone()),
                fact("Scenario", source.catalog.scenario.clone()),
                fact("Traces", source.catalog.trace_count.to_string()),
                fact(
                    "Source preparation",
                    source_preparation(
                        schema_version,
                        source.preparation_attempt_id.as_deref(),
                        None,
                    ),
                ),
            ],
        };
    }
    FactSection {
        title: "RECORDED SOURCE",
        rows: vec![
            fact("Source evidence", "unavailable for this record schema"),
            fact(
                "Schema version",
                schema_version.map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
            ),
        ],
    }
}

pub(in crate::tui) fn case_evidence(evidence: CaseEvidence<'_>) -> (Vec<FactSection>, Vec<String>) {
    let case = evidence.id.unwrap_or("unnamed case");
    let mut sections = Vec::new();
    let mut common = vec![fact("Case", case)];
    if let Some(slice) = evidence.population_slice {
        common.extend(population_slice(slice));
    }
    if let Some(value) = evidence.completed_requests {
        common.push(fact("Completed requests", value.to_string()));
    }
    if let Some(value) = evidence.failed_requests {
        common.push(fact("Failed requests", value.to_string()));
    }
    if let Some(value) = evidence.normalization_schema {
        common.push(fact("Normalization", value));
    }
    if let Some(preparation) = evidence.cache_preparation {
        common.push(fact("Cache start", preparation.start.clone()));
        common.push(fact("Prefix-cache reset", prefix_reset(&preparation.reset)));
        if let Some(conditioning) = &preparation.conditioning {
            let summary = if conditioning.succeeded {
                let ranks = if conditioning.ranks.len() > 1 {
                    format!(" · {} ranks", conditioning.ranks.len())
                } else {
                    String::new()
                };
                format!("succeeded · {} tok{ranks}", conditioning.prompt_tokens)
            } else {
                let reason = conditioning
                    .error
                    .as_deref()
                    .map(str::to_owned)
                    .or_else(|| {
                        conditioning
                            .ranks
                            .iter()
                            .find(|rank| rank.error.is_some())
                            .map(|rank| {
                                let status = rank.http_status.map_or_else(
                                    || "no response".to_owned(),
                                    |code| format!("HTTP {code}"),
                                );
                                let target = rank
                                    .target
                                    .as_deref()
                                    .map_or_else(String::new, |target| format!("{target} · "));
                                format!(
                                    "{target}rank {} · {status} · {}",
                                    rank.rank,
                                    rank.error.as_deref().unwrap_or("unknown error")
                                )
                            })
                    });
                format!(
                    "failed · {}",
                    reason.unwrap_or_else(|| "unknown error".to_owned())
                )
            };
            common.push(fact("Prefix conditioning", summary));
        }
    }
    if !evidence.prompt_token_reconciliation.is_empty() {
        let reconciled = evidence
            .prompt_token_reconciliation
            .iter()
            .filter(|value| value.reconciled)
            .count();
        common.push(fact(
            "Prompt-token reconciliation",
            format!(
                "{reconciled}/{} reconciled",
                evidence.prompt_token_reconciliation.len()
            ),
        ));
    }
    if let Some(slo) = evidence.slo {
        common.push(fact(
            "SLO",
            format!(
                "{} · {} aggregate constraint(s)",
                if slo.passed { "passed" } else { "failed" },
                slo.aggregate_slos.len()
            ),
        ));
    }
    sections.push(FactSection {
        title: "CASE EVIDENCE",
        rows: common,
    });
    if evidence.request_unavailable {
        sections.push(unavailable_case_source("REQUEST RESULT", case));
    }
    if let Some(session) = evidence.session {
        sections.push(session_result(case, session));
    } else if evidence.session_unavailable {
        sections.push(unavailable_case_source("LINEAR SESSION RESULT", case));
    }
    let mut artifacts = evidence
        .raw_artifacts
        .iter()
        .map(|artifact| {
            format!(
                "{} · {} · {}",
                artifact.name,
                artifact.kind,
                artifact.path.display()
            )
        })
        .collect::<Vec<_>>();
    if let Some(agentic) = evidence.agentic {
        sections.push(agentic_result(case, agentic));
        if let Some(run) = agentic.run.as_deref() {
            artifacts.push(format!(
                "agentic aggregate · {}",
                run.aggregate_artifact.display()
            ));
            if let Some(raw_records) = run.raw_records_artifact.as_ref() {
                artifacts.push(format!("agentic raw records · {}", raw_records.display()));
            }
        }
    } else if evidence.agentic_unavailable {
        sections.push(unavailable_case_source("AGENTIC REPLAY RESULT", case));
    }
    (sections, artifacts)
}

pub(in crate::tui) fn capture_summary(capture: &CaptureProjection) -> String {
    let mut summary = format!(
        "{} · {} window(s) · {} report(s)",
        capture.status,
        capture.windows.len(),
        capture.reports.len()
    );
    if let Some(error) = capture.error.as_deref() {
        summary.push_str(" · ");
        summary.push_str(error);
    }
    summary
}

fn request_record_source(
    schema_version: Option<u32>,
    source: &RequestSourceProjection,
) -> FactSection {
    let rows = match source {
        RequestSourceProjection::Random {
            input_tokens,
            output_tokens,
            prefix_sharing,
            shared_system_content,
            preparation,
        } => vec![
            fact("Generator", "random"),
            fact("Input tokens", token_selector(input_tokens)),
            fact("Output tokens", token_selector(output_tokens)),
            fact("Prefix sharing", prefix_summary(prefix_sharing.as_ref())),
            fact(
                "Shared system content",
                shared_system_summary(shared_system_content.as_ref()),
            ),
            fact(
                "Population preparation",
                preparation
                    .as_ref()
                    .and_then(|value| value.result.as_ref())
                    .map_or("—".to_owned(), |value| {
                        format!("{} admitted", value.admitted_entries)
                    }),
            ),
        ],
        RequestSourceProjection::RandomMixture {
            shapes,
            total_weight,
            prefix_sharing,
            preparation,
        } => vec![
            fact("Generator", "random mixture"),
            fact("Shapes", shapes.len().to_string()),
            fact("Total weight", total_weight.to_string()),
            fact("Prefix sharing", prefix_summary(prefix_sharing.as_ref())),
            fact(
                "Population preparation",
                preparation
                    .as_ref()
                    .and_then(|value| value.result.as_ref())
                    .map_or("—".to_owned(), |value| {
                        format!("{} admitted", value.admitted_entries)
                    }),
            ),
        ],
        RequestSourceProjection::Dataset(source) => vec![
            fact("Generator", "dataset"),
            fact("Dataset", source.catalog.dataset.clone()),
            fact("Profile", optional_text(source.catalog.profile.as_deref())),
            fact(
                "Upstream identity",
                source.catalog.upstream_identity.clone(),
            ),
            fact("Population digest", source.catalog.sha256.clone()),
            fact(
                "Source preparation",
                source_preparation(
                    schema_version,
                    source.preparation_attempt_id.as_deref(),
                    source.acquisition.as_ref(),
                ),
            ),
            fact(
                "Preparation",
                source.preparation.as_ref().map_or("—".to_owned(), |value| {
                    format!("{} admitted", value.admitted_entries)
                }),
            ),
        ],
    };
    FactSection {
        title: "RECORDED SOURCE · REQUESTS",
        rows,
    }
}

fn source_preparation(
    schema_version: Option<u32>,
    preparation_attempt_id: Option<&str>,
    historical_acquisition: Option<&AcquisitionProjection>,
) -> String {
    match schema_version {
        Some(crate::workload::EVIDENCE_WORKLOAD_SCHEMA_VERSION) => {
            optional_text(preparation_attempt_id)
        }
        Some(version) if version < crate::workload::EVIDENCE_WORKLOAD_SCHEMA_VERSION => {
            historical_acquisition.map_or_else(
                || "unavailable for this record schema".to_owned(),
                acquisition,
            )
        }
        Some(_) => "unavailable for unsupported record schema".to_owned(),
        None => "unavailable for unknown record schema".to_owned(),
    }
}

fn acquisition(value: &AcquisitionProjection) -> String {
    let outcome = match value.outcome {
        AcquisitionOutcomeProjection::Reused => "reused",
        AcquisitionOutcomeProjection::Downloaded => "downloaded",
        AcquisitionOutcomeProjection::Failed => "failed",
    };
    value.error.as_deref().map_or_else(
        || outcome.to_owned(),
        |error| format!("{outcome} · {error}"),
    )
}

fn population_slice(value: &PopulationSliceProjection) -> Vec<(String, String)> {
    match value {
        PopulationSliceProjection::Requests {
            population_sha256,
            warmup_count,
            profiling_count,
            ..
        } => vec![
            fact("Population digest", population_sha256.clone()),
            fact("Warmup requests", warmup_count.to_string()),
            fact("Profiling requests", profiling_count.to_string()),
        ],
        PopulationSliceProjection::Sessions {
            population_sha256,
            warmup_session_count,
            warmup_request_count,
            profiling_session_count,
            profiling_request_count,
            ..
        } => vec![
            fact("Population digest", population_sha256.clone()),
            fact(
                "Warmup slice",
                format!("{warmup_session_count} sessions · {warmup_request_count} requests"),
            ),
            fact(
                "Profiling slice",
                format!("{profiling_session_count} sessions · {profiling_request_count} requests"),
            ),
        ],
    }
}

fn prefix_reset(value: &PrefixCacheResetProjection) -> String {
    let outcome = if value.succeeded {
        "succeeded"
    } else {
        "failed"
    };
    value.error.as_deref().map_or_else(
        || outcome.to_owned(),
        |error| format!("{outcome} · {error}"),
    )
}

fn session_result(case: &str, value: &BenchSessionResultEvidence) -> FactSection {
    let mut rows = vec![fact("Case", case)];
    rows.extend(phase_summary("Warmup", &value.warmup));
    rows.extend(phase_summary("Profiling", &value.profiling));
    rows.extend([
        fact(
            "Population slice reconciled",
            yes_no(value.population_slice_reconciled),
        ),
        fact("Sessions reconciled", yes_no(value.sessions_reconciled)),
        fact("Turn order reconciled", yes_no(value.turn_order_reconciled)),
        fact(
            "Inter-turn delays reconciled",
            yes_no(value.inter_turn_delays_reconciled),
        ),
        fact(
            "Native requests reconciled",
            value.native_requests_reconciled.map_or_else(
                || "unavailable".to_owned(),
                |reconciled| yes_no(reconciled).to_owned(),
            ),
        ),
        fact("Counts reconciled", yes_no(value.counts_reconciled)),
    ]);
    if !value.unavailable_dimensions.is_empty() {
        rows.push(fact(
            "Unavailable dimensions",
            value.unavailable_dimensions.join(", "),
        ));
    }
    if let Some(failure) = value
        .sessions
        .iter()
        .find(|session| session.status == ClientStatus::Failed)
    {
        rows.push(fact(
            "Failed session",
            format!(
                "{} · {} · {}",
                failure.runtime_session_id,
                failure
                    .failure_classification
                    .as_deref()
                    .unwrap_or("failed"),
                failure.diagnostic.as_deref().unwrap_or("no diagnostic")
            ),
        ));
    }
    FactSection {
        title: "LINEAR SESSION RESULT",
        rows,
    }
}

fn phase_summary(prefix: &str, value: &BenchSessionPhaseSummary) -> Vec<(String, String)> {
    vec![
        fact(
            format!("{prefix} sessions"),
            format!(
                "{} planned · {} started · {} succeeded · {} failed",
                value.planned_sessions,
                value.started_sessions,
                value.succeeded_sessions,
                value.failed_sessions
            ),
        ),
        fact(
            format!("{prefix} requests"),
            format!(
                "{} planned · {} attempted · {} completed · {} failed",
                value.planned_requests,
                value.attempted_requests,
                value.completed_requests,
                value.failed_requests
            ),
        ),
        fact(format!("{prefix} reconciled"), yes_no(value.reconciled)),
    ]
}

fn unavailable_count(value: Option<u64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), |count| count.to_string())
}

fn agentic_result(case: &str, value: &BenchAgenticResultEvidence) -> FactSection {
    let mut rows = vec![
        fact("Case", case),
        fact("Repository", value.source.repository.clone()),
        fact("Expected revision", value.source.expected_revision.clone()),
        fact(
            "Observed revision",
            optional_text(value.source.observed_revision.as_deref()),
        ),
        fact("Expected digest", value.source.expected_sha256.clone()),
        fact(
            "Observed digest",
            optional_text(value.source.observed_sha256.as_deref()),
        ),
        fact(
            "Acquisition",
            value.source.acquisition_outcome.map_or_else(
                || "—".to_owned(),
                |outcome| format!("{outcome:?}").to_lowercase(),
            ),
        ),
    ];
    if let Some(run) = value.run.as_deref() {
        rows.extend([
            fact("Native run", run.native_run_id.clone()),
            fact("Scenario", run.scenario.clone()),
            fact(
                "Submission",
                if run.submission_valid {
                    "valid".to_owned()
                } else {
                    format!("invalid · {}", run.submission_invalid_reasons.join(", "))
                },
            ),
            fact(
                "Warmup",
                format!(
                    "{} records · {} errors · {}",
                    run.warmup_records,
                    run.warmup_error_records,
                    if run.warmup_succeeded {
                        "succeeded"
                    } else {
                        "failed"
                    }
                ),
            ),
            fact(
                "Warmup source coordinates",
                unavailable_count(run.warmup_source_coordinate_records),
            ),
            fact("Profiling records", run.profiling_records.to_string()),
            fact(
                "Profiling source coordinates",
                unavailable_count(run.source_coordinate_records),
            ),
            fact(
                "Source traces",
                unavailable_count(run.distinct_source_traces),
            ),
            fact(
                "Runtime conversations",
                run.distinct_runtime_conversations.to_string(),
            ),
            fact(
                "Transport requests",
                run.distinct_transport_requests.to_string(),
            ),
            fact("Cache busts", unavailable_count(run.cache_bust_records)),
            fact("Context overflows", run.context_overflow_count.to_string()),
            fact("Ordinary failures", run.ordinary_failure_count.to_string()),
            fact(
                "Branches",
                format!(
                    "{} spawned · {} completed · {} errored · {} truncated · {} delayed",
                    run.branch_stats.children_spawned,
                    run.branch_stats.children_completed,
                    run.branch_stats.children_errored,
                    run.branch_stats.children_truncated,
                    run.branch_stats.children_delayed
                ),
            ),
            fact(
                "Parent / join",
                format!(
                    "{} suspended · {} resumed · {} child-failed · {} joins suppressed",
                    run.branch_stats.parents_suspended,
                    run.branch_stats.parents_resumed,
                    run.branch_stats.parents_failed_due_to_child_error,
                    run.branch_stats.joins_suppressed
                ),
            ),
            fact(
                "Unavailable dimensions",
                if run.unavailable_dimensions.is_empty() {
                    "none".to_owned()
                } else {
                    run.unavailable_dimensions.join(", ")
                },
            ),
        ]);
    } else {
        rows.push(fact("Native run", "not produced"));
    }
    FactSection {
        title: "AGENTIC REPLAY RESULT",
        rows,
    }
}

fn unavailable_case_source(title: &'static str, case: &str) -> FactSection {
    FactSection {
        title,
        rows: vec![
            fact("Case", case),
            fact("Evidence", "unavailable for this record schema"),
        ],
    }
}
