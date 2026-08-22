use super::bench_detail::{
    AgenticSourceProjection, CachePreparationProjection, CaptureProjection, CaseSloProjection,
    PopulationSliceProjection, RequestSourceProjection, SessionSourceProjection,
};
use super::{CaseView, LOG_TAIL_BYTES, RecordView, State};
use inferlab_protocol::{
    BenchAgenticResultEvidence, BenchPromptTokenReconciliation, BenchSessionResultEvidence,
    RawArtifact,
};
use inferlab_protocol::{BenchClientRequest, BenchLoadInput};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Default, Deserialize)]
struct RecordProjection {
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    definition_id: Option<String>,
    #[serde(default)]
    started_unix_ms: Option<u64>,
    #[serde(default)]
    finished_unix_ms: Option<u64>,
    #[serde(default)]
    passed: Option<bool>,
    #[serde(default)]
    skip_reason: Option<String>,
    #[serde(default)]
    capture: Option<serde_json::Value>,
    #[serde(default)]
    resolved: Option<ResolvedProjection>,
    #[serde(default)]
    process_evidence: BTreeMap<String, ProcessProjection>,
    #[serde(default)]
    server: Option<IdProjection>,
    #[serde(default)]
    evals: Option<Vec<IdProjection>>,
    #[serde(default)]
    benches: Option<Vec<IdProjection>>,
    #[serde(default)]
    assemblies: Option<serde_json::Value>,
    #[serde(default)]
    validations: Vec<ValidationProjection>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    errors: Vec<String>,
    #[serde(default)]
    failure: Option<FailureProjection>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    request_source: Option<serde_json::Value>,
    #[serde(default)]
    session_source: Option<serde_json::Value>,
    #[serde(default)]
    agentic_source: Option<serde_json::Value>,
    #[serde(default)]
    cases: Vec<CaseProjection>,
}

#[derive(Default, Deserialize)]
struct ResolvedProjection {
    #[serde(default)]
    workflow: Option<String>,
    #[serde(default)]
    recipe: Option<IdProjection>,
    #[serde(default)]
    server: Option<ServerProjection>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    measurements: Option<MeasurementProjection>,
    #[serde(default)]
    image: Option<IdProjection>,
    #[serde(default)]
    execution: Option<serde_json::Value>,
    #[serde(default)]
    bench: Option<BenchPlanProjection>,
    #[serde(default)]
    client: Option<ClientPlanProjection>,
}

#[derive(Default, Deserialize)]
struct ClientPlanProjection {
    #[serde(default)]
    effective_definition: Option<EffectiveBenchProjection>,
}

#[derive(Default, Deserialize)]
struct EffectiveBenchProjection {
    #[serde(default)]
    prompt: Option<PromptProjection>,
}

#[derive(Default, Deserialize)]
struct PromptProjection {
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    rendering_authority: Option<String>,
    #[serde(default)]
    declared: Option<serde_json::Value>,
}

#[derive(Default, Deserialize)]
struct BenchPlanProjection {
    #[serde(default)]
    execution: Option<serde_json::Value>,
    #[serde(default)]
    client: Option<ClientPlanProjection>,
}

#[derive(Deserialize)]
struct IdProjection {
    id: String,
}

#[derive(Deserialize)]
struct ServerProjection {
    id: String,
    #[serde(default)]
    case: Option<IdProjection>,
    #[serde(default)]
    topology: Option<inferlab_protocol::ServeTopology>,
}

#[derive(Default, Deserialize)]
struct ProcessProjection {
    #[serde(default)]
    stdout: Option<String>,
    #[serde(default)]
    stderr: Option<String>,
}

#[derive(Deserialize)]
struct FailureProjection {
    message: String,
}

#[derive(Default, Deserialize)]
struct CaseProjection {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    request: Option<PathBuf>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    stdout: Option<String>,
    #[serde(default)]
    stderr: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    metrics: Option<BTreeMap<String, f64>>,
    #[serde(default)]
    normalized_metrics: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    cache_preparation: Option<serde_json::Value>,
    #[serde(default)]
    slo: Option<serde_json::Value>,
    #[serde(default)]
    population_slice: Option<serde_json::Value>,
    #[serde(default)]
    completed_requests: Option<u64>,
    #[serde(default)]
    failed_requests: Option<u64>,
    #[serde(default)]
    normalization_schema: Option<String>,
    #[serde(default)]
    session: Option<serde_json::Value>,
    #[serde(default)]
    agentic: Option<serde_json::Value>,
    #[serde(default)]
    prompt_token_reconciliation: Vec<serde_json::Value>,
    #[serde(default)]
    raw_artifacts: Vec<serde_json::Value>,
}

#[derive(Default, Deserialize)]
struct MeasurementProjection {
    #[serde(default)]
    evals: Vec<IdProjection>,
    #[serde(default)]
    benches: Vec<IdProjection>,
}

#[derive(Default, Deserialize)]
struct ValidationProjection {
    #[serde(default)]
    outcome: serde_json::Value,
}

pub(super) struct RecordCollection {
    pub records: Vec<RecordView>,
    pub child_servers: Vec<RecordView>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RecordStamp {
    len: u64,
    modified: SystemTime,
}

#[derive(Clone)]
struct CachedRecord {
    stamp: RecordStamp,
    record: RecordView,
}

#[derive(Default)]
pub(super) struct RecordReader {
    finalized: HashMap<PathBuf, CachedRecord>,
    #[cfg(test)]
    body_reads: usize,
}

impl RecordReader {
    pub(super) fn read(&mut self, root: &Path, observed_unix_ms: u64) -> RecordCollection {
        let directory = root.join(crate::record::RECORDS_DIR);
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.finalized.clear();
                return RecordCollection {
                    records: Vec::new(),
                    child_servers: Vec::new(),
                    error: None,
                };
            }
            Err(error) => {
                return RecordCollection {
                    records: Vec::new(),
                    child_servers: Vec::new(),
                    error: Some(format!("failed to read {}: {error}", directory.display())),
                };
            }
        };
        let mut collection_error = None;
        let mut seen = HashSet::new();
        let mut records = Vec::new();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    collection_error = Some(format!(
                        "failed to enumerate {}: {error}",
                        directory.display()
                    ));
                    continue;
                }
            };
            let path = entry.path().join(crate::record::RECORD_FILE);
            seen.insert(path.clone());
            records.push(self.read_one(root, path, observed_unix_ms));
        }
        if collection_error.is_none() {
            self.finalized.retain(|path, _| seen.contains(path));
        }
        organize_records(records, collection_error)
    }

    fn read_one(&mut self, root: &Path, path: PathBuf, observed_unix_ms: u64) -> RecordView {
        let before = record_stamp(&path);
        if let (Some(stamp), Some(cached)) = (before.as_ref(), self.finalized.get(&path))
            && &cached.stamp == stamp
        {
            let mut record = cached.record.clone();
            record.state = State::Live;
            record.reason = None;
            record.observed_unix_ms = observed_unix_ms;
            record.last_success_unix_ms = Some(observed_unix_ms);
            return record;
        }

        #[cfg(test)]
        {
            self.body_reads = self.body_reads.saturating_add(1);
        }
        let record = read_record(root, path.clone(), observed_unix_ms);
        if record.state != State::Live {
            self.finalized.remove(&path);
            return record;
        }
        if record.finished_unix_ms.is_some()
            && let (Some(before), Some(after)) = (before, record_stamp(&path))
            && before == after
        {
            self.finalized.insert(
                path,
                CachedRecord {
                    stamp: after,
                    record: record.clone(),
                },
            );
        } else {
            self.finalized.remove(&path);
        }
        record
    }

    #[cfg(test)]
    fn body_reads(&self) -> usize {
        self.body_reads
    }
}

fn record_stamp(path: &Path) -> Option<RecordStamp> {
    let metadata = fs::metadata(path).ok()?;
    Some(RecordStamp {
        len: metadata.len(),
        modified: metadata.modified().ok()?,
    })
}

fn organize_records(
    mut records: Vec<RecordView>,
    collection_error: Option<String>,
) -> RecordCollection {
    let child_ids = records
        .iter()
        .flat_map(|record| record.child_refs.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let mut child_servers = records
        .iter()
        .filter(|record| {
            record.kind == "server" && record.id.as_ref().is_some_and(|id| child_ids.contains(id))
        })
        .cloned()
        .collect::<Vec<_>>();
    records.retain(|record| record.id.as_ref().is_none_or(|id| !child_ids.contains(id)));
    records.sort_by_key(|record| std::cmp::Reverse(record.started_unix_ms.unwrap_or(0)));
    child_servers.sort_by_key(|record| std::cmp::Reverse(record.started_unix_ms.unwrap_or(0)));
    RecordCollection {
        records,
        child_servers,
        error: collection_error,
    }
}

#[cfg(test)]
fn read_records(root: &Path, observed_unix_ms: u64) -> RecordCollection {
    RecordReader::default().read(root, observed_unix_ms)
}

fn read_record(root: &Path, path: PathBuf, observed_unix_ms: u64) -> RecordView {
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return unavailable_record(path, format!("read failed: {error}"), observed_unix_ms);
        }
    };
    let projection = match serde_json::from_slice::<RecordProjection>(&bytes) {
        Ok(projection) => projection,
        Err(error) => {
            return unavailable_record(
                path,
                format!("invalid record JSON: {error}"),
                observed_unix_ms,
            );
        }
    };
    let kind = if projection.assemblies.is_some() {
        "image"
    } else if projection.server.is_some()
        && projection.evals.is_some()
        && projection.benches.is_some()
    {
        "recipe"
    } else if !projection.process_evidence.is_empty() {
        "server"
    } else {
        projection
            .kind
            .as_deref()
            .or_else(|| {
                projection
                    .resolved
                    .as_ref()
                    .and_then(|resolved| resolved.kind.as_deref())
            })
            .unwrap_or("workload")
    };
    let mut definition_ids = projection.definition_id.into_iter().collect::<Vec<_>>();
    definition_ids.extend(
        projection
            .resolved
            .as_ref()
            .and_then(|resolved| resolved.recipe.as_ref().map(|recipe| recipe.id.clone())),
    );
    if let Some(resolved) = &projection.resolved {
        definition_ids.extend(resolved.server.iter().map(|server| server.id.clone()));
        definition_ids.extend(resolved.image.iter().map(|image| image.id.clone()));
        if let Some(measurements) = &resolved.measurements {
            definition_ids.extend(
                measurements
                    .evals
                    .iter()
                    .chain(&measurements.benches)
                    .map(|definition| definition.id.clone()),
            );
        }
    }
    definition_ids.sort();
    definition_ids.dedup();
    let case = projection
        .resolved
        .as_ref()
        .and_then(|resolved| resolved.server.as_ref())
        .and_then(|server| server.case.as_ref())
        .map(|case| case.id.clone());
    let workflow = projection
        .resolved
        .as_ref()
        .and_then(|resolved| resolved.workflow.clone());
    let topology = projection
        .resolved
        .as_ref()
        .and_then(|resolved| resolved.server.as_ref())
        .and_then(|server| server.topology.as_ref())
        .and_then(|topology| serde_json::to_string(topology).ok());
    let mut child_refs = projection
        .server
        .iter()
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    child_refs.extend(
        projection
            .evals
            .iter()
            .flatten()
            .chain(projection.benches.iter().flatten())
            .map(|record| record.id.clone()),
    );
    child_refs.extend(projection.validations.iter().filter_map(|validation| {
        validation
            .outcome
            .get("recipe_record_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    }));
    let mut seen_children = std::collections::BTreeSet::new();
    child_refs.retain(|id| seen_children.insert(id.clone()));
    let resolved_case_loads = resolved_case_loads(projection.resolved.as_ref());
    let request_source = projection
        .request_source
        .as_ref()
        .and_then(|value| serde_json::from_value::<RequestSourceProjection>(value.clone()).ok());
    let session_source = projection
        .session_source
        .as_ref()
        .and_then(|value| serde_json::from_value::<SessionSourceProjection>(value.clone()).ok());
    let agentic_source = projection
        .agentic_source
        .as_ref()
        .and_then(|value| serde_json::from_value::<AgenticSourceProjection>(value.clone()).ok());
    let request_source_known = request_source.is_some();
    let session_source_known = session_source.is_some();
    let agentic_source_known = agentic_source.is_some();
    let source_known = request_source_known || session_source_known || agentic_source_known;
    let mut bench_details = if kind == "bench" {
        let mut source = super::bench_detail::record_source(
            projection.schema_version,
            request_source.as_ref(),
            session_source.as_ref(),
            agentic_source.as_ref(),
        );
        if request_source.is_some()
            && let Some(prompt) = projection.resolved.as_ref().and_then(resolved_prompt)
        {
            source
                .rows
                .push(("Prompt authority".to_owned(), prompt_summary(prompt)));
        }
        vec![source]
    } else {
        Vec::new()
    };
    let mut artifact_refs = Vec::new();
    let prompt_authorities = projection
        .cases
        .iter()
        .flat_map(|case| case.normalized_metrics.values())
        .filter_map(|metric| {
            metric
                .get("prompt_authority")
                .and_then(|authority| authority.get("kind"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    let cases = projection
        .cases
        .into_iter()
        .map(|case| {
            let cache_preparation = case.cache_preparation.as_ref().and_then(|value| {
                serde_json::from_value::<CachePreparationProjection>(value.clone()).ok()
            });
            let slo = case
                .slo
                .as_ref()
                .and_then(|value| serde_json::from_value::<CaseSloProjection>(value.clone()).ok());
            let population_slice = case.population_slice.as_ref().and_then(|value| {
                serde_json::from_value::<PopulationSliceProjection>(value.clone()).ok()
            });
            let session_present = case.session.is_some();
            let session = case.session.as_ref().and_then(|value| {
                serde_json::from_value::<BenchSessionResultEvidence>(value.clone()).ok()
            });
            let agentic_present = case.agentic.is_some();
            let agentic = case.agentic.as_ref().and_then(|value| {
                serde_json::from_value::<BenchAgenticResultEvidence>(value.clone()).ok()
            });
            let prompt_token_reconciliation = case
                .prompt_token_reconciliation
                .iter()
                .filter_map(|value| {
                    serde_json::from_value::<BenchPromptTokenReconciliation>(value.clone()).ok()
                })
                .collect::<Vec<_>>();
            let raw_artifacts = case
                .raw_artifacts
                .iter()
                .filter_map(|value| serde_json::from_value::<RawArtifact>(value.clone()).ok())
                .collect::<Vec<_>>();
            let request_evidence_unavailable = request_source_known && population_slice.is_none();
            let has_bench_evidence = source_known
                || case.cache_preparation.is_some()
                || case.slo.is_some()
                || case.population_slice.is_some()
                || case.completed_requests.is_some()
                || case.failed_requests.is_some()
                || case.normalization_schema.is_some()
                || case.session.is_some()
                || case.agentic.is_some()
                || !case.prompt_token_reconciliation.is_empty()
                || !case.raw_artifacts.is_empty();
            if kind == "bench" && has_bench_evidence {
                let (sections, artifacts) =
                    super::bench_detail::case_evidence(super::bench_detail::CaseEvidence {
                        id: case.id.as_deref(),
                        cache_preparation: cache_preparation.as_ref(),
                        slo: slo.as_ref(),
                        population_slice: population_slice.as_ref(),
                        completed_requests: case.completed_requests,
                        failed_requests: case.failed_requests,
                        normalization_schema: case.normalization_schema.as_deref(),
                        request_unavailable: request_evidence_unavailable,
                        session: session.as_ref(),
                        session_unavailable: session_source_known && !session_present
                            || session_present && session.is_none(),
                        agentic: agentic.as_ref(),
                        agentic_unavailable: agentic_source_known && !agentic_present
                            || agentic_present && agentic.is_none(),
                        prompt_token_reconciliation: &prompt_token_reconciliation,
                        raw_artifacts: &raw_artifacts,
                    });
                bench_details.extend(sections);
                artifact_refs.extend(artifacts);
            }
            let resolved_load = case
                .id
                .as_ref()
                .and_then(|id| resolved_case_loads.get(id))
                .cloned();
            CaseView {
                id: case.id,
                load: read_case_load(root, case.request.as_deref())
                    .or(resolved_load)
                    .unwrap_or(super::CaseLoad::Unknown),
                status: case.status,
                stdout: case.stdout,
                stderr: case.stderr,
                error: case.error,
                metrics: case.metrics.unwrap_or_default(),
            }
        })
        .collect::<Vec<_>>();
    let error = projection
        .error
        .or_else(|| (!projection.errors.is_empty()).then(|| projection.errors.join("; ")))
        .or_else(|| projection.failure.map(|failure| failure.message));
    let mut outcome_facts = Vec::new();
    if kind == "eval"
        && let Some(authority) = prompt_authorities.first()
    {
        // A metric scored under another authority is a different measurement,
        // so the authority is shown wherever the values are.
        outcome_facts.push(("Prompt authority".to_owned(), authority.clone()));
    }
    if kind == "bench" {
        if let Some(passed) = projection.passed {
            outcome_facts.push((
                "Passed".to_owned(),
                if passed { "yes" } else { "no" }.to_owned(),
            ));
        }
        if let Some(reason) = projection.skip_reason.as_deref() {
            outcome_facts.push(("Skip reason".to_owned(), reason.to_owned()));
        }
        if let Some(capture) = projection
            .capture
            .as_ref()
            .and_then(|value| serde_json::from_value::<CaptureProjection>(value.clone()).ok())
        {
            outcome_facts.push((
                "Capture".to_owned(),
                super::bench_detail::capture_summary(&capture),
            ));
        } else if projection.capture.is_some() {
            outcome_facts.push((
                "Capture".to_owned(),
                "unavailable for this record schema".to_owned(),
            ));
        }
    }
    let mut log_refs = projection
        .process_evidence
        .into_values()
        .flat_map(|process| process.stdout.into_iter().chain(process.stderr))
        .collect::<Vec<_>>();
    log_refs.extend(
        cases
            .iter()
            .flat_map(|case| case.stdout.iter().chain(case.stderr.iter()).cloned()),
    );
    let mut seen_logs = std::collections::BTreeSet::new();
    log_refs.retain(|reference| seen_logs.insert(reference.clone()));
    RecordView {
        path,
        state: State::Live,
        reason: None,
        id: projection.id,
        kind: kind.to_owned(),
        status: projection.status,
        definition_ids,
        case,
        workflow,
        error,
        started_unix_ms: projection.started_unix_ms,
        finished_unix_ms: projection.finished_unix_ms,
        log_refs,
        observed_unix_ms,
        last_success_unix_ms: Some(observed_unix_ms),
        child_refs,
        topology,
        cases,
        outcome_facts,
        bench_details,
        artifact_refs,
        process_observation: None,
    }
}

fn prompt_summary(prompt: &PromptProjection) -> String {
    let kind = prompt.kind.as_deref().unwrap_or("unknown");
    let authority = prompt.rendering_authority.as_deref().unwrap_or("unknown");
    let provenance = if prompt.declared.is_some() {
        "declared"
    } else {
        "defaulted"
    };
    format!("{kind} · {authority} · {provenance}")
}

fn resolved_prompt(resolved: &ResolvedProjection) -> Option<&PromptProjection> {
    resolved
        .client
        .as_ref()
        .or_else(|| {
            resolved
                .bench
                .as_ref()
                .and_then(|bench| bench.client.as_ref())
        })
        .and_then(|client| client.effective_definition.as_ref())
        .and_then(|definition| definition.prompt.as_ref())
}

fn read_case_load(root: &Path, reference: Option<&Path>) -> Option<super::CaseLoad> {
    let reference = reference?;
    let path = if reference.is_absolute() {
        reference.to_path_buf()
    } else {
        root.join(reference)
    };
    let Ok(bytes) = fs::read(path) else {
        return None;
    };
    let Ok(request) = serde_json::from_slice::<BenchClientRequest>(&bytes) else {
        return None;
    };
    Some(match request.case.load_shape {
        BenchLoadInput::ConcurrencyLimited { concurrency } => {
            super::CaseLoad::Concurrency(concurrency)
        }
        BenchLoadInput::RequestRateLimited { request_rate, .. } => {
            super::CaseLoad::RequestRate(request_rate)
        }
        BenchLoadInput::UnboundedRequestRate => super::CaseLoad::UnboundedRequestRate,
    })
}

fn resolved_case_loads(resolved: Option<&ResolvedProjection>) -> BTreeMap<String, super::CaseLoad> {
    let execution = resolved.and_then(|resolved| {
        resolved
            .bench
            .as_ref()
            .and_then(|bench| bench.execution.as_ref())
            .or(resolved.execution.as_ref())
    });
    let Some(execution) = execution else {
        return BTreeMap::new();
    };
    let Ok(execution) =
        serde_json::from_value::<crate::workload::BenchExecutionPlan>(execution.clone())
    else {
        return BTreeMap::new();
    };
    match execution {
        crate::workload::BenchExecutionPlan::Matrix { cases } => cases
            .into_iter()
            .map(|case| (case.id, case_load(case.load_shape)))
            .collect(),
        crate::workload::BenchExecutionPlan::Adaptive { .. } => BTreeMap::new(),
    }
}

fn case_load(load: crate::workload::LoadShape) -> super::CaseLoad {
    match load {
        crate::workload::LoadShape::ConcurrencyLimited { concurrency } => {
            super::CaseLoad::Concurrency(concurrency)
        }
        crate::workload::LoadShape::RequestRateLimited { request_rate, .. } => match request_rate {
            crate::workspace::RequestRate::Finite(rate) => super::CaseLoad::RequestRate(rate),
            crate::workspace::RequestRate::Unbounded => super::CaseLoad::UnboundedRequestRate,
        },
    }
}

fn unavailable_record(path: PathBuf, reason: String, observed_unix_ms: u64) -> RecordView {
    RecordView {
        path,
        state: State::Unavailable,
        reason: Some(reason),
        id: None,
        kind: "record".to_owned(),
        status: None,
        definition_ids: Vec::new(),
        case: None,
        workflow: None,
        error: None,
        started_unix_ms: None,
        finished_unix_ms: None,
        log_refs: Vec::new(),
        observed_unix_ms,
        last_success_unix_ms: None,
        child_refs: Vec::new(),
        topology: None,
        cases: Vec::new(),
        outcome_facts: Vec::new(),
        bench_details: Vec::new(),
        artifact_refs: Vec::new(),
        process_observation: None,
    }
}

pub(super) fn read_log_tail(root: &Path, reference: &str) -> String {
    let reference = Path::new(reference);
    let path = if reference.is_absolute() {
        reference.to_path_buf()
    } else {
        root.join(reference)
    };
    let mut file = match fs::File::open(&path) {
        Ok(file) => file,
        Err(error) => return format!("[unavailable] {}: {error}", path.display()),
    };
    let length = file.metadata().map_or(0, |metadata| metadata.len());
    let start = length.saturating_sub(LOG_TAIL_BYTES);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return format!("[unavailable] could not seek {}", path.display());
    }
    let mut bytes = Vec::new();
    if file.take(LOG_TAIL_BYTES).read_to_end(&mut bytes).is_err() {
        return format!("[unavailable] could not read {}", path.display());
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::{RecordReader, State, read_log_tail, read_record, read_records};
    use crate::tui::CaseLoad;

    #[test]
    fn an_eval_record_surfaces_the_prompt_authority_that_produced_its_metric()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(
            &path,
            br#"{"id":"e1","kind":"eval","status":"succeeded","started_unix_ms":1,"definition_id":"gsm8k","cases":[{"id":"trial-1","status":"succeeded","metrics":{"gsm8k:exact_match,strict-match":0.91},"normalized_metrics":{"gsm8k:exact_match,strict-match":{"source_identity":"gsm8k","metric":"exact_match","filter":"strict-match","native_metric_key":"exact_match,strict-match","value":0.91,"higher_is_better":true,"prompt_authority":{"kind":"flat"}}}}]}"#,
        )?;

        let record = read_record(root.path(), path, 42);

        assert!(
            record
                .outcome_facts
                .contains(&("Prompt authority".to_owned(), "flat".to_owned())),
            "{:?}",
            record.outcome_facts
        );
        Ok(())
    }

    #[test]
    fn projection_extracts_typed_search_fields_and_log_references()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(&path, br#"{"id":"r1","kind":"eval","status":"running","started_unix_ms":12,"finished_unix_ms":40,"definition_id":"quality","resolved":{"workflow":"serve_start","recipe":{"id":"recipe-def"},"server":{"id":"qwen","case":{"id":"long"}},"measurements":{"evals":[{"id":"eval-def"}],"benches":[{"id":"bench-def"}]}},"process_evidence":{"worker":{"stdout":"out.log","stderr":"err.log"}},"cases":[{"id":"trial-1","status":"failed","stdout":"case.out","stderr":"case.err","error":"bad answer","metrics":{"pass":0.0}}]}"#)?;
        let record = read_record(root.path(), path, 42);
        assert_eq!(record.state, State::Live);
        assert_eq!(record.kind, "server");
        assert_eq!(
            record.definition_ids,
            ["bench-def", "eval-def", "quality", "qwen", "recipe-def"]
        );
        assert_eq!(record.case.as_deref(), Some("long"));
        assert_eq!(record.finished_unix_ms, Some(40));
        assert_eq!(
            record.log_refs,
            ["out.log", "err.log", "case.out", "case.err"]
        );
        assert_eq!(record.cases[0].id.as_deref(), Some("trial-1"));
        assert_eq!(record.cases[0].metrics.get("pass"), Some(&0.0));
        Ok(())
    }

    #[test]
    fn bench_projection_preserves_source_outcome_and_raw_artifact_evidence()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(
            &path,
            r#"{
  "schema_version": 11,
  "id": "bench-1",
  "kind": "bench",
  "status": "failed",
  "definition_id": "shared-prefix",
  "started_unix_ms": 1,
  "finished_unix_ms": 2,
  "passed": false,
  "capture": {"status":"succeeded","plan":null,"arm":[],"windows":[],"finalization":[],"reports":[],"error":null},
  "resolved": {"client":{"effective_definition":{"prompt":{"kind":"server_chat","rendering_authority":"server","declared":{"kind":"server_chat"}}}}},
  "request_source": {
    "kind": "random",
    "input_tokens": {"kind":"inclusive_uniform","min":512,"max":1024},
    "output_tokens": 128,
    "prefix_sharing": {"shared_prefix_ratio":0.5},
    "shared_system_content": null
  },
  "cases": [{
    "id": "c1",
    "status": "failed",
    "request": "request.json",
    "metrics": {},
    "population_slice": {"kind":"requests","population_sha256":"abc","warmup_start":0,"warmup_count":1,"profiling_start":1,"profiling_count":4},
    "completed_requests": 3,
    "failed_requests": 1,
    "normalization_schema": "aiperf-summary-v1",
    "cache_preparation": {"start":"cold","transitions":[],"reset":{"method":"post","url":"http://server/flush_cache","succeeded":true,"http_status":200,"elapsed_ms":2,"error":null}},
    "prompt_token_reconciliation": [{"population_index":1,"native_session_num":2,"planned_prompt_tokens":512,"observed_prompt_tokens":512,"reconciled":true}],
    "raw_artifacts": [{"name":"requests","kind":"jsonl","path":"cases/c1/requests.jsonl"}]
  }]
}"#,
        )?;

        let record = read_record(root.path(), path, 3);

        assert_eq!(
            record.outcome_facts,
            [
                ("Passed".to_owned(), "no".to_owned()),
                (
                    "Capture".to_owned(),
                    "succeeded · 0 window(s) · 0 report(s)".to_owned()
                )
            ]
        );
        let source = &record.bench_details[0];
        assert_eq!(source.title, "RECORDED SOURCE · REQUESTS");
        assert!(
            source
                .rows
                .iter()
                .any(|(label, value)| { label == "Prefix sharing" && value == "50%" })
        );
        assert!(source.rows.iter().any(|(label, value)| {
            label == "Prompt authority" && value == "server_chat · server · declared"
        }));
        assert!(record.bench_details.iter().any(|section| {
            section.rows.iter().any(|(label, value)| {
                label == "Prompt-token reconciliation" && value == "1/1 reconciled"
            })
        }));
        assert_eq!(
            record.artifact_refs,
            ["requests · jsonl · cases/c1/requests.jsonl"]
        );
        Ok(())
    }

    #[test]
    fn historical_bench_without_tagged_source_stays_visible_without_inference()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(
            &path,
            r#"{"schema_version":7,"id":"old","kind":"bench","status":"succeeded","started_unix_ms":1,"cases":[]}"#,
        )?;

        let record = read_record(root.path(), path, 2);

        assert_eq!(record.state, State::Live);
        assert_eq!(record.bench_details[0].title, "RECORDED SOURCE");
        assert_eq!(
            record.bench_details[0].rows[0].1,
            "unavailable for this record schema"
        );
        Ok(())
    }

    #[test]
    fn dataset_source_summary_reads_new_preparation_reference_and_historical_acquisition()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let current = root.path().join("current.json");
        let historical = root.path().join("historical.json");
        let future = root.path().join("future.json");
        let catalog = r#"{"dataset":"speed_bench","profile":"default","upstream_identity":"release","sha256":"abc"}"#;
        std::fs::write(
            &current,
            format!(
                r#"{{"schema_version":19,"id":"current","kind":"bench","status":"succeeded","started_unix_ms":1,"request_source":{{"kind":"dataset","catalog":{catalog},"preparation_attempt_id":"data-asset-1"}},"cases":[]}}"#,
            ),
        )?;
        std::fs::write(
            &historical,
            format!(
                r#"{{"schema_version":12,"id":"historical","kind":"bench","status":"succeeded","started_unix_ms":1,"request_source":{{"kind":"dataset","catalog":{catalog},"preparation_attempt_id":"must-not-be-read","acquisition":{{"outcome":"reused"}}}},"cases":[]}}"#,
            ),
        )?;
        std::fs::write(
            &future,
            format!(
                r#"{{"schema_version":20,"id":"future","kind":"bench","status":"succeeded","started_unix_ms":1,"request_source":{{"kind":"dataset","catalog":{catalog},"preparation_attempt_id":"must-not-be-read"}},"cases":[]}}"#,
            ),
        )?;

        let current = read_record(root.path(), current, 2);
        let historical = read_record(root.path(), historical, 2);
        let future = read_record(root.path(), future, 2);
        assert!(
            current.bench_details[0]
                .rows
                .iter()
                .any(|(label, value)| { label == "Source preparation" && value == "data-asset-1" })
        );
        assert!(
            historical.bench_details[0]
                .rows
                .iter()
                .any(|(label, value)| { label == "Source preparation" && value == "reused" })
        );
        assert!(future.bench_details[0].rows.iter().any(|(label, value)| {
            label == "Source preparation" && value == "unavailable for unsupported record schema"
        }));
        Ok(())
    }

    #[test]
    fn unsupported_tagged_source_evidence_does_not_hide_the_record()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(
            &path,
            r#"{"schema_version":8,"id":"old-session","kind":"bench","status":"succeeded","started_unix_ms":1,"session_source":{"catalog":{"dataset":"legacy"}},"cases":[]}"#,
        )?;

        let record = read_record(root.path(), path, 2);

        assert_eq!(record.state, State::Live);
        assert_eq!(record.bench_details[0].title, "RECORDED SOURCE");
        assert_eq!(
            record.bench_details[0].rows[0].1,
            "unavailable for this record schema"
        );
        Ok(())
    }

    #[test]
    fn unsupported_case_evidence_degrades_only_its_source_specific_detail()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(
            &path,
            r#"{"schema_version":9,"id":"future","kind":"bench","status":"succeeded","started_unix_ms":1,"cases":[{"id":"session","status":"failed","session":{"future_field":true}},{"id":"agentic","status":"failed","agentic":{"future_field":true}}]}"#,
        )?;

        let record = read_record(root.path(), path, 2);

        assert_eq!(record.state, State::Live);
        for title in ["LINEAR SESSION RESULT", "AGENTIC REPLAY RESULT"] {
            let section = record
                .bench_details
                .iter()
                .find(|section| section.title == title)
                .ok_or("missing degraded source-specific detail")?;
            assert!(section.rows.iter().any(|(label, value)| {
                label == "Evidence" && value == "unavailable for this record schema"
            }));
        }
        Ok(())
    }

    #[test]
    fn generic_request_case_facts_do_not_stand_in_for_recorded_population()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(
            &path,
            r#"{"schema_version":8,"id":"old-request","kind":"bench","status":"succeeded","started_unix_ms":1,"request_source":{"kind":"random","input_tokens":128,"output_tokens":32},"cases":[{"id":"c1","status":"succeeded","completed_requests":1,"normalization_schema":"legacy","raw_artifacts":[{"name":"summary","kind":"json","path":"summary.json"}]}]}"#,
        )?;

        let record = read_record(root.path(), path, 2);

        assert_eq!(record.state, State::Live);
        let section = record
            .bench_details
            .iter()
            .find(|section| section.title == "REQUEST RESULT")
            .ok_or("missing unavailable request population detail")?;
        assert!(section.rows.iter().any(|(label, value)| {
            label == "Evidence" && value == "unavailable for this record schema"
        }));
        assert!(record.bench_details.iter().any(|section| {
            section
                .rows
                .iter()
                .any(|(label, value)| label == "Completed requests" && value == "1")
        }));
        Ok(())
    }

    #[test]
    fn recipe_children_stay_under_their_explicit_parent() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = tempfile::tempdir()?;
        let records = root.path().join(".inferlab/records");
        for (id, json) in [
            (
                "recipe-1",
                r#"{"id":"recipe-1","status":"running","started_unix_ms":3,"server":{"id":"serve-1"},"evals":[{"id":"eval-1"}],"benches":[]}"#,
            ),
            (
                "serve-1",
                r#"{"id":"serve-1","status":"running","started_unix_ms":2,"process_evidence":{"p":{"stdout":"out","stderr":"err"}}}"#,
            ),
            (
                "eval-1",
                r#"{"id":"eval-1","kind":"eval","status":"succeeded","started_unix_ms":1,"definition_id":"quality"}"#,
            ),
        ] {
            let directory = records.join(id);
            std::fs::create_dir_all(&directory)?;
            std::fs::write(directory.join("record.json"), json)?;
        }
        let collection = read_records(root.path(), 10);
        assert!(collection.error.is_none());
        assert_eq!(collection.records.len(), 1);
        assert_eq!(collection.records[0].id.as_deref(), Some("recipe-1"));
        assert_eq!(collection.records[0].child_refs, ["serve-1", "eval-1"]);
        assert_eq!(collection.child_servers.len(), 1);
        assert_eq!(collection.child_servers[0].id.as_deref(), Some("serve-1"));
        Ok(())
    }

    #[test]
    fn referenced_logs_resolve_from_the_workspace_root() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let relative = ".inferlab/records/r1/cases/c1/stderr.log";
        let path = root.path().join(relative);
        std::fs::create_dir_all(path.parent().ok_or("missing log parent")?)?;
        std::fs::write(&path, "workspace log")?;

        assert_eq!(read_log_tail(root.path(), relative), "workspace log");
        Ok(())
    }

    #[test]
    fn image_validation_keeps_its_explicit_recipe_record_child()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(
            &path,
            br#"{"id":"image-1","status":"succeeded","started_unix_ms":1,"resolved":{"image":{"id":"runtime-image"}},"assemblies":[],"validations":[{"outcome":{"kind":"validated","recipe_record_id":"recipe-1"}}]}"#,
        )?;

        let record = read_record(root.path(), path, 2);

        assert_eq!(record.kind, "image");
        assert_eq!(record.child_refs, ["recipe-1"]);
        assert_eq!(record.definition_ids, ["runtime-image"]);
        Ok(())
    }

    #[test]
    fn case_loads_come_from_typed_record_references_instead_of_case_ids()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let record_dir = root.path().join(".inferlab/records/bench-1");
        let concurrency_dir = record_dir.join("cases/arbitrary-a");
        let rate_dir = record_dir.join("cases/arbitrary-b");
        std::fs::create_dir_all(&concurrency_dir)?;
        std::fs::create_dir_all(&rate_dir)?;
        let request = |load_shape: &str, artifact_dir: &std::path::Path| {
            format!(
                r#"{{"protocol_version":"8","endpoint":{{"protocol":"http","host":"127.0.0.1","port":8000,"completions_path":"/v1/completions","chat_completions_path":"/v1/chat/completions","server_metrics":null}},"model":{{"locator":"/models/test","served_name":"test"}},"definition":{{"request_source":{{"kind":"random","input_tokens":8,"output_tokens":1,"prefix_sharing":null,"shared_system_content":null}},"prompt":{{"kind":"server_chat","request_representation":"structured_messages","route":"chat_completions","rendering_authority":"server"}},"server_metrics":false,"seed":7,"request_body":{{}},"request_slo":null,"timeout_seconds":120,"cache_start":"uncontrolled"}},"case":{{"load_shape":{load_shape},"request_count":4,"warmup_request_count":0}},"case_budget_seconds":120.0,"artifact_dir":{}}}"#,
                serde_json::to_string(artifact_dir).unwrap_or_else(|_| "\"artifacts\"".to_owned())
            )
        };
        std::fs::write(
            concurrency_dir.join("request.json"),
            request(
                r#"{"kind":"concurrency_limited","concurrency":8}"#,
                &concurrency_dir.join("artifacts"),
            ),
        )?;
        std::fs::write(
            rate_dir.join("request.json"),
            request(
                r#"{"kind":"request_rate_limited","request_rate":3.5,"burstiness":null}"#,
                &rate_dir.join("artifacts"),
            ),
        )?;
        std::fs::write(
            record_dir.join("record.json"),
            r#"{"id":"bench-1","kind":"bench","status":"succeeded","started_unix_ms":1,"definition_id":"load","cases":[{"id":"arbitrary-a","status":"succeeded","request":".inferlab/records/bench-1/cases/arbitrary-a/request.json","result":"result-a.json","metrics":{"request_throughput":7.0}},{"id":"arbitrary-b","status":"succeeded","request":".inferlab/records/bench-1/cases/arbitrary-b/request.json","result":"result-b.json","metrics":{"request_throughput":3.0}}]}"#,
        )?;

        let collection = read_records(root.path(), 10);

        assert!(collection.error.is_none());
        assert_eq!(collection.records.len(), 1);
        assert_eq!(
            collection.records[0].cases[0].load,
            CaseLoad::Concurrency(8)
        );
        assert_eq!(
            collection.records[0].cases[1].load,
            CaseLoad::RequestRate(3.5)
        );
        Ok(())
    }

    #[test]
    fn static_case_uses_the_frozen_matrix_when_request_evidence_is_not_yet_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let record_dir = root.path().join(".inferlab/records/bench-1");
        std::fs::create_dir_all(&record_dir)?;
        std::fs::write(
            record_dir.join("record.json"),
            r#"{"id":"bench-1","kind":"bench","status":"failed","started_unix_ms":1,"definition_id":"load","resolved":{"execution":{"mode":"matrix","cases":[{"id":"opaque-static-id","load_shape":{"kind":"concurrency-limited","concurrency":16},"request_count":4,"warmup_request_count":0}]}},"cases":[{"id":"opaque-static-id","status":"failed","request":".inferlab/records/bench-1/cases/opaque-static-id/request.json","result":"result.json","metrics":{}}]}"#,
        )?;

        let collection = read_records(root.path(), 10);

        assert_eq!(
            collection.records[0].cases[0].load,
            CaseLoad::Concurrency(16)
        );
        Ok(())
    }

    #[test]
    fn unchanged_finalized_records_reuse_the_parsed_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let record_dir = root.path().join(".inferlab/records/record-1");
        std::fs::create_dir_all(&record_dir)?;
        std::fs::write(
            record_dir.join("record.json"),
            r#"{"id":"record-1","kind":"bench","status":"succeeded","started_unix_ms":1,"finished_unix_ms":2}"#,
        )?;
        let mut reader = RecordReader::default();

        let first = reader.read(root.path(), 10);
        let second = reader.read(root.path(), 20);

        assert_eq!(reader.body_reads(), 1);
        assert_eq!(first.records[0].observed_unix_ms, 10);
        assert_eq!(second.records[0].observed_unix_ms, 20);
        assert_eq!(second.records[0].last_success_unix_ms, Some(20));
        Ok(())
    }

    #[test]
    fn non_finalized_records_are_reread_on_every_refresh() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = tempfile::tempdir()?;
        let record_dir = root.path().join(".inferlab/records/record-1");
        std::fs::create_dir_all(&record_dir)?;
        let path = record_dir.join("record.json");
        std::fs::write(
            &path,
            r#"{"id":"record-1","kind":"bench","status":"running","started_unix_ms":1}"#,
        )?;
        let mut reader = RecordReader::default();

        let _ = reader.read(root.path(), 10);
        std::fs::write(
            &path,
            r#"{"id":"record-1","kind":"bench","status":"running","started_unix_ms":1,"error":"new evidence"}"#,
        )?;
        let second = reader.read(root.path(), 20);

        assert_eq!(reader.body_reads(), 2);
        assert_eq!(second.records[0].error.as_deref(), Some("new evidence"));
        Ok(())
    }

    #[test]
    fn changed_finalized_record_invalidates_only_its_cached_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let records = root.path().join(".inferlab/records");
        for id in ["record-1", "record-2"] {
            let record_dir = records.join(id);
            std::fs::create_dir_all(&record_dir)?;
            std::fs::write(
                record_dir.join("record.json"),
                format!(
                    r#"{{"id":"{id}","kind":"bench","status":"succeeded","started_unix_ms":1,"finished_unix_ms":2}}"#,
                ),
            )?;
        }
        let mut reader = RecordReader::default();

        let _ = reader.read(root.path(), 10);
        std::fs::write(
            records.join("record-1/record.json"),
            r#"{"id":"record-1","kind":"bench","status":"failed","started_unix_ms":1,"finished_unix_ms":2,"error":"changed"}"#,
        )?;
        let second = reader.read(root.path(), 20);

        assert_eq!(reader.body_reads(), 3);
        assert!(second.records.iter().any(|record| {
            record.id.as_deref() == Some("record-1")
                && record.status.as_deref() == Some("failed")
                && record.error.as_deref() == Some("changed")
        }));
        Ok(())
    }
}
