//! Dataset acquisition and tokenizer-dependent request-population preparation.

use super::client::{accept_client_result, run_client, wait_for_interrupt};
use super::{
    BenchDatasetRequestSourceEvidence, BenchExecutionPlan, BenchPlan, BenchPopulation,
    BenchPopulationPreparationEvidence, BenchRequestSourceEvidence, BenchSessionSourceEvidence,
    BenchSessionTemplate, DatasetAcquisitionEvidence, DatasetAcquisitionOutcome,
    ResolvedBenchRequestSource, ResolvedBenchSource, SYNTHETIC_MATERIALIZATION_IDENTITY,
    WorkloadRecordSession,
};
use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::workload::plan::session_population_layout;
use crate::workload::wire;
use crate::workspace::BenchPrompt;
use inferlab_protocol::{
    BenchPopulationPreparationRequest, BenchPopulationPreparationResult, ClientStatus,
    ProtocolVersion,
};
use inferlab_runtime::operation_bound::OperationBound;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

pub(super) fn prepare_bench_request_source(
    plan: &mut BenchPlan,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<(), InferlabError> {
    let source = plan.client.effective_definition.source.clone();
    match source {
        ResolvedBenchSource::Requests { request_source } => match request_source {
            ResolvedBenchRequestSource::Random {
                input_tokens,
                output_tokens,
                prefix_sharing,
                shared_system_content,
            } => {
                let preparation = run_population_preparation(plan, session, progress, None)?;
                session.set_bench_request_source(BenchRequestSourceEvidence::Random {
                    input_tokens,
                    output_tokens,
                    prefix_sharing,
                    shared_system_content,
                    preparation: Some(preparation.0.clone()),
                })?;
                finish_population_preparation(
                    plan,
                    SYNTHETIC_MATERIALIZATION_IDENTITY,
                    &preparation.0,
                    preparation.1,
                )?;
                Ok(())
            }
            ResolvedBenchRequestSource::RandomMixture {
                shapes,
                total_weight,
                prefix_sharing,
            } => {
                let preparation = run_population_preparation(plan, session, progress, None)?;
                session.set_bench_request_source(BenchRequestSourceEvidence::RandomMixture {
                    shapes,
                    total_weight,
                    prefix_sharing,
                    preparation: Some(preparation.0.clone()),
                })?;
                finish_population_preparation(
                    plan,
                    SYNTHETIC_MATERIALIZATION_IDENTITY,
                    &preparation.0,
                    preparation.1,
                )?;
                Ok(())
            }
            ResolvedBenchRequestSource::Dataset { catalog, .. } => {
                let phase = if catalog.cache_path.is_file() {
                    "dataset snapshot verification"
                } else {
                    "dataset snapshot download"
                };
                progress.phase(Phase::named(phase).current_item(&catalog.upstream_identity))?;
                let acquisition = match acquire_dataset_snapshot(
                    &catalog.cache_path,
                    &catalog.url,
                    &catalog.sha256,
                ) {
                    Ok(evidence) => evidence,
                    Err(failure) => {
                        let (evidence, error) = *failure;
                        session.set_bench_request_source(BenchRequestSourceEvidence::Dataset(
                            Box::new(BenchDatasetRequestSourceEvidence {
                                catalog: *catalog,
                                acquisition: evidence,
                                preparation: None,
                                preparation_process: None,
                                preparation_request: None,
                                preparation_result: None,
                                preparation_stdout: None,
                                preparation_stderr: None,
                            }),
                        ))?;
                        return Err(error);
                    }
                };
                let (preparation, decode_error) = run_population_preparation(
                    plan,
                    session,
                    progress,
                    Some(catalog.cache_path.clone()),
                )?;
                session.set_bench_request_source(BenchRequestSourceEvidence::Dataset(Box::new(
                    BenchDatasetRequestSourceEvidence {
                        catalog: (*catalog).clone(),
                        acquisition,
                        preparation: preparation.result.clone(),
                        preparation_process: preparation.process.clone(),
                        preparation_request: Some(preparation.request.clone()),
                        preparation_result: Some(preparation.result_path.clone()),
                        preparation_stdout: Some(preparation.stdout.clone()),
                        preparation_stderr: Some(preparation.stderr.clone()),
                    },
                )))?;
                finish_population_preparation(
                    plan,
                    &catalog.materialization_identity,
                    &preparation,
                    decode_error,
                )?;
                Ok(())
            }
        },
        ResolvedBenchSource::Sessions { session_source } => {
            let catalog = session_source.catalog;
            let phase = if catalog.cache_path.is_file() {
                "dataset snapshot verification"
            } else {
                "dataset snapshot download"
            };
            progress.phase(Phase::named(phase).current_item(&catalog.upstream_identity))?;
            let acquisition = match acquire_dataset_snapshot(
                &catalog.cache_path,
                &catalog.url,
                &catalog.sha256,
            ) {
                Ok(evidence) => evidence,
                Err(failure) => {
                    let (evidence, error) = *failure;
                    session.set_bench_session_source(BenchSessionSourceEvidence {
                        catalog: *catalog,
                        acquisition: evidence,
                        preparation: None,
                        preparation_process: None,
                        preparation_request: None,
                        preparation_result: None,
                        preparation_stdout: None,
                        preparation_stderr: None,
                    })?;
                    return Err(error);
                }
            };
            let (preparation, decode_error) = run_population_preparation(
                plan,
                session,
                progress,
                Some(catalog.cache_path.clone()),
            )?;
            session.set_bench_session_source(BenchSessionSourceEvidence {
                catalog: (*catalog).clone(),
                acquisition,
                preparation: preparation.result.clone(),
                preparation_process: preparation.process.clone(),
                preparation_request: Some(preparation.request.clone()),
                preparation_result: Some(preparation.result_path.clone()),
                preparation_stdout: Some(preparation.stdout.clone()),
                preparation_stderr: Some(preparation.stderr.clone()),
            })?;
            finish_population_preparation(
                plan,
                &catalog.materialization_identity,
                &preparation,
                decode_error,
            )
        }
    }
}

pub(super) fn run_population_preparation(
    plan: &BenchPlan,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
    source_path: Option<PathBuf>,
) -> Result<(BenchPopulationPreparationEvidence, Option<String>), InferlabError> {
    let paths = session.case_paths("request-source")?;
    progress.phase(
        Phase::named("request population materialization")
            .current_item(&plan.id)
            .log(session.absolute(&paths.stderr)),
    )?;
    let (request_source, session_source) =
        wire::bench_source_inputs(&plan.client.effective_definition)?;
    let request = BenchPopulationPreparationRequest {
        protocol_version: ProtocolVersion::V7,
        model: wire::model_input(&plan.client.model),
        tokenizer_backend: plan.client.tokenizer_backend.clone(),
        transformers_version: plan.client.toolchain.transformers_version.clone(),
        request_source,
        session_source,
        prompt: wire::prompt_input(&plan.client.effective_definition.prompt)?,
        source_path,
        required_entries: plan.client.required_population_count,
        seed: plan.client.effective_definition.seed,
        request_body: wire::bench_request_body_input(&plan.client.effective_definition)?,
        artifact_dir: paths.artifact_dir.clone(),
    };
    let mut command = plan.client.command.clone();
    command.argv.push("--prepare".to_owned());
    let bound = OperationBound::unbounded();
    let run = run_client(&command, &request, session, &paths, &bound)?;
    let mut accepted = accept_client_result::<BenchPopulationPreparationResult>(
        &session.absolute(&paths.result),
        "Bench population preparation client",
        run,
        &bound,
    );
    let evidence = BenchPopulationPreparationEvidence {
        result: accepted.result.take(),
        process: accepted.run.process.clone(),
        request: paths.request,
        result_path: paths.result,
        stdout: paths.stdout,
        stderr: paths.stderr,
        artifact_dir: paths.artifact_dir,
    };
    let decode_error = accepted.decode_error.take();
    accepted.run.finish_cleanup();
    Ok((evidence, decode_error))
}

pub(super) fn finish_population_preparation(
    plan: &mut BenchPlan,
    expected_materialization_identity: &str,
    evidence: &BenchPopulationPreparationEvidence,
    decode_error: Option<String>,
) -> Result<(), InferlabError> {
    if let Some(error) = decode_error {
        return Err(InferlabError::DatasetPreparation { message: error });
    }
    let result = evidence
        .result
        .as_ref()
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: "population preparation returned no result".to_owned(),
        })?;
    validate_population_preparation(plan, expected_materialization_identity, result)?;
    let evidence_path =
        result
            .evidence_path
            .as_ref()
            .ok_or_else(|| InferlabError::DatasetPreparation {
                message: "successful population preparation omitted its evidence path".to_owned(),
            })?;
    plan.client.population = result
        .population
        .as_ref()
        .map(|population| BenchPopulation {
            path: population.path.clone(),
            evidence_path: evidence_path.clone(),
            sha256: population.sha256.clone(),
            entries: population.entries,
            tpot_applicable: population.tpot_applicable,
            session_templates: population
                .session_templates
                .iter()
                .map(|template| BenchSessionTemplate {
                    template_identity: template.template_identity.clone(),
                    turn_count: template.turn_count,
                })
                .collect(),
        });
    resolve_session_case_request_counts(plan)?;
    Ok(())
}

fn resolve_session_case_request_counts(plan: &mut BenchPlan) -> Result<(), InferlabError> {
    if !matches!(
        plan.client.effective_definition.source,
        ResolvedBenchSource::Sessions { .. }
    ) {
        return Ok(());
    }
    let templates = plan
        .client
        .population
        .as_ref()
        .map(|population| population.session_templates.as_slice())
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: "linear-session preparation omitted its population".to_owned(),
        })?;
    let BenchExecutionPlan::Matrix { cases } = &mut plan.execution else {
        return Err(InferlabError::DatasetPreparation {
            message: "linear sessions require a static Bench matrix".to_owned(),
        });
    };
    for case in cases {
        let warmup_sessions = case.warmup_session_count.unwrap_or(0);
        let profiling_sessions =
            case.session_count
                .ok_or_else(|| InferlabError::DatasetPreparation {
                    message: format!("linear-session case {:?} omitted session_count", case.id),
                })?;
        let layout =
            session_population_layout(warmup_sessions, profiling_sessions).ok_or_else(|| {
                InferlabError::DatasetPreparation {
                    message: format!("linear-session case {:?} slice exceeds u32", case.id),
                }
            })?;
        let end = usize::try_from(layout.required_entries).map_err(|_| {
            InferlabError::DatasetPreparation {
                message: format!("linear-session case {:?} slice is not addressable", case.id),
            }
        })?;
        if end > templates.len() {
            return Err(InferlabError::DatasetPreparation {
                message: format!(
                    "linear-session case {:?} requires {} templates, population has {}",
                    case.id,
                    layout.required_entries,
                    templates.len()
                ),
            });
        }
        let warmup_end =
            usize::try_from(warmup_sessions).map_err(|_| InferlabError::DatasetPreparation {
                message: format!(
                    "linear-session case {:?} warmup is not addressable",
                    case.id
                ),
            })?;
        case.warmup_request_count = sum_session_turns(&templates[..warmup_end], &case.id)?;
        let profiling_start = usize::try_from(layout.profiling_start).map_err(|_| {
            InferlabError::DatasetPreparation {
                message: format!(
                    "linear-session case {:?} profiling slice is not addressable",
                    case.id
                ),
            }
        })?;
        case.request_count = sum_session_turns(&templates[profiling_start..end], &case.id)?;
    }
    Ok(())
}

fn sum_session_turns(
    templates: &[BenchSessionTemplate],
    case_id: &str,
) -> Result<u32, InferlabError> {
    templates.iter().try_fold(0_u32, |total, template| {
        total
            .checked_add(template.turn_count)
            .ok_or_else(|| InferlabError::DatasetPreparation {
                message: format!(
                    "linear-session case {case_id:?} transport-request count exceeds u32"
                ),
            })
    })
}

pub(super) fn acquire_dataset_snapshot(
    cache_path: &Path,
    url: &str,
    expected_sha256: &str,
) -> Result<DatasetAcquisitionEvidence, Box<(DatasetAcquisitionEvidence, InferlabError)>> {
    if cache_path.is_file() {
        let (observed_bytes, observed_sha256) = match hash_dataset_file(cache_path) {
            Ok(observed) => observed,
            Err(error) => {
                let evidence = failed_acquisition(None, None, &error);
                return Err(Box::new((evidence, error)));
            }
        };
        if observed_sha256 != expected_sha256 {
            let error = InferlabError::DatasetDigest {
                path: cache_path.to_path_buf(),
                expected: expected_sha256.to_owned(),
                observed: observed_sha256.clone(),
            };
            return Err(Box::new((
                failed_acquisition(Some(observed_bytes), Some(observed_sha256), &error),
                error,
            )));
        }
        return Ok(DatasetAcquisitionEvidence {
            outcome: DatasetAcquisitionOutcome::Reused,
            observed_bytes: Some(observed_bytes),
            observed_sha256: Some(observed_sha256),
            error: None,
        });
    }
    let parent = cache_path
        .parent()
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: format!("dataset cache path {} has no parent", cache_path.display()),
        })
        .map_err(|error| Box::new((failed_acquisition(None, None, &error), error)))?;
    fs::create_dir_all(parent)
        .map_err(|source| InferlabError::DatasetIo {
            operation: "create",
            path: parent.to_path_buf(),
            source,
        })
        .map_err(|error| Box::new((failed_acquisition(None, None, &error), error)))?;
    let mut temporary = NamedTempFile::new_in(parent)
        .map_err(|source| InferlabError::DatasetIo {
            operation: "create temporary dataset snapshot in",
            path: parent.to_path_buf(),
            source,
        })
        .map_err(|error| Box::new((failed_acquisition(None, None, &error), error)))?;
    let mut observed_bytes = 0_u64;
    let mut digest = Sha256::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| InferlabError::DatasetPreparation {
            message: format!("failed to initialize dataset download runtime: {error}"),
        })
        .map_err(|error| Box::new((failed_acquisition(None, None, &error), error)))?;
    let download = runtime.block_on(async {
        let client = reqwest::Client::new();
        let send = client.get(url).send();
        let mut response = tokio::select! {
            response = send => response,
            () = wait_for_interrupt() => {
                return Err(InferlabError::DatasetPreparation {
                    message: "dataset download was interrupted".to_owned(),
                });
            }
        }
        .and_then(reqwest::Response::error_for_status)
        .map_err(|source| InferlabError::DatasetHttp {
            url: url.to_owned(),
            source,
        })?;
        loop {
            let chunk = tokio::select! {
                chunk = response.chunk() => chunk.map_err(|source| InferlabError::DatasetHttp {
                    url: url.to_owned(),
                    source,
                })?,
                () = wait_for_interrupt() => {
                    return Err(InferlabError::DatasetPreparation {
                        message: "dataset download was interrupted".to_owned(),
                    });
                }
            };
            let Some(chunk) = chunk else {
                break;
            };
            temporary
                .write_all(&chunk)
                .map_err(|source| InferlabError::DatasetIo {
                    operation: "write",
                    path: temporary.path().to_path_buf(),
                    source,
                })?;
            observed_bytes = observed_bytes.saturating_add(chunk.len() as u64);
            digest.update(&chunk);
        }
        Ok::<(), InferlabError>(())
    });
    if let Err(error) = download {
        return Err(Box::new((
            failed_acquisition(Some(observed_bytes), None, &error),
            error,
        )));
    }
    let observed_sha256 = format!("{:x}", digest.finalize());
    if observed_sha256 != expected_sha256 {
        let error = InferlabError::DatasetDigest {
            path: cache_path.to_path_buf(),
            expected: expected_sha256.to_owned(),
            observed: observed_sha256.clone(),
        };
        return Err(Box::new((
            failed_acquisition(Some(observed_bytes), Some(observed_sha256), &error),
            error,
        )));
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| InferlabError::DatasetIo {
            operation: "sync",
            path: temporary.path().to_path_buf(),
            source,
        })
        .map_err(|error| {
            Box::new((
                failed_acquisition(Some(observed_bytes), Some(observed_sha256.clone()), &error),
                error,
            ))
        })?;
    temporary.persist(cache_path).map_err(|error| {
        let error = InferlabError::DatasetIo {
            operation: "publish",
            path: cache_path.to_path_buf(),
            source: error.error,
        };
        Box::new((
            failed_acquisition(Some(observed_bytes), Some(observed_sha256.clone()), &error),
            error,
        ))
    })?;
    Ok(DatasetAcquisitionEvidence {
        outcome: DatasetAcquisitionOutcome::Downloaded,
        observed_bytes: Some(observed_bytes),
        observed_sha256: Some(observed_sha256),
        error: None,
    })
}

pub(super) fn hash_dataset_file(path: &Path) -> Result<(u64, String), InferlabError> {
    let mut file = File::open(path).map_err(|source| InferlabError::DatasetIo {
        operation: "open",
        path: path.to_path_buf(),
        source,
    })?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| InferlabError::DatasetIo {
                operation: "read",
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        bytes = bytes.saturating_add(read as u64);
        digest.update(&buffer[..read]);
    }
    Ok((bytes, format!("{:x}", digest.finalize())))
}

pub(super) fn failed_acquisition(
    observed_bytes: Option<u64>,
    observed_sha256: Option<String>,
    error: &InferlabError,
) -> DatasetAcquisitionEvidence {
    DatasetAcquisitionEvidence {
        outcome: DatasetAcquisitionOutcome::Failed,
        observed_bytes,
        observed_sha256,
        error: Some(error.to_string()),
    }
}

pub(super) fn validate_population_preparation(
    plan: &BenchPlan,
    expected_materialization_identity: &str,
    result: &BenchPopulationPreparationResult,
) -> Result<(), InferlabError> {
    if result.status != ClientStatus::Succeeded {
        return Err(InferlabError::DatasetPreparation {
            message: result
                .error
                .clone()
                .unwrap_or_else(|| "population preparation client reported failure".to_owned()),
        });
    }
    if result.materialization_identity != expected_materialization_identity {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "population preparation returned materialization identity {:?}, expected {:?}",
                result.materialization_identity, expected_materialization_identity
            ),
        });
    }
    if result.requested_entries != plan.client.required_population_count {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "population preparation returned {} requested entries, expected {}",
                result.requested_entries, plan.client.required_population_count
            ),
        });
    }
    let population =
        result
            .population
            .as_ref()
            .ok_or_else(|| InferlabError::DatasetPreparation {
                message: "successful population preparation omitted its population".to_owned(),
            })?;
    if population.entries != plan.client.required_population_count {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "request population has {} entries, expected {}",
                population.entries, plan.client.required_population_count
            ),
        });
    }
    let (synthetic, prefix_declared, shared_system_declared) =
        match &plan.client.effective_definition.source {
            ResolvedBenchSource::Requests {
                request_source:
                    ResolvedBenchRequestSource::Random {
                        prefix_sharing,
                        shared_system_content,
                        ..
                    },
            } => (
                true,
                prefix_sharing.is_some(),
                shared_system_content.is_some(),
            ),
            ResolvedBenchSource::Requests {
                request_source: ResolvedBenchRequestSource::RandomMixture { prefix_sharing, .. },
            } => (true, prefix_sharing.is_some(), false),
            _ => (false, false, false),
        };
    let synthetic_prompt = synthetic.then_some(&plan.client.effective_definition.prompt.definition);
    match (synthetic, result.prompt_token_targeting.as_ref()) {
        (true, Some(targeting)) => {
            let Some(total_entries) = targeting
                .exact_entries
                .checked_add(targeting.fallback_entries)
            else {
                return Err(InferlabError::DatasetPreparation {
                    message: "synthetic prompt-targeting entry counts overflow".to_owned(),
                });
            };
            let fallback_reason_entries =
                targeting
                    .fallback_reasons
                    .iter()
                    .try_fold(0_u32, |total, (reason, count)| {
                        if reason.is_empty() || *count == 0 {
                            return None;
                        }
                        total.checked_add(*count)
                    });
            let projection_template_valid =
                targeting
                    .projection_template
                    .as_ref()
                    .is_none_or(|template| {
                        template.sha256
                            == format!("{:x}", Sha256::digest(template.content.as_bytes()))
                    });
            let template_policy_valid = match synthetic_prompt {
                Some(BenchPrompt::Flat) => {
                    targeting.exact_entries == population.entries
                        && targeting.fallback_entries == 0
                        && targeting.projection_template.is_none()
                }
                Some(BenchPrompt::RenderedChat { .. }) => {
                    targeting.exact_entries == population.entries
                        && targeting.fallback_entries == 0
                        && targeting.projection_template.is_some()
                }
                Some(BenchPrompt::ServerChat) => {
                    targeting.exact_entries == 0 || targeting.projection_template.is_some()
                }
                None => false,
            };
            if total_entries != population.entries
                || fallback_reason_entries != Some(targeting.fallback_entries)
                || !valid_token_summary(&targeting.selected_prompt_tokens)
                || !valid_token_summary(&targeting.pre_template_content_tokens)
                || result.input_tokens.as_ref() != Some(&targeting.selected_prompt_tokens)
                || !template_policy_valid
                || !projection_template_valid
            {
                return Err(InferlabError::DatasetPreparation {
                    message: "synthetic prompt-targeting summary is not reconciled".to_owned(),
                });
            }
        }
        (true, None) => {
            return Err(InferlabError::DatasetPreparation {
                message: "synthetic population omitted prompt-targeting evidence".to_owned(),
            });
        }
        (false, Some(_)) => {
            return Err(InferlabError::DatasetPreparation {
                message: "source-owned population returned synthetic prompt-targeting evidence"
                    .to_owned(),
            });
        }
        (false, None) => {}
    }
    match (prefix_declared, result.prefix_geometry.as_ref()) {
        (true, Some(summary))
            if valid_token_summary_allow_zero(&summary.shared_prefix_tokens)
                && valid_token_summary_allow_zero(&summary.unique_suffix_tokens)
                && summary.maximum_shared_prefix_tokens == summary.shared_prefix_tokens.maximum
                && summary.full_prompt_entries <= population.entries
                && summary.canonical_prefix_sha256.len() == 64 => {}
        (false, None) => {}
        _ => {
            return Err(InferlabError::DatasetPreparation {
                message: "synthetic prefix-geometry summary is not reconciled".to_owned(),
            });
        }
    }
    match (
        shared_system_declared,
        result.shared_system_content.as_ref(),
    ) {
        (true, Some(summary))
            if valid_token_summary(&summary.system_content_tokens)
                && valid_token_summary(&summary.user_content_tokens)
                && summary.canonical_system_content_sha256.len() == 64 => {}
        (false, None) => {}
        _ => {
            return Err(InferlabError::DatasetPreparation {
                message: "synthetic shared-system-content summary is not reconciled".to_owned(),
            });
        }
    }
    let session_backed = matches!(
        plan.client.effective_definition.source,
        ResolvedBenchSource::Sessions { .. }
    );
    if session_backed {
        if population.session_templates.len() != population.entries as usize {
            return Err(InferlabError::DatasetPreparation {
                message: format!(
                    "linear-session population has {} template summaries, expected {}",
                    population.session_templates.len(),
                    population.entries
                ),
            });
        }
        let mut identities = BTreeSet::new();
        for template in &population.session_templates {
            if template.template_identity.is_empty()
                || template.turn_count < 2
                || !identities.insert(&template.template_identity)
            {
                return Err(InferlabError::DatasetPreparation {
                    message:
                        "linear-session population has an invalid or duplicate template summary"
                            .to_owned(),
                });
            }
        }
    } else if !population.session_templates.is_empty() {
        return Err(InferlabError::DatasetPreparation {
            message: "independent-request population returned linear-session templates".to_owned(),
        });
    }
    let (_, observed_sha256) = hash_dataset_file(&population.path)?;
    if observed_sha256 != population.sha256 {
        return Err(InferlabError::DatasetDigest {
            path: population.path.clone(),
            expected: population.sha256.clone(),
            observed: observed_sha256,
        });
    }
    let expected_tpot = plan.client.tpot_applicability.is_applicable();
    if population.tpot_applicable != expected_tpot {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "request population TPOT applicability is {}, expected {} from the resolved request source",
                population.tpot_applicable, expected_tpot
            ),
        });
    }
    if !result
        .evidence_path
        .as_ref()
        .is_some_and(|path| path.is_file())
    {
        return Err(InferlabError::DatasetPreparation {
            message: "successful population preparation omitted its evidence artifact".to_owned(),
        });
    }
    if result.evidence_path.as_ref() != Some(&population.evidence_path) {
        return Err(InferlabError::DatasetPreparation {
            message: "request population and preparation result disagree on evidence path"
                .to_owned(),
        });
    }
    Ok(())
}

fn valid_token_summary(summary: &inferlab_protocol::BenchTokenCountSummary) -> bool {
    summary.minimum > 0
        && summary.maximum >= summary.minimum
        && summary.mean.is_finite()
        && summary.mean >= f64::from(summary.minimum)
        && summary.mean <= f64::from(summary.maximum)
}

fn valid_token_summary_allow_zero(summary: &inferlab_protocol::BenchTokenCountSummary) -> bool {
    summary.maximum >= summary.minimum
        && summary.mean.is_finite()
        && summary.mean >= f64::from(summary.minimum)
        && summary.mean <= f64::from(summary.maximum)
}
