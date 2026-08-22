//! Bench definition lowering and execution-case planning.

mod execution;

use super::overrides::{apply_definition_override, override_plan};
use crate::InferlabError;
use crate::bench_agentic_catalog;
use crate::bench_dataset_catalog;
use crate::toml_override::InvocationOverride;
use crate::toolchain::InstalledBenchToolchain;
use crate::workload::domain::{
    AggregateSloBound, BenchAgenticCatalog, BenchDatasetCatalog, BenchDatasetFilter,
    BenchSessionDatasetCatalog, DatasetCacheState, ResolvedAggregateSlo,
    ResolvedBenchAgenticSource, ResolvedBenchDefinition, ResolvedBenchPrompt,
    ResolvedBenchRandomShape, ResolvedBenchRequestSource, ResolvedBenchSessionSource,
    ResolvedBenchSloPolicy, ResolvedBenchSource, WorkloadHttpAction,
};
use crate::workload::plan::{
    BenchClientPlan, BenchPlan, BenchPrefixCacheConditioningPlan, ClientCommandPlan,
    ConditioningServingShape, MeasurementOverridePlan, MeasurementResolveContext,
};
use crate::workspace::{
    AggregateSlo, BenchAgenticSource, BenchCacheStart, BenchDefinition, BenchPrefixSharing,
    BenchPrompt, BenchRequestSource, BenchSessionSource, BenchTokenSelector,
    BenchTpotApplicability, validate_bench,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub(super) fn resolve_bench(
    id: &str,
    definitions: &BTreeMap<String, BenchDefinition>,
    overrides: &[InvocationOverride],
    context: &MeasurementResolveContext<'_>,
    toolchain: &InstalledBenchToolchain,
) -> Result<BenchPlan, InferlabError> {
    let declared_definition =
        definitions
            .get(id)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown selected bench {id:?}"),
            })?;
    let (definition, override_plan) =
        apply_bench_overrides(id, declared_definition.clone(), overrides)?;
    build_bench_plan(
        id,
        declared_definition,
        definition,
        override_plan,
        context,
        toolchain,
    )
}

pub(super) fn build_bench_plan(
    id: &str,
    declared_definition: BenchDefinition,
    definition: BenchDefinition,
    overrides: Vec<MeasurementOverridePlan>,
    context: &MeasurementResolveContext<'_>,
    toolchain: &InstalledBenchToolchain,
) -> Result<BenchPlan, InferlabError> {
    let tpot_applicability = match &definition {
        BenchDefinition::Serving {
            request_source,
            session_source,
            agentic_source,
            ..
        } => {
            if agentic_source.is_some() {
                BenchTpotApplicability::Inapplicable
            } else {
                request_source.as_ref().map_or_else(
                    || {
                        session_source.as_ref().map_or(
                            BenchTpotApplicability::Inapplicable,
                            BenchSessionSource::tpot_applicability,
                        )
                    },
                    BenchRequestSource::tpot_applicability,
                )
            }
        }
        BenchDefinition::AdaptiveServing { request_source, .. } => {
            request_source.tpot_applicability()
        }
    };
    let resolved_definition = resolve_bench_definition(&declared_definition, &definition)?;
    if resolved_definition.server_metrics && context.endpoint.server_metrics.is_none() {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "bench {id:?} enables server_metrics, but the resolved server endpoint exposes no server-metrics capability"
            ),
        });
    }
    if resolved_definition.requires_prompt_cache_evidence()
        && context
            .endpoint
            .prompt_cache_read_zero_representation
            .is_none()
    {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "bench {id:?} requires backend prompt cache-read usage, but the resolved server endpoint exposes no prompt cache-read capability; enable the serving integration's cache-read reporting setting and rebuild the server"
            ),
        });
    }
    let slo = resolve_bench_slo_policy(&definition)?;
    let prefix_cache_reset = if matches!(
        resolved_definition.cache_start,
        BenchCacheStart::Cold | BenchCacheStart::Primed
    ) {
        Some(
            context
                .prefix_cache_reset
                .clone()
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "bench {id:?} selects cache.start = {:?}, but the server exposes no prefix-cache reset capability",
                        resolved_definition.cache_start
                    ),
                })?,
        )
    } else {
        None
    };
    let prefix_cache_conditioning = prefix_cache_conditioning_plan(
        id,
        &resolved_definition,
        &context.endpoint.completions_path,
        &context.model.served_name,
        context.prefix_cache_conditioning.as_ref(),
        context.conditioning_serving,
    )?;
    let mut env = context.command_env.clone();
    env.remove("HF_HUB_OFFLINE");
    env.insert(
        "PYTHONPATH".to_owned(),
        toolchain.python_path.to_string_lossy().into_owned(),
    );
    env.insert("PYTHONNOUSERSITE".to_owned(), "1".to_owned());
    let execution = resolve_bench_execution(id, &definition)?;
    let required_population_count = required_population_count(id, &execution)?;
    Ok(BenchPlan {
        id: id.to_owned(),
        capture: context.capture_ids.iter().any(|capture| capture == id),
        declared_definition,
        execution,
        definition,
        overrides,
        client: BenchClientPlan {
            toolchain: toolchain.identity.clone(),
            tokenizer_backend: "huggingface".to_owned(),
            endpoint: context.endpoint.clone(),
            model: context.model.clone(),
            effective_definition: resolved_definition,
            tpot_applicability,
            slo,
            required_population_count,
            population: None,
            command: ClientCommandPlan {
                argv: vec![
                    toolchain.python.to_string_lossy().into_owned(),
                    "-m".to_owned(),
                    "inferlab_bench_runner.bench_client".to_owned(),
                ],
                env,
                cwd: context.command_cwd.to_path_buf(),
            },
            prefix_cache_reset,
            prefix_cache_conditioning,
        },
    })
}

fn prefix_cache_conditioning_plan(
    id: &str,
    definition: &ResolvedBenchDefinition,
    completions_path: &str,
    model: &str,
    frontend_conditioning: Option<&WorkloadHttpAction>,
    serving: ConditioningServingShape,
) -> Result<Option<BenchPrefixCacheConditioningPlan>, InferlabError> {
    if definition.cache_start != BenchCacheStart::Primed {
        return Ok(None);
    }
    let Some(request_source) = definition.source.request_source() else {
        return Ok(None);
    };
    let (maximum_input_tokens, sharing) = match request_source {
        ResolvedBenchRequestSource::Random {
            input_tokens,
            prefix_sharing: Some(sharing),
            ..
        } => (token_selector_maximum(input_tokens), sharing),
        ResolvedBenchRequestSource::RandomMixture {
            shapes,
            prefix_sharing: Some(sharing),
            ..
        } => {
            let Some(maximum) = shapes.iter().map(|shape| shape.input_tokens).max() else {
                return Ok(None);
            };
            (maximum, sharing)
        }
        ResolvedBenchRequestSource::Random { .. }
        | ResolvedBenchRequestSource::RandomMixture { .. }
        | ResolvedBenchRequestSource::Dataset { .. } => return Ok(None),
    };
    let maximum_shared_prefix_tokens = match sharing {
        BenchPrefixSharing::Tokens {
            shared_prefix_tokens,
        } => *shared_prefix_tokens,
        BenchPrefixSharing::Ratio {
            shared_prefix_ratio,
        } => (f64::from(maximum_input_tokens) * shared_prefix_ratio).floor() as u32,
    };
    let (route, frontend_fanout) = if serving.gateway_frontend {
        let Some(action) = frontend_conditioning else {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "bench {id:?} selects cache.start = \"primed\", but the Gateway frontend does not declare a prefix-cache conditioning fan-out capability: frontend routing cannot address an individual replica or data-parallel rank, so conditioning cannot cover every cache-owning rank"
                ),
            });
        };
        (action.path.clone(), true)
    } else {
        (completions_path.to_owned(), false)
    };
    Ok(Some(BenchPrefixCacheConditioningPlan {
        route,
        model: model.to_owned(),
        prompt: definition.prompt.clone(),
        request_body: definition.request_body.clone(),
        maximum_shared_prefix_tokens,
        output_tokens: 1,
        consumes_population_entry: false,
        attention_data_parallel_size: serving.attention_data_parallel_size,
        frontend_fanout,
    }))
}

fn token_selector_maximum(selector: &BenchTokenSelector) -> u32 {
    match selector {
        BenchTokenSelector::Fixed(value) => *value,
        BenchTokenSelector::InclusiveUniform { max, .. } => *max,
    }
}

pub(super) fn apply_bench_overrides(
    id: &str,
    definition: BenchDefinition,
    overrides: &[InvocationOverride],
) -> Result<(BenchDefinition, Vec<MeasurementOverridePlan>), InferlabError> {
    let session_backed = matches!(
        &definition,
        BenchDefinition::Serving {
            session_source: Some(_),
            ..
        }
    );
    let agentic_backed = matches!(
        &definition,
        BenchDefinition::Serving {
            agentic_source: Some(_),
            ..
        }
    );
    let mut value =
        toml::Value::try_from(definition).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare bench {id:?} for overrides: {error}"),
        })?;
    for item in overrides {
        if agentic_backed
            && ["request_source", "session_source", "agentic_source"]
                .into_iter()
                .any(|source| {
                    item.path() == source || item.path().starts_with(&format!("{source}."))
                })
        {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: "agentic Bench overrides cannot change the selected source boundary, dataset, or profile"
                    .to_owned(),
            });
        }
        if session_backed
            && (item.path() == "session_source"
                || (item.path().starts_with("session_source.")
                    && item.path() != "session_source.inter_turn_delay_scale"
                    && item.path() != "session_source.max_inter_turn_delay_seconds")
                || item.path() == "request_source"
                || item.path().starts_with("request_source."))
        {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: "linear-session Bench overrides preserve the source boundary and may change only its inter-turn delay controls"
                    .to_owned(),
            });
        }
        if !session_backed
            && (item.path() == "session_source" || item.path().starts_with("session_source."))
        {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: "Bench invocation overrides cannot change the selected source boundary"
                    .to_owned(),
            });
        }
        if !agentic_backed
            && (item.path() == "agentic_source" || item.path().starts_with("agentic_source."))
        {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: "Bench invocation overrides cannot change the selected source boundary"
                    .to_owned(),
            });
        }
        if item.path() == "request_source.kind" {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: "Bench request_source.kind cannot be overridden".to_owned(),
            });
        }
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
            message: format!("invalid effective Bench definition: {error}"),
        })?;
    validate_bench(id, &definition)?;
    Ok((definition, override_plan(overrides)))
}

fn resolve_bench_definition(
    declared_definition: &BenchDefinition,
    definition: &BenchDefinition,
) -> Result<ResolvedBenchDefinition, InferlabError> {
    match definition {
        BenchDefinition::Serving {
            request_source,
            session_source,
            agentic_source,
            seed,
            server_metrics,
            artifact_level,
            request_body,
            request_slo,
            cache,
            timeout_seconds,
            ..
        } => {
            let (source, prompt) = match (request_source, session_source, agentic_source) {
                (Some(request_source), None, None) => (
                    ResolvedBenchSource::Requests {
                        request_source: resolve_bench_request_source(request_source)?,
                    },
                    resolved_request_source_prompt(
                        declared_request_source(declared_definition),
                        request_source,
                    ),
                ),
                (None, Some(session_source), None) => (
                    ResolvedBenchSource::Sessions {
                        session_source: resolve_bench_session_source(session_source)?,
                    },
                    ResolvedBenchPrompt::from_definition(&BenchPrompt::ServerChat),
                ),
                (None, None, Some(agentic_source)) => (
                    ResolvedBenchSource::Agentic {
                        agentic_source: resolve_bench_agentic_source(agentic_source)?,
                    },
                    ResolvedBenchPrompt::from_definition(&BenchPrompt::ServerChat),
                ),
                _ => {
                    return Err(InferlabError::InvalidConfig {
                        message:
                            "resolved serving Bench requires exactly one request, session, or agentic source"
                                .to_owned(),
                    });
                }
            };
            Ok(ResolvedBenchDefinition {
                source,
                prompt,
                server_metrics: *server_metrics,
                artifact_level: *artifact_level,
                seed: *seed,
                request_body: request_body.clone(),
                request_slo: request_slo.clone(),
                timeout_seconds: *timeout_seconds,
                cache_start: cache.map_or(BenchCacheStart::Uncontrolled, |cache| cache.start),
            })
        }
        BenchDefinition::AdaptiveServing {
            request_source,
            seed,
            server_metrics,
            artifact_level,
            request_body,
            request_slo,
            cache,
            timeout_seconds,
            ..
        } => Ok(ResolvedBenchDefinition {
            source: ResolvedBenchSource::Requests {
                request_source: resolve_bench_request_source(request_source)?,
            },
            prompt: resolved_request_source_prompt(
                declared_request_source(declared_definition),
                request_source,
            ),
            server_metrics: *server_metrics,
            artifact_level: *artifact_level,
            seed: *seed,
            request_body: request_body.clone(),
            request_slo: request_slo.clone(),
            timeout_seconds: *timeout_seconds,
            cache_start: cache.map_or(BenchCacheStart::Uncontrolled, |cache| cache.start),
        }),
    }
}

fn resolve_bench_agentic_source(
    source: &BenchAgenticSource,
) -> Result<ResolvedBenchAgenticSource, InferlabError> {
    let resolved = bench_agentic_catalog::resolve(&source.dataset, &source.profile)?;
    Ok(ResolvedBenchAgenticSource {
        dataset: resolved.dataset,
        profile: resolved.profile,
        catalog: Box::new(BenchAgenticCatalog {
            repository: resolved.source.repository,
            revision: resolved.source.revision,
            filename: resolved.source.filename,
            sha256: resolved.source.sha256,
            cache_path: None,
            cache_state: None,
            trace_count: resolved.source.trace_count,
            approximate_bytes: resolved.source.approximate_bytes,
            license: resolved.source.license,
            source_format: resolved.source.source_format,
            aiperf_loader: resolved.source.aiperf_loader,
            materialization_identity: resolved.source.materialization_identity,
            scenario: resolved.policy.scenario,
            concurrency_semantics: resolved.policy.concurrency_semantics,
            replay_semantics: resolved.policy.replay_semantics,
            cache_bust: resolved.policy.cache_bust,
            trajectory_start_min: resolved.policy.trajectory_start_min,
            trajectory_start_max: resolved.policy.trajectory_start_max,
            global_idle_gap_cap_seconds: resolved.policy.global_idle_gap_cap_seconds,
            cache_warmup_seconds: resolved.policy.cache_warmup_seconds,
            warmup_grace_seconds: resolved.policy.warmup_grace_seconds,
            dataset_configuration_timeout_seconds: resolved
                .policy
                .dataset_configuration_timeout_seconds,
            service_profile_configuration_timeout_seconds: resolved
                .policy
                .service_profile_configuration_timeout_seconds,
            default_duration_seconds: resolved.policy.default_duration_seconds,
            minimum_duration_seconds: resolved.policy.minimum_duration_seconds,
            failure_threshold: resolved.policy.failure_threshold,
            dataset_entries: resolved.policy.dataset_entries,
            streaming: resolved.policy.streaming,
            ignore_eos: resolved.policy.ignore_eos,
            use_server_token_count: resolved.policy.use_server_token_count,
            gpu_telemetry: resolved.policy.gpu_telemetry,
            server_metric_slice_seconds: resolved.policy.server_metric_slice_seconds,
            required_artifacts: resolved.policy.required_artifacts,
            unavailable_dimensions: resolved.policy.unavailable_dimensions,
            inferencex_repository: resolved.qualification.inferencex_repository,
            inferencex_revision: resolved.qualification.inferencex_revision,
            inferencex_reference: resolved.qualification.inferencex_reference,
            aiperf_revision: resolved.qualification.aiperf_revision,
            aiperf_version: resolved.qualification.aiperf_version,
        }),
    })
}

fn declared_request_source(definition: &BenchDefinition) -> Option<&BenchRequestSource> {
    match definition {
        BenchDefinition::Serving { request_source, .. } => request_source.as_ref(),
        BenchDefinition::AdaptiveServing { request_source, .. } => Some(request_source),
    }
}

fn resolved_request_source_prompt(
    declared_source: Option<&BenchRequestSource>,
    source: &BenchRequestSource,
) -> ResolvedBenchPrompt {
    let declared_prompt = match declared_source {
        Some(BenchRequestSource::Random { prompt, .. })
        | Some(BenchRequestSource::RandomMixture { prompt, .. }) => Some(prompt),
        Some(BenchRequestSource::Dataset { .. }) | None => None,
    };
    match source {
        BenchRequestSource::Random { prompt, .. }
        | BenchRequestSource::RandomMixture { prompt, .. } => {
            ResolvedBenchPrompt::from_declared_and_effective(declared_prompt, prompt)
        }
        BenchRequestSource::Dataset { .. } => {
            ResolvedBenchPrompt::from_definition(&BenchPrompt::ServerChat)
        }
    }
}

fn resolve_bench_session_source(
    source: &BenchSessionSource,
) -> Result<ResolvedBenchSessionSource, InferlabError> {
    let resolved =
        bench_dataset_catalog::resolve_session(&source.dataset, source.profile.as_deref())?;
    let cache_path = dataset_cache_home()?
        .join("inferlab/datasets/sha256")
        .join(&resolved.sha256);
    let cache_state = if cache_path.is_file() {
        DatasetCacheState::Present
    } else {
        DatasetCacheState::Missing
    };
    Ok(ResolvedBenchSessionSource {
        dataset: source.dataset.clone(),
        profile: source.profile.clone(),
        max_input_tokens: source.max_input_tokens,
        output_tokens: source.output_tokens,
        inter_turn_delay_scale: source.inter_turn_delay_scale,
        max_inter_turn_delay_seconds: source.max_inter_turn_delay_seconds,
        catalog: Box::new(BenchSessionDatasetCatalog {
            dataset: resolved.dataset,
            profile: resolved.profile,
            source: resolved.source,
            upstream_identity: resolved.upstream_identity,
            url: resolved.url,
            sha256: resolved.sha256,
            source_format: resolved.source_format,
            configuration: resolved.configuration,
            split: resolved.split,
            filter: resolved.filter.map(|filter| BenchDatasetFilter {
                field: filter.field,
                value: filter.value,
            }),
            license: resolved.license,
            cache_path,
            cache_state,
            materialization_identity: resolved.materialization_identity,
            provides_output_targets: resolved.provides_output_targets,
        }),
    })
}

fn resolve_bench_request_source(
    source: &BenchRequestSource,
) -> Result<ResolvedBenchRequestSource, InferlabError> {
    match source {
        BenchRequestSource::Random {
            prompt: _,
            input_tokens,
            output_tokens,
            prefix_sharing,
            shared_system_content,
        } => Ok(ResolvedBenchRequestSource::Random {
            input_tokens: input_tokens.clone(),
            output_tokens: output_tokens.clone(),
            prefix_sharing: prefix_sharing.clone(),
            shared_system_content: shared_system_content.clone(),
        }),
        BenchRequestSource::RandomMixture {
            prompt: _,
            shapes,
            prefix_sharing,
        } => {
            let total_weight = shapes.iter().try_fold(0_u64, |total, shape| {
                total.checked_add(u64::from(shape.weight)).ok_or_else(|| {
                    InferlabError::InvalidConfig {
                        message:
                            "resolved Bench random_mixture total weight exceeds the supported unsigned 64-bit range"
                                .to_owned(),
                    }
                })
            })?;
            Ok(ResolvedBenchRequestSource::RandomMixture {
                shapes: shapes
                    .iter()
                    .map(|shape| ResolvedBenchRandomShape {
                        input_tokens: shape.input_tokens,
                        output_tokens: shape.output_tokens,
                        weight: shape.weight,
                    })
                    .collect(),
                total_weight,
                prefix_sharing: prefix_sharing.clone(),
            })
        }
        BenchRequestSource::Dataset {
            dataset,
            profile,
            max_input_tokens,
            output_tokens,
        } => {
            let resolved = bench_dataset_catalog::resolve(dataset, profile.as_deref())?;
            let cache_path = dataset_cache_home()?
                .join("inferlab/datasets/sha256")
                .join(&resolved.sha256);
            let cache_state = if cache_path.is_file() {
                DatasetCacheState::Present
            } else {
                DatasetCacheState::Missing
            };
            Ok(ResolvedBenchRequestSource::Dataset {
                dataset: dataset.clone(),
                profile: profile.clone(),
                max_input_tokens: *max_input_tokens,
                output_tokens: *output_tokens,
                catalog: Box::new(BenchDatasetCatalog {
                    dataset: resolved.dataset,
                    profile: resolved.profile,
                    source: resolved.source,
                    upstream_identity: resolved.upstream_identity,
                    url: resolved.url,
                    sha256: resolved.sha256,
                    source_format: resolved.source_format,
                    aiperf_format: resolved.aiperf_format,
                    configuration: resolved.configuration,
                    split: resolved.split,
                    filter: resolved.filter.map(|filter| BenchDatasetFilter {
                        field: filter.field,
                        value: filter.value,
                    }),
                    license: resolved.license,
                    cache_path,
                    cache_state,
                    materialization_identity: resolved.materialization_identity,
                    provides_output_targets: resolved.provides_output_targets,
                }),
            })
        }
    }
}

fn resolve_bench_slo_policy(
    definition: &BenchDefinition,
) -> Result<ResolvedBenchSloPolicy, InferlabError> {
    let (aggregate, request) = match definition {
        BenchDefinition::Serving {
            aggregate_slos,
            request_slo,
            ..
        }
        | BenchDefinition::AdaptiveServing {
            aggregate_slos,
            request_slo,
            ..
        } => (aggregate_slos, request_slo),
    };
    Ok(ResolvedBenchSloPolicy {
        aggregate: aggregate
            .iter()
            .map(resolve_aggregate_slo)
            .collect::<Result<Vec<_>, _>>()?,
        request: request.clone(),
    })
}

fn resolve_aggregate_slo(slo: &AggregateSlo) -> Result<ResolvedAggregateSlo, InferlabError> {
    let metric = slo.metric;
    let bound = match (slo.at_most, slo.at_least) {
        (Some(value), None) => AggregateSloBound::AtMost(value),
        (None, Some(value)) => AggregateSloBound::AtLeast(value),
        _ => {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "resolved Bench metric {:?} has no unique SLO bound",
                    metric.name()
                ),
            });
        }
    };
    Ok(ResolvedAggregateSlo { metric, bound })
}

fn dataset_cache_home() -> Result<PathBuf, InferlabError> {
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path));
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".cache"))
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: "neither XDG_CACHE_HOME nor HOME is set for the dataset cache".to_owned(),
        })
}

pub(crate) use execution::resolved_request_count;
use execution::{required_population_count, resolve_bench_execution};
#[cfg(test)]
mod tests {
    use super::{
        ResolvedBenchPrompt, ResolvedBenchRequestSource, apply_bench_overrides,
        required_population_count, resolve_bench_definition, resolve_bench_execution,
        resolve_bench_request_source,
    };
    use crate::toml_override::InvocationOverride;
    use crate::workload::domain::{
        BenchPromptRoute, BenchRenderingAuthority, BenchRequestRepresentation,
    };
    use crate::workspace::{
        BenchDefinition, BenchPrefixSharing, BenchPrompt, BenchPromptSelection, BenchRandomShape,
        BenchRequestSource, BenchTokenSelector, validate_bench,
    };

    #[test]
    fn synthetic_request_sources_resolve_effective_prefix_and_total_weight()
    -> Result<(), Box<dyn std::error::Error>> {
        let prefix = resolve_bench_request_source(&BenchRequestSource::Random {
            prompt: BenchPromptSelection::explicit(BenchPrompt::Flat),
            input_tokens: BenchTokenSelector::Fixed(8000),
            output_tokens: BenchTokenSelector::Fixed(1000),
            prefix_sharing: Some(BenchPrefixSharing::Ratio {
                shared_prefix_ratio: 0.75,
            }),
            shared_system_content: None,
        })?;
        let mixture = resolve_bench_request_source(&BenchRequestSource::RandomMixture {
            prompt: BenchPromptSelection::explicit(BenchPrompt::ServerChat),
            shapes: vec![
                BenchRandomShape {
                    input_tokens: 1024,
                    output_tokens: 128,
                    weight: 7,
                },
                BenchRandomShape {
                    input_tokens: 8192,
                    output_tokens: 1024,
                    weight: 3,
                },
            ],
            prefix_sharing: None,
        })?;
        let prompt = ResolvedBenchPrompt::from_definition(&BenchPrompt::Flat);

        assert!(matches!(
            prompt,
            ResolvedBenchPrompt {
                declared: None,
                definition: BenchPrompt::Flat,
                request_representation: BenchRequestRepresentation::FlatPrompt,
                route: BenchPromptRoute::Completions,
                rendering_authority: BenchRenderingAuthority::LocalFlat,
            }
        ));
        assert!(matches!(
            prefix,
            ResolvedBenchRequestSource::Random {
                prefix_sharing: Some(BenchPrefixSharing::Ratio {
                    shared_prefix_ratio: 0.75,
                }),
                ..
            }
        ));
        assert!(matches!(
            mixture,
            ResolvedBenchRequestSource::RandomMixture {
                total_weight: 10,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn linear_session_definition_resolves_session_counts_and_delay_controls()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
session_source = { dataset = "sharegpt", max_input_tokens = 8192, inter_turn_delay_scale = 0.25, max_inter_turn_delay_seconds = 3.0 }
concurrency = [2, 4]
sessions_per_concurrency = 3
warmup_sessions_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;

        validate_bench("agentic", &definition)?;
        let resolved = serde_json::to_value(resolve_bench_definition(&definition, &definition)?)?;
        assert_eq!(resolved["session_source"]["dataset"], "sharegpt");
        assert_eq!(resolved["session_source"]["inter_turn_delay_scale"], 0.25);
        assert_eq!(
            resolved["session_source"]["max_inter_turn_delay_seconds"],
            3.0
        );

        let execution_plan = resolve_bench_execution("agentic", &definition)?;
        assert_eq!(required_population_count("agentic", &execution_plan)?, 17);
        let execution = serde_json::to_value(execution_plan)?;
        assert_eq!(execution["cases"][0]["session_count"], 6);
        assert_eq!(execution["cases"][0]["warmup_session_count"], 2);
        assert_eq!(execution["cases"][1]["session_count"], 12);
        assert_eq!(execution["cases"][1]["warmup_session_count"], 4);
        Ok(())
    }

    #[test]
    fn agentic_definition_resolves_profile_default_duration_and_catalog_policy()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
agentic_source = { dataset = "semianalysis_agentx_062126_256k", profile = "inferencex" }
concurrency = [2]
timeout_seconds = 3600
"#,
        )?;

        validate_bench("agentx", &definition)?;
        let resolved = serde_json::to_value(resolve_bench_definition(&definition, &definition)?)?;
        assert_eq!(
            resolved["agentic_source"]["catalog"]["revision"],
            "8fecd2fc56694469f758f0afbbb6335ad3043740"
        );
        assert_eq!(
            resolved["agentic_source"]["catalog"]["scenario"],
            "inferencex-agentx-mvp"
        );

        let execution = serde_json::to_value(resolve_bench_execution("agentx", &definition)?)?;
        assert_eq!(execution["cases"][0]["duration_seconds"], 1800);
        assert_eq!(execution["cases"][0]["load_shape"]["concurrency"], 2);
        assert_eq!(
            required_population_count("agentx", &resolve_bench_execution("agentx", &definition)?)?,
            0
        );
        Ok(())
    }

    #[test]
    fn linear_session_overrides_change_delay_controls_without_replacing_the_source()
    -> Result<(), Box<dyn std::error::Error>> {
        let definition = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
session_source = { dataset = "sharegpt", max_input_tokens = 8192 }
concurrency = [1]
sessions_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let delays = InvocationOverride::parse_all(&[
            "session_source.inter_turn_delay_scale=0.0".to_owned(),
            "session_source.max_inter_turn_delay_seconds=1.5".to_owned(),
        ])?;

        let (effective, _) = apply_bench_overrides("agentic", definition.clone(), &delays)?;
        let BenchDefinition::Serving {
            session_source: Some(source),
            ..
        } = effective
        else {
            return Err(std::io::Error::other("session source boundary changed").into());
        };
        assert_eq!(source.inter_turn_delay_scale, 0.0);
        assert_eq!(source.max_inter_turn_delay_seconds, Some(1.5));

        let replacement =
            InvocationOverride::parse_all(&["session_source.dataset=\"speed_bench\"".to_owned()])?;
        let error = apply_bench_overrides("agentic", definition, &replacement)
            .err()
            .ok_or("session source replacement unexpectedly succeeded")?;
        assert!(
            error
                .to_string()
                .contains("may change only its inter-turn delay controls")
        );
        Ok(())
    }

    #[test]
    fn artifact_level_defaults_to_diagnostic_and_preserves_an_explicit_level()
    -> Result<(), Box<dyn std::error::Error>> {
        let defaulted = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
request_source = { kind = "random", input_tokens = 128, output_tokens = 32 }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let explicit = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
artifact_level = "performance"
request_source = { kind = "random", input_tokens = 128, output_tokens = 32 }
concurrency = [1]
prompts_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;

        validate_bench("defaulted", &defaulted)?;
        validate_bench("explicit", &explicit)?;
        let defaulted = serde_json::to_value(resolve_bench_definition(&defaulted, &defaulted)?)?;
        let explicit = serde_json::to_value(resolve_bench_definition(&explicit, &explicit)?)?;
        assert_eq!(defaulted["artifact_level"], "diagnostic");
        assert_eq!(explicit["artifact_level"], "performance");
        Ok(())
    }

    #[test]
    fn performance_artifact_level_remains_valid_for_session_and_agentic_sources()
    -> Result<(), Box<dyn std::error::Error>> {
        let session = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
artifact_level = "performance"
session_source = { dataset = "sharegpt", max_input_tokens = 8192 }
concurrency = [1]
sessions_per_concurrency = 1
timeout_seconds = 60
"#,
        )?;
        let agentic = toml::from_str::<BenchDefinition>(
            r#"
kind = "serving"
artifact_level = "performance"
agentic_source = { dataset = "semianalysis_agentx_062126_256k", profile = "inferencex" }
concurrency = [2]
timeout_seconds = 3600
"#,
        )?;

        validate_bench("session", &session)?;
        validate_bench("agentic", &agentic)?;
        assert_eq!(
            serde_json::to_value(resolve_bench_definition(&session, &session)?)?["artifact_level"],
            "performance"
        );
        assert_eq!(
            serde_json::to_value(resolve_bench_definition(&agentic, &agentic)?)?["artifact_level"],
            "performance"
        );
        Ok(())
    }
}
