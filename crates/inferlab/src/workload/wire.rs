use super::domain::{
    BenchDatasetCatalog, BenchPopulation, BenchPromptRoute, BenchRenderingAuthority,
    BenchRequestRepresentation, DatasetCacheState, MeasurementModel, ResolvedBenchDefinition,
    ResolvedBenchPrompt, ResolvedBenchRequestSource, ResolvedBenchSessionSource,
    ResolvedBenchSource, WorkloadEndpoint, WorkloadEndpointProtocol,
};
use crate::InferlabError;
use crate::adapter::project_setting_values;
use crate::toolchain::BundledEvalTask;
use crate::workspace::{
    BenchPrefixSharing, BenchPrompt, BenchSharedSystemContent, BenchTokenSelector, EvalDefinition,
    EvalTaskSource, RequestSlo,
};
use inferlab_protocol::{
    BenchDatasetCacheState, BenchDatasetCatalogInput, BenchDatasetFilterInput,
    BenchDefinitionInput, BenchInclusiveUniformInput, BenchPopulationInput,
    BenchPrefixSharingInput, BenchPromptInput, BenchPromptRouteInput, BenchRandomShapeInput,
    BenchRenderingAuthorityInput, BenchRequestRepresentationInput, BenchRequestSloInput,
    BenchRequestSourceInput, BenchSessionDatasetCatalogInput, BenchSessionSourceInput,
    BenchSessionTemplateInput, BenchSharedSystemContentInput, BenchTokenDistributionKindInput,
    BenchTokenSelectorInput, ClientEndpointInput, EndpointProtocol, EvalDefinitionInput,
    EvalTaskSourceInput, MeasurementModelInput, ServerMetricsEndpointInput, SettingValue,
};
use std::collections::BTreeMap;

pub(super) fn endpoint_input(endpoint: &WorkloadEndpoint) -> ClientEndpointInput {
    ClientEndpointInput {
        protocol: match endpoint.protocol {
            WorkloadEndpointProtocol::Http => EndpointProtocol::Http,
        },
        host: endpoint.host.clone(),
        port: endpoint.port,
        completions_path: endpoint.completions_path.clone(),
        chat_completions_path: endpoint.chat_completions_path.clone(),
        server_metrics: endpoint.server_metrics.as_ref().map(|metrics| {
            ServerMetricsEndpointInput {
                path: metrics.path.clone(),
                port_name: metrics.port_name.clone(),
                url: metrics.url.clone(),
            }
        }),
    }
}

pub(super) fn model_input(model: &MeasurementModel) -> MeasurementModelInput {
    MeasurementModelInput {
        locator: model.locator.clone(),
        served_name: model.served_name.clone(),
    }
}

pub(super) fn bench_source_inputs(
    definition: &ResolvedBenchDefinition,
) -> Result<
    (
        Option<BenchRequestSourceInput>,
        Option<BenchSessionSourceInput>,
    ),
    InferlabError,
> {
    match &definition.source {
        ResolvedBenchSource::Requests { request_source } => {
            Ok((Some(bench_request_source_input(request_source)?), None))
        }
        ResolvedBenchSource::Sessions { session_source } => {
            Ok((None, Some(bench_session_source_input(session_source))))
        }
    }
}

pub(super) fn bench_request_body_input(
    definition: &ResolvedBenchDefinition,
) -> Result<BTreeMap<String, SettingValue>, InferlabError> {
    project_setting_values("Bench request body", &definition.request_body)
}

pub(super) fn bench_definition_input(
    definition: &ResolvedBenchDefinition,
) -> Result<BenchDefinitionInput, InferlabError> {
    let (request_source, session_source) = bench_source_inputs(definition)?;
    Ok(BenchDefinitionInput {
        request_source,
        session_source,
        prompt: prompt_input(&definition.prompt)?,
        server_metrics: definition.server_metrics,
        seed: definition.seed,
        request_body: bench_request_body_input(definition)?,
        request_slo: definition.request_slo.as_ref().map(request_slo_input),
        timeout_seconds: definition.timeout_seconds,
        reset_prefix_cache: definition.reset_prefix_cache,
    })
}

pub(super) fn bench_session_source_input(
    source: &ResolvedBenchSessionSource,
) -> BenchSessionSourceInput {
    BenchSessionSourceInput {
        dataset: source.dataset.clone(),
        profile: source.profile.clone(),
        max_input_tokens: source.max_input_tokens,
        output_tokens: source.output_tokens,
        inter_turn_delay_scale: source.inter_turn_delay_scale,
        max_inter_turn_delay_seconds: source.max_inter_turn_delay_seconds,
        catalog: Box::new(BenchSessionDatasetCatalogInput {
            dataset: source.catalog.dataset.clone(),
            profile: source.catalog.profile.clone(),
            source: source.catalog.source.clone(),
            upstream_identity: source.catalog.upstream_identity.clone(),
            url: source.catalog.url.clone(),
            sha256: source.catalog.sha256.clone(),
            source_format: source.catalog.source_format.clone(),
            configuration: source.catalog.configuration.clone(),
            split: source.catalog.split.clone(),
            filter: source
                .catalog
                .filter
                .as_ref()
                .map(|filter| BenchDatasetFilterInput {
                    field: filter.field.clone(),
                    value: filter.value.clone(),
                }),
            license: source.catalog.license.clone(),
            cache_path: source.catalog.cache_path.clone(),
            cache_state: match source.catalog.cache_state {
                DatasetCacheState::Missing => BenchDatasetCacheState::Missing,
                DatasetCacheState::Present => BenchDatasetCacheState::Present,
            },
            materialization_identity: source.catalog.materialization_identity.clone(),
            provides_output_targets: source.catalog.provides_output_targets,
        }),
    }
}

pub(super) fn bench_request_source_input(
    source: &ResolvedBenchRequestSource,
) -> Result<BenchRequestSourceInput, InferlabError> {
    Ok(match source {
        ResolvedBenchRequestSource::Random {
            input_tokens,
            output_tokens,
            prefix_sharing,
            shared_system_content,
        } => BenchRequestSourceInput::Random {
            input_tokens: token_selector_input(input_tokens),
            output_tokens: token_selector_input(output_tokens),
            prefix_sharing: prefix_sharing.as_ref().map(prefix_sharing_input),
            shared_system_content: shared_system_content
                .as_ref()
                .map(shared_system_content_input),
        },
        ResolvedBenchRequestSource::RandomMixture {
            shapes,
            total_weight,
            prefix_sharing,
        } => BenchRequestSourceInput::RandomMixture {
            shapes: shapes
                .iter()
                .map(|shape| BenchRandomShapeInput {
                    input_tokens: shape.input_tokens,
                    output_tokens: shape.output_tokens,
                    weight: shape.weight,
                })
                .collect(),
            total_weight: *total_weight,
            prefix_sharing: prefix_sharing.as_ref().map(prefix_sharing_input),
        },
        ResolvedBenchRequestSource::Dataset {
            dataset,
            profile,
            max_input_tokens,
            output_tokens,
            catalog,
        } => BenchRequestSourceInput::Dataset {
            dataset: dataset.clone(),
            profile: profile.clone(),
            max_input_tokens: *max_input_tokens,
            output_tokens: *output_tokens,
            catalog: Box::new(catalog_input(catalog)),
        },
    })
}

pub(super) fn prompt_input(
    prompt: &ResolvedBenchPrompt,
) -> Result<BenchPromptInput, InferlabError> {
    let request_representation = match prompt.request_representation {
        BenchRequestRepresentation::FlatPrompt => BenchRequestRepresentationInput::FlatPrompt,
        BenchRequestRepresentation::StructuredMessages => {
            BenchRequestRepresentationInput::StructuredMessages
        }
    };
    let route = match prompt.route {
        BenchPromptRoute::Completions => BenchPromptRouteInput::Completions,
        BenchPromptRoute::ChatCompletions => BenchPromptRouteInput::ChatCompletions,
    };
    let rendering_authority = match prompt.rendering_authority {
        BenchRenderingAuthority::LocalFlat => BenchRenderingAuthorityInput::LocalFlat,
        BenchRenderingAuthority::LocalTemplate => BenchRenderingAuthorityInput::LocalTemplate,
        BenchRenderingAuthority::Server => BenchRenderingAuthorityInput::Server,
    };
    Ok(match &prompt.definition {
        BenchPrompt::Flat => BenchPromptInput::Flat {
            request_representation,
            route,
            rendering_authority,
        },
        BenchPrompt::RenderedChat {
            chat_template,
            chat_template_kwargs,
        } => BenchPromptInput::RenderedChat {
            chat_template: chat_template.clone(),
            chat_template_kwargs: project_setting_values(
                "Bench rendered-chat template kwargs",
                chat_template_kwargs,
            )?,
            request_representation,
            route,
            rendering_authority,
        },
        BenchPrompt::ServerChat => BenchPromptInput::ServerChat {
            request_representation,
            route,
            rendering_authority,
        },
    })
}

fn prefix_sharing_input(sharing: &BenchPrefixSharing) -> BenchPrefixSharingInput {
    match sharing {
        BenchPrefixSharing::Tokens {
            shared_prefix_tokens,
        } => BenchPrefixSharingInput::Tokens {
            shared_prefix_tokens: *shared_prefix_tokens,
        },
        BenchPrefixSharing::Ratio {
            shared_prefix_ratio,
        } => BenchPrefixSharingInput::Ratio {
            shared_prefix_ratio: *shared_prefix_ratio,
        },
    }
}

fn shared_system_content_input(
    sharing: &BenchSharedSystemContent,
) -> BenchSharedSystemContentInput {
    match sharing {
        BenchSharedSystemContent::Tokens { tokens } => {
            BenchSharedSystemContentInput::Tokens { tokens: *tokens }
        }
        BenchSharedSystemContent::Ratio { ratio } => {
            BenchSharedSystemContentInput::Ratio { ratio: *ratio }
        }
    }
}

fn token_selector_input(selector: &BenchTokenSelector) -> BenchTokenSelectorInput {
    match selector {
        BenchTokenSelector::Fixed(value) => BenchTokenSelectorInput::Fixed(*value),
        BenchTokenSelector::InclusiveUniform { min, max } => {
            BenchTokenSelectorInput::InclusiveUniform(BenchInclusiveUniformInput {
                kind: BenchTokenDistributionKindInput::InclusiveUniform,
                min: *min,
                max: *max,
            })
        }
    }
}

pub(super) fn catalog_input(catalog: &BenchDatasetCatalog) -> BenchDatasetCatalogInput {
    BenchDatasetCatalogInput {
        dataset: catalog.dataset.clone(),
        profile: catalog.profile.clone(),
        source: catalog.source.clone(),
        upstream_identity: catalog.upstream_identity.clone(),
        url: catalog.url.clone(),
        sha256: catalog.sha256.clone(),
        source_format: catalog.source_format.clone(),
        aiperf_format: catalog.aiperf_format.clone(),
        configuration: catalog.configuration.clone(),
        split: catalog.split.clone(),
        filter: catalog
            .filter
            .as_ref()
            .map(|filter| BenchDatasetFilterInput {
                field: filter.field.clone(),
                value: filter.value.clone(),
            }),
        license: catalog.license.clone(),
        cache_path: catalog.cache_path.clone(),
        cache_state: match catalog.cache_state {
            DatasetCacheState::Missing => BenchDatasetCacheState::Missing,
            DatasetCacheState::Present => BenchDatasetCacheState::Present,
        },
        materialization_identity: catalog.materialization_identity.clone(),
        provides_output_targets: catalog.provides_output_targets,
    }
}

pub(super) fn population_input(population: &BenchPopulation) -> BenchPopulationInput {
    BenchPopulationInput {
        path: population.path.clone(),
        evidence_path: population.evidence_path.clone(),
        sha256: population.sha256.clone(),
        entries: population.entries,
        tpot_applicable: population.tpot_applicable,
        session_templates: population
            .session_templates
            .iter()
            .map(|template| BenchSessionTemplateInput {
                template_identity: template.template_identity.clone(),
                turn_count: template.turn_count,
            })
            .collect(),
    }
}

pub(super) fn eval_definition_input(
    definition: &EvalDefinition,
    bundled_task: Option<&BundledEvalTask>,
) -> Result<EvalDefinitionInput, InferlabError> {
    Ok(match definition {
        EvalDefinition::OpenAiSmoke {
            prompt,
            max_tokens,
            timeout_seconds,
        } => EvalDefinitionInput::OpenAiSmoke {
            prompt: prompt.clone(),
            max_tokens: *max_tokens,
            timeout_seconds: *timeout_seconds,
        },
        EvalDefinition::LmEval {
            task,
            request_body,
            limit,
            few_shot,
            seed,
            trials,
            max_tokens,
            concurrency,
            metric,
            metric_filter,
            threshold,
            timeout_seconds,
        } => EvalDefinitionInput::LmEval {
            task: Box::new(match task {
                EvalTaskSource::BuiltIn(name) => {
                    EvalTaskSourceInput::BuiltIn { name: name.clone() }
                }
                EvalTaskSource::Bundled { bundled } => {
                    let task = bundled_task
                        .filter(|task| &task.name == bundled)
                        .ok_or_else(|| InferlabError::InvalidConfig {
                            message: format!(
                                "bundled Eval task {bundled:?} has no matching toolchain resolution"
                            ),
                        })?;
                    EvalTaskSourceInput::Bundled {
                        name: task.name.clone(),
                        task_identity: task.task_identity.clone(),
                        path: task.path.clone(),
                        task_closure_sha256: task.task_closure_sha256.clone(),
                        task_definition_sha256: task.task_definition_sha256.clone(),
                        prompt_asset_sha256: task.prompt_asset_sha256.clone(),
                        dataset_asset_sha256: task.dataset_asset_sha256.clone(),
                        scorer_sha256: task.scorer_sha256.clone(),
                    }
                }
                EvalTaskSource::WorkspaceYaml { yaml } => {
                    EvalTaskSourceInput::WorkspaceYaml { path: yaml.clone() }
                }
            }),
            request_body: project_setting_values("Eval request body", request_body)?,
            limit: *limit,
            few_shot: *few_shot,
            seed: *seed,
            trials: *trials,
            max_tokens: *max_tokens,
            concurrency: *concurrency,
            metric: metric.clone(),
            metric_filter: metric_filter.clone(),
            threshold: *threshold,
            timeout_seconds: *timeout_seconds,
        },
    })
}

fn request_slo_input(slo: &RequestSlo) -> BenchRequestSloInput {
    BenchRequestSloInput {
        request_latency_ms: slo.request_latency_ms,
        ttft_ms: slo.ttft_ms,
        tpot_ms: slo.tpot_ms,
        minimum_good_request_ratio: slo.minimum_good_request_ratio,
    }
}
