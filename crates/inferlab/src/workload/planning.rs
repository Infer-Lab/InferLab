//! Measurement selection, overrides, and resolution into immutable plans.

use super::domain::{
    AggregateSloBound, BenchDatasetCatalog, BenchDatasetFilter, BenchSessionDatasetCatalog,
    DatasetCacheState, ResolvedAggregateSlo, ResolvedBenchDefinition, ResolvedBenchPrompt,
    ResolvedBenchRandomShape, ResolvedBenchRequestSource, ResolvedBenchSessionSource,
    ResolvedBenchSloPolicy, ResolvedBenchSource,
};
use super::plan::{
    BenchCasePlan, BenchClientPlan, BenchExecutionPlan, BenchPlan, ClientCommandPlan,
    EvalExecutionPlan, EvalPlan, LoadShape, ManualBenchPlan, ManualBenchTarget,
    MeasurementOverridePlan, MeasurementPlan, MeasurementResolveContext, session_population_layout,
};
use super::{
    MeasurementModel, WorkloadEndpoint, WorkloadEndpointProtocol, WorkloadHttpAction,
    WorkloadHttpMethod, WorkloadServerMetricsEndpoint,
};
use crate::InferlabError;
use crate::bench_dataset_catalog;
use crate::resolve::current_environment;
use crate::server::ServerRecord;
use crate::toml_override::InvocationOverride;
use crate::toolchain::{self, InstalledBenchToolchain, InstalledEvalToolchain};
use crate::workspace::{
    AggregateSlo, BenchDefinition, BenchPrompt, BenchRequestSource, BenchSessionSource,
    BenchTpotApplicability, EvalDefinition, RequestRate, WorkloadSuiteDefinition, WorkspaceConfig,
    WorkspaceSnapshot, validate_bench, validate_eval,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn resolve_manual_bench(
    root: &Path,
    config: &WorkspaceConfig,
    snapshot: &WorkspaceSnapshot,
    server: &ServerRecord,
    bench_id: &str,
    overrides: &[String],
    capture: bool,
) -> Result<ManualBenchPlan, InferlabError> {
    if server.schema_version != ServerRecord::SCHEMA_VERSION {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "server record {:?} has unsupported schema version {}",
                server.id, server.schema_version
            ),
        });
    }
    if capture
        && !server
            .process_evidence
            .values()
            .any(|process| process.profiler.is_some())
    {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "server record {:?} was not started with profiling target preparation",
                server.id
            ),
        });
    }
    let recorded = &server.resolved;
    let declared_definition =
        config
            .benches
            .get(bench_id)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown selected bench {bench_id:?}"),
            })?;
    let indexed = InvocationOverride::parse_all(overrides)?;
    let (definition, override_plan) =
        apply_bench_overrides(bench_id, declared_definition.clone(), &indexed)?;
    let model_locator = recorded
        .server
        .roles
        .iter()
        .flat_map(|role| &role.replicas)
        .flat_map(|replica| &replica.ranks)
        .filter(|rank| rank.rank() == Some(0))
        .find_map(|rank| rank.allocation.model_locator.clone())
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!(
                "server record {:?} has no model locator usable by measurements",
                server.id
            ),
        })?;
    let toolchain = toolchain::require_bench()?;
    let command_env = current_environment()?;
    let capture_ids = if capture {
        vec![bench_id.to_owned()]
    } else {
        Vec::new()
    };
    let context =
        MeasurementResolveContext {
            workspace_root: root,
            workspace_source_exclusions: &snapshot.source_exclusions,
            endpoint: WorkloadEndpoint {
                protocol: match recorded.server.endpoint.protocol {
                    inferlab_protocol::EndpointProtocol::Http => WorkloadEndpointProtocol::Http,
                },
                host: recorded.server.endpoint.host.clone(),
                port: recorded.server.endpoint.port,
                completions_path: recorded.server.endpoint.completions_path.clone(),
                chat_completions_path: recorded.server.endpoint.chat_completions_path.clone(),
                server_metrics: recorded
                    .server
                    .endpoint
                    .server_metrics
                    .as_ref()
                    .map(|metrics| WorkloadServerMetricsEndpoint {
                        path: metrics.path.clone(),
                        port_name: metrics.port_name.clone(),
                        url: metrics.url.clone(),
                    }),
            },
            model: MeasurementModel {
                locator: model_locator,
                served_name: recorded.server.model.served_name.clone(),
            },
            prefix_cache_reset: recorded.server.endpoint.prefix_cache_reset.as_ref().map(
                |action| WorkloadHttpAction {
                    method: match action.method {
                        inferlab_protocol::HttpMethod::Post => WorkloadHttpMethod::Post,
                    },
                    path: action.path.clone(),
                },
            ),
            capture_ids: &capture_ids,
            command_env: &command_env,
            command_cwd: &root.join(".inferlab"),
        };
    Ok(ManualBenchPlan {
        invoking_inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
        target: ManualBenchTarget {
            server_record_id: server.id.clone(),
            producing_inferlab_version: server.inferlab_version.clone(),
            serving_snapshot: server.resolved.clone(),
        },
        measurement_workspace: snapshot.clone(),
        overrides: overrides.to_vec(),
        bench: build_bench_plan(
            bench_id,
            declared_definition,
            definition,
            override_plan,
            &context,
            &toolchain,
        )?,
    })
}

pub fn resolve_measurements(
    suite: &WorkloadSuiteDefinition,
    evals: &BTreeMap<String, EvalDefinition>,
    benches: &BTreeMap<String, BenchDefinition>,
    overrides: &[InvocationOverride],
    context: &MeasurementResolveContext<'_>,
) -> Result<MeasurementPlan, InferlabError> {
    validate_recipe_measurement_overrides(suite, evals, benches, overrides)?;
    for id in context.capture_ids {
        if !suite.evals.contains(id) && !suite.benches.contains(id) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "capture selects workload {id:?}, which is not in the workload suite"
                ),
            });
        }
    }
    let eval_toolchain = if suite
        .evals
        .iter()
        .any(|id| definitions_are_lm_eval(evals, id))
    {
        Some(toolchain::require_eval()?)
    } else {
        None
    };
    let bench_toolchain = if suite.benches.is_empty() {
        None
    } else {
        Some(toolchain::require_bench()?)
    };
    Ok(MeasurementPlan {
        gate: suite.gate.clone(),
        evals: suite
            .evals
            .iter()
            .map(|id| {
                resolve_eval(
                    id,
                    evals,
                    &recipe_measurement_overrides("evals", id, overrides),
                    context,
                    eval_toolchain.as_ref(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
        benches: suite
            .benches
            .iter()
            .map(|id| {
                resolve_bench(
                    id,
                    benches,
                    &recipe_measurement_overrides("benches", id, overrides),
                    context,
                    bench_toolchain
                        .as_ref()
                        .ok_or_else(|| InferlabError::InvalidConfig {
                            message: "Bench toolchain was not resolved".to_owned(),
                        })?,
                )
            })
            .collect::<Result<Vec<_>, InferlabError>>()?,
    })
}

fn resolve_bench(
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

fn build_bench_plan(
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
            ..
        } => request_source.as_ref().map_or_else(
            || {
                session_source.as_ref().map_or(
                    BenchTpotApplicability::Inapplicable,
                    BenchSessionSource::tpot_applicability,
                )
            },
            BenchRequestSource::tpot_applicability,
        ),
        BenchDefinition::AdaptiveServing { request_source, .. } => {
            request_source.tpot_applicability()
        }
    };
    let resolved_definition = resolve_bench_definition(&definition)?;
    if resolved_definition.server_metrics && context.endpoint.server_metrics.is_none() {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "bench {id:?} enables server_metrics, but the resolved server endpoint exposes no server-metrics capability"
            ),
        });
    }
    let slo = resolve_bench_slo_policy(&definition)?;
    let prefix_cache_reset = if resolved_definition.reset_prefix_cache {
        Some(
            context
                .prefix_cache_reset
                .clone()
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "bench {id:?} requests prefix-cache reset, but the server exposes no reset capability"
                    ),
                })?,
        )
    } else {
        None
    };
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
                    toolchain.runner.to_string_lossy().into_owned(),
                ],
                env,
                cwd: context.command_cwd.to_path_buf(),
            },
            prefix_cache_reset,
        },
    })
}

fn apply_bench_overrides(
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
    let mut value =
        toml::Value::try_from(definition).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare bench {id:?} for overrides: {error}"),
        })?;
    for item in overrides {
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

fn apply_definition_override(
    definition: &mut toml::Value,
    item: &InvocationOverride,
) -> Result<(), InferlabError> {
    let assignment = item.assignment()?;
    if assignment.root_key() == "kind" {
        return Err(InferlabError::InvalidOverride {
            value: item.raw().to_owned(),
            message: "measurement kind cannot be overridden".to_owned(),
        });
    }
    assignment.apply_to(definition, item.raw())
}

fn apply_eval_overrides(
    id: &str,
    definition: EvalDefinition,
    overrides: &[InvocationOverride],
) -> Result<(EvalDefinition, Vec<MeasurementOverridePlan>), InferlabError> {
    let mut value =
        toml::Value::try_from(definition).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare eval {id:?} for overrides: {error}"),
        })?;
    for item in overrides {
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
            message: format!("invalid effective Eval definition: {error}"),
        })?;
    validate_eval(id, &definition)?;
    Ok((definition, override_plan(overrides)))
}

fn recipe_measurement_overrides(
    section: &str,
    id: &str,
    overrides: &[InvocationOverride],
) -> Vec<InvocationOverride> {
    let prefix = format!("{section}.{id}.");
    overrides
        .iter()
        .filter_map(|item| item.under(&prefix))
        .collect()
}

fn validate_recipe_measurement_overrides(
    suite: &WorkloadSuiteDefinition,
    evals: &BTreeMap<String, EvalDefinition>,
    benches: &BTreeMap<String, BenchDefinition>,
    overrides: &[InvocationOverride],
) -> Result<(), InferlabError> {
    for item in overrides {
        let path = item.path();
        if path.starts_with("server.") {
            continue;
        }
        let (section, remaining, selected) = if let Some(remaining) = path.strip_prefix("evals.") {
            ("evals", remaining, &suite.evals)
        } else if let Some(remaining) = path.strip_prefix("benches.") {
            ("benches", remaining, &suite.benches)
        } else {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: "recipe override must be under server., evals.<id>., or benches.<id>."
                    .to_owned(),
            });
        };
        let Some((id, field)) = remaining.split_once('.') else {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: format!("expected {section}.<id>.<field>=<TOML-value>"),
            });
        };
        let declared = match section {
            "evals" => evals.contains_key(id),
            "benches" => benches.contains_key(id),
            _ => false,
        };
        if id.is_empty()
            || field.is_empty()
            || !declared
            || !selected.iter().any(|selected| selected == id)
        {
            return Err(InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: format!(
                    "{section} override must name a definition selected by the recipe's workload suite"
                ),
            });
        }
    }
    Ok(())
}

fn override_plan(overrides: &[InvocationOverride]) -> Vec<MeasurementOverridePlan> {
    overrides
        .iter()
        .map(|item| MeasurementOverridePlan {
            invocation_index: item.index(),
            value: item.raw().to_owned(),
        })
        .collect()
}

fn resolve_eval(
    id: &str,
    definitions: &BTreeMap<String, EvalDefinition>,
    overrides: &[InvocationOverride],
    context: &MeasurementResolveContext<'_>,
    toolchain: Option<&InstalledEvalToolchain>,
) -> Result<EvalPlan, InferlabError> {
    let declared_definition =
        definitions
            .get(id)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown selected eval definition {id:?}"),
            })?;
    let (mut definition, override_plan) =
        apply_eval_overrides(id, declared_definition.clone(), overrides)?;
    crate::workspace::validate_eval_task_source(context.workspace_root, id, &definition)?;
    if let EvalDefinition::LmEval {
        task: crate::workspace::EvalTaskSource::WorkspaceYaml { yaml },
        ..
    } = &mut definition
    {
        *yaml = context.workspace_root.join(&*yaml);
    }
    let execution = match &definition {
        EvalDefinition::OpenAiSmoke { .. } => EvalExecutionPlan::NativeOpenAiSmoke,
        EvalDefinition::LmEval { .. } => {
            let toolchain = toolchain.ok_or_else(|| InferlabError::InvalidConfig {
                message: "lm-eval toolchain was not resolved".to_owned(),
            })?;
            let bundled_task = match &definition {
                EvalDefinition::LmEval {
                    task: crate::workspace::EvalTaskSource::Bundled { bundled },
                    ..
                } => Some(Box::new(toolchain.bundled_task(bundled)?)),
                _ => None,
            };
            let mut env = context.command_env.clone();
            env.insert(
                "PYTHONPATH".to_owned(),
                toolchain.python_path.to_string_lossy().into_owned(),
            );
            env.insert("PYTHONNOUSERSITE".to_owned(), "1".to_owned());
            EvalExecutionPlan::LmEval {
                toolchain: Box::new(toolchain.identity.clone()),
                bundled_task,
                command: ClientCommandPlan {
                    argv: vec![
                        toolchain.python.to_string_lossy().into_owned(),
                        toolchain.runner.to_string_lossy().into_owned(),
                    ],
                    env,
                    cwd: context.command_cwd.to_path_buf(),
                },
            }
        }
    };
    Ok(EvalPlan {
        id: id.to_owned(),
        capture: context.capture_ids.iter().any(|capture| capture == id),
        declared_definition,
        definition,
        overrides: override_plan,
        endpoint: context.endpoint.clone(),
        model: context.model.clone(),
        workspace_source_exclusions: context.workspace_source_exclusions.to_vec(),
        execution,
    })
}

fn definitions_are_lm_eval(definitions: &BTreeMap<String, EvalDefinition>, id: &str) -> bool {
    matches!(definitions.get(id), Some(EvalDefinition::LmEval { .. }))
}

fn resolve_bench_definition(
    definition: &BenchDefinition,
) -> Result<ResolvedBenchDefinition, InferlabError> {
    match definition {
        BenchDefinition::Serving {
            request_source,
            session_source,
            seed,
            server_metrics,
            request_body,
            request_slo,
            reset_prefix_cache,
            timeout_seconds,
            ..
        } => {
            let (source, prompt) = match (request_source, session_source) {
                (Some(request_source), None) => (
                    ResolvedBenchSource::Requests {
                        request_source: resolve_bench_request_source(request_source)?,
                    },
                    resolved_request_source_prompt(request_source),
                ),
                (None, Some(session_source)) => (
                    ResolvedBenchSource::Sessions {
                        session_source: resolve_bench_session_source(session_source)?,
                    },
                    ResolvedBenchPrompt::from_definition(&BenchPrompt::ServerChat),
                ),
                _ => {
                    return Err(InferlabError::InvalidConfig {
                        message:
                            "resolved serving Bench requires exactly one request or session source"
                                .to_owned(),
                    });
                }
            };
            Ok(ResolvedBenchDefinition {
                source,
                prompt,
                server_metrics: *server_metrics,
                seed: *seed,
                request_body: request_body.clone(),
                request_slo: request_slo.clone(),
                timeout_seconds: *timeout_seconds,
                reset_prefix_cache: *reset_prefix_cache,
            })
        }
        BenchDefinition::AdaptiveServing {
            request_source,
            seed,
            server_metrics,
            request_body,
            request_slo,
            reset_prefix_cache,
            timeout_seconds,
            ..
        } => Ok(ResolvedBenchDefinition {
            source: ResolvedBenchSource::Requests {
                request_source: resolve_bench_request_source(request_source)?,
            },
            prompt: resolved_request_source_prompt(request_source),
            server_metrics: *server_metrics,
            seed: *seed,
            request_body: request_body.clone(),
            request_slo: request_slo.clone(),
            timeout_seconds: *timeout_seconds,
            reset_prefix_cache: *reset_prefix_cache,
        }),
    }
}

fn resolved_request_source_prompt(source: &BenchRequestSource) -> ResolvedBenchPrompt {
    match source {
        BenchRequestSource::Random { prompt, .. }
        | BenchRequestSource::RandomMixture { prompt, .. } => {
            ResolvedBenchPrompt::from_definition(prompt)
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

fn resolve_bench_execution(
    id: &str,
    definition: &BenchDefinition,
) -> Result<BenchExecutionPlan, InferlabError> {
    match definition {
        BenchDefinition::Serving {
            session_source,
            concurrency,
            prompts_per_concurrency,
            warmup_prompts_per_concurrency,
            sessions_per_concurrency,
            warmup_sessions_per_concurrency,
            request_rates,
            request_count,
            duration_seconds,
            burstiness,
            ..
        } => {
            let mut cases = Vec::with_capacity(concurrency.len() + request_rates.len());
            for (index, concurrency) in concurrency.iter().copied().enumerate() {
                if session_source.is_some() {
                    let multiplier =
                        sessions_per_concurrency.ok_or_else(|| InferlabError::InvalidConfig {
                            message: format!("bench {id:?} is missing sessions_per_concurrency"),
                        })?;
                    let session_count = concurrency.checked_mul(multiplier).ok_or_else(|| {
                        InferlabError::InvalidConfig {
                            message: format!("bench {id:?} profiling session count exceeds u32"),
                        }
                    })?;
                    let warmup_session_count = concurrency
                        .checked_mul(*warmup_sessions_per_concurrency)
                        .ok_or_else(|| InferlabError::InvalidConfig {
                            message: format!("bench {id:?} warmup session count exceeds u32"),
                        })?;
                    cases.push(BenchCasePlan {
                        id: format!("concurrency-{index:03}"),
                        load_shape: LoadShape::ConcurrencyLimited { concurrency },
                        request_count: 0,
                        warmup_request_count: 0,
                        session_count: Some(session_count),
                        warmup_session_count: Some(warmup_session_count),
                    });
                    continue;
                }
                let multiplier =
                    prompts_per_concurrency.ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!("bench {id:?} is missing prompts_per_concurrency"),
                    })?;
                let request_count = concurrency.checked_mul(multiplier).ok_or_else(|| {
                    InferlabError::InvalidConfig {
                        message: format!("bench {id:?} concurrency request count exceeds u32"),
                    }
                })?;
                let warmup_request_count = concurrency
                    .checked_mul(*warmup_prompts_per_concurrency)
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!("bench {id:?} warmup request count exceeds u32"),
                    })?;
                cases.push(BenchCasePlan {
                    id: format!("concurrency-{index:03}"),
                    load_shape: LoadShape::ConcurrencyLimited { concurrency },
                    request_count,
                    warmup_request_count,
                    session_count: None,
                    warmup_session_count: None,
                });
            }
            for (index, rate) in request_rates.iter().cloned().enumerate() {
                let count = resolved_request_count(id, &rate, *request_count, *duration_seconds)?;
                cases.push(BenchCasePlan {
                    id: format!("request-rate-{index:03}"),
                    load_shape: LoadShape::RequestRateLimited {
                        request_rate: rate,
                        burstiness: *burstiness,
                    },
                    request_count: count,
                    warmup_request_count: 0,
                    session_count: None,
                    warmup_session_count: None,
                });
            }
            Ok(BenchExecutionPlan::Matrix { cases })
        }
        BenchDefinition::AdaptiveServing {
            initial_request_rates,
            max_search_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
            ..
        } => {
            let mut initial_request_rates = initial_request_rates.clone();
            initial_request_rates.sort_by(f64::total_cmp);
            initial_request_rates.dedup();
            Ok(BenchExecutionPlan::Adaptive {
                policy: "highest-feasible-rate-v1".to_owned(),
                initial_request_rates,
                max_search_steps: *max_search_steps,
                min_rate_resolution: *min_rate_resolution,
                request_count: *request_count,
                duration_seconds: *duration_seconds,
            })
        }
    }
}

fn required_population_count(
    id: &str,
    execution: &BenchExecutionPlan,
) -> Result<u32, InferlabError> {
    match execution {
        BenchExecutionPlan::Matrix { cases } => cases.iter().try_fold(0_u32, |largest, case| {
            let entries = match (case.warmup_session_count, case.session_count) {
                (Some(warmup), Some(profiling)) => session_population_layout(warmup, profiling)
                    .map(|layout| layout.required_entries),
                _ => case.warmup_request_count.checked_add(case.request_count),
            }
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("bench {id:?} population exceeds u32"),
            })?;
            Ok(largest.max(entries))
        }),
        BenchExecutionPlan::Adaptive {
            initial_request_rates,
            max_search_steps,
            request_count,
            duration_seconds,
            ..
        } => {
            if let Some(request_count) = request_count {
                return Ok(*request_count);
            }
            let initial = initial_request_rates
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!("adaptive bench {id:?} has no initial request rate"),
                })?;
            let factor = 2_f64.powf(f64::from(*max_search_steps));
            let largest_rate = initial * factor;
            if !largest_rate.is_finite() {
                return Err(InferlabError::InvalidConfig {
                    message: format!("adaptive bench {id:?} request population exceeds u32"),
                });
            }
            resolved_request_count(
                id,
                &RequestRate::Finite(largest_rate),
                None,
                *duration_seconds,
            )
        }
    }
}

pub fn resolved_request_count(
    bench_id: &str,
    rate: &RequestRate,
    request_count: Option<u32>,
    duration_seconds: Option<u64>,
) -> Result<u32, InferlabError> {
    if let Some(request_count) = request_count {
        return Ok(request_count);
    }
    let rate = rate.finite().ok_or_else(|| InferlabError::InvalidConfig {
        message: format!("bench {bench_id:?} cannot derive request count for an unbounded rate"),
    })?;
    let duration = duration_seconds.ok_or_else(|| InferlabError::InvalidConfig {
        message: format!("bench {bench_id:?} has no request count policy"),
    })?;
    let count = (rate * duration as f64).ceil().max(1.0);
    if count > f64::from(u32::MAX) {
        return Err(InferlabError::InvalidConfig {
            message: format!("bench {bench_id:?} request count exceeds u32"),
        });
    }
    Ok(count as u32)
}

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
        BenchDefinition, BenchPrefixSharing, BenchPrompt, BenchRandomShape, BenchRequestSource,
        BenchTokenSelector, validate_bench,
    };

    #[test]
    fn synthetic_request_sources_resolve_effective_prefix_and_total_weight()
    -> Result<(), Box<dyn std::error::Error>> {
        let prefix = resolve_bench_request_source(&BenchRequestSource::Random {
            prompt: BenchPrompt::Flat,
            input_tokens: BenchTokenSelector::Fixed(8000),
            output_tokens: BenchTokenSelector::Fixed(1000),
            prefix_sharing: Some(BenchPrefixSharing::Ratio {
                shared_prefix_ratio: 0.75,
            }),
            shared_system_content: None,
        })?;
        let mixture = resolve_bench_request_source(&BenchRequestSource::RandomMixture {
            prompt: BenchPrompt::ServerChat,
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
        let resolved = serde_json::to_value(resolve_bench_definition(&definition)?)?;
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
}
