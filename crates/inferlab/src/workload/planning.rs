//! Measurement selection and composition into immutable plans.

mod bench;
mod eval;
mod overrides;

use super::plan::{ManualBenchPlan, ManualBenchTarget, MeasurementPlan, MeasurementResolveContext};
use super::{
    MeasurementModel, WorkloadEndpoint, WorkloadEndpointProtocol, WorkloadHttpAction,
    WorkloadHttpMethod, WorkloadServerMetricsEndpoint,
};
use crate::InferlabError;
use crate::resolve::current_environment;
use crate::server::ServerRecord;
use crate::toml_override::InvocationOverride;
use crate::toolchain;
use crate::workspace::{
    BenchDefinition, EvalDefinition, WorkloadSuiteDefinition, WorkspaceConfig, WorkspaceSnapshot,
};
use std::collections::BTreeMap;

pub(crate) use bench::resolved_request_count;
use bench::{apply_bench_overrides, build_bench_plan, resolve_bench};
use eval::{definitions_are_lm_eval, resolve_eval};
use overrides::{recipe_measurement_overrides, validate_recipe_measurement_overrides};
use std::path::Path;

pub(crate) fn resolve_manual_bench(
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

pub(crate) fn resolve_measurements(
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
