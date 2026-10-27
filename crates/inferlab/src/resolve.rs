mod allocation;
mod integration;
pub(crate) mod occupancy;
mod realization;
mod selection;
mod synthetic_acceptance;
pub(crate) mod topology;

use crate::InferlabError;
use crate::adapter::{AdapterClient, executable_name};
use crate::execution::{
    CasePlan, EndpointPlan, IntegrationPlan, ModelPlan, PlacementPlan, ResolvedExecution,
    ResourcePlan, ServerPlan, StackPlan, SyntheticAcceptancePlan, Workflow,
};
use crate::record::STATE_DIR;
use crate::toml_override::InvocationOverride;
use crate::workload::{
    ConditioningServingShape, MeasurementPlan, MeasurementResolveContext, resolve_measurements,
};
use crate::workspace::LoadedWorkspace;
use inferlab_protocol::{EndpointProtocol, ProtocolVersion, ServeProcessAllocation};
use std::collections::BTreeMap;

use inferlab_serve_domain::{
    ModelLocatorSource, ResolvedProcessAllocation, RuntimeRealizationParts,
};
use integration::{plan_integration, render_integration};
use realization::{ProcessHierarchy, assemble_process_hierarchy, realize_runtime};
use selection::{WorkflowSelection, resolve_effective_server_input, select_workflow};
use topology::profiler_escapes_plan;

#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionTarget<'a> {
    Server(&'a str),
    Recipe(&'a str),
}

pub(crate) struct ResolveRequest<'a> {
    pub workflow: Workflow,
    pub target: ExecutionTarget<'a>,
    pub case: Option<&'a str>,
    pub placement: Option<&'a str>,
    pub overrides: &'a [String],
    pub captures: &'a [String],
    /// A validated image selection ([[RFC-0003:C-RUNTIME-WORKFLOWS]]):
    /// resolution keys realization-dependent facts (adapter execution,
    /// runtime cache identity) on it and applies the containerized
    /// substitution before returning.
    pub image: Option<&'a crate::image::launch::ImageLaunchPlan>,
    /// A validated external-image selection, mutually exclusive with
    /// `image`: the same substitution, launched through an explicit command
    /// override as an explicitly not-qualified realization
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub external: Option<&'a crate::image::launch::ExternalImagePlan>,
    /// Wait for occupied assigned devices to free instead of failing the
    /// launch ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub wait_for_devices: bool,
}

#[allow(clippy::too_many_arguments)]
fn compose_measurements(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    overrides: &[InvocationOverride],
    selection: &WorkflowSelection<'_>,
    synthetic_acceptance: bool,
    public_endpoint: &EndpointPlan,
    conditioning_serving: ConditioningServingShape,
    serve_state_probe: Option<crate::workload::WorkloadServeStateProbe>,
    allocations: &[ResolvedProcessAllocation],
) -> Result<Option<MeasurementPlan>, InferlabError> {
    if matches!(request.workflow, Workflow::ServeStart) {
        return Ok(None);
    }
    let command_env = current_environment()?;
    let command_cwd = workspace.root.join(STATE_DIR);
    let suite = selection
        .suite
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: "recipe workflow has no workload suite".to_owned(),
        })?;
    let model_locator = inferlab_serve_domain::measurement_model_locator(
        allocations
            .iter()
            .filter_map(|allocation| {
                let locator = allocation.model_locator()?;
                let controller_local = matches!(
                    allocation.wire(),
                    ServeProcessAllocation::ModelRank {
                        launch: inferlab_protocol::AllocationLaunch::Local,
                        ..
                    }
                );
                Some((
                    controller_local,
                    allocation.model_locator_source(),
                    locator,
                ))
            })
            .chain(
                selection
                    .weight
                    .locator
                    .as_deref()
                    .map(|locator| (false, Some(ModelLocatorSource::Fallback), locator)),
            ),
    )
    .ok_or_else(|| InferlabError::InvalidConfig {
        message: format!(
            "recipe target server {:?} declares no model locator usable by measurements on the controller machine (model weight binding {:?})",
            selection.server_id, selection.server.model
        ),
    })?
    .to_owned();
    resolve_measurements(
        suite,
        &workspace.config.evals,
        &workspace.config.benches,
        overrides,
        &MeasurementResolveContext {
            workspace_root: &workspace.root,
            workspace_source_exclusions: &workspace.snapshot.source_exclusions,
            endpoint: crate::workload::WorkloadEndpoint {
                protocol: match public_endpoint.protocol {
                    EndpointProtocol::Http => crate::workload::WorkloadEndpointProtocol::Http,
                },
                host: public_endpoint.host.clone(),
                port: public_endpoint.port,
                completions_path: public_endpoint.completions_path.clone(),
                chat_completions_path: public_endpoint.chat_completions_path.clone(),
                server_metrics: public_endpoint.server_metrics.as_ref().map(|metrics| {
                    crate::workload::WorkloadServerMetricsEndpoint {
                        path: metrics.path.clone(),
                        port_name: metrics.port_name.clone(),
                        url: metrics.url.clone(),
                    }
                }),
                prompt_cache_read_zero_representation: public_endpoint
                    .prompt_cache_read_zero_representation,
            },
            model: crate::workload::MeasurementModel {
                locator: model_locator,
                served_name: selection.model.served_name.clone(),
            },
            model_id: &selection.server.model,
            prefix_cache_reset: public_endpoint
                .prefix_cache_reset
                .as_ref()
                .map(crate::workload::WorkloadHttpAction::from_wire),
            replica_prefix_cache_resets: public_endpoint
                .replica_prefix_cache_resets
                .iter()
                .map(crate::workload::WorkloadReplicaReset::from_plan)
                .collect(),
            prefix_cache_conditioning: public_endpoint
                .prefix_cache_conditioning
                .as_ref()
                .map(crate::workload::WorkloadHttpAction::from_wire),
            conditioning_serving,
            serve_state_probe,
            synthetic_acceptance,
            capture_ids: request.captures,
            command_env: &command_env,
            command_cwd: &command_cwd,
        },
    )
    .map(Some)
}

pub(crate) fn resolve<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    adapter: &C,
) -> Result<ResolvedExecution, InferlabError> {
    let overrides = InvocationOverride::parse_all(request.overrides)?;
    let selection = select_workflow(workspace, request)?;
    let effective =
        resolve_effective_server_input(&selection, request, &overrides, &workspace.root)?;
    let server = selection.server;
    let stack = selection.stack;
    let server_id = selection.server_id.as_str();
    let case_id = selection.case_id.as_deref();
    let topology = effective.topology;
    let profiling = effective.profiling.is_some();

    let served_name = selection.model.served_name.clone();
    let planned_stage = plan_integration(workspace, &selection, &effective, adapter)?;
    let (rendered_stage, unavailable) = render_integration(
        workspace,
        request,
        &selection,
        &effective,
        &planned_stage,
        adapter,
    )?;
    let planned = planned_stage.planned();
    // The accepted plan's outcome was validated against the request
    // declaration during planning; pair them into the evidence object.
    let synthetic_acceptance = match (
        &effective.synthetic_acceptance,
        planned.synthetic_acceptance,
    ) {
        (Some(resolved), Some(outcome)) => Some(SyntheticAcceptancePlan {
            declared: resolved.declared.clone(),
            acceptance_length: outcome.acceptance_length,
            thinking_mode: resolved.effective_thinking_mode().map(str::to_owned),
            draft_count: outcome.draft_count,
        }),
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => {
            return Err(InferlabError::AdapterSemantics {
                message: format!(
                    "integration {:?} synthetic acceptance outcome does not match the request declaration",
                    stack.integration
                ),
            });
        }
    };
    let RuntimeRealizationParts {
        processes,
        public_endpoint,
        device_count,
        selected_machines,
        network,
        remote_workspaces,
        remote_containers,
    } = realize_runtime(
        workspace,
        request,
        &selection,
        &effective,
        &planned_stage,
        &rendered_stage,
    )?;
    let serve_state_probe = serve_state_probe_plan(
        workspace.local.adapter.process_timeout().as_secs(),
        &stack.integration,
        planned,
        rendered_stage.allocations(),
        effective.readiness_attempt_timeout_seconds,
        request.image.is_some() || request.external.is_some(),
    )?;
    let workload_serve_state_probe = serve_state_probe.as_ref().map(|plan| {
        crate::workload::WorkloadServeStateProbe::new(
            &stack.integration,
            &stack.pixi_environment,
            inferlab_protocol::ServeModelInput {
                id: server.model.clone(),
                served_name: selection.model.served_name.clone(),
            },
            effective.topology,
            plan,
        )
    });
    let conditioning_serving = ConditioningServingShape::resolve(
        planned.gateway.is_some(),
        planned.roles.iter().map(|role| {
            (
                role.public_endpoint.is_some(),
                role.kind,
                role.effective_replica_count,
                role.effective_parallelism
                    .attention
                    .as_ref()
                    .and_then(|attention| attention.data_parallel_size)
                    .unwrap_or(1),
            )
        }),
    )
    .shared(
        integration::shared_prefix_cache(planned),
        integration::prefix_publication_ms(planned),
    );
    let measurements = compose_measurements(
        workspace,
        request,
        &overrides,
        &selection,
        effective.synthetic_acceptance.is_some(),
        &public_endpoint,
        conditioning_serving,
        workload_serve_state_probe,
        rendered_stage.allocations(),
    )?;
    let ProcessHierarchy {
        roles: role_plans,
        frontend,
        discovery,
        kv_store,
    } = assemble_process_hierarchy(&stack.integration, &effective, &planned_stage, processes)?;
    let mut execution = ResolvedExecution {
        workflow: request.workflow,
        workspace: workspace.snapshot.clone(),
        recipe: selection.recipe,
        stack: StackPlan {
            id: server.stack.clone(),
            integration: stack.integration.clone(),
            pixi_environment: stack.pixi_environment.clone(),
            source_paths: stack.source_paths.clone(),
            realization: if request.image.is_some() {
                crate::environment::CheckRealization::Image
            } else if request.external.is_some() {
                crate::environment::CheckRealization::ExternalImage
            } else {
                crate::environment::CheckRealization::LocalWorkspace
            },
            checks: selection.stack_checks,
        },
        server: ServerPlan {
            id: server_id.to_owned(),
            wait_for_devices: request.wait_for_devices,
            case: case_id
                .zip(selection.case_selection)
                .map(|(id, selection)| CasePlan {
                    id: id.to_owned(),
                    selection,
                }),
            explicit_overrides: effective
                .override_patches
                .iter()
                .map(|item| item.invocation.raw().to_owned())
                .collect(),
            declarations: effective.declarations,
            topology,
            profiling,
            readiness_timeout_seconds: effective.readiness_timeout_seconds,
            readiness_attempt_timeout_seconds: effective.readiness_attempt_timeout_seconds,
            capture_arm_deadline_seconds: effective.capture_arm_deadline_seconds,
            capture_control_deadline_seconds: effective.capture_control_deadline_seconds,
            capture_finalization_deadline_seconds: effective.capture_finalization_deadline_seconds,
            kv_transfer: effective.kv_transfer,
            synthetic_acceptance,
            auxiliary_models: effective
                .auxiliary_models
                .iter()
                .map(|auxiliary| crate::execution::AuxiliaryModelPlan {
                    kind: auxiliary.kind,
                    model: ModelPlan {
                        id: auxiliary.model.id.clone(),
                        served_name: auxiliary.model.served_name.clone(),
                        fallback_locator: None,
                    },
                })
                .collect(),
            serve_state_probe,
            shared_prefix_cache: integration::shared_prefix_cache(planned),
            prefix_publication_ms: integration::prefix_publication_ms(planned),
            stack_settings: planned.stack_settings.clone(),
            frontend,
            discovery,
            kv_store,
            profiler_escapes: profiler_escapes_plan(server),
            model: ModelPlan {
                id: server.model.clone(),
                served_name,
                fallback_locator: selection.weight.locator.clone(),
            },
            image: None,
            external_image: None,
            integration: IntegrationPlan {
                id: stack.integration.clone(),
                adapter_id: planned.integration.adapter_id.clone(),
                adapter_version: planned.integration.adapter_version.clone(),
                framework: planned.integration.framework.clone(),
                framework_version: planned.integration.framework_version.clone(),
                executable: executable_name(&stack.integration),
                protocol_version: ProtocolVersion::CURRENT,
                plan_request_sha256: planned_stage.evidence().request_sha256().to_owned(),
                plan_response_sha256: planned_stage.evidence().response_sha256().to_owned(),
                render_request_sha256: rendered_stage.evidence().request_sha256().to_owned(),
                render_response_sha256: rendered_stage.evidence().response_sha256().to_owned(),
                plan_timing: Some(planned_stage.evidence().timing().clone()),
                render_timing: Some(rendered_stage.evidence().timing().clone()),
            },
            resources: ResourcePlan { device_count },
            placement: PlacementPlan {
                id: selection.placement_id,
                selection: selection.placement_selection,
                machines: selected_machines,
                unavailable_ports: unavailable.ports,
                unavailable_devices: unavailable.devices,
                remote_workspaces,
                remote_containers,
            },
            network,
            roles: role_plans,
            links: planned.links.clone(),
            endpoint: public_endpoint,
        },
        measurements,
    };
    if let Some(image) = request.image {
        crate::image::launch::apply(&mut execution, image, &workspace.local.machines)?;
    } else if let Some(external) = request.external {
        crate::image::launch::apply_external(
            &mut execution,
            external,
            &workspace.local.machines,
            &workspace.local.adapter,
        )?;
    }
    // After containerization, so each process's command is the one that
    // launches ([[RFC-0002:C-LOCAL-PLACEMENT]]).
    crate::server::apply_numa_pinning(execution.server.processes_mut(), &workspace.local.machines)?;
    Ok(execution)
}

pub(crate) fn current_environment() -> Result<BTreeMap<String, String>, InferlabError> {
    std::env::vars_os()
        .map(|(key, value)| {
            let key = key
                .into_string()
                .map_err(|_| InferlabError::InvalidConfig {
                    message: "process environment contains a non-UTF-8 variable name".to_owned(),
                })?;
            let value = value
                .into_string()
                .map_err(|_| InferlabError::InvalidConfig {
                    message: format!("process environment variable {key:?} is not valid UTF-8"),
                })?;
            Ok((key, value))
        })
        .collect()
}

/// Freeze the serve-state probe the integration declared: every model-serving
/// rank-zero process and the store endpoint it was rendered with
/// ([[RFC-0003:C-SERVE-KV-STORE]]). The probe runs through the workspace
/// adapter environment, so an image-launched server cannot declare one yet.
fn serve_state_probe_plan(
    adapter_timeout_seconds: u64,
    integration: &str,
    planned: &inferlab_protocol::PlanServeResult,
    allocations: &[inferlab_serve_domain::ResolvedProcessAllocation],
    attempt_bound_seconds: u64,
    image_launched: bool,
) -> Result<Option<crate::execution::ServeStateProbePlan>, InferlabError> {
    if !planned.serve_state_probe {
        return Ok(None);
    }
    let Some(store_plan) = planned
        .kv_store
        .as_ref()
        .filter(|store| !store.role_access.is_empty())
    else {
        return Err(InferlabError::AdapterSemantics {
            message: format!(
                "integration {integration:?} declared a serve-state probe without a KV store that a role uses"
            ),
        });
    };
    if image_launched {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} declares a serve-state probe, which image-launched servers do not support yet"
            ),
        });
    }
    let mut processes = Vec::new();
    let mut kv_store = None;
    for allocation in allocations {
        if let inferlab_protocol::ServeProcessAllocation::ModelRank {
            process,
            role,
            replica,
            rank: 0,
            endpoint: Some(endpoint),
            kv_store: store,
            ..
        } = allocation.wire()
            && store_plan.role_access.contains_key(role)
        {
            processes.push(inferlab_protocol::ProbeServeTarget {
                process: process.clone(),
                role: role.clone(),
                replica: *replica,
                endpoint: endpoint.clone(),
            });
            if let Some(store) = store {
                kv_store = Some((**store).clone());
            }
        }
    }
    if processes.is_empty() {
        return Err(InferlabError::AdapterSemantics {
            message: format!(
                "integration {integration:?} declared a serve-state probe, but no model-serving rank-zero process of a role that uses the KV store has an endpoint to probe"
            ),
        });
    }
    Ok(Some(crate::execution::ServeStateProbePlan {
        processes,
        kv_store,
        attempt_bound_seconds,
        adapter_timeout_seconds,
    }))
}

#[cfg(test)]
mod tests {
    use super::serve_state_probe_plan;

    /// A probe with nothing to read would report every process engaged, so
    /// it is rejected while resolving ([[RFC-0003:C-SERVE-KV-STORE]]).
    #[test]
    fn a_serve_state_probe_without_a_probeable_process_is_rejected()
    -> Result<(), Box<dyn std::error::Error>> {
        let planned: inferlab_protocol::PlanServeResult =
            serde_json::from_value(serde_json::json!({
                "integration": {
                    "adapter_id": "fixture",
                    "adapter_version": "1",
                    "framework": "fixture",
                    "framework_version": "1"
                },
                "roles": [],
                "replicas": [],
                "links": [],
                "kv_store": {
                    "backend": "mooncake",
                    "implementation": "mooncake",
                    "implementation_version": "1",
                    "effective_settings": {},
                    "role_access": { "serve": "read_write" },
                    "readiness": { "kind": "process_alive" }
                },
                "serve_state_probe": true
            }))?;

        let error = serve_state_probe_plan(30, "fixture", &planned, &[], 5, false)
            .err()
            .ok_or("a probe without targets resolved")?;

        assert!(
            error
                .to_string()
                .contains("no model-serving rank-zero process"),
            "{error}"
        );
        Ok(())
    }
}
