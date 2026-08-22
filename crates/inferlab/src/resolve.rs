mod allocation;
mod integration;
mod realization;
mod selection;
mod topology;

use crate::InferlabError;
use crate::adapter::{AdapterClient, executable_name};
use crate::execution::{
    CasePlan, EndpointPlan, IntegrationPlan, ModelPlan, PlacementPlan, ResolvedExecution,
    ResourcePlan, ServerPlan, StackPlan, Workflow,
};
use crate::toml_override::InvocationOverride;
use crate::workload::{
    ConditioningServingShape, MeasurementPlan, MeasurementResolveContext, resolve_measurements,
};
use crate::workspace::LoadedWorkspace;
use inferlab_protocol::{EndpointProtocol, ProtocolVersion, ServeProcessAllocation};
use std::collections::BTreeMap;

use inferlab_serve_domain::{ResolvedProcessAllocation, RuntimeRealizationParts};
use integration::{plan_integration, render_integration};
use realization::{assemble_process_hierarchy, realize_runtime};
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
}

fn compose_measurements(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    overrides: &[InvocationOverride],
    selection: &WorkflowSelection<'_>,
    public_endpoint: &EndpointPlan,
    conditioning_serving: ConditioningServingShape,
    allocations: &[ResolvedProcessAllocation],
) -> Result<Option<MeasurementPlan>, InferlabError> {
    if matches!(request.workflow, Workflow::ServeStart) {
        return Ok(None);
    }
    let command_env = current_environment()?;
    let command_cwd = workspace.root.join(".inferlab");
    let suite = selection
        .suite
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: "recipe workflow has no workload suite".to_owned(),
        })?;
    let model_locator = selection
        .weight
        .locator
        .clone()
        .or_else(|| {
            allocations
                .iter()
                .find(|allocation| {
                    matches!(
                        allocation.wire(),
                        ServeProcessAllocation::ModelRank { rank: 0, .. }
                    )
                })
                .and_then(|allocation| allocation.model_locator().map(str::to_owned))
        })
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!(
                "recipe target server {:?} has no model locator usable by its measurements",
                selection.server_id
            ),
        })?;
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
            prefix_cache_reset: public_endpoint.prefix_cache_reset.as_ref().map(|action| {
                crate::workload::WorkloadHttpAction {
                    method: match action.method {
                        inferlab_protocol::HttpMethod::Post => {
                            crate::workload::WorkloadHttpMethod::Post
                        }
                    },
                    path: action.path.clone(),
                }
            }),
            prefix_cache_conditioning: public_endpoint.prefix_cache_conditioning.as_ref().map(
                |action| crate::workload::WorkloadHttpAction {
                    method: match action.method {
                        inferlab_protocol::HttpMethod::Post => {
                            crate::workload::WorkloadHttpMethod::Post
                        }
                    },
                    path: action.path.clone(),
                },
            ),
            conditioning_serving,
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
    let effective = resolve_effective_server_input(&selection, request, &overrides)?;
    let server = selection.server;
    let stack = selection.stack;
    let server_id = selection.server_id.as_str();
    let case_id = selection.case_id.as_deref();
    let topology = effective.topology;
    let profiling = effective.profiling.is_some();

    let served_name = selection.model.served_name.clone();
    let planned_stage = plan_integration(workspace, &selection, &effective, adapter)?;
    let rendered_stage = render_integration(
        workspace,
        request,
        &selection,
        &effective,
        &planned_stage,
        adapter,
    )?;
    let planned = planned_stage.planned();
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
    )?
    .into_parts();
    let conditioning_serving = ConditioningServingShape::resolve(
        planned.gateway.is_some(),
        planned.roles.iter().map(|role| {
            (
                role.public_endpoint.is_some(),
                role.effective_parallelism
                    .attention
                    .as_ref()
                    .and_then(|attention| attention.data_parallel_size)
                    .unwrap_or(1),
            )
        }),
    );
    let measurements = compose_measurements(
        workspace,
        request,
        &overrides,
        &selection,
        &public_endpoint,
        conditioning_serving,
        rendered_stage.allocations(),
    )?;
    let (role_plans, frontend) =
        assemble_process_hierarchy(&stack.integration, &effective, &planned_stage, processes)?;
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
            frontend,
            profiler_escapes: profiler_escapes_plan(server),
            model: ModelPlan {
                id: server.model.clone(),
                served_name,
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
                protocol_version: ProtocolVersion::V8,
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
