use super::allocation::{launch_plan, pixi_command, readiness_plan};
use super::integration::{rendered_process_id, validate_launch_file_declarations};
use super::selection::{EffectiveServerInput, WorkflowSelection, validate_effective_settings};
use super::topology::{endpoint_url, public_endpoint_requirement, resolved_gateway_process_id};
use super::{ResolveRequest, current_environment};
use crate::InferlabError;
use crate::execution::{
    AllocationPlan, EndpointPlan, FrontendPlan, GatewayComponentPlan, PdRouterComponentPlan,
    ProcessCommandSource, ProcessIdentityPlan, ProcessPlan, RolePlan, RoleReplicaPlan,
    ServerMetricsEndpointPlan,
};
use crate::workspace::{LaunchBinding, LoadedWorkspace};
use inferlab_profiler::plan::{
    CaptureWindowActionPlan, CaptureWindowControlEndpointPlan, ProcessCapturePlan,
};
use inferlab_protocol::{
    EndpointAssignment, RenderSource, RenderedServeProcess, ServeProcessAllocation,
};
use inferlab_runtime::plan::{CommandPlan, ProcessEndpointPlan};
use inferlab_serve_domain::{
    PendingCaptureWindowActionPlan, PlannedServeStage, ProcessRequirement,
    ProcessRequirementIdentity, RenderedServeStage, ResolvedProcessAllocation,
    RuntimeRealizationParts, RuntimeRealizationStage,
};
use std::collections::{BTreeMap, BTreeSet};

fn resolve_capture_target(
    requirement: &ProcessRequirement,
    gateway_process_id: Option<&str>,
    allocations: &[ResolvedProcessAllocation],
) -> Result<Option<ProcessCapturePlan>, InferlabError> {
    requirement
        .capture_target()
        .map(|target| {
            let control_process_id = match target.window_control_endpoint() {
                CaptureWindowControlEndpointPlan::ReplicaEntry => {
                    target.replica_entry_process_id().to_owned()
                }
                CaptureWindowControlEndpointPlan::Gateway => gateway_process_id
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: "Gateway profiling window control has no resolved Gateway process"
                            .to_owned(),
                    })?
                    .to_owned(),
            };
            let control_endpoint = allocations
                .iter()
                .find(|allocation| allocation.process() == control_process_id)
                .and_then(ResolvedProcessAllocation::endpoint)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "profiling window-control process {control_process_id:?} has no resolved endpoint"
                    ),
                })?;
            Ok(ProcessCapturePlan {
                window_control_endpoint: target.window_control_endpoint(),
                control_process_id,
                start: resolve_capture_window_action(target.start(), control_endpoint),
                stop: resolve_capture_window_action(target.stop(), control_endpoint),
                escapes: target.escapes().clone(),
            })
        })
        .transpose()
}

fn resolve_capture_window_action(
    action: &PendingCaptureWindowActionPlan,
    endpoint: &EndpointAssignment,
) -> CaptureWindowActionPlan {
    CaptureWindowActionPlan {
        method: action.method(),
        path: action.path().to_owned(),
        body: action.body().cloned(),
        effective_url: format!("{}{}", endpoint_url(endpoint), action.path()),
    }
}

pub(super) fn realize_runtime(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    selection: &WorkflowSelection<'_>,
    effective: &EffectiveServerInput,
    planned_stage: &PlannedServeStage,
    rendered_stage: &RenderedServeStage,
) -> Result<RuntimeRealizationStage, InferlabError> {
    let planned = planned_stage.planned();
    let requirements = planned_stage.requirements();
    let public_process = planned_stage.public_process();
    let allocations = rendered_stage.allocations();
    let endpoint_requirement =
        public_endpoint_requirement(&selection.stack.integration, effective.topology, planned)?;
    let mut processes = Vec::with_capacity(requirements.len());
    let mut public_endpoint = None;
    let mut device_count = 0_u32;
    let gateway_process_id = resolved_gateway_process_id(requirements)?;
    for ((requirement, allocation), rendered_process) in requirements
        .iter()
        .zip(allocations)
        .zip(rendered_stage.rendered_processes())
    {
        let (identity, command, launch_file_declarations, integration_rendered) = match (
            requirement.identity(),
            allocation.wire(),
            rendered_process,
        ) {
            (
                ProcessRequirementIdentity::ModelRank {
                    role_id,
                    replica_index,
                    rank,
                    ..
                },
                ServeProcessAllocation::ModelRank {
                    process,
                    role,
                    replica,
                    rank: allocated_rank,
                    rank_count,
                    ..
                },
                RenderedServeProcess::ModelRank {
                    process: rendered_id,
                    role: rendered_role,
                    replica: rendered_replica,
                    rank: rendered_rank,
                    rank_count: rendered_rank_count,
                    command,
                    launch_files,
                },
            ) if process == requirement.id()
                && rendered_id == requirement.id()
                && role == role_id
                && rendered_role == role_id
                && replica == replica_index
                && rendered_replica == replica_index
                && allocated_rank == rank
                && rendered_rank == rank
                && rank_count == rendered_rank_count =>
            {
                (
                    ProcessIdentityPlan::ModelRank {
                        rank: *rank,
                        rank_count: *rank_count,
                    },
                    command,
                    launch_files,
                    true,
                )
            }
            (
                ProcessRequirementIdentity::Frontend {
                    process_role,
                    components,
                    gateway,
                    ..
                },
                ServeProcessAllocation::Frontend {
                    process,
                    process_role: allocated_role,
                    components: allocated_components,
                    ..
                },
                RenderedServeProcess::Frontend {
                    process: rendered_id,
                    process_role: rendered_role,
                    components: rendered_components,
                    command,
                    launch_files,
                },
            ) if process == requirement.id()
                && rendered_id == requirement.id()
                && allocated_role == process_role
                && rendered_role == process_role
                && allocated_components == components
                && rendered_components == components =>
            {
                (
                    ProcessIdentityPlan::Frontend {
                        process_role: *process_role,
                        components: components.clone(),
                    },
                    command,
                    launch_files,
                    gateway.render_source == RenderSource::Integration,
                )
            }
            _ => {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {:?} rendered process {:?} with an identity different from allocation {:?}",
                        selection.stack.integration,
                        rendered_process_id(rendered_process),
                        requirement.id()
                    ),
                });
            }
        };
        if command.argv.is_empty() {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} rendered an empty argv for process {:?}",
                    selection.stack.integration,
                    requirement.id()
                ),
            });
        }
        if command.env.contains_key("CUDA_VISIBLE_DEVICES") {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} attempted to select devices for process {:?}",
                    selection.stack.integration,
                    requirement.id()
                ),
            });
        }
        let launch_files = validate_launch_file_declarations(
            &selection.stack.integration,
            requirement.id(),
            &allocation.runtime_cache().path,
            command,
            launch_file_declarations,
        )?;
        let machine_id = allocation.machine();
        let machine = workspace.local.machines.get(machine_id).ok_or_else(|| {
            InferlabError::InvalidConfig {
                message: format!("unknown machine {machine_id:?}"),
            }
        })?;
        if !integration_rendered && !matches!(machine.launch, LaunchBinding::Local) {
            return Err(InferlabError::InvalidConfig {
                message: "a control-plane-rendered frontend must use a local machine binding"
                    .to_owned(),
            });
        }
        let workspace_root = machine
            .workspace
            .clone()
            .unwrap_or_else(|| workspace.root.clone());
        let runtime_cwd = workspace_root.join(".inferlab");
        let mut env = match machine.launch {
            LaunchBinding::Local => current_environment()?,
            LaunchBinding::Ssh { .. } => BTreeMap::new(),
        };
        let mut explicit_env = command.env.keys().cloned().collect::<Vec<_>>();
        env.extend(command.env.clone());
        env.insert(
            "CUDA_VISIBLE_DEVICES".to_owned(),
            allocation
                .devices()
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
        explicit_env.push("CUDA_VISIBLE_DEVICES".to_owned());
        env.insert("PWD".to_owned(), runtime_cwd.to_string_lossy().into_owned());
        explicit_env.push("PWD".to_owned());
        explicit_env.sort();
        explicit_env.dedup();
        let allocation_endpoint =
            allocation
                .endpoint()
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!("allocation {:?} has no endpoint", allocation.process()),
                })?;
        let endpoint = ProcessEndpointPlan {
            host: allocation_endpoint.host.clone(),
            port: allocation_endpoint.port,
        };
        if requirement.id() == public_process {
            let server_metrics = endpoint_requirement
                .server_metrics
                .as_ref()
                .map(|metrics| -> Result<ServerMetricsEndpointPlan, InferlabError> {
                    let port = if let Some(name) = &metrics.port {
                        allocation
                            .ports()
                            .get(name)
                            .ok_or_else(|| InferlabError::InvalidConfig {
                                message: format!(
                                    "integration {:?} selected undeclared server-metrics port {name:?}",
                                    selection.stack.integration
                                ),
                            })?
                            .port
                    } else {
                        endpoint.port
                    };
                    let metrics_endpoint = EndpointAssignment {
                        host: endpoint.host.clone(),
                        port,
                    };
                    Ok(ServerMetricsEndpointPlan {
                        path: metrics.path.clone(),
                        port_name: metrics.port.clone(),
                        url: format!("{}{}", endpoint_url(&metrics_endpoint), metrics.path),
                    })
                })
                .transpose()?;
            public_endpoint = Some(EndpointPlan {
                host: endpoint.host.clone(),
                port: endpoint.port,
                protocol: endpoint_requirement.protocol,
                completions_path: endpoint_requirement.completions_path.clone(),
                chat_completions_path: endpoint_requirement.chat_completions_path.clone(),
                server_metrics,
                prefix_cache_reset: endpoint_requirement.prefix_cache_reset.clone(),
            });
        }
        device_count += requirement.device_count();
        processes.push(ProcessPlan {
            id: requirement.id().to_owned(),
            identity,
            command_source: if integration_rendered {
                ProcessCommandSource::Integration
            } else {
                ProcessCommandSource::ControlPlane
            },
            machine: machine_id.to_owned(),
            launch: launch_plan(&machine.launch),
            launch_dependencies: requirement.launch_dependencies().to_vec(),
            allocation: AllocationPlan {
                devices: allocation.devices().to_vec(),
                model_locator: allocation.model_locator().map(str::to_owned),
                model_locator_source: allocation.model_locator_source(),
                ports: allocation.ports().clone(),
                runtime_cache: allocation.runtime_cache().clone(),
                communication_interface: None,
            },
            command: CommandPlan {
                argv: if integration_rendered {
                    pixi_command(&selection.stack.pixi_environment, command.argv.clone())
                } else {
                    command.argv.clone()
                },
                env,
                explicit_env,
                pass_env: Vec::new(),
                cwd: runtime_cwd,
            },
            launch_files,
            readiness: readiness_plan(
                requirement.readiness(),
                effective.readiness_timeout_seconds,
                effective.readiness_attempt_timeout_seconds,
                effective.profiling,
                allocations,
            )?,
            endpoint,
            container: None,
            capture_target: resolve_capture_target(requirement, gateway_process_id, allocations)?,
        });
    }
    let public_endpoint = public_endpoint.ok_or_else(|| InferlabError::InvalidConfig {
        message: format!(
            "integration {:?} did not plan a public endpoint",
            selection.stack.integration
        ),
    })?;
    if request.image.is_some() {
        crate::image::launch::gate_placement(&processes)?;
    }
    let network = crate::server::resolve_network(&processes)?;
    if let Some(network) = &network {
        for process in &mut processes {
            process.command.env.insert(
                "NCCL_SOCKET_IFNAME".to_owned(),
                network.selected_interface.clone(),
            );
            let explicit = &mut process.command.explicit_env;
            if !explicit.iter().any(|name| name == "NCCL_SOCKET_IFNAME") {
                explicit.push("NCCL_SOCKET_IFNAME".to_owned());
                explicit.sort();
            }
            process.allocation.communication_interface = Some(network.selected_interface.clone());
        }
    }
    let (remote_workspaces, remote_containers) = if let Some(external) = request.external {
        (
            BTreeMap::new(),
            crate::server::preflight_container_targets(
                &mut processes,
                &workspace.local.machines,
                &external.id,
                &external.reference,
            )?,
        )
    } else {
        (
            crate::server::preflight_targets(
                &mut processes,
                &workspace.snapshot,
                &selection.stack.pixi_environment,
            )?,
            BTreeMap::new(),
        )
    };
    Ok(RuntimeRealizationStage::new(RuntimeRealizationParts {
        processes,
        public_endpoint,
        device_count,
        selected_machines: allocations
            .iter()
            .map(|allocation| allocation.machine().to_owned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        network,
        remote_workspaces,
        remote_containers,
    }))
}

pub(super) fn assemble_process_hierarchy(
    integration: &str,
    effective: &EffectiveServerInput,
    planned_stage: &PlannedServeStage,
    processes: Vec<ProcessPlan>,
) -> Result<(Vec<RolePlan>, Option<FrontendPlan>), InferlabError> {
    let planned = planned_stage.planned();
    let requirements = planned_stage.requirements();
    let process_count = processes.len();
    let mut processes_by_id = processes
        .into_iter()
        .map(|process| (process.id.clone(), process))
        .collect::<BTreeMap<_, _>>();
    if processes_by_id.len() != process_count {
        return Err(InferlabError::InvalidConfig {
            message: "resolved topology contains duplicate process identities".to_owned(),
        });
    }
    let role_plans = planned
        .roles
        .iter()
        .map(|role| {
            let resolution = effective
                .role_resolutions
                .iter()
                .find(|resolution| resolution.input.id == role.id);
            if let Some(resolution) = resolution {
                validate_effective_settings(
                    &resolution.input.settings,
                    &role.effective_settings,
                    integration,
                )?;
            }
            let replicas = planned
                .replicas
                .iter()
                .filter(|replica| replica.role_id == role.id)
                .map(|replica| {
                    let mut ranks = requirements
                        .iter()
                        .filter_map(|requirement| {
                            let ProcessRequirementIdentity::ModelRank {
                                role_id,
                                replica_index,
                                rank,
                                ..
                            } = requirement.identity()
                            else {
                                return None;
                            };
                            (role_id == &role.id && replica_index == &replica.replica_index)
                                .then_some((requirement, *rank))
                        })
                        .collect::<Vec<_>>();
                    ranks.sort_by_key(|(_, rank)| *rank);
                    let entry_process = ranks
                        .first()
                        .map(|(requirement, _)| requirement.id().to_owned())
                        .ok_or_else(|| InferlabError::InvalidConfig {
                            message: format!("resolved replica {:?} contains no ranks", replica.id),
                        })?;
                    let ranks = ranks
                        .into_iter()
                        .map(|(requirement, _)| {
                            processes_by_id.remove(requirement.id()).ok_or_else(|| {
                                InferlabError::InvalidConfig {
                                    message: format!(
                                        "resolved replica {:?} references missing process {:?}",
                                        replica.id,
                                        requirement.id()
                                    ),
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(RoleReplicaPlan {
                        id: replica.id.clone(),
                        index: replica.replica_index,
                        device_count: replica.device_count,
                        ports: replica.ports.clone(),
                        primary_ports: replica.primary_ports.clone(),
                        primary_readiness: replica.primary_readiness.clone(),
                        worker_readiness: replica.worker_readiness.clone(),
                        capture_target: replica.capture_target.clone(),
                        entry_process,
                        ranks,
                    })
                })
                .collect::<Result<Vec<_>, InferlabError>>()?;
            Ok(RolePlan {
                id: role.id.clone(),
                kind: role.kind,
                declared_replica_count: role.declared_replica_count,
                effective_replica_count: role.effective_replica_count,
                declared_parallelism: resolution
                    .map(|resolution| resolution.input.parallelism.clone())
                    .unwrap_or_default(),
                effective_parallelism: role.effective_parallelism.clone(),
                declared_settings: resolution
                    .map(|resolution| resolution.input.settings.clone())
                    .unwrap_or_default(),
                effective_settings: role.effective_settings.clone(),
                public_endpoint: role.public_endpoint.clone(),
                render_inputs: role.render_inputs.clone(),
                replicas,
            })
        })
        .collect::<Result<Vec<_>, InferlabError>>()?;
    let frontend_process_ids = requirements
        .iter()
        .filter_map(|requirement| {
            matches!(
                requirement.identity(),
                ProcessRequirementIdentity::Frontend { .. }
            )
            .then_some(requirement.id())
        })
        .collect::<Vec<_>>();
    let frontend_processes = frontend_process_ids
        .iter()
        .map(|id| {
            processes_by_id
                .remove(*id)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "resolved frontend component references missing process {id:?}"
                    ),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let frontend = match &planned.gateway {
        Some(gateway) => {
            let [process] = frontend_processes.as_slice() else {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "current protocol requires one frontend process, resolved {}",
                        frontend_processes.len()
                    ),
                });
            };
            let process_id = process.id.clone();
            Some(FrontendPlan {
                gateway: GatewayComponentPlan {
                    plan: gateway.clone(),
                    process_id: process_id.clone(),
                },
                pd_router: planned
                    .pd_router
                    .clone()
                    .map(|plan| PdRouterComponentPlan { plan, process_id }),
                processes: frontend_processes,
            })
        }
        None if frontend_processes.is_empty() => None,
        None => {
            return Err(InferlabError::InvalidConfig {
                message: "resolved frontend processes without a Gateway component".to_owned(),
            });
        }
    };
    if !processes_by_id.is_empty() {
        return Err(InferlabError::InvalidConfig {
            message: "resolved topology contains a process outside its owning hierarchy".to_owned(),
        });
    }
    Ok((role_plans, frontend))
}
