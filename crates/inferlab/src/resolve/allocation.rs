use crate::InferlabError;
use crate::execution::{
    ModelLocatorSource, RuntimeCacheNamespacePlan, RuntimeCachePlan, RuntimeCacheRootSource,
};
use crate::workspace::{LaunchBinding, LoadedWorkspace, PlacementBinding};
use inferlab_protocol::{
    AllocationLaunch, EndpointAssignment, ReadinessProbe, ServeProcessAllocation, ServeRoleKind,
    TargetEndpointScheme,
};
use inferlab_runtime::plan::{LaunchPlan, ReadinessPlan, TargetRegistryExpectedTarget};
use inferlab_serve_domain::{
    FixedDeviceAssignment, ProcessRequirement, ProcessRequirementIdentity,
    ResolvedProcessAllocation,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[allow(clippy::too_many_arguments)]
pub(super) fn allocate_processes(
    workspace: &LoadedWorkspace,
    placement_id: &str,
    placement: &crate::workspace::PlacementBinding,
    weight: &crate::workspace::ModelWeightBinding,
    pixi_environment: &str,
    image_identity: Option<&str>,
    requirements: &[ProcessRequirement],
    local_process: Option<&str>,
) -> Result<Vec<ResolvedProcessAllocation>, InferlabError> {
    let mut process_ids = BTreeSet::new();
    let mut usage = BTreeMap::<String, MachineAllocationUsage>::new();
    let mut allocations = Vec::with_capacity(requirements.len());

    for requirement in requirements {
        if requirement.id().is_empty() || !process_ids.insert(requirement.id().to_owned()) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration returned invalid or duplicate process id {:?}",
                    requirement.id()
                ),
            });
        }
        let placement_role = requirement.placement_role();
        if placement_role.is_empty() {
            return Err(InferlabError::InvalidConfig {
                message: format!("process {:?} has an empty placement role", requirement.id()),
            });
        }
        let mut port_names = BTreeSet::new();
        if requirement
            .ports()
            .iter()
            .any(|name| name.is_empty() || !port_names.insert(name))
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration returned invalid or duplicate port requirements for process {:?}",
                    requirement.id()
                ),
            });
        }

        let mut candidates = if let Some(fixed) = requirement.fixed_devices() {
            vec![fixed.machine().to_owned()]
        } else if let Some(role_machines) = placement
            .roles
            .get(placement_role)
            .and_then(|role| role.machines())
            .filter(|machines| !machines.is_empty())
        {
            role_machines.to_vec()
        } else if !placement.machines.is_empty() {
            placement.machines.clone()
        } else {
            placement_machine_pool(placement)
        };
        if local_process == Some(requirement.id()) {
            candidates.retain(|machine_id| {
                workspace
                    .local
                    .machines
                    .get(machine_id)
                    .is_some_and(|machine| matches!(machine.launch, LaunchBinding::Local))
            });
            if candidates.is_empty() {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement {placement_id:?} has no local machine for control-plane-rendered frontend {:?}",
                        requirement.id()
                    ),
                });
            }
        }
        let machine_id = candidates
            .iter()
            .find(|machine_id| {
                let Some(machine) = workspace.local.machines.get(*machine_id) else {
                    return false;
                };
                machine_capacity(
                    machine,
                    usage.get(*machine_id),
                    requirement.device_count() as usize,
                    requirement.ports().len() + 1,
                )
            })
            .cloned();
        let machine_id = match machine_id {
            Some(machine_id) => machine_id,
            None if candidates.len() == 1 => {
                let candidate = &candidates[0];
                let machine = workspace.local.machines.get(candidate).ok_or_else(|| {
                    InferlabError::InvalidConfig {
                        message: format!("unknown machine {candidate:?}"),
                    }
                })?;
                let available = free_device_count(machine, usage.get(candidate));
                if available < requirement.device_count() as usize {
                    return Err(InferlabError::InsufficientDevices {
                        machine: candidate.clone(),
                        required: requirement.device_count(),
                        available,
                    });
                }
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "machine {candidate:?} has insufficient free ports for process {:?}",
                        requirement.id()
                    ),
                });
            }
            None => {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement {placement_id:?} has no machine with {} free devices and {} free ports for process {:?} in role {placement_role:?}",
                        requirement.device_count(),
                        requirement.ports().len() + 1,
                        requirement.id()
                    ),
                });
            }
        };
        let machine = workspace.local.machines.get(&machine_id).ok_or_else(|| {
            InferlabError::InvalidConfig {
                message: format!("unknown machine {machine_id:?}"),
            }
        })?;
        let used = usage.entry(machine_id.clone()).or_default();
        let devices =
            if let Some(fixed) = requirement.fixed_devices() {
                if fixed.devices().iter().any(|device| {
                    !machine.devices.contains(device) || used.devices.contains(device)
                }) {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "placement assigns unavailable or overlapping devices to process {:?}",
                            requirement.id()
                        ),
                    });
                }
                fixed.devices().to_vec()
            } else {
                machine
                    .devices
                    .iter()
                    .filter(|device| !used.devices.contains(device))
                    .take(requirement.device_count() as usize)
                    .copied()
                    .collect::<Vec<_>>()
            };
        used.devices.extend(&devices);
        let endpoint_port = requirement
            .fixed_devices()
            .and_then(FixedDeviceAssignment::endpoint_port);
        let endpoint_port = match endpoint_port {
            Some(port) => {
                if !machine.ports.contains(&port) || used.ports.contains(&port) {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "placement assigns unavailable endpoint port {port} to process {:?}",
                            requirement.id()
                        ),
                    });
                }
                used.ports.insert(port);
                port
            }
            None => machine
                .ports
                .iter()
                .find(|port| !used.ports.contains(port))
                .copied()
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "machine {machine_id:?} has no free endpoint port for process {:?}",
                        requirement.id()
                    ),
                })?,
        };
        used.ports.insert(endpoint_port);
        let selected_ports = machine
            .ports
            .iter()
            .filter(|port| !used.ports.contains(port))
            .take(requirement.ports().len())
            .copied()
            .collect::<Vec<_>>();
        if selected_ports.len() != requirement.ports().len() {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "machine {machine_id:?} has insufficient free named ports for process {:?}",
                    requirement.id()
                ),
            });
        }
        used.ports.extend(&selected_ports);
        let named_ports = requirement
            .ports()
            .iter()
            .zip(&selected_ports)
            .map(|(name, port)| {
                (
                    name.clone(),
                    EndpointAssignment {
                        host: machine.host.clone(),
                        port: *port,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let endpoint = EndpointAssignment {
            host: machine.host.clone(),
            port: endpoint_port,
        };
        let runtime_cache = runtime_cache_plan(
            workspace,
            machine,
            &machine_id,
            requirement.id(),
            pixi_environment,
            image_identity,
        );
        let cache = runtime_cache
            .path
            .to_str()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "runtime cache path for process {:?} is not valid UTF-8",
                    requirement.id()
                ),
            })?
            .to_owned();
        let launch = match &machine.launch {
            LaunchBinding::Local => AllocationLaunch::Local,
            LaunchBinding::Ssh { target } => AllocationLaunch::Ssh {
                target: target.clone(),
            },
        };
        let (wire, model_locator_source) = match requirement.identity() {
            ProcessRequirementIdentity::ModelRank {
                role_id,
                role_kind,
                replica_id,
                replica_index,
                rank,
                effective_settings,
                effective_parallelism,
                links,
                render_inputs,
            } => {
                let (model_locator, source) = if let Some(locator) =
                    weight.machine_locators.get(&machine_id)
                {
                    (locator.clone(), ModelLocatorSource::Machine)
                } else if let Some(locator) = &weight.locator {
                    (locator.clone(), ModelLocatorSource::Fallback)
                } else {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "model weights have no locator for Engine process {:?} on machine {machine_id:?}",
                            requirement.id()
                        ),
                    });
                };
                let rank_count = u32::try_from(
                    requirements
                        .iter()
                        .filter(|candidate| {
                            matches!(
                                candidate.identity(),
                                ProcessRequirementIdentity::ModelRank {
                                    replica_id: candidate_replica,
                                    ..
                                } if candidate_replica == replica_id
                            )
                        })
                        .count(),
                )
                .map_err(|_| InferlabError::InvalidConfig {
                    message: format!("replica {replica_id:?} has too many ranks"),
                })?;
                (
                    ServeProcessAllocation::ModelRank {
                        process: requirement.id().to_owned(),
                        role: role_id.clone(),
                        role_kind: *role_kind,
                        replica: *replica_index,
                        rank: *rank,
                        rank_count,
                        machine: machine_id.clone(),
                        devices,
                        model_locator,
                        endpoint: Some(endpoint),
                        ports: named_ports,
                        cache,
                        launch,
                        effective_settings: effective_settings.clone(),
                        effective_parallelism: effective_parallelism.clone(),
                        links: links.clone(),
                        dependencies: requirement.launch_dependencies().to_vec(),
                        render_inputs: render_inputs.clone(),
                    },
                    Some(source),
                )
            }
            ProcessRequirementIdentity::Frontend {
                process_role,
                components,
                gateway,
                pd_router,
                links,
                render_inputs,
            } => {
                if requirement.device_count() != 0 || !devices.is_empty() {
                    return Err(InferlabError::InvalidConfig {
                        message: "frontend process must not allocate model devices".to_owned(),
                    });
                }
                (
                    ServeProcessAllocation::Frontend {
                        process: requirement.id().to_owned(),
                        process_role: *process_role,
                        components: components.clone(),
                        machine: machine_id.clone(),
                        devices,
                        endpoint,
                        ports: named_ports,
                        cache,
                        launch,
                        gateway: gateway.clone(),
                        pd_router: pd_router.clone(),
                        links: links.clone(),
                        dependencies: requirement.launch_dependencies().to_vec(),
                        render_inputs: render_inputs.clone(),
                    },
                    None,
                )
            }
        };
        allocations.push(ResolvedProcessAllocation::new(
            wire,
            runtime_cache,
            model_locator_source,
        ));
    }
    Ok(allocations)
}

fn placement_machine_pool(placement: &PlacementBinding) -> Vec<String> {
    placement
        .roles
        .values()
        .filter_map(|role| role.machines())
        .flat_map(|machines| machines.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[derive(Default)]
struct MachineAllocationUsage {
    devices: BTreeSet<u32>,
    ports: BTreeSet<u16>,
}

fn machine_capacity(
    machine: &crate::workspace::MachineBinding,
    usage: Option<&MachineAllocationUsage>,
    devices: usize,
    ports: usize,
) -> bool {
    let free_devices = free_device_count(machine, usage);
    let available_ports = machine.ports.len();
    let used_ports = usage.map_or(0, |usage| usage.ports.len());
    free_devices >= devices && available_ports - used_ports >= ports
}

fn free_device_count(
    machine: &crate::workspace::MachineBinding,
    usage: Option<&MachineAllocationUsage>,
) -> usize {
    machine.devices.len()
        - usage.map_or(0, |usage| {
            machine
                .devices
                .iter()
                .filter(|device| usage.devices.contains(device))
                .count()
        })
}

fn runtime_cache_plan(
    workspace: &LoadedWorkspace,
    machine: &crate::workspace::MachineBinding,
    machine_id: &str,
    process_id: &str,
    pixi_environment: &str,
    image_identity: Option<&str>,
) -> RuntimeCachePlan {
    let (storage_root, storage_root_source) = machine.cache_root.as_ref().map_or_else(
        || {
            let workspace_root = machine.workspace.as_ref().unwrap_or(&workspace.root);
            (
                workspace_root.join(".inferlab/cache/runtime"),
                RuntimeCacheRootSource::WorkspaceDefault,
            )
        },
        |root| (root.clone(), RuntimeCacheRootSource::MachineBinding),
    );
    let namespace = RuntimeCacheNamespacePlan {
        workspace_source_digest: workspace.snapshot.source_digest.clone(),
        pixi_environment: pixi_environment.to_owned(),
        image_id: image_identity.map(str::to_owned),
        machine: machine_id.to_owned(),
        process: process_id.to_owned(),
    };
    // For an image-backed launch, the image is the software identity that
    // generates and consumes the cached JIT artifacts, so its immutable
    // identity keys the namespace in place of the invoking checkout's
    // source state and environment ([[RFC-0002:C-RUNTIME-CACHE]]). The
    // discriminant keeps the two key families in disjoint domains.
    let key_inputs: [&str; 2] = match &namespace.image_id {
        Some(image_id) => ["image-realization", image_id.as_str()],
        None => [
            namespace.workspace_source_digest.as_str(),
            namespace.pixi_environment.as_str(),
        ],
    };
    let mut hasher = Sha256::new();
    for value in key_inputs {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    let environment_key = format!("{:x}", hasher.finalize());
    let path = storage_root
        .join("v1")
        .join(environment_key)
        .join(machine_id)
        .join(process_id);
    RuntimeCachePlan {
        storage_root,
        storage_root_source,
        namespace,
        path,
    }
}

pub(super) fn launch_plan(binding: &LaunchBinding) -> LaunchPlan {
    match binding {
        LaunchBinding::Local => LaunchPlan::Local,
        LaunchBinding::Ssh { target } => LaunchPlan::Ssh {
            target: target.clone(),
        },
    }
}

pub(super) fn pixi_command(environment: &str, process: Vec<String>) -> Vec<String> {
    let mut argv = vec![
        "pixi".to_owned(),
        "run".to_owned(),
        "--as-is".to_owned(),
        "--executable".to_owned(),
        "-e".to_owned(),
        environment.to_owned(),
        "--".to_owned(),
    ];
    argv.extend(process);
    argv
}

pub(super) fn readiness_plan(
    probe: &ReadinessProbe,
    timeout: u64,
    attempt_timeout: u64,
    capture_armed: bool,
    allocations: &[ResolvedProcessAllocation],
) -> Result<ReadinessPlan, InferlabError> {
    match probe {
        ReadinessProbe::Http { path } => Ok(ReadinessPlan::Http {
            path: path.clone(),
            // A capture-armed server's readiness wait is unbounded
            // ([[RFC-0004:C-WORKLOAD-PROFILING]]): instrumentation multiplies
            // startup unpredictably, and the wait still terminates on process
            // death or interruption.
            timeout_seconds: (!capture_armed).then_some(timeout),
            attempt_timeout_seconds: attempt_timeout,
        }),
        ReadinessProbe::HttpTargetRegistry(registry) => {
            let expected_targets = allocations
                .iter()
                .filter_map(|allocation| {
                    let ServeProcessAllocation::ModelRank {
                        role_kind,
                        rank: 0,
                        ..
                    } = allocation.wire()
                    else {
                        return None;
                    };
                    Some((allocation, *role_kind))
                })
                .filter_map(|(allocation, kind)| match kind {
                    ServeRoleKind::Prefill => Some((
                        allocation,
                        registry.prefill_role_value.as_str(),
                        Some(registry.prefill_bootstrap_port.as_str()),
                    )),
                    ServeRoleKind::Decode => {
                        Some((allocation, registry.decode_role_value.as_str(), None))
                    }
                    ServeRoleKind::Serve => None,
                })
                .map(|(allocation, role, bootstrap_port)| {
                    let bootstrap_port = bootstrap_port
                        .map(|port| {
                            allocation
                                .ports()
                                .get(port)
                                .map(|endpoint| endpoint.port)
                                .ok_or_else(|| InferlabError::InvalidConfig {
                                    message: format!(
                                        "prefill process {:?} has no registry bootstrap port {port:?}",
                                        allocation.process()
                                    ),
                                })
                        })
                        .transpose()?;
                    let endpoint = allocation.endpoint().ok_or_else(|| {
                        InferlabError::InvalidConfig {
                            message: format!(
                                "process {:?} has no endpoint for target-aware readiness",
                                allocation.process()
                            ),
                        }
                    })?;
                    Ok(TargetRegistryExpectedTarget {
                        url: target_endpoint_url(endpoint, registry.target_scheme),
                        role: role.to_owned(),
                        bootstrap_port,
                    })
                })
                .collect::<Result<Vec<_>, InferlabError>>()?;
            Ok(ReadinessPlan::HttpTargetRegistry {
                readiness_path: registry.readiness_path.clone(),
                registry_path: registry.registry_path.clone(),
                targets_field: registry.targets_field.clone(),
                target_url_field: registry.target_url_field.clone(),
                target_role_field: registry.target_role_field.clone(),
                target_healthy_field: registry.target_healthy_field.clone(),
                target_bootstrap_port_field: registry.target_bootstrap_port_field.clone(),
                expected_targets,
                timeout_seconds: (!capture_armed).then_some(timeout),
                attempt_timeout_seconds: attempt_timeout,
            })
        }
        ReadinessProbe::ProcessAlive => Ok(ReadinessPlan::ProcessAlive {
            timeout_seconds: (!capture_armed).then_some(timeout),
            attempt_timeout_seconds: attempt_timeout,
        }),
    }
}

fn target_endpoint_url(endpoint: &EndpointAssignment, scheme: TargetEndpointScheme) -> String {
    let scheme = match scheme {
        TargetEndpointScheme::Http => "http",
        TargetEndpointScheme::Grpc => "grpc",
    };
    format!("{scheme}://{}:{}", endpoint.host, endpoint.port)
}
