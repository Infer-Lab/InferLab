use crate::InferlabError;
use crate::execution::ProfilerEscapesPlan;
use crate::workspace::{PlacementBinding, PlacementRoleBinding, ServerDefinition};
use inferlab_profiler::plan::{CaptureWindowControlEndpointPlan, CaptureWindowHttpMethodPlan};
use inferlab_protocol::{
    CaptureMechanism, CaptureWindowControlEndpoint, EndpointAssignment, EndpointRequirement,
    FrontendComponents, FrontendProcessRole, GatewayTarget, KvTransferMechanism, PlanServeResult,
    RenderSource, ServeReplicaRequirement, ServeRoleInput, ServeRoleKind, ServeRoleLink,
    ServeTopology, SuppliedRenderInput, SyntheticAcceptanceInput, SyntheticAcceptanceOutcome,
};
use inferlab_serve_domain::{
    FixedDeviceAssignment, PendingCaptureTargetPlan, PendingCaptureWindowActionPlan,
    ProcessRequirement, ProcessRequirementIdentity,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate_workload_endpoint(
    integration: &str,
    endpoint: &inferlab_protocol::EndpointRequirement,
    declared_ports: &[String],
) -> Result<(), InferlabError> {
    const COMPLETIONS_PATH: &str = "/v1/completions";
    const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

    if endpoint.completions_path != COMPLETIONS_PATH {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} declared completions_path {:?}; expected {COMPLETIONS_PATH:?}",
                endpoint.completions_path
            ),
        });
    }
    if endpoint.chat_completions_path != CHAT_COMPLETIONS_PATH {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} declared chat_completions_path {:?}; expected {CHAT_COMPLETIONS_PATH:?}",
                endpoint.chat_completions_path
            ),
        });
    }
    if let Some(metrics) = &endpoint.server_metrics {
        if !is_absolute_origin_path(&metrics.path) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} declared server-metrics path {:?}; expected an absolute origin path without scheme, authority, query, or fragment",
                    metrics.path
                ),
            });
        }
        if let Some(port) = &metrics.port {
            let matches = declared_ports
                .iter()
                .filter(|declared| *declared == port)
                .count();
            if port.is_empty() || matches != 1 {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} selected server-metrics port {port:?}, but the public process must declare that non-empty named port exactly once"
                    ),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn endpoint_url(endpoint: &EndpointAssignment) -> String {
    format!("http://{}:{}", endpoint.host, endpoint.port)
}

fn is_absolute_origin_path(path: &str) -> bool {
    path.starts_with('/')
        && !path.starts_with("//")
        && !path.contains('?')
        && !path.contains('#')
        && !path.contains('\\')
        && !path
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
}

pub(super) fn validate_capture_targets(
    integration: &str,
    profiling: Option<CaptureMechanism>,
    has_gateway: bool,
    replicas: &[ServeReplicaRequirement],
) -> Result<(), InferlabError> {
    for replica in replicas {
        if let Some(target) = &replica.capture_target {
            if let Some(mechanism) = profiling
                && target.mechanism != mechanism
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} declared capture mechanism {:?} for replica {:?}, but the effective request mechanism is {mechanism:?}",
                        target.mechanism, replica.id
                    ),
                });
            }
            if target.window_control.endpoint == CaptureWindowControlEndpoint::Gateway
                && !has_gateway
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} selected Gateway profiling window control without planning a Gateway"
                    ),
                });
            }
            for (operation, action) in [
                ("start", &target.window_control.start),
                ("stop", &target.window_control.stop),
            ] {
                if !is_absolute_origin_path(&action.path) {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "integration {integration:?} declared capture {operation} path {:?} for replica {:?}; expected an absolute origin path without scheme, authority, query, fragment, backslash, control character, or whitespace",
                            action.path, replica.id
                        ),
                    });
                }
            }
        }
        if profiling.is_none() {
            continue;
        }
        if replica.capture_target.is_none() {
            if replica.device_count == 0 {
                continue;
            }
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} did not prepare model-serving replica {:?} as a profiling target",
                    replica.id
                ),
            });
        }
    }
    Ok(())
}

const fn capture_window_control_endpoint_plan(
    endpoint: CaptureWindowControlEndpoint,
) -> CaptureWindowControlEndpointPlan {
    match endpoint {
        CaptureWindowControlEndpoint::ReplicaEntry => {
            CaptureWindowControlEndpointPlan::ReplicaEntry
        }
        CaptureWindowControlEndpoint::Gateway => CaptureWindowControlEndpointPlan::Gateway,
    }
}

fn capture_window_action_plan(
    action: &inferlab_protocol::CaptureWindowHttpActionSpec,
) -> PendingCaptureWindowActionPlan {
    PendingCaptureWindowActionPlan::new(
        match action.method {
            inferlab_protocol::HttpMethod::Post => CaptureWindowHttpMethodPlan::Post,
        },
        action.path.clone(),
        action.body.clone(),
    )
}

pub(super) fn resolved_gateway_process_id(
    requirements: &[ProcessRequirement],
) -> Result<Option<&str>, InferlabError> {
    let mut gateway_process_ids = requirements.iter().filter_map(|requirement| {
        let ProcessRequirementIdentity::Frontend { components, .. } = requirement.identity() else {
            return None;
        };
        match components {
            FrontendComponents::Gateway(_) | FrontendComponents::GatewayPdRouter(_) => {
                Some(requirement.id())
            }
        }
    });
    let gateway_process_id = gateway_process_ids.next();
    if gateway_process_ids.next().is_some() {
        return Err(InferlabError::InvalidConfig {
            message: "resolved topology binds the Gateway component to multiple processes"
                .to_owned(),
        });
    }
    Ok(gateway_process_id)
}

pub(super) fn profiler_escapes_plan(server: &ServerDefinition) -> Option<ProfilerEscapesPlan> {
    let roles = server
        .roles
        .iter()
        .filter(|(_, role)| !role.profiler.nsys.is_empty())
        .map(|(id, role)| (id.clone(), role.profiler.nsys.clone()))
        .collect::<BTreeMap<_, _>>();
    if server.profiler.is_empty() && roles.is_empty() {
        return None;
    }
    Some(ProfilerEscapesPlan {
        mechanism: server.profiler.mechanism,
        common: server.profiler.nsys.clone(),
        roles,
    })
}

pub(super) fn uses_explicit_replica_placement(placement: &PlacementBinding) -> bool {
    placement
        .roles
        .values()
        .any(PlacementRoleBinding::uses_explicit_replicas)
}

pub(super) fn links_for_node(links: &[ServeRoleLink], node: &str) -> Vec<ServeRoleLink> {
    links
        .iter()
        .filter(|link| match link {
            ServeRoleLink::RequestRouting { source, targets } => {
                source == node || targets.iter().any(|target| target == node)
            }
            ServeRoleLink::KvTransfer { source, target, .. }
            | ServeRoleLink::Bootstrap { source, target, .. }
            | ServeRoleLink::SideChannel { source, target, .. } => source == node || target == node,
        })
        .cloned()
        .collect()
}

pub(super) fn expand_replica_requirements(
    integration: &str,
    plan: &PlanServeResult,
    placement: &PlacementBinding,
    server: &ServerDefinition,
    role_render_inputs: &BTreeMap<String, Vec<SuppliedRenderInput>>,
) -> Result<Vec<ProcessRequirement>, InferlabError> {
    let uses_explicit_replicas = uses_explicit_replica_placement(placement);
    let role_replica_counts = plan
        .replicas
        .iter()
        .fold(BTreeMap::new(), |mut counts, replica| {
            counts
                .entry(replica.role_id.as_str())
                .and_modify(|count: &mut u32| {
                    *count = (*count).max(replica.replica_index + 1);
                })
                .or_insert(replica.replica_index + 1);
            counts
        });
    for role in placement.roles.keys() {
        let valid = role_replica_counts.contains_key(role.as_str())
            || role == "gateway" && plan.gateway.is_some();
        if !valid {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "placement references role {role:?}, which is not part of the resolved topology"
                ),
            });
        }
    }
    if uses_explicit_replicas {
        for (role, replica_count) in &role_replica_counts {
            let expected =
                usize::try_from(*replica_count).map_err(|_| InferlabError::InvalidConfig {
                    message: format!("role {role:?} has too many replicas"),
                })?;
            let actual = placement
                .roles
                .get(*role)
                .and_then(PlacementRoleBinding::replica_count)
                .unwrap_or(0);
            if actual != expected {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement assigns {actual} replicas to role {role:?}, which requires {expected}"
                    ),
                });
            }
        }
        if plan.gateway.is_some()
            && placement
                .roles
                .get("gateway")
                .and_then(PlacementRoleBinding::replica_count)
                != Some(1)
        {
            return Err(InferlabError::InvalidConfig {
                message: "explicit routed placement must bind exactly one Gateway process"
                    .to_owned(),
            });
        }
    }

    let mut processes = Vec::new();
    for replica in &plan.replicas {
        let role = plan
            .roles
            .iter()
            .find(|role| role.id == replica.role_id)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} replica {:?} has no owning Engine role",
                    replica.id
                ),
            })?;
        let explicit_ranks = if uses_explicit_replicas {
            let replica_index = usize::try_from(replica.replica_index).map_err(|_| {
                InferlabError::InvalidConfig {
                    message: format!("replica {:?} has an invalid index", replica.id),
                }
            })?;
            placement
                .roles
                .get(&replica.role_id)
                .and_then(|role| role.ranks_for_replica(replica_index))
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "placement does not assign ranks for replica {:?}",
                        replica.id
                    ),
                })?
        } else {
            &[]
        };
        let assigned_devices = explicit_ranks
            .iter()
            .map(|rank| rank.devices.len())
            .sum::<usize>();
        if uses_explicit_replicas && assigned_devices != replica.device_count as usize {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "placement assigns {assigned_devices} devices to replica {:?}, which requires {}",
                    replica.id, replica.device_count
                ),
            });
        }
        let rank_count = if uses_explicit_replicas {
            explicit_ranks.len()
        } else {
            1
        };
        let primary_id = process_id(&replica.id, 0, rank_count);
        for rank in 0..rank_count {
            let rank_index = u32::try_from(rank).map_err(|_| InferlabError::InvalidConfig {
                message: format!("replica {:?} has too many ranks", replica.id),
            })?;
            let fixed_devices = explicit_ranks.get(rank).map(|assignment| {
                FixedDeviceAssignment::new(
                    assignment.machine.clone(),
                    assignment.devices.clone(),
                    assignment.endpoint_port,
                )
            });
            let device_count = fixed_devices.as_ref().map_or_else(
                || Ok(replica.device_count),
                |fixed| {
                    u32::try_from(fixed.devices().len()).map_err(|_| InferlabError::InvalidConfig {
                        message: format!("replica {:?} rank has too many devices", replica.id),
                    })
                },
            )?;
            let mut ports = replica.ports.clone();
            if rank == 0 && rank_count > 1 {
                ports.extend(replica.primary_ports.iter().cloned());
            }
            let capture_target = replica.capture_target.as_ref().map(|target| {
                PendingCaptureTargetPlan::new(
                    target.mechanism,
                    capture_window_control_endpoint_plan(target.window_control.endpoint),
                    primary_id.clone(),
                    // Engine-trace coverage is verified against the replica's
                    // declared whole-replica device count: engine-internal
                    // profilers write one artifact per worker, which the
                    // device count bounds, while the rank model counts entry
                    // processes ([[RFC-0004:C-WORKLOAD-PROFILING]]).
                    replica.device_count,
                    capture_window_action_plan(&target.window_control.start),
                    capture_window_action_plan(&target.window_control.stop),
                    server.roles.get(&replica.role_id).map_or_else(
                        || server.profiler.nsys.clone(),
                        |role| server.profiler.merged_with(&role.profiler).nsys,
                    ),
                )
            });
            processes.push(ProcessRequirement::new(
                process_id(&replica.id, rank_index, rank_count),
                ProcessRequirementIdentity::ModelRank {
                    role_id: replica.role_id.clone(),
                    role_kind: role.kind,
                    replica_id: replica.id.clone(),
                    replica_index: replica.replica_index,
                    rank: rank_index,
                    effective_settings: role.effective_settings.clone(),
                    effective_parallelism: role.effective_parallelism.clone(),
                    links: links_for_node(&plan.links, &replica.role_id),
                    render_inputs: role_render_inputs
                        .get(&replica.role_id)
                        .cloned()
                        .unwrap_or_default(),
                },
                device_count,
                ports,
                if rank == 0 {
                    replica.primary_readiness.clone()
                } else {
                    replica.worker_readiness.clone()
                },
                if rank == 0 {
                    Vec::new()
                } else {
                    vec![primary_id.clone()]
                },
                capture_target,
                fixed_devices,
            ));
        }
    }
    Ok(processes)
}

fn process_id(replica_id: &str, rank: u32, rank_count: usize) -> String {
    if rank_count == 1 {
        replica_id.to_owned()
    } else {
        format!("{replica_id}-rank-{rank:03}")
    }
}

pub(super) fn public_endpoint_requirement<'a>(
    integration: &str,
    topology: ServeTopology,
    plan: &'a PlanServeResult,
) -> Result<&'a EndpointRequirement, InferlabError> {
    if let Some(gateway) = &plan.gateway {
        return Ok(&gateway.endpoint);
    }
    if topology == ServeTopology::Single {
        return plan
            .roles
            .iter()
            .find_map(|role| role.public_endpoint.as_ref())
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} did not declare a direct Engine public endpoint"
                ),
            });
    }
    Err(InferlabError::InvalidConfig {
        message: format!("integration {integration:?} did not declare a Gateway public endpoint"),
    })
}

fn public_endpoint_ports<'a>(
    integration: &str,
    topology: ServeTopology,
    plan: &'a PlanServeResult,
) -> Result<&'a [String], InferlabError> {
    if let Some(gateway) = &plan.gateway {
        return Ok(&gateway.ports);
    }
    if topology == ServeTopology::Single {
        let role = plan
            .roles
            .iter()
            .find(|role| role.public_endpoint.is_some())
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} did not declare a direct Engine public endpoint"
                ),
            })?;
        return plan
            .replicas
            .iter()
            .find(|replica| replica.role_id == role.id && replica.replica_index == 0)
            .map(|replica| replica.ports.as_slice())
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} did not declare the public Engine replica"
                ),
            });
    }
    Err(InferlabError::InvalidConfig {
        message: format!("integration {integration:?} did not declare a Gateway public endpoint"),
    })
}

pub(super) fn validate_serve_graph(
    integration: &str,
    topology: ServeTopology,
    requested_roles: &[ServeRoleInput],
    gateway_backend: Option<&str>,
    pd_router_backend: Option<&str>,
    kv_transfer: Option<KvTransferMechanism>,
    plan: &PlanServeResult,
) -> Result<(), InferlabError> {
    let mut role_kinds = BTreeMap::new();
    for role in &plan.roles {
        if role.id.is_empty()
            || role.declared_replica_count == 0
            || role.effective_replica_count == 0
            || role_kinds.insert(role.id.as_str(), role.kind).is_some()
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned a duplicate or empty Engine role id"
                ),
            });
        }
    }
    if plan.roles.len() != requested_roles.len()
        || requested_roles.iter().any(|requested| {
            role_kinds.get(requested.id.as_str()) != Some(&requested.kind)
                || !plan.roles.iter().any(|role| {
                    role.id == requested.id
                        && role.declared_replica_count == requested.replica_count
                })
        })
    {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} did not preserve the requested Engine role set"
            ),
        });
    }

    let role_replica_counts = plan
        .roles
        .iter()
        .map(|role| (role.id.as_str(), role.effective_replica_count))
        .collect::<BTreeMap<_, _>>();
    let mut replica_ids = BTreeSet::new();
    let mut role_replicas = BTreeSet::new();
    for replica in &plan.replicas {
        let Some(replica_count) = role_replica_counts.get(replica.role_id.as_str()) else {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} replica {:?} references unknown Engine role {:?}",
                    replica.id, replica.role_id
                ),
            });
        };
        if replica.id.is_empty()
            || replica.replica_index >= *replica_count
            || replica.device_count == 0
            || !replica_ids.insert(replica.id.as_str())
            || !role_replicas.insert((replica.role_id.as_str(), replica.replica_index))
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned an invalid or duplicate Engine replica"
                ),
            });
        }
    }
    for role in &plan.roles {
        for index in 0..role.effective_replica_count {
            if !role_replicas.contains(&(role.id.as_str(), index)) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} omitted replica {index} for Engine role {:?}",
                        role.id
                    ),
                });
            }
        }
    }

    let mut graph_nodes = role_kinds
        .keys()
        .map(|role| (*role).to_owned())
        .collect::<BTreeSet<_>>();
    let routing_source = match topology {
        ServeTopology::Single => {
            if plan.pd_router.is_some() || pd_router_backend.is_some() {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} returned a P/D Router for single topology"
                    ),
                });
            }
            let serve = plan
                .roles
                .iter()
                .find(|role| role.kind == ServeRoleKind::Serve)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} did not return the single Engine role"
                    ),
                })?;
            match (gateway_backend, plan.gateway.as_ref()) {
                (None, None) => {
                    if serve.public_endpoint.is_none()
                        || plan
                            .roles
                            .iter()
                            .filter(|role| role.public_endpoint.is_some())
                            .count()
                            != 1
                    {
                        return Err(InferlabError::InvalidConfig {
                            message: format!(
                                "integration {integration:?} direct single must expose exactly one Engine endpoint"
                            ),
                        });
                    }
                    serve.id.clone()
                }
                (Some(selected), Some(gateway)) if gateway.backend == selected => {
                    if gateway.render_source != RenderSource::Integration
                        || plan.roles.iter().any(|role| role.public_endpoint.is_some())
                        || !matches!(
                            gateway.targets.as_slice(),
                            [GatewayTarget::Engine { role }] if role == &serve.id
                        )
                    {
                        return Err(InferlabError::InvalidConfig {
                            message: format!(
                                "integration {integration:?} returned an incompatible routed-single Gateway"
                            ),
                        });
                    }
                    graph_nodes.insert("gateway".to_owned());
                    "gateway".to_owned()
                }
                _ => {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "integration {integration:?} Gateway plan does not match selected gateway_backend {gateway_backend:?}"
                        ),
                    });
                }
            }
        }
        ServeTopology::PrefillDecode => {
            let gateway = plan
                .gateway
                .as_ref()
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} did not return a Gateway for prefill_decode"
                    ),
                })?;
            let pd_router = plan.pd_router.as_ref().ok_or_else(|| {
                InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} did not return a P/D Router for prefill_decode"
                    ),
                }
            })?;
            if Some(gateway.backend.as_str()) != gateway_backend
                || Some(pd_router.backend.as_str()) != pd_router_backend
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} frontend backends do not match gateway_backend {gateway_backend:?} and pd_router_backend {pd_router_backend:?}"
                    ),
                });
            }
            if gateway.render_source != pd_router.render_source
                || gateway.implementation != pd_router.implementation
                || gateway.implementation_version != pd_router.implementation_version
                || gateway.co_rendering.process_role != FrontendProcessRole::Gateway
                || gateway.co_rendering != pd_router.co_rendering
                || plan.roles.iter().any(|role| role.public_endpoint.is_some())
                || !matches!(gateway.targets.as_slice(), [GatewayTarget::PdRouter])
                || role_kinds.get(pd_router.prefill_role.as_str()) != Some(&ServeRoleKind::Prefill)
                || role_kinds.get(pd_router.decode_role.as_str()) != Some(&ServeRoleKind::Decode)
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} returned incompatible fused frontend component plans"
                    ),
                });
            }
            graph_nodes.insert("gateway".to_owned());
            graph_nodes.insert("pd_router".to_owned());
            "pd_router".to_owned()
        }
    };

    validate_workload_endpoint(
        integration,
        public_endpoint_requirement(integration, topology, plan)?,
        public_endpoint_ports(integration, topology, plan)?,
    )?;
    if kv_transfer.is_some()
        && !plan.links.iter().any(|link| {
            matches!(
                link,
                ServeRoleLink::KvTransfer { mechanism, .. } if Some(*mechanism) == kv_transfer
            )
        })
    {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} did not link the planned KV-transfer mechanism"
            ),
        });
    }
    for link in &plan.links {
        let valid = match link {
            ServeRoleLink::RequestRouting { source, targets } => {
                graph_nodes.contains(source)
                    && !targets.is_empty()
                    && targets.iter().all(|target| graph_nodes.contains(target))
            }
            ServeRoleLink::KvTransfer {
                source,
                target,
                mechanism,
            } => {
                graph_nodes.contains(source)
                    && graph_nodes.contains(target)
                    && Some(*mechanism) == kv_transfer
            }
            ServeRoleLink::Bootstrap {
                source,
                target,
                port,
            } => {
                graph_nodes.contains(source)
                    && graph_nodes.contains(target)
                    && role_all_have_port(&plan.replicas, target, port)
            }
            ServeRoleLink::SideChannel {
                source,
                target,
                port,
            } => {
                graph_nodes.contains(source)
                    && graph_nodes.contains(target)
                    && role_all_have_port(&plan.replicas, source, port)
                    && role_all_have_port(&plan.replicas, target, port)
            }
        };
        if !valid {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned a link with unknown component or Engine endpoints"
                ),
            });
        }
    }

    match topology {
        ServeTopology::Single if plan.gateway.is_some() => {
            let serve = plan
                .roles
                .iter()
                .find(|role| role.kind == ServeRoleKind::Serve)
                .map(|role| role.id.as_str())
                .unwrap_or("serve");
            if !plan.links.iter().any(|link| {
                matches!(
                    link,
                    ServeRoleLink::RequestRouting { source, targets }
                        if source == "gateway" && targets == &[serve.to_owned()]
                )
            }) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} did not link Gateway to its single Engine"
                    ),
                });
            }
        }
        ServeTopology::Single => {}
        ServeTopology::PrefillDecode => {
            let pd_router =
                plan.pd_router
                    .as_ref()
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: "validated prefill_decode plan lost its P/D Router".to_owned(),
                    })?;
            let gateway_handoff = plan.links.iter().any(|link| {
                matches!(
                    link,
                    ServeRoleLink::RequestRouting { source, targets }
                        if source == "gateway" && targets == &[String::from("pd_router")]
                )
            });
            let request_routing = plan.links.iter().any(|link| {
                matches!(
                    link,
                    ServeRoleLink::RequestRouting { source, targets }
                        if source == &routing_source
                            && targets.iter().any(|target| target == &pd_router.prefill_role)
                            && targets.iter().any(|target| target == &pd_router.decode_role)
                )
            });
            let kv_link = plan.links.iter().any(|link| {
                matches!(
                    link,
                    ServeRoleLink::KvTransfer { source, target, mechanism }
                        if source == &pd_router.prefill_role
                            && target == &pd_router.decode_role
                            && Some(*mechanism) == kv_transfer
                )
            });
            if !gateway_handoff || !request_routing || !kv_link {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} did not declare the required Gateway, P/D routing, and KV links"
                    ),
                });
            }
            let transport_link = match (integration, kv_transfer) {
                ("tensorrt-llm", Some(KvTransferMechanism::Nixl)) => {
                    if plan.links.iter().any(|link| {
                        matches!(
                            link,
                            ServeRoleLink::Bootstrap { .. } | ServeRoleLink::SideChannel { .. }
                        )
                    }) {
                        return Err(InferlabError::InvalidConfig {
                            message: format!(
                                "integration {integration:?} declared a bootstrap or side-channel link for in-band NIXL transfer"
                            ),
                        });
                    }
                    true
                }
                ("sglang", Some(KvTransferMechanism::Mooncake | KvTransferMechanism::Nixl))
                | (_, Some(KvTransferMechanism::Mooncake)) => plan.links.iter().any(|link| {
                    matches!(
                        link,
                        ServeRoleLink::Bootstrap { source, target, port }
                            if source == &routing_source
                                && target == &pd_router.prefill_role
                                && role_all_have_port(
                                    &plan.replicas,
                                    &pd_router.prefill_role,
                                    port
                                )
                    )
                }),
                (_, Some(KvTransferMechanism::Nixl)) => plan.links.iter().any(|link| {
                    matches!(
                        link,
                        ServeRoleLink::SideChannel { source, target, port }
                            if source == &pd_router.prefill_role
                                && target == &pd_router.decode_role
                                && role_all_have_port(
                                    &plan.replicas,
                                    &pd_router.prefill_role,
                                    port
                                )
                                && role_all_have_port(
                                    &plan.replicas,
                                    &pd_router.decode_role,
                                    port
                                )
                    )
                }),
                (_, None) => false,
            };
            if !transport_link {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} did not declare the required KV transport link and process ports"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn role_all_have_port(replicas: &[ServeReplicaRequirement], role: &str, port: &str) -> bool {
    let mut role_replicas = replicas.iter().filter(|replica| replica.role_id == role);
    let Some(first) = role_replicas.next() else {
        return false;
    };
    first.ports.iter().any(|candidate| candidate == port)
        && role_replicas.all(|replica| replica.ports.iter().any(|candidate| candidate == port))
}

pub(super) fn validate_launch_dependencies(
    integration: &str,
    processes: &[ProcessRequirement],
) -> Result<(), InferlabError> {
    let mut prior = BTreeSet::new();
    for process in processes {
        if process.id().is_empty() || !prior.insert(process.id()) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned a duplicate or empty process id"
                ),
            });
        }
        let mut dependencies = BTreeSet::new();
        for dependency in process.launch_dependencies() {
            if !dependencies.insert(dependency) || !prior.contains(dependency.as_str()) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} process {:?} has an invalid or unordered launch dependency {dependency:?}",
                        process.id()
                    ),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn validate_integration_identity(
    expected: &str,
    actual: &str,
) -> Result<(), InferlabError> {
    if actual == expected {
        Ok(())
    } else {
        Err(InferlabError::InvalidConfig {
            message: format!("integration {expected:?} returned framework identity {actual:?}"),
        })
    }
}

/// [[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]: the accepted plan must return
/// the effective acceptance length whenever the request carried the
/// declaration — a finite value of at least one — and must not return the
/// outcome when the request carried none. The curve form additionally
/// requires the determined draft count; the explicit form must not carry one.
pub(super) fn validate_synthetic_acceptance_outcome(
    integration: &str,
    request: Option<&SyntheticAcceptanceInput>,
    outcome: Option<SyntheticAcceptanceOutcome>,
) -> Result<(), InferlabError> {
    match (request, outcome) {
        (Some(_), None) => Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} omitted the synthetic acceptance outcome although the plan request carried the declaration"
            ),
        }),
        (None, Some(_)) => Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} returned a synthetic acceptance outcome although the plan request carried no declaration"
            ),
        }),
        (Some(_), Some(outcome))
            if !outcome.acceptance_length.is_finite() || outcome.acceptance_length < 1.0 =>
        {
            Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned synthetic acceptance length {}; it must be a finite number of at least one",
                    outcome.acceptance_length
                ),
            })
        }
        (Some(SyntheticAcceptanceInput::Curve(_)), Some(outcome))
            if outcome.draft_count.is_none() =>
        {
            Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} omitted the determined draft count although the plan request carried the curve form"
                ),
            })
        }
        (Some(SyntheticAcceptanceInput::Explicit { .. }), Some(outcome))
            if outcome.draft_count.is_some() =>
        {
            Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned a draft count although the plan request carried the explicit form"
                ),
            })
        }
        (Some(_), Some(_)) | (None, None) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferlab_protocol::{
        EndpointProtocol, FrontendCoRendering, FrontendHandoff, FrontendProcessRole, GatewayPlan,
        IntegrationIdentity, Parallelism, PdRouterPlan, PdRoutingPolicies, ReadinessProbe,
        ServeRoleResult, SyntheticAcceptanceCurveInput, SyntheticAcceptanceOutcome,
        TargetEndpointScheme,
    };

    // [[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]: the outcome is required with
    // the declaration, forbidden without it, and always a finite value >= 1;
    // the curve form additionally requires the determined draft count and the
    // explicit form forbids it.
    #[test]
    fn synthetic_acceptance_outcome_matches_the_declaration_and_is_valid() {
        let outcome = |acceptance_length, draft_count| SyntheticAcceptanceOutcome {
            acceptance_length,
            draft_count,
        };
        let explicit = SyntheticAcceptanceInput::Explicit {
            acceptance_length: 2.0,
        };
        let curve = SyntheticAcceptanceInput::Curve(SyntheticAcceptanceCurveInput {
            model_key: "model".to_owned(),
            thinking_mode: None,
            text: "model:\n  - 4: 3.5\n".to_owned(),
            sha256: "a".repeat(64),
        });

        assert!(
            validate_synthetic_acceptance_outcome(
                "fixture",
                Some(&explicit),
                Some(outcome(2.0, None))
            )
            .is_ok()
        );
        assert!(
            validate_synthetic_acceptance_outcome(
                "fixture",
                Some(&curve),
                Some(outcome(3.5, Some(4)))
            )
            .is_ok()
        );
        assert!(validate_synthetic_acceptance_outcome("fixture", None, None).is_ok());

        for (request, value, expected) in [
            (
                Some(&explicit),
                None,
                "omitted the synthetic acceptance outcome",
            ),
            (None, Some(outcome(2.0, None)), "carried no declaration"),
            (
                Some(&explicit),
                Some(outcome(f64::NAN, None)),
                "finite number of at least one",
            ),
            (
                Some(&curve),
                Some(outcome(f64::INFINITY, Some(4))),
                "finite number of at least one",
            ),
            (
                Some(&explicit),
                Some(outcome(0.5, None)),
                "finite number of at least one",
            ),
            (
                Some(&curve),
                Some(outcome(3.5, None)),
                "omitted the determined draft count",
            ),
            (
                Some(&explicit),
                Some(outcome(2.0, Some(4))),
                "returned a draft count",
            ),
        ] {
            let error = validate_synthetic_acceptance_outcome("fixture", request, value)
                .err()
                .map(|error| error.to_string());
            assert!(
                error
                    .as_deref()
                    .is_some_and(|error| error.contains(expected)),
                "{expected}: {error:?}"
            );
        }
    }

    #[test]
    fn rejects_an_integration_that_rebinds_a_named_workload_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let endpoint = EndpointRequirement {
            protocol: EndpointProtocol::Http,
            completions_path: "/v1/completions".to_owned(),
            chat_completions_path: "/v1/completions".to_owned(),
            server_metrics: None,
            prefix_cache_reset: None,
            prefix_cache_conditioning: None,
            prompt_cache_read_zero_representation: None,
        };

        let error = validate_workload_endpoint("fixture", &endpoint, &[])
            .err()
            .ok_or("rebound chat-completions path was accepted")?;

        assert!(error.to_string().contains("chat_completions_path"));
        assert!(error.to_string().contains("/v1/chat/completions"));
        Ok(())
    }

    #[test]
    fn server_metrics_capability_is_an_origin_path_not_a_concrete_url()
    -> Result<(), Box<dyn std::error::Error>> {
        let endpoint = |server_metrics_path: Option<&str>| EndpointRequirement {
            protocol: EndpointProtocol::Http,
            completions_path: "/v1/completions".to_owned(),
            chat_completions_path: "/v1/chat/completions".to_owned(),
            server_metrics: server_metrics_path.map(|path| {
                inferlab_protocol::ServerMetricsEndpointRequirement {
                    path: path.to_owned(),
                    port: None,
                }
            }),
            prefix_cache_reset: None,
            prefix_cache_conditioning: None,
            prompt_cache_read_zero_representation: None,
        };

        validate_workload_endpoint("fixture", &endpoint(Some("/metrics")), &[])?;
        let error = validate_workload_endpoint(
            "fixture",
            &endpoint(Some("http://private.example/metrics")),
            &[],
        )
        .err()
        .ok_or("concrete server-metrics URL was accepted")?;

        assert!(error.to_string().contains("absolute origin path"));
        let whitespace_error =
            validate_workload_endpoint("fixture", &endpoint(Some("/metrics bad")), &[])
                .err()
                .ok_or("server-metrics path with whitespace was accepted")?;
        assert!(
            whitespace_error
                .to_string()
                .contains("absolute origin path"),
            "{whitespace_error}"
        );
        Ok(())
    }

    #[test]
    fn server_metrics_named_port_must_belong_to_the_public_process()
    -> Result<(), Box<dyn std::error::Error>> {
        let endpoint = EndpointRequirement {
            protocol: EndpointProtocol::Http,
            completions_path: "/v1/completions".to_owned(),
            chat_completions_path: "/v1/chat/completions".to_owned(),
            server_metrics: Some(inferlab_protocol::ServerMetricsEndpointRequirement {
                path: "/metrics".to_owned(),
                port: Some("prometheus".to_owned()),
            }),
            prefix_cache_reset: None,
            prefix_cache_conditioning: None,
            prompt_cache_read_zero_representation: None,
        };

        validate_workload_endpoint("fixture", &endpoint, &["prometheus".to_owned()])?;
        let error = validate_workload_endpoint("fixture", &endpoint, &[])
            .err()
            .ok_or("unknown server-metrics port was accepted")?;

        assert!(error.to_string().contains("public process"), "{error}");
        Ok(())
    }

    fn bootstrap_prefill_decode_plan(framework: &str) -> (Vec<ServeRoleInput>, PlanServeResult) {
        let role = |id: &str, kind| ServeRoleInput {
            id: id.to_owned(),
            kind,
            replica_count: 1,
            parallelism: Parallelism::default(),
            settings: BTreeMap::new(),
        };
        let requested_roles = vec![
            role("prefill", ServeRoleKind::Prefill),
            role("decode", ServeRoleKind::Decode),
        ];
        let roles = requested_roles
            .iter()
            .map(|role| ServeRoleResult {
                id: role.id.clone(),
                kind: role.kind,
                declared_replica_count: role.replica_count,
                effective_replica_count: role.replica_count,
                effective_settings: BTreeMap::new(),
                effective_parallelism: Parallelism::default(),
                public_endpoint: None,
                render_inputs: Vec::new(),
            })
            .collect();
        let replicas = requested_roles
            .iter()
            .map(|role| ServeReplicaRequirement {
                id: role.id.clone(),
                role_id: role.id.clone(),
                replica_index: 0,
                device_count: 1,
                ports: if role.kind == ServeRoleKind::Prefill {
                    vec!["bootstrap".to_owned()]
                } else {
                    Vec::new()
                },
                primary_ports: vec!["master".to_owned()],
                primary_readiness: ReadinessProbe::Http {
                    path: "/v1/models".to_owned(),
                },
                worker_readiness: ReadinessProbe::ProcessAlive,
                capture_target: None,
            })
            .collect();
        let implementation = match framework {
            "sglang" => "sglang",
            "tensorrt-llm" => "trtllm",
            _ => "vllm_nixl",
        };
        let co_rendering = FrontendCoRendering {
            process_role: FrontendProcessRole::Gateway,
        };
        let readiness = ReadinessProbe::Http {
            path: "/healthcheck".to_owned(),
        };
        let plan = PlanServeResult {
            integration: IntegrationIdentity {
                adapter_id: format!("inferlab-{framework}"),
                adapter_version: "1".to_owned(),
                framework: framework.to_owned(),
                framework_version: "test".to_owned(),
            },
            synthetic_acceptance: None,
            roles,
            replicas,
            links: vec![
                ServeRoleLink::RequestRouting {
                    source: "gateway".to_owned(),
                    targets: vec!["pd_router".to_owned()],
                },
                ServeRoleLink::RequestRouting {
                    source: "pd_router".to_owned(),
                    targets: vec!["prefill".to_owned(), "decode".to_owned()],
                },
                ServeRoleLink::KvTransfer {
                    source: "prefill".to_owned(),
                    target: "decode".to_owned(),
                    mechanism: KvTransferMechanism::Nixl,
                },
                ServeRoleLink::Bootstrap {
                    source: "pd_router".to_owned(),
                    target: "prefill".to_owned(),
                    port: "bootstrap".to_owned(),
                },
            ],
            gateway: Some(GatewayPlan {
                backend: "builtin".to_owned(),
                implementation: implementation.to_owned(),
                implementation_version: "1".to_owned(),
                effective_settings: BTreeMap::new(),
                endpoint: EndpointRequirement {
                    protocol: EndpointProtocol::Http,
                    completions_path: "/v1/completions".to_owned(),
                    chat_completions_path: "/v1/chat/completions".to_owned(),
                    server_metrics: None,
                    prefix_cache_reset: None,
                    prefix_cache_conditioning: None,
                    prompt_cache_read_zero_representation: None,
                },
                readiness: readiness.clone(),
                ports: Vec::new(),
                targets: vec![GatewayTarget::PdRouter],
                render_inputs: Vec::new(),
                render_source: RenderSource::ControlPlane,
                co_rendering: co_rendering.clone(),
            }),
            pd_router: Some(PdRouterPlan {
                backend: "builtin".to_owned(),
                implementation: implementation.to_owned(),
                implementation_version: "1".to_owned(),
                effective_settings: BTreeMap::new(),
                policies: PdRoutingPolicies {
                    prefill: "round_robin".to_owned(),
                    decode: "round_robin".to_owned(),
                },
                prefill_role: "prefill".to_owned(),
                decode_role: "decode".to_owned(),
                target_scheme: TargetEndpointScheme::Http,
                ports: Vec::new(),
                readiness,
                handoff: FrontendHandoff::InProcess,
                render_inputs: Vec::new(),
                render_source: RenderSource::ControlPlane,
                co_rendering,
            }),
        };
        (requested_roles, plan)
    }

    fn native_trtllm_prefill_decode_plan() -> (Vec<ServeRoleInput>, PlanServeResult) {
        let (requested_roles, mut plan) = bootstrap_prefill_decode_plan("tensorrt-llm");
        plan.links
            .retain(|link| !matches!(link, ServeRoleLink::Bootstrap { .. }));
        for replica in &mut plan.replicas {
            replica.ports.clear();
        }
        if let Some(gateway) = &mut plan.gateway {
            gateway.backend = "trtllm-disaggregated".to_owned();
            gateway.implementation = "trtllm-disaggregated".to_owned();
            gateway.render_source = RenderSource::Integration;
        }
        if let Some(pd_router) = &mut plan.pd_router {
            pd_router.backend = "trtllm-disaggregated".to_owned();
            pd_router.implementation = "trtllm-disaggregated".to_owned();
            pd_router.render_source = RenderSource::Integration;
        }
        (requested_roles, plan)
    }

    #[test]
    fn gateway_and_pd_router_backend_facts_are_validated_independently() {
        let (roles, mut plan) = bootstrap_prefill_decode_plan("sglang");
        if let Some(gateway) = &mut plan.gateway {
            gateway.backend = "gateway-provider".to_owned();
        }
        if let Some(pd_router) = &mut plan.pd_router {
            pd_router.backend = "pd-provider".to_owned();
        }

        assert!(
            validate_serve_graph(
                "sglang",
                ServeTopology::PrefillDecode,
                &roles,
                Some("gateway-provider"),
                Some("pd-provider"),
                Some(KvTransferMechanism::Nixl),
                &plan,
            )
            .is_ok()
        );
        assert!(
            validate_serve_graph(
                "sglang",
                ServeTopology::PrefillDecode,
                &roles,
                Some("pd-provider"),
                Some("gateway-provider"),
                Some(KvTransferMechanism::Nixl),
                &plan,
            )
            .is_err()
        );
    }

    fn add_second_replica(
        requested_roles: &mut [ServeRoleInput],
        plan: &mut PlanServeResult,
        role_id: &str,
        ports: Vec<String>,
    ) -> Result<(), String> {
        requested_roles
            .iter_mut()
            .find(|role| role.id == role_id)
            .ok_or_else(|| format!("missing requested role {role_id:?}"))?
            .replica_count = 2;
        plan.roles
            .iter_mut()
            .find(|role| role.id == role_id)
            .ok_or_else(|| format!("missing planned role {role_id:?}"))?
            .declared_replica_count = 2;
        plan.roles
            .iter_mut()
            .find(|role| role.id == role_id)
            .ok_or_else(|| format!("missing planned role {role_id:?}"))?
            .effective_replica_count = 2;
        let mut replica = plan
            .replicas
            .iter()
            .find(|replica| replica.role_id == role_id)
            .cloned()
            .ok_or_else(|| format!("missing planned replica for role {role_id:?}"))?;
        replica.id = format!("{role_id}-001");
        replica.replica_index = 1;
        replica.ports = ports;
        plan.replicas.push(replica);
        Ok(())
    }

    fn add_first_replica_port(
        plan: &mut PlanServeResult,
        role_id: &str,
        port: &str,
    ) -> Result<(), String> {
        plan.replicas
            .iter_mut()
            .find(|replica| replica.role_id == role_id && replica.replica_index == 0)
            .ok_or_else(|| format!("missing first replica for role {role_id:?}"))?
            .ports
            .push(port.to_owned());
        Ok(())
    }

    #[test]
    fn nixl_transport_link_is_framework_specific() -> Result<(), String> {
        let (sglang_roles, sglang_plan) = bootstrap_prefill_decode_plan("sglang");
        assert!(
            validate_serve_graph(
                "sglang",
                ServeTopology::PrefillDecode,
                &sglang_roles,
                Some("builtin"),
                Some("builtin"),
                Some(KvTransferMechanism::Nixl),
                &sglang_plan,
            )
            .is_ok()
        );

        let (vllm_roles, vllm_plan) = bootstrap_prefill_decode_plan("vllm");
        let result = validate_serve_graph(
            "vllm",
            ServeTopology::PrefillDecode,
            &vllm_roles,
            Some("builtin"),
            Some("builtin"),
            Some(KvTransferMechanism::Nixl),
            &vllm_plan,
        );
        assert!(result.is_err());
        assert!(
            result
                .err()
                .is_some_and(|error| error.to_string().contains("required KV transport link"))
        );

        let (trtllm_roles, trtllm_plan) = native_trtllm_prefill_decode_plan();
        assert!(
            validate_serve_graph(
                "tensorrt-llm",
                ServeTopology::PrefillDecode,
                &trtllm_roles,
                Some("trtllm-disaggregated"),
                Some("trtllm-disaggregated"),
                Some(KvTransferMechanism::Nixl),
                &trtllm_plan,
            )
            .is_ok()
        );

        let mut endpoint_link_plan = trtllm_plan.clone();
        add_first_replica_port(&mut endpoint_link_plan, "prefill", "bootstrap")?;
        endpoint_link_plan.links.push(ServeRoleLink::Bootstrap {
            source: "pd_router".to_owned(),
            target: "prefill".to_owned(),
            port: "bootstrap".to_owned(),
        });
        let result = validate_serve_graph(
            "tensorrt-llm",
            ServeTopology::PrefillDecode,
            &trtllm_roles,
            Some("trtllm-disaggregated"),
            Some("trtllm-disaggregated"),
            Some(KvTransferMechanism::Nixl),
            &endpoint_link_plan,
        );
        assert!(result.is_err_and(|error| error.to_string().contains("in-band NIXL")));
        Ok(())
    }

    #[test]
    fn bootstrap_link_requires_every_target_replica_endpoint() -> Result<(), String> {
        let (mut roles, mut plan) = bootstrap_prefill_decode_plan("sglang");
        add_first_replica_port(&mut plan, "decode", "diagnostic")?;
        add_second_replica(&mut roles, &mut plan, "decode", Vec::new())?;
        plan.links.push(ServeRoleLink::Bootstrap {
            source: "pd_router".to_owned(),
            target: "decode".to_owned(),
            port: "diagnostic".to_owned(),
        });

        let result = validate_serve_graph(
            "sglang",
            ServeTopology::PrefillDecode,
            &roles,
            Some("builtin"),
            Some("builtin"),
            Some(KvTransferMechanism::Nixl),
            &plan,
        );

        assert!(result.is_err_and(|error| error.to_string().contains("unknown component")));
        Ok(())
    }

    #[test]
    fn side_channel_link_requires_every_source_and_target_replica_endpoint() -> Result<(), String> {
        for missing_role in ["prefill", "decode"] {
            let (mut roles, mut plan) = bootstrap_prefill_decode_plan("sglang");
            add_first_replica_port(&mut plan, "prefill", "diagnostic")?;
            add_first_replica_port(&mut plan, "decode", "diagnostic")?;
            let second_ports = if missing_role == "prefill" {
                vec!["bootstrap".to_owned()]
            } else {
                Vec::new()
            };
            add_second_replica(&mut roles, &mut plan, missing_role, second_ports)?;
            plan.links.push(ServeRoleLink::SideChannel {
                source: "prefill".to_owned(),
                target: "decode".to_owned(),
                port: "diagnostic".to_owned(),
            });

            let result = validate_serve_graph(
                "sglang",
                ServeTopology::PrefillDecode,
                &roles,
                Some("builtin"),
                Some("builtin"),
                Some(KvTransferMechanism::Nixl),
                &plan,
            );

            assert!(
                result.is_err_and(|error| error.to_string().contains("unknown component")),
                "side channel accepted a missing {missing_role} replica endpoint"
            );
        }
        Ok(())
    }
}
