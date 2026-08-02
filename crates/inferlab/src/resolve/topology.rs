use crate::InferlabError;
use crate::execution::ProfilerEscapesPlan;
use crate::workspace::{PlacementBinding, PlacementRoleBinding, ServerDefinition};
use inferlab_profiler::plan::{CaptureWindowControlEndpointPlan, CaptureWindowHttpMethodPlan};
use inferlab_protocol::{
    CaptureWindowControlEndpoint, EndpointAssignment, EndpointRequirement, FrontendComponents,
    FrontendProcessRole, GatewayTarget, KvTransferMechanism, PlanServeResult, RenderSource,
    ServeReplicaRequirement, ServeRoleInput, ServeRoleKind, ServeRoleLink, ServeTopology,
    SuppliedRenderInput,
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
    profiling: bool,
    has_gateway: bool,
    replicas: &[ServeReplicaRequirement],
) -> Result<(), InferlabError> {
    for replica in replicas {
        if let Some(target) = &replica.capture_target {
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
        if !profiling {
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
    if server.profiler.nsys.is_empty() && roles.is_empty() {
        return None;
    }
    Some(ProfilerEscapesPlan {
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
                    capture_window_control_endpoint_plan(target.window_control.endpoint),
                    primary_id.clone(),
                    capture_window_action_plan(&target.window_control.start),
                    capture_window_action_plan(&target.window_control.stop),
                    server.roles.get(&replica.role_id).map_or_else(
                        || server.profiler.nsys.clone(),
                        |role| server.profiler.nsys.merged_with(&role.profiler.nsys),
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
