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
use crate::workload::{MeasurementPlan, MeasurementResolveContext, resolve_measurements};
use crate::workspace::LoadedWorkspace;
use inferlab_protocol::{EndpointProtocol, ProtocolVersion, ServeProcessAllocation};
use std::collections::BTreeMap;
#[cfg(test)]
use std::path::{Path, PathBuf};

use inferlab_serve_domain::{ResolvedProcessAllocation, RuntimeRealizationParts};
use integration::{plan_integration, render_integration};
use realization::{assemble_process_hierarchy, realize_runtime};
use selection::{WorkflowSelection, resolve_effective_server_input, select_workflow};
use topology::profiler_escapes_plan;

#[cfg(test)]
use crate::execution::{RuntimeCacheNamespacePlan, RuntimeCacheRootSource};
#[cfg(test)]
use allocation::readiness_plan;
#[cfg(test)]
use inferlab_protocol::{
    AllocationLaunch, ProcessSpec, RenderSource, ServeReplicaRequirement, TargetEndpointScheme,
};
#[cfg(test)]
use inferlab_runtime::plan::{ReadinessPlan, TargetRegistryExpectedTarget};
#[cfg(test)]
use integration::{load_render_inputs, validate_launch_file_declarations};
#[cfg(test)]
use selection::validate_effective_parallelism;
#[cfg(test)]
use sha2::{Digest, Sha256};
#[cfg(test)]
use topology::{validate_serve_graph, validate_workload_endpoint};

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
    let profiling = effective.profiling;

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
    let measurements = compose_measurements(
        workspace,
        request,
        &overrides,
        &selection,
        &public_endpoint,
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
                protocol_version: ProtocolVersion::V7,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{ModelLocatorSource, RuntimeCachePlan};
    use inferlab_protocol::{
        EndpointAssignment, EndpointRequirement, FrontendCoRendering, FrontendHandoff,
        FrontendProcessRole, GatewayPlan, GatewayTarget, HttpTargetRegistryReadiness,
        IntegrationIdentity, KvTransferMechanism, LaunchFileDeclaration, Parallelism,
        ParallelismAttention, ParallelismExperts, ParallelismOuter, PdRouterPlan,
        PdRoutingPolicies, PlanServeResult, ReadinessProbe, RenderInputDeclaration, ServeRoleInput,
        ServeRoleKind, ServeRoleLink, ServeRoleResult, ServeTopology,
    };
    use std::error::Error;

    #[test]
    fn rejects_an_integration_that_rebinds_a_named_workload_path() -> Result<(), Box<dyn Error>> {
        let endpoint = EndpointRequirement {
            protocol: EndpointProtocol::Http,
            completions_path: "/v1/completions".to_owned(),
            chat_completions_path: "/v1/completions".to_owned(),
            server_metrics: None,
            prefix_cache_reset: None,
        };

        let error = validate_workload_endpoint("fixture", &endpoint, &[])
            .err()
            .ok_or("rebound chat-completions path was accepted")?;

        assert!(error.to_string().contains("chat_completions_path"));
        assert!(error.to_string().contains("/v1/chat/completions"));
        Ok(())
    }

    #[test]
    fn server_metrics_capability_is_an_origin_path_not_a_concrete_url() -> Result<(), Box<dyn Error>>
    {
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
    fn server_metrics_named_port_must_belong_to_the_public_process() -> Result<(), Box<dyn Error>> {
        let endpoint = EndpointRequirement {
            protocol: EndpointProtocol::Http,
            completions_path: "/v1/completions".to_owned(),
            chat_completions_path: "/v1/chat/completions".to_owned(),
            server_metrics: Some(inferlab_protocol::ServerMetricsEndpointRequirement {
                path: "/metrics".to_owned(),
                port: Some("prometheus".to_owned()),
            }),
            prefix_cache_reset: None,
        };

        validate_workload_endpoint("fixture", &endpoint, &["prometheus".to_owned()])?;
        let error = validate_workload_endpoint("fixture", &endpoint, &[])
            .err()
            .ok_or("unknown server-metrics port was accepted")?;

        assert!(error.to_string().contains("public process"), "{error}");
        Ok(())
    }

    fn launch_file(text: &str, name: &str) -> LaunchFileDeclaration {
        let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
        LaunchFileDeclaration {
            relative_path: format!("launch-files/{sha256}/{name}"),
            text: text.to_owned(),
            sha256,
        }
    }

    fn launch_process(argv: Vec<String>, env: BTreeMap<String, String>) -> ProcessSpec {
        ProcessSpec { argv, env }
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
    fn effective_parallelism_preserves_explicit_role_components() {
        let declared = Parallelism {
            outer: Some(ParallelismOuter {
                tensor_parallel_size: Some(4),
                pipeline_parallel_size: None,
            }),
            ..Parallelism::default()
        };
        let effective = Parallelism {
            outer: Some(ParallelismOuter {
                tensor_parallel_size: Some(2),
                pipeline_parallel_size: Some(1),
            }),
            attention: Some(ParallelismAttention {
                tensor_parallel_size: Some(2),
                data_parallel_size: Some(1),
                context_parallel_size: Some(1),
            }),
            experts: Some(ParallelismExperts {
                tensor_parallel_size: Some(2),
                data_parallel_size: Some(1),
                expert_parallel_size: Some(1),
                dense_tensor_parallel_size: Some(1),
            }),
        };

        let result =
            validate_effective_parallelism("fixture", "role \"prefill\"", &declared, &effective);

        assert!(result.is_err());
        assert!(
            result
                .err()
                .is_some_and(|error| error.to_string().contains("outer.tensor_parallel_size"))
        );
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

    #[test]
    fn target_registry_readiness_derives_rank_zero_serving_targets() -> Result<(), InferlabError> {
        let allocation =
            |process_id: &str, role_id: &str, rank: u32, port: u16, bootstrap_port: Option<u16>| {
                let mut ports = BTreeMap::new();
                if let Some(bootstrap_port) = bootstrap_port {
                    ports.insert(
                        "bootstrap".to_owned(),
                        EndpointAssignment {
                            host: "node.example".to_owned(),
                            port: bootstrap_port,
                        },
                    );
                }
                ResolvedProcessAllocation::new(
                    ServeProcessAllocation::ModelRank {
                        process: process_id.to_owned(),
                        role: role_id.to_owned(),
                        role_kind: if role_id == "prefill" {
                            ServeRoleKind::Prefill
                        } else {
                            ServeRoleKind::Decode
                        },
                        replica: 0,
                        rank,
                        rank_count: if role_id == "prefill" { 2 } else { 1 },
                        machine: "node".to_owned(),
                        model_locator: "/models/example".to_owned(),
                        devices: vec![0],
                        endpoint: Some(EndpointAssignment {
                            host: "node.example".to_owned(),
                            port,
                        }),
                        ports,
                        cache: format!("/cache/{process_id}"),
                        launch: AllocationLaunch::Local,
                        effective_settings: BTreeMap::new(),
                        effective_parallelism: Parallelism::default(),
                        links: Vec::new(),
                        dependencies: Vec::new(),
                        render_inputs: Vec::new(),
                    },
                    RuntimeCachePlan {
                        storage_root: PathBuf::from("/cache"),
                        storage_root_source: RuntimeCacheRootSource::WorkspaceDefault,
                        namespace: RuntimeCacheNamespacePlan {
                            workspace_source_digest: "source".to_owned(),
                            pixi_environment: "sglang".to_owned(),
                            image_id: None,
                            machine: "node".to_owned(),
                            process: process_id.to_owned(),
                        },
                        path: PathBuf::from(format!("/cache/{process_id}")),
                    },
                    Some(ModelLocatorSource::Fallback),
                )
            };
        let allocations = vec![
            allocation("prefill", "prefill", 0, 8000, Some(9000)),
            allocation("prefill-rank-001", "prefill", 1, 8001, Some(9001)),
            allocation("decode", "decode", 0, 8100, None),
        ];
        let probe = ReadinessProbe::HttpTargetRegistry(Box::new(HttpTargetRegistryReadiness {
            target_scheme: inferlab_protocol::TargetEndpointScheme::Grpc,
            readiness_path: "/readiness".to_owned(),
            registry_path: "/workers".to_owned(),
            targets_field: "workers".to_owned(),
            target_url_field: "url".to_owned(),
            target_role_field: "worker_type".to_owned(),
            target_healthy_field: "is_healthy".to_owned(),
            target_bootstrap_port_field: "bootstrap_port".to_owned(),
            prefill_role_value: "prefill".to_owned(),
            decode_role_value: "decode".to_owned(),
            prefill_bootstrap_port: "bootstrap".to_owned(),
        }));

        let readiness = readiness_plan(&probe, 900, 30, false, &allocations)?;
        assert!(matches!(
            &readiness,
            ReadinessPlan::HttpTargetRegistry { .. }
        ));
        if let ReadinessPlan::HttpTargetRegistry {
            expected_targets, ..
        } = readiness
        {
            assert_eq!(
                expected_targets,
                vec![
                    TargetRegistryExpectedTarget {
                        url: "grpc://node.example:8000".to_owned(),
                        role: "prefill".to_owned(),
                        bootstrap_port: Some(9000),
                    },
                    TargetRegistryExpectedTarget {
                        url: "grpc://node.example:8100".to_owned(),
                        role: "decode".to_owned(),
                        bootstrap_port: None,
                    },
                ]
            );
        }
        Ok(())
    }

    #[test]
    fn render_inputs_preserve_original_paths_exact_text_and_digest()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace = tempfile::tempdir()?;
        let relative_path = "configs/operator.yaml";
        let relative_text = "batch_scheduler:\n  enable_chunked_context: true\n";
        std::fs::create_dir_all(workspace.path().join("configs"))?;
        std::fs::write(workspace.path().join(relative_path), relative_text)?;

        let absolute = workspace.path().join("absolute.yaml");
        let absolute_text = "kv_cache_config:\n  enable_block_reuse: false\n";
        std::fs::write(&absolute, absolute_text)?;
        let absolute_path = absolute.to_string_lossy().into_owned();
        let declarations = vec![
            RenderInputDeclaration {
                source_path: relative_path.to_owned(),
            },
            RenderInputDeclaration {
                source_path: absolute_path.clone(),
            },
        ];

        let supplied = load_render_inputs(workspace.path(), "tensorrt-llm", &declarations)?;

        assert_eq!(supplied[0].source_path, relative_path);
        assert_eq!(supplied[0].text, relative_text);
        assert_eq!(
            supplied[0].sha256,
            format!("{:x}", Sha256::digest(relative_text.as_bytes()))
        );
        assert_eq!(supplied[1].source_path, absolute_path);
        assert_eq!(supplied[1].text, absolute_text);
        assert_eq!(
            supplied[1].sha256,
            format!("{:x}", Sha256::digest(absolute_text.as_bytes()))
        );
        Ok(())
    }

    #[test]
    fn unreadable_render_input_is_a_typed_resolution_error()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace = tempfile::tempdir()?;
        let missing = RenderInputDeclaration {
            source_path: "configs/missing.yaml".to_owned(),
        };

        let result = load_render_inputs(workspace.path(), "tensorrt-llm", &[missing]);

        assert!(matches!(result, Err(InferlabError::RenderInputRead { .. })));
        Ok(())
    }

    #[test]
    fn non_utf8_render_input_is_a_typed_resolution_error() -> Result<(), Box<dyn std::error::Error>>
    {
        let workspace = tempfile::tempdir()?;
        std::fs::write(workspace.path().join("operator.yaml"), [0xff, 0xfe])?;
        let declaration = RenderInputDeclaration {
            source_path: "operator.yaml".to_owned(),
        };

        let result = load_render_inputs(workspace.path(), "tensorrt-llm", &[declaration]);

        assert!(matches!(result, Err(InferlabError::RenderInputUtf8 { .. })));
        Ok(())
    }

    #[test]
    fn launch_files_preserve_valid_argv_and_env_references() -> Result<(), InferlabError> {
        let cache_root = Path::new("/does/not/need/to/exist/cache/worker");
        let argv_file = launch_file("worker: argv\n", "worker.yaml");
        let env_file = launch_file("worker: 零\n", "environment.yaml");
        let argv_path = cache_root.join(&argv_file.relative_path);
        let env_path = cache_root.join(&env_file.relative_path);
        let process = launch_process(
            vec![
                "server".to_owned(),
                "--config".to_owned(),
                argv_path.to_string_lossy().into_owned(),
            ],
            BTreeMap::from([(
                "SERVER_CONFIG".to_owned(),
                env_path.to_string_lossy().into_owned(),
            )]),
        );

        let plans = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[argv_file.clone(), env_file.clone()],
        )?;

        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].relative_path, argv_file.relative_path);
        assert_eq!(plans[0].resolved_path, argv_path);
        assert_eq!(plans[0].text, argv_file.text);
        assert_eq!(plans[0].sha256, argv_file.sha256);
        assert_eq!(plans[1].resolved_path, env_path);
        Ok(())
    }

    #[test]
    fn launch_file_path_must_be_canonical() {
        let cache_root = Path::new("/cache/worker");
        let mut declaration = launch_file("worker: invalid-path\n", "worker.yaml");
        declaration.relative_path =
            format!("launch-files/{}/nested/worker.yaml", declaration.sha256);
        let resolved = cache_root.join(&declaration.relative_path);
        let process = launch_process(
            vec![resolved.to_string_lossy().into_owned()],
            BTreeMap::new(),
        );

        let result = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[declaration],
        );

        assert!(result.is_err_and(|error| error.to_string().contains("canonical path")));
    }

    #[test]
    fn launch_file_digest_must_match_utf8_text() {
        let cache_root = Path::new("/cache/worker");
        let mut declaration = launch_file("worker: original\n", "worker.yaml");
        declaration.text = "worker: changed\n".to_owned();
        let resolved = cache_root.join(&declaration.relative_path);
        let process = launch_process(
            vec![resolved.to_string_lossy().into_owned()],
            BTreeMap::new(),
        );

        let result = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[declaration],
        );

        assert!(result.is_err_and(|error| error.to_string().contains("content digest")));
    }

    #[test]
    fn launch_file_digest_must_be_complete_lowercase_hex() {
        let cache_root = Path::new("/cache/worker");
        let mut declaration = launch_file("worker: uppercase-digest\n", "worker.yaml");
        declaration.sha256.make_ascii_uppercase();
        declaration.relative_path = format!("launch-files/{}/worker.yaml", declaration.sha256);
        let resolved = cache_root.join(&declaration.relative_path);
        let process = launch_process(
            vec![resolved.to_string_lossy().into_owned()],
            BTreeMap::new(),
        );

        let result = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[declaration],
        );

        assert!(result.is_err_and(|error| error.to_string().contains("64-lowercase-sha256")));
    }

    #[test]
    fn launch_file_requires_an_exact_invocation_reference() {
        let cache_root = Path::new("/cache/worker");
        let declaration = launch_file("worker: unreferenced\n", "worker.yaml");
        let resolved = cache_root.join(&declaration.relative_path);
        let process = launch_process(
            vec![format!("--config={}", resolved.to_string_lossy())],
            BTreeMap::new(),
        );

        let result = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[declaration],
        );

        assert!(result.is_err_and(|error| error.to_string().contains("exact argv or environment")));
    }
}
