use super::ResolveRequest;
use super::allocation::{allocate_processes, gate_engine_trace_placement};
use super::selection::{EffectiveServerInput, WorkflowSelection, validate_effective_parallelism};
use super::topology::{
    endpoint_url, expand_replica_requirements, links_for_node, uses_explicit_replica_placement,
    validate_capture_targets, validate_integration_identity, validate_launch_dependencies,
    validate_serve_graph,
};
use crate::InferlabError;
use crate::adapter::{AdapterClient, AdapterLowering};
use crate::workspace::LoadedWorkspace;
use inferlab_protocol::{
    FrontendComponents, FrontendProcessRole, GatewayPlan, KvTransferMechanism,
    LaunchFileDeclaration, PdRouterPlan, PlanServeInput, ProcessSpec, RenderInputDeclaration,
    RenderServeInput, RenderSource, RenderedServeProcess, ServeModelInput, ServeProcessAllocation,
    SuppliedRenderInput,
};
use inferlab_runtime::plan::LaunchFilePlan;
use inferlab_serve_domain::{
    FixedDeviceAssignment, LoweringEvidence, PlannedServeStage, ProcessRequirement,
    ProcessRequirementIdentity, RenderedServeStage, ResolvedProcessAllocation,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuiltinProxyKind {
    VllmMooncake,
    VllmNixl,
    Sglang,
    Trtllm,
}

impl BuiltinProxyKind {
    const fn command_name(self) -> &'static str {
        match self {
            Self::VllmMooncake => "vllm-mooncake",
            Self::VllmNixl => "vllm-nixl",
            Self::Sglang => "sglang",
            Self::Trtllm => "trtllm",
        }
    }
}

fn builtin_proxy_kind(
    framework: &str,
    transport: Option<KvTransferMechanism>,
) -> Result<BuiltinProxyKind, InferlabError> {
    match (framework, transport) {
        ("vllm", Some(KvTransferMechanism::Mooncake)) => Ok(BuiltinProxyKind::VllmMooncake),
        ("vllm", Some(KvTransferMechanism::Nixl)) => Ok(BuiltinProxyKind::VllmNixl),
        ("sglang", Some(KvTransferMechanism::Mooncake | KvTransferMechanism::Nixl)) => {
            Ok(BuiltinProxyKind::Sglang)
        }
        ("tensorrt-llm", Some(KvTransferMechanism::Nixl)) => Ok(BuiltinProxyKind::Trtllm),
        (_, None) => Err(InferlabError::InvalidConfig {
            message: "built-in prefill/decode proxy requires a KV-transfer mechanism".to_owned(),
        }),
        _ => Err(InferlabError::InvalidConfig {
            message: format!(
                "framework {framework:?} does not support the built-in prefill/decode proxy"
            ),
        }),
    }
}

fn render_builtin_frontend(
    gateway: &GatewayPlan,
    pd_router: &PdRouterPlan,
    framework: &str,
    transport: Option<KvTransferMechanism>,
    prefill_data_parallel_size: u32,
    allocations: &[ResolvedProcessAllocation],
) -> Result<RenderedServeProcess, InferlabError> {
    if gateway.render_source != RenderSource::ControlPlane
        || pd_router.render_source != RenderSource::ControlPlane
    {
        return Err(InferlabError::InvalidConfig {
            message: "expected control-plane-rendered frontend component plans".to_owned(),
        });
    }
    let process_id = "gateway";
    let proxy = allocations
        .iter()
        .find(|allocation| allocation.process() == process_id)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("built-in frontend process {process_id:?} was not allocated"),
        })?;
    let prefill = allocations
        .iter()
        .filter(|allocation| {
            matches!(
                allocation.wire(),
                ServeProcessAllocation::ModelRank { role, rank: 0, .. }
                    if role == &pd_router.prefill_role
            )
        })
        .collect::<Vec<_>>();
    let decode = allocations
        .iter()
        .filter(|allocation| {
            matches!(
                allocation.wire(),
                ServeProcessAllocation::ModelRank { role, rank: 0, .. }
                    if role == &pd_router.decode_role
            )
        })
        .collect::<Vec<_>>();
    if prefill.is_empty() || decode.is_empty() {
        return Err(InferlabError::InvalidConfig {
            message: "built-in proxy requires prefill and decode replica entry points".to_owned(),
        });
    }
    let proxy_kind = builtin_proxy_kind(framework, transport)?;
    let declared_kind = match gateway.implementation.as_str() {
        "vllm_mooncake" => BuiltinProxyKind::VllmMooncake,
        "vllm_nixl" => BuiltinProxyKind::VllmNixl,
        "sglang" => BuiltinProxyKind::Sglang,
        "trtllm" => BuiltinProxyKind::Trtllm,
        implementation => {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration returned unknown control-plane frontend implementation {implementation:?}"
                ),
            });
        }
    };
    if proxy_kind != declared_kind || pd_router.implementation != gateway.implementation {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration returned control-plane frontend implementation {:?}, which is incompatible with framework {framework:?} and transport {transport:?}",
                gateway.implementation
            ),
        });
    }
    let proxy_endpoint = proxy
        .endpoint()
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: "built-in frontend allocation has no endpoint".to_owned(),
        })?;
    let executable = std::env::current_exe().map_err(|source| InferlabError::Read {
        path: PathBuf::from("/proc/self/exe"),
        source,
    })?;
    let mut argv = vec![
        executable.to_string_lossy().into_owned(),
        "__internal".to_owned(),
        "proxy".to_owned(),
        proxy_kind.command_name().to_owned(),
        "--host".to_owned(),
        proxy_endpoint.host.clone(),
        "--port".to_owned(),
        proxy_endpoint.port.to_string(),
    ];
    for replica in prefill {
        let endpoint = replica
            .endpoint()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("prefill allocation {:?} has no endpoint", replica.process()),
            })?;
        argv.extend(["--prefill".to_owned(), endpoint_url(endpoint)]);
        if matches!(
            proxy_kind,
            BuiltinProxyKind::VllmMooncake | BuiltinProxyKind::Sglang
        ) {
            let bootstrap =
                replica
                    .ports()
                    .get("bootstrap")
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!(
                            "prefill allocation {:?} has no bootstrap endpoint",
                            replica.process()
                        ),
                    })?;
            match proxy_kind {
                BuiltinProxyKind::VllmMooncake => argv.push(endpoint_url(bootstrap)),
                BuiltinProxyKind::Sglang => {
                    argv.extend([bootstrap.host.clone(), bootstrap.port.to_string()]);
                }
                BuiltinProxyKind::VllmNixl | BuiltinProxyKind::Trtllm => {}
            }
        }
        // The NIXL and SGLang proxies have no rank discovery: the control
        // plane issues each prefill replica's effective attention
        // data-parallel size so the conditioning fan-out can pin every rank
        // ([[RFC-0004:C-BENCH-CACHE-STATE]]). The Mooncake proxy discovers
        // its ranks and engine ids from the bootstrap query instead.
        if matches!(
            proxy_kind,
            BuiltinProxyKind::VllmNixl | BuiltinProxyKind::Sglang
        ) {
            argv.extend([
                "--prefill-dp".to_owned(),
                prefill_data_parallel_size.to_string(),
            ]);
        }
    }
    for replica in decode {
        let endpoint = replica
            .endpoint()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("decode allocation {:?} has no endpoint", replica.process()),
            })?;
        argv.extend(["--decode".to_owned(), endpoint_url(endpoint)]);
    }
    Ok(RenderedServeProcess::Frontend {
        process: process_id.to_owned(),
        process_role: FrontendProcessRole::Gateway,
        components: FrontendComponents::gateway_pd_router(),
        command: ProcessSpec {
            argv,
            env: BTreeMap::new(),
        },
        launch_files: Vec::new(),
    })
}

pub(super) fn load_render_inputs(
    workspace_root: &Path,
    integration: &str,
    declarations: &[RenderInputDeclaration],
) -> Result<Vec<SuppliedRenderInput>, InferlabError> {
    declarations
        .iter()
        .map(|declaration| {
            let source = Path::new(&declaration.source_path);
            let path = if source.is_absolute() {
                source.to_owned()
            } else {
                workspace_root.join(source)
            };
            let bytes = std::fs::read(&path).map_err(|source| InferlabError::RenderInputRead {
                integration: integration.to_owned(),
                source_path: declaration.source_path.clone(),
                path: path.clone(),
                source,
            })?;
            let text =
                String::from_utf8(bytes).map_err(|source| InferlabError::RenderInputUtf8 {
                    integration: integration.to_owned(),
                    source_path: declaration.source_path.clone(),
                    path,
                    source,
                })?;
            let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
            Ok(SuppliedRenderInput {
                source_path: declaration.source_path.clone(),
                text,
                sha256,
            })
        })
        .collect()
}

pub(super) fn validate_launch_file_declarations(
    integration: &str,
    process_id: &str,
    runtime_cache_root: &Path,
    process: &ProcessSpec,
    declarations: &[LaunchFileDeclaration],
) -> Result<Vec<LaunchFilePlan>, InferlabError> {
    declarations
        .iter()
        .map(|declaration| {
            let relative_path = Path::new(&declaration.relative_path);
            let components = relative_path.components().collect::<Vec<_>>();
            let name = match components.as_slice() {
                [
                    Component::Normal(root),
                    Component::Normal(digest),
                    Component::Normal(name),
                ] if root.to_str() == Some("launch-files")
                    && digest.to_str() == Some(declaration.sha256.as_str())
                    && is_lowercase_sha256(&declaration.sha256) =>
                {
                    name.to_str()
                }
                _ => None,
            };
            let Some(name) = name else {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {:?} for process \
                         {process_id:?} without canonical path \
                         launch-files/<64-lowercase-sha256>/<name>",
                        declaration.relative_path
                    ),
                });
            };
            let canonical = format!("launch-files/{}/{name}", declaration.sha256);
            if declaration.relative_path != canonical {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {:?} for process \
                         {process_id:?} without canonical path \
                         launch-files/<64-lowercase-sha256>/<name>",
                        declaration.relative_path
                    ),
                });
            }
            let actual_sha256 = format!("{:x}", Sha256::digest(declaration.text.as_bytes()));
            if declaration.sha256 != actual_sha256 {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {:?} for process \
                         {process_id:?} with content digest {:?}, expected {actual_sha256:?}",
                        declaration.relative_path, declaration.sha256
                    ),
                });
            }
            let resolved_path = runtime_cache_root.join(relative_path);
            if !matches!(
                resolved_path.strip_prefix(runtime_cache_root),
                Ok(path) if path == relative_path
            ) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {:?} outside process \
                         {process_id:?} runtime cache {:?}",
                        declaration.relative_path, runtime_cache_root
                    ),
                });
            }
            let resolved = resolved_path.to_str().ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "process {process_id:?} launch-file path {resolved_path:?} is not valid UTF-8"
                ),
            })?;
            if !process.argv.iter().any(|argument| argument == resolved)
                && !process.env.values().any(|value| value == resolved)
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {resolved_path:?} for \
                         process {process_id:?} without an exact argv or environment reference"
                    ),
                });
            }
            Ok(LaunchFilePlan {
                relative_path: declaration.relative_path.clone(),
                resolved_path,
                text: declaration.text.clone(),
                sha256: declaration.sha256.clone(),
            })
        })
        .collect()
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn split_lowering<T>(lowering: AdapterLowering<T>) -> (T, LoweringEvidence) {
    (
        lowering.output,
        LoweringEvidence::new(
            lowering.request_sha256,
            lowering.response_sha256,
            lowering.timing,
        ),
    )
}

pub(super) fn plan_integration<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    selection: &WorkflowSelection<'_>,
    effective: &EffectiveServerInput,
    adapter: &C,
) -> Result<PlannedServeStage, InferlabError> {
    let stack = selection.stack;
    let served_name = selection.model.served_name.clone();
    let lowering = adapter.plan_serve(
        &workspace.root,
        &stack.integration,
        &stack.pixi_environment,
        PlanServeInput {
            model: ServeModelInput {
                id: selection.server.model.clone(),
                served_name,
            },
            topology: effective.topology,
            gateway_backend: effective.gateway_backend.clone(),
            pd_router_backend: effective.pd_router_backend.clone(),
            kv_transfer: effective.kv_transfer,
            roles: effective.role_inputs.clone(),
            profiling: effective.profiling,
        },
    )?;
    let (planned, evidence) = split_lowering(lowering);
    validate_integration_identity(&stack.integration, &planned.integration.framework)?;
    validate_serve_graph(
        &stack.integration,
        effective.topology,
        &effective.role_inputs,
        effective.gateway_backend.as_deref(),
        effective.pd_router_backend.as_deref(),
        effective.kv_transfer,
        &planned,
    )?;
    for resolution in &effective.role_resolutions {
        let role = planned
            .roles
            .iter()
            .find(|role| role.id == resolution.input.id)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} omitted Engine role {:?}",
                    stack.integration, resolution.input.id
                ),
            })?;
        validate_effective_parallelism(
            &stack.integration,
            &format!("role {:?}", resolution.input.id),
            &resolution.input.parallelism,
            &role.effective_parallelism,
        )?;
    }
    validate_capture_targets(
        &stack.integration,
        effective.profiling,
        planned.gateway.is_some(),
        &planned.replicas,
    )?;

    let role_render_inputs = planned
        .roles
        .iter()
        .map(|role| {
            load_render_inputs(&workspace.root, &stack.integration, &role.render_inputs)
                .map(|inputs| (role.id.clone(), inputs))
        })
        .collect::<Result<BTreeMap<_, _>, InferlabError>>()?;
    let mut requirements = expand_replica_requirements(
        &stack.integration,
        &planned,
        selection.placement,
        selection.server,
        &role_render_inputs,
    )?;
    let mut integration_rendered_process_ids = requirements
        .iter()
        .map(|requirement| requirement.id().to_owned())
        .collect::<BTreeSet<_>>();

    let public_process = if let Some(gateway) = &planned.gateway {
        let fixed_gateway = if uses_explicit_replica_placement(selection.placement) {
            let rank = selection
                .placement
                .roles
                .get("gateway")
                .and_then(|role| role.ranks_for_replica(0))
                .and_then(|ranks| ranks.first())
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: "explicit routed placement must bind Gateway process".to_owned(),
                })?;
            Some(FixedDeviceAssignment::new(
                rank.machine.clone(),
                Vec::new(),
                rank.endpoint_port,
            ))
        } else {
            None
        };
        let mut ports = gateway.ports.clone();
        if let Some(pd_router) = &planned.pd_router {
            for port in &pd_router.ports {
                if !ports.contains(port) {
                    ports.push(port.clone());
                }
            }
        }
        let mut render_inputs =
            load_render_inputs(&workspace.root, &stack.integration, &gateway.render_inputs)?;
        if let Some(pd_router) = &planned.pd_router {
            for supplied in load_render_inputs(
                &workspace.root,
                &stack.integration,
                &pd_router.render_inputs,
            )? {
                if !render_inputs
                    .iter()
                    .any(|current| current.source_path == supplied.source_path)
                {
                    render_inputs.push(supplied);
                }
            }
        }
        let components = if planned.pd_router.is_some() {
            FrontendComponents::gateway_pd_router()
        } else {
            FrontendComponents::gateway()
        };
        let readiness = planned.pd_router.as_ref().map_or_else(
            || gateway.readiness.clone(),
            |router| router.readiness.clone(),
        );
        let dependencies = requirements
            .iter()
            .map(|requirement| requirement.id().to_owned())
            .collect();
        let mut links = links_for_node(&planned.links, "gateway");
        for link in links_for_node(&planned.links, "pd_router") {
            if !links.contains(&link) {
                links.push(link);
            }
        }
        requirements.push(ProcessRequirement::new(
            "gateway".to_owned(),
            ProcessRequirementIdentity::Frontend {
                process_role: FrontendProcessRole::Gateway,
                components,
                gateway: Box::new(gateway.clone()),
                pd_router: planned.pd_router.clone().map(Box::new),
                links,
                render_inputs,
            },
            0,
            ports,
            readiness,
            dependencies,
            None,
            fixed_gateway,
        ));
        if gateway.render_source == RenderSource::Integration {
            integration_rendered_process_ids.insert("gateway".to_owned());
        }
        "gateway".to_owned()
    } else {
        let role = planned
            .roles
            .iter()
            .find(|role| role.public_endpoint.is_some())
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} did not select a direct public Engine",
                    stack.integration
                ),
            })?;
        requirements
            .iter()
            .find(|requirement| {
                matches!(
                    requirement.identity(),
                    ProcessRequirementIdentity::ModelRank {
                        role_id,
                        replica_index: 0,
                        rank: 0,
                        ..
                    } if role_id == &role.id
                )
            })
            .map(|requirement| requirement.id().to_owned())
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} selected unknown direct public Engine role {:?}",
                    stack.integration, role.id
                ),
            })?
    };
    validate_launch_dependencies(&stack.integration, &requirements)?;
    Ok(PlannedServeStage::new(
        planned,
        evidence,
        requirements,
        integration_rendered_process_ids,
        public_process,
    ))
}

pub(super) fn rendered_process_id(process: &RenderedServeProcess) -> &str {
    match process {
        RenderedServeProcess::ModelRank { process, .. }
        | RenderedServeProcess::Frontend { process, .. } => process,
    }
}

pub(super) fn render_integration<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    selection: &WorkflowSelection<'_>,
    effective: &EffectiveServerInput,
    planned_stage: &PlannedServeStage,
    adapter: &C,
) -> Result<RenderedServeStage, InferlabError> {
    let stack = selection.stack;
    let planned = planned_stage.planned();
    let control_plane_frontend = planned
        .gateway
        .as_ref()
        .is_some_and(|gateway| gateway.render_source == RenderSource::ControlPlane);
    let allocations = allocate_processes(
        workspace,
        &selection.server_id,
        &selection.placement_id,
        selection.placement,
        selection.weight,
        &stack.pixi_environment,
        request
            .image
            .map(|image| image.image_id.as_str())
            .or_else(|| request.external.map(|external| external.digest.as_str())),
        planned_stage.requirements(),
        control_plane_frontend.then_some(planned_stage.public_process()),
    )?;
    gate_engine_trace_placement(
        workspace,
        request,
        planned_stage.requirements(),
        &allocations,
    )?;
    let render_allocations = allocations
        .iter()
        .filter(|allocation| {
            planned_stage
                .integration_rendered_process_ids()
                .contains(allocation.process())
        })
        .map(|allocation| allocation.wire().clone())
        .collect::<Vec<_>>();
    let lowering = adapter.render_serve(
        &workspace.root,
        &stack.integration,
        &stack.pixi_environment,
        RenderServeInput {
            model: ServeModelInput {
                id: selection.server.model.clone(),
                served_name: selection.model.served_name.clone(),
            },
            topology: effective.topology,
            gateway_backend: effective.gateway_backend.clone(),
            pd_router_backend: effective.pd_router_backend.clone(),
            kv_transfer: effective.kv_transfer,
            allocations: render_allocations,
            profiling: effective.profiling,
        },
    )?;
    let (rendered, evidence) = split_lowering(lowering);
    if rendered.integration != planned.integration {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {:?} changed identity between serve planning and rendering",
                stack.integration
            ),
        });
    }
    if rendered.processes.len() != planned_stage.integration_rendered_process_ids().len() {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {:?} rendered {} processes for {} integration-rendered allocations",
                stack.integration,
                rendered.processes.len(),
                planned_stage.integration_rendered_process_ids().len()
            ),
        });
    }
    let mut rendered_by_id = BTreeMap::new();
    for process in rendered.processes {
        let id = rendered_process_id(&process).to_owned();
        if !planned_stage
            .integration_rendered_process_ids()
            .contains(&id)
            || rendered_by_id.insert(id.clone(), process).is_some()
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} returned duplicate or unknown process {id:?}",
                    stack.integration
                ),
            });
        }
    }
    if control_plane_frontend {
        let gateway = planned
            .gateway
            .as_ref()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: "control-plane frontend lost its Gateway plan".to_owned(),
            })?;
        let pd_router = planned
            .pd_router
            .as_ref()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: "control-plane frontend requires a P/D Router plan".to_owned(),
            })?;
        let prefill_data_parallel_size = planned
            .roles
            .iter()
            .find(|role| role.id == pd_router.prefill_role)
            .and_then(|role| role.effective_parallelism.attention.as_ref())
            .and_then(|attention| attention.data_parallel_size)
            .unwrap_or(1);
        let process = render_builtin_frontend(
            gateway,
            pd_router,
            &planned.integration.framework,
            effective.kv_transfer,
            prefill_data_parallel_size,
            &allocations,
        )?;
        let id = rendered_process_id(&process).to_owned();
        if rendered_by_id.insert(id.clone(), process).is_some() {
            return Err(InferlabError::InvalidConfig {
                message: format!("frontend renderer duplicated process {id:?}"),
            });
        }
    }
    let rendered_processes = planned_stage
        .requirements()
        .iter()
        .map(|requirement| {
            rendered_by_id
                .remove(requirement.id())
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "resolved topology has no rendered process for allocation {:?}",
                        requirement.id()
                    ),
                })
        })
        .collect::<Result<Vec<_>, InferlabError>>()?;
    if !rendered_by_id.is_empty() {
        return Err(InferlabError::InvalidConfig {
            message: "rendering returned processes outside the resolved topology".to_owned(),
        });
    }
    Ok(RenderedServeStage::new(
        evidence,
        allocations,
        rendered_processes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
