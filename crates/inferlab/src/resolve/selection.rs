use super::{ExecutionTarget, ResolveRequest};
use crate::InferlabError;
use crate::execution::{
    CaseSelectionSource, CommonDeclarationPlan, DeclarationSource, PlacementSelectionSource,
    RecipePlan, RoleDeclarationPlan, ServerDeclarationPlan, Workflow,
};
use crate::toml_override::{ExactTomlOverride, InvocationOverride, apply_toml_patch};
use crate::workspace::{
    DEFAULT_CAPTURE_ARM_DEADLINE_SECONDS, DEFAULT_CAPTURE_CONTROL_DEADLINE_SECONDS,
    DEFAULT_CAPTURE_FINALIZATION_DEADLINE_SECONDS, DEFAULT_READINESS_ATTEMPT_TIMEOUT_SECONDS,
    JsonValue, LoadedWorkspace, ModelDefinition, ModelWeightBinding, PlacementBinding,
    RecipeDefinition, ServerCaseDefinition, ServerDefinition, StackDefinition,
    WorkloadSuiteDefinition,
};
use inferlab_protocol::{
    KvTransferMechanism, Parallelism, ServeRoleInput, ServeRoleKind, ServeTopology, SettingValue,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct ServerOverridePatch {
    pub(super) topology: Option<ServeTopology>,
    pub(super) readiness_timeout_seconds: Option<u64>,
    pub(super) readiness_attempt_timeout_seconds: Option<u64>,
    pub(super) capture_arm_deadline_seconds: Option<u64>,
    pub(super) capture_control_deadline_seconds: Option<u64>,
    pub(super) capture_finalization_deadline_seconds: Option<u64>,
    pub(super) gateway_backend: Option<String>,
    pub(super) pd_router_backend: Option<String>,
    pub(super) kv_transfer: Option<KvTransferMechanism>,
    pub(super) profiling: Option<bool>,
    pub(super) parallelism: Parallelism,
    pub(super) roles: BTreeMap<String, ServerRoleOverridePatch>,
    pub(super) settings: BTreeMap<String, JsonValue>,
}

pub(super) struct IndexedServerOverride {
    pub(super) invocation: InvocationOverride,
    pub(super) assignment: ExactTomlOverride,
    pub(super) patch: ServerOverridePatch,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct ServerRoleOverridePatch {
    pub(super) replicas: Option<u32>,
    pub(super) parallelism: Parallelism,
    pub(super) settings: BTreeMap<String, JsonValue>,
}

pub(super) struct ResolvedRoleInput {
    pub(super) input: ServeRoleInput,
}

/// Selected public definitions and local bindings. `LoadedWorkspace` already
/// owns loading, semantic validation, and source identity; this stage only
/// selects the exact workflow inputs and never reconstructs those facts.
pub(super) struct WorkflowSelection<'a> {
    pub(super) server_id: String,
    pub(super) recipe: Option<RecipePlan>,
    pub(super) server: &'a ServerDefinition,
    pub(super) model: &'a ModelDefinition,
    pub(super) stack: &'a StackDefinition,
    pub(super) stack_checks: Vec<crate::environment::PlannedEnvironmentCheck>,
    pub(super) suite: Option<&'a WorkloadSuiteDefinition>,
    pub(super) case_id: Option<String>,
    pub(super) case: Option<&'a ServerCaseDefinition>,
    pub(super) case_selection: Option<CaseSelectionSource>,
    pub(super) weight: &'a ModelWeightBinding,
    pub(super) placement_id: String,
    pub(super) placement_selection: PlacementSelectionSource,
    pub(super) placement: &'a PlacementBinding,
}

/// Effective server input after case and invocation precedence, before the
/// integration is allowed to plan framework-specific roles and processes.
pub(super) struct EffectiveServerInput {
    pub(super) topology: ServeTopology,
    pub(super) readiness_timeout_seconds: u64,
    pub(super) readiness_attempt_timeout_seconds: u64,
    pub(super) gateway_backend: Option<String>,
    pub(super) pd_router_backend: Option<String>,
    pub(super) kv_transfer: Option<KvTransferMechanism>,
    pub(super) profiling: bool,
    pub(super) capture_arm_deadline_seconds: u64,
    pub(super) capture_control_deadline_seconds: u64,
    pub(super) capture_finalization_deadline_seconds: u64,
    pub(super) override_patches: Vec<IndexedServerOverride>,
    pub(super) role_resolutions: Vec<ResolvedRoleInput>,
    pub(super) declarations: Vec<ServerDeclarationPlan>,
    pub(super) role_inputs: Vec<ServeRoleInput>,
}

fn role_declarations(
    server: &ServerDefinition,
    topology: ServeTopology,
) -> Result<Vec<(String, ServeRoleKind)>, InferlabError> {
    let required = match topology {
        ServeTopology::Single => [ServeRoleKind::Serve].as_slice(),
        ServeTopology::PrefillDecode => [ServeRoleKind::Prefill, ServeRoleKind::Decode].as_slice(),
    };
    let declarations = required
        .iter()
        .map(|kind| (kind_name(*kind).to_owned(), *kind))
        .collect::<Vec<_>>();
    for role in server.roles.keys() {
        if !declarations.iter().any(|(id, _)| id == role) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "server role {role:?} is not valid for topology {topology:?}; roles are canonical"
                ),
            });
        }
    }
    Ok(declarations)
}

const fn kind_name(kind: ServeRoleKind) -> &'static str {
    match kind {
        ServeRoleKind::Serve => "serve",
        ServeRoleKind::Prefill => "prefill",
        ServeRoleKind::Decode => "decode",
    }
}

fn resolve_role_inputs(
    server: &ServerDefinition,
    topology: ServeTopology,
) -> Result<Vec<ResolvedRoleInput>, InferlabError> {
    let declarations = role_declarations(server, topology)?;
    declarations
        .into_iter()
        .map(|(id, kind)| {
            let role = server.roles.get(&id);
            let mut parallelism = server.parallelism.clone();
            if let Some(role) = role {
                parallelism.merge_from(&role.parallelism);
            }
            let settings = effective_role_settings(
                &format!("effective role {id:?}"),
                &server.settings,
                role.map(|role| &role.settings),
            )?;
            let replica_count = role.and_then(|role| role.replicas).unwrap_or(1);
            if replica_count == 0 {
                return Err(InferlabError::InvalidConfig {
                    message: format!("role {id:?} replica count must be nonzero"),
                });
            }
            Ok(ResolvedRoleInput {
                input: ServeRoleInput {
                    id,
                    kind,
                    replica_count,
                    parallelism,
                    settings,
                },
            })
        })
        .collect()
}

fn server_declarations(
    server_id: &str,
    server: &ServerDefinition,
    case_id: Option<&str>,
    case: Option<&ServerCaseDefinition>,
    overrides: &[IndexedServerOverride],
) -> Result<Vec<ServerDeclarationPlan>, InferlabError> {
    let mut declarations = vec![ServerDeclarationPlan {
        source: DeclarationSource::Server {
            id: server_id.to_owned(),
        },
        common: CommonDeclarationPlan {
            readiness_timeout_seconds: Some(server.readiness_timeout_seconds),
            readiness_attempt_timeout_seconds: server.readiness_attempt_timeout_seconds,
            capture_arm_deadline_seconds: server.capture_arm_deadline_seconds,
            gateway_backend: server.gateway_backend.clone(),
            pd_router_backend: server.pd_router_backend.clone(),
            kv_transfer: server.kv_transfer,
            profiling: server.profiling,
            capture_control_deadline_seconds: server.capture_control_deadline_seconds,
            capture_finalization_deadline_seconds: server.capture_finalization_deadline_seconds,
            parallelism: server.parallelism.clone(),
            settings: declaration_settings("server common", &server.settings)?,
        },
        roles: server
            .roles
            .iter()
            .map(|(id, role)| {
                Ok((
                    id.clone(),
                    RoleDeclarationPlan {
                        replicas: role.replicas,
                        parallelism: role.parallelism.clone(),
                        settings: declaration_settings(
                            &format!("server role {id:?}"),
                            &role.settings,
                        )?,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, InferlabError>>()?,
    }];

    if let Some(case) = case {
        let case_id = case_id.ok_or_else(|| InferlabError::InvalidConfig {
            message: "selected server case has no identity".to_owned(),
        })?;
        declarations.push(ServerDeclarationPlan {
            source: DeclarationSource::Case {
                id: case_id.to_owned(),
            },
            common: CommonDeclarationPlan {
                readiness_timeout_seconds: case.readiness_timeout_seconds,
                readiness_attempt_timeout_seconds: case.readiness_attempt_timeout_seconds,
                capture_arm_deadline_seconds: case.capture_arm_deadline_seconds,
                gateway_backend: case.gateway_backend.clone(),
                pd_router_backend: case.pd_router_backend.clone(),
                kv_transfer: case.kv_transfer,
                profiling: case.profiling,
                capture_control_deadline_seconds: case.capture_control_deadline_seconds,
                capture_finalization_deadline_seconds: case.capture_finalization_deadline_seconds,
                parallelism: case.parallelism.clone(),
                settings: declaration_settings("case common", &case.settings)?,
            },
            roles: case
                .roles
                .iter()
                .map(|(id, role)| {
                    Ok((
                        id.clone(),
                        RoleDeclarationPlan {
                            replicas: role.replicas,
                            parallelism: role.parallelism.clone(),
                            settings: declaration_settings(
                                &format!("case role {id:?}"),
                                &role.settings,
                            )?,
                        },
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, InferlabError>>()?,
        });
    }

    for item in overrides {
        let index = item.invocation.index();
        let patch = &item.patch;
        declarations.push(ServerDeclarationPlan {
            source: DeclarationSource::Invocation { index },
            common: CommonDeclarationPlan {
                readiness_timeout_seconds: patch.readiness_timeout_seconds,
                readiness_attempt_timeout_seconds: patch.readiness_attempt_timeout_seconds,
                capture_arm_deadline_seconds: patch.capture_arm_deadline_seconds,
                gateway_backend: patch.gateway_backend.clone(),
                pd_router_backend: patch.pd_router_backend.clone(),
                kv_transfer: patch.kv_transfer,
                profiling: patch.profiling,
                capture_control_deadline_seconds: patch.capture_control_deadline_seconds,
                capture_finalization_deadline_seconds: patch.capture_finalization_deadline_seconds,
                parallelism: patch.parallelism.clone(),
                settings: declaration_settings("invocation common", &patch.settings)?,
            },
            roles: patch
                .roles
                .iter()
                .map(|(id, role)| {
                    Ok((
                        id.clone(),
                        RoleDeclarationPlan {
                            replicas: role.replicas,
                            parallelism: role.parallelism.clone(),
                            settings: declaration_settings(
                                &format!("invocation role {id:?}"),
                                &role.settings,
                            )?,
                        },
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, InferlabError>>()?,
        });
    }
    Ok(declarations)
}

fn declaration_settings(
    scope: &str,
    settings: &BTreeMap<String, JsonValue>,
) -> Result<BTreeMap<String, SettingValue>, InferlabError> {
    crate::adapter::project_setting_values(&format!("{scope} settings"), settings)
}

fn effective_role_settings(
    scope: &str,
    common: &BTreeMap<String, JsonValue>,
    role: Option<&BTreeMap<String, JsonValue>>,
) -> Result<BTreeMap<String, SettingValue>, InferlabError> {
    let mut settings =
        toml::Value::try_from(common).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare {scope} settings: {error}"),
        })?;
    if let Some(role) = role {
        let patch = toml::Value::try_from(role).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare {scope} role settings: {error}"),
        })?;
        apply_toml_patch(&mut settings, patch).map_err(|message| InferlabError::InvalidConfig {
            message: format!("invalid {scope} settings composition: {message}"),
        })?;
    }
    let settings: BTreeMap<String, JsonValue> =
        settings
            .try_into()
            .map_err(|error| InferlabError::InvalidConfig {
                message: format!("failed to resolve {scope} settings: {error}"),
            })?;
    declaration_settings(scope, &settings)
}

pub(super) fn select_workflow<'a>(
    workspace: &'a LoadedWorkspace,
    request: &ResolveRequest<'_>,
) -> Result<WorkflowSelection<'a>, InferlabError> {
    let (server_id, recipe_definition): (&str, Option<(&str, &RecipeDefinition)>) =
        match request.target {
            ExecutionTarget::Server(server) if matches!(request.workflow, Workflow::ServeStart) => {
                (server, None)
            }
            ExecutionTarget::Recipe(recipe) if matches!(request.workflow, Workflow::RecipeRun) => {
                let definition = lookup("recipe", recipe, &workspace.config.recipes)?;
                (definition.server.as_str(), Some((recipe, definition)))
            }
            ExecutionTarget::Server(_) => {
                return Err(InferlabError::InvalidConfig {
                    message: "recipe run requires a recipe target".to_owned(),
                });
            }
            ExecutionTarget::Recipe(_) => {
                return Err(InferlabError::InvalidConfig {
                    message: "serve start requires a server target".to_owned(),
                });
            }
        };
    let server = lookup("server", server_id, &workspace.config.servers)?;
    let model = lookup("model", &server.model, &workspace.config.models)?;
    let stack = lookup("stack", &server.stack, &workspace.config.stacks)?;
    let (stack_checks, _image_postprocess) =
        crate::environment::plan_environment_checks(&workspace.root, stack)?;
    let suite = recipe_definition
        .map(|(_, recipe)| {
            lookup(
                "workload suite",
                &recipe.workload_suite,
                &workspace.config.workload_suites,
            )
        })
        .transpose()?;
    let (case_id, case, case_selection) = match request.case {
        Some(selected) => (
            Some(selected.to_owned()),
            Some(
                server
                    .cases
                    .get(selected)
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!("unknown case {selected:?} for server {server_id:?}"),
                    })?,
            ),
            Some(CaseSelectionSource::Explicit),
        ),
        None => {
            if let Some(selected) = server.default_case.as_deref() {
                (
                    Some(selected.to_owned()),
                    Some(&server.cases[selected]),
                    Some(CaseSelectionSource::Default),
                )
            } else {
                match (server.cases.iter().next(), server.cases.iter().nth(1)) {
                    (None, _) => (None, None, None),
                    (Some((id, definition)), None) => (
                        Some(id.clone()),
                        Some(definition),
                        Some(CaseSelectionSource::Sole),
                    ),
                    (Some(_), Some(_)) => {
                        return Err(InferlabError::InvalidConfig {
                            message: format!(
                                "server {server_id:?} declares multiple cases {:?}; select one with --case or set default_case",
                                server.cases.keys().collect::<Vec<_>>()
                            ),
                        });
                    }
                }
            }
        }
    };
    let weight = workspace
        .local
        .model_weights
        .get(&server.model)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("missing model weight binding {:?}", server.model),
        })?;
    let (placement_id, placement_selection) = if let Some(selected) = request.placement {
        (selected, PlacementSelectionSource::Explicit)
    } else if let Some(selected) = workspace.local.default_placement.as_deref() {
        (selected, PlacementSelectionSource::Default)
    } else {
        match (
            workspace.local.placements.keys().next(),
            workspace.local.placements.keys().nth(1),
        ) {
            (Some(only), None) => (only.as_str(), PlacementSelectionSource::Sole),
            (None, _) => {
                return Err(InferlabError::InvalidConfig {
                    message: "no local placement is declared".to_owned(),
                });
            }
            (Some(_), Some(_)) => {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "local bindings declare multiple placements {:?}; select one with --placement or set default_placement",
                        workspace.local.placements.keys().collect::<Vec<_>>()
                    ),
                });
            }
        }
    };
    let placement = workspace
        .local
        .placements
        .get(placement_id)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("unknown placement {placement_id:?}"),
        })?;
    Ok(WorkflowSelection {
        server_id: server_id.to_owned(),
        recipe: recipe_definition.map(|(id, recipe)| RecipePlan {
            id: id.to_owned(),
            workload_suite: recipe.workload_suite.clone(),
        }),
        server,
        model,
        stack,
        stack_checks,
        suite,
        case_id,
        case,
        case_selection,
        weight,
        placement_id: placement_id.to_owned(),
        placement_selection,
        placement,
    })
}

pub(super) fn resolve_effective_server_input(
    selection: &WorkflowSelection<'_>,
    request: &ResolveRequest<'_>,
    overrides: &[InvocationOverride],
) -> Result<EffectiveServerInput, InferlabError> {
    let server = selection.server;
    let case = selection.case;
    let topology = server.topology;
    let mut override_patches = Vec::new();
    let selected_roles = role_declarations(server, topology)?
        .into_iter()
        .map(|(id, _)| id)
        .collect::<BTreeSet<_>>();
    for item in overrides {
        if !item.path().starts_with("server.")
            && matches!(request.workflow, Workflow::RecipeRun)
            && (item.path().starts_with("evals.") || item.path().starts_with("benches."))
        {
            continue;
        }
        let invocation = item
            .under("server.")
            .ok_or_else(|| InferlabError::InvalidOverride {
                value: item.raw().to_owned(),
                message: "only paths under server. may be overridden".to_owned(),
            })?;
        let assignment = invocation.assignment()?;
        let patch = parse_server_override(&invocation, &assignment)?;
        if patch.topology.is_some() {
            return Err(InferlabError::InvalidConfig {
                message: "invocation overrides must not change server topology".to_owned(),
            });
        }
        if patch.gateway_backend.is_some() && server.gateway_backend.is_none() {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "cannot add gateway_backend because server {:?} does not declare a Gateway",
                    selection.server_id
                ),
            });
        }
        if patch.pd_router_backend.is_some() && server.pd_router_backend.is_none() {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "cannot add pd_router_backend because server {:?} does not declare a P/D Router",
                    selection.server_id
                ),
            });
        }
        if patch.readiness_timeout_seconds == Some(0) {
            return Err(InferlabError::InvalidConfig {
                message: "readiness_timeout_seconds must be nonzero".to_owned(),
            });
        }
        if patch.readiness_attempt_timeout_seconds == Some(0) {
            return Err(InferlabError::InvalidConfig {
                message: "readiness_attempt_timeout_seconds must be nonzero".to_owned(),
            });
        }
        for (name, value) in [
            (
                "capture_arm_deadline_seconds",
                patch.capture_arm_deadline_seconds,
            ),
            (
                "capture_control_deadline_seconds",
                patch.capture_control_deadline_seconds,
            ),
            (
                "capture_finalization_deadline_seconds",
                patch.capture_finalization_deadline_seconds,
            ),
        ] {
            if value == Some(0) {
                return Err(InferlabError::InvalidConfig {
                    message: format!("{name} must be nonzero"),
                });
            }
        }
        for id in patch.roles.keys() {
            if !selected_roles.contains(id) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "invocation configures role {id:?}, which is not part of the selected topology"
                    ),
                });
            }
        }
        override_patches.push(IndexedServerOverride {
            invocation,
            assignment,
            patch,
        });
    }
    let effective_server = compose_server_definition(server, case, &override_patches)?;
    let readiness_timeout_seconds = effective_server.readiness_timeout_seconds;
    let readiness_attempt_timeout_seconds = effective_server
        .readiness_attempt_timeout_seconds
        .unwrap_or(DEFAULT_READINESS_ATTEMPT_TIMEOUT_SECONDS);
    let capture_arm_deadline_seconds = effective_server
        .capture_arm_deadline_seconds
        .unwrap_or(DEFAULT_CAPTURE_ARM_DEADLINE_SECONDS);
    let capture_control_deadline_seconds = effective_server
        .capture_control_deadline_seconds
        .unwrap_or(DEFAULT_CAPTURE_CONTROL_DEADLINE_SECONDS);
    let capture_finalization_deadline_seconds = effective_server
        .capture_finalization_deadline_seconds
        .unwrap_or(DEFAULT_CAPTURE_FINALIZATION_DEADLINE_SECONDS);
    let gateway_backend = effective_server.gateway_backend.clone();
    let pd_router_backend = effective_server.pd_router_backend.clone();
    let kv_transfer = effective_server.kv_transfer;
    let mut profiling = effective_server.profiling.unwrap_or(false);
    if !request.captures.is_empty() {
        profiling = true;
    }
    match topology {
        ServeTopology::Single if pd_router_backend.is_some() || kv_transfer.is_some() => {
            return Err(InferlabError::InvalidConfig {
                message: "single topology does not accept pd_router_backend or kv_transfer"
                    .to_owned(),
            });
        }
        ServeTopology::PrefillDecode
            if gateway_backend.is_none() || pd_router_backend.is_none() =>
        {
            return Err(InferlabError::InvalidConfig {
                message:
                    "prefill_decode topology requires both gateway_backend and pd_router_backend"
                        .to_owned(),
            });
        }
        ServeTopology::Single | ServeTopology::PrefillDecode => {}
    }

    let case_id = selection.case_id.as_deref();
    let role_resolutions = resolve_role_inputs(&effective_server, topology)?;
    let declarations = server_declarations(
        &selection.server_id,
        server,
        case_id,
        case,
        &override_patches,
    )?;
    let role_inputs = role_resolutions
        .iter()
        .map(|role| role.input.clone())
        .collect();
    Ok(EffectiveServerInput {
        topology,
        readiness_timeout_seconds,
        readiness_attempt_timeout_seconds,
        gateway_backend,
        pd_router_backend,
        kv_transfer,
        profiling,
        capture_arm_deadline_seconds,
        capture_control_deadline_seconds,
        capture_finalization_deadline_seconds,
        override_patches,
        role_resolutions,
        declarations,
        role_inputs,
    })
}

fn compose_server_definition(
    server: &ServerDefinition,
    case: Option<&ServerCaseDefinition>,
    overrides: &[IndexedServerOverride],
) -> Result<ServerDefinition, InferlabError> {
    let mut definition =
        toml::Value::try_from(server).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare the selected server definition: {error}"),
        })?;
    if let Some(case) = case {
        let patch = toml::Value::try_from(case).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare the selected server case: {error}"),
        })?;
        apply_toml_patch(&mut definition, patch).map_err(|message| {
            InferlabError::InvalidConfig {
                message: format!("invalid selected server case composition: {message}"),
            }
        })?;
    }
    for item in overrides {
        item.assignment
            .clone()
            .apply_to(&mut definition, item.invocation.raw())?;
    }
    definition
        .try_into()
        .map_err(|error| InferlabError::InvalidConfig {
            message: format!("invalid effective server definition: {error}"),
        })
}

pub(super) fn validate_effective_parallelism(
    integration: &str,
    scope: &str,
    declared: &Parallelism,
    effective: &Parallelism,
) -> Result<(), InferlabError> {
    if let Some((field, value)) = parallelism_values(effective)
        .into_iter()
        .find(|(_, value)| !value.is_some_and(|value| value > 0))
    {
        return non_concrete_parallelism(integration, scope, field, value);
    }
    for ((field, declared), (_, effective)) in parallelism_values(declared)
        .into_iter()
        .zip(parallelism_values(effective))
    {
        if declared.is_some() && declared != effective {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} changed explicitly declared {scope} parallelism.{field} from {declared:?} to {effective:?}"
                ),
            });
        }
    }
    Ok(())
}

fn parallelism_values(parallelism: &Parallelism) -> [(&'static str, Option<u32>); 9] {
    [
        (
            "outer.tensor_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|value| value.tensor_parallel_size),
        ),
        (
            "outer.pipeline_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|value| value.pipeline_parallel_size),
        ),
        (
            "attention.tensor_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|value| value.tensor_parallel_size),
        ),
        (
            "attention.data_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|value| value.data_parallel_size),
        ),
        (
            "attention.context_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|value| value.context_parallel_size),
        ),
        (
            "experts.tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|value| value.tensor_parallel_size),
        ),
        (
            "experts.data_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|value| value.data_parallel_size),
        ),
        (
            "experts.expert_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|value| value.expert_parallel_size),
        ),
        (
            "experts.dense_tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|value| value.dense_tensor_parallel_size),
        ),
    ]
}

fn non_concrete_parallelism(
    integration: &str,
    scope: &str,
    field: &str,
    value: Option<u32>,
) -> Result<(), InferlabError> {
    Err(InferlabError::InvalidConfig {
        message: format!(
            "integration {integration:?} returned non-concrete effective {scope} parallelism.{field}={value:?}"
        ),
    })
}

fn parse_server_override(
    override_: &InvocationOverride,
    assignment: &ExactTomlOverride,
) -> Result<ServerOverridePatch, InferlabError> {
    assignment
        .clone()
        .into_patch()
        .try_into()
        .map_err(|error| InferlabError::InvalidOverride {
            value: override_.raw().to_owned(),
            message: format!("invalid server setting: {error}"),
        })
}

pub(super) fn validate_effective_settings(
    requested: &BTreeMap<String, SettingValue>,
    effective: &BTreeMap<String, SettingValue>,
    integration: &str,
) -> Result<(), InferlabError> {
    let requested = flattened_settings(requested);
    let effective = flattened_settings(effective);
    for path in requested.keys() {
        if !effective.contains_key(path) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} omitted effective server setting {path:?}"
                ),
            });
        }
    }
    Ok(())
}

fn flattened_settings(settings: &BTreeMap<String, SettingValue>) -> BTreeMap<String, SettingValue> {
    let mut flattened = BTreeMap::new();
    for (key, value) in settings {
        flatten_setting(&mut flattened, key, value);
    }
    flattened
}

fn flatten_setting(
    flattened: &mut BTreeMap<String, SettingValue>,
    path: &str,
    value: &SettingValue,
) {
    if let SettingValue::Object(values) = value
        && !values.is_empty()
    {
        for (key, value) in values {
            flatten_setting(flattened, &format!("{path}.{key}"), value);
        }
    } else {
        flattened.insert(path.to_owned(), value.clone());
    }
}

fn lookup<'a, T>(
    label: &str,
    id: &str,
    definitions: &'a BTreeMap<String, T>,
) -> Result<&'a T, InferlabError> {
    definitions
        .get(id)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("unknown {label} {id:?}"),
        })
}
