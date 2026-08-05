use crate::workload::MeasurementPlan;
use crate::workspace::WorkspaceSnapshot;
use inferlab_protocol::{
    CaptureTargetRequirement, EndpointRequirement, GatewayPlan, KvTransferMechanism, Parallelism,
    PdRouterPlan, ProtocolVersion, ReadinessProbe, RenderInputDeclaration, ServeRoleKind,
    ServeRoleLink, ServeTopology, SettingValue,
};
pub use inferlab_serve_domain::{
    ActiveRdmaInterfacePlan, AllocationPlan, ContainerPlan, EndpointPlan, ModelLocatorSource,
    NetworkMachinePlan, NetworkPlan, NetworkSelectionReason, ProcessCommandSource,
    ProcessIdentityPlan, ProcessPlan, ProfilerEscapesPlan, RemoteContainerFacts,
    RemoteWorkspacePlan, RuntimeCacheNamespacePlan, RuntimeCachePlan, RuntimeCacheRootSource,
    ServerMetricsEndpointPlan,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Workflow {
    ServeStart,
    RecipeRun,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolvedExecution {
    pub workflow: Workflow,
    pub workspace: WorkspaceSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe: Option<RecipePlan>,
    pub stack: StackPlan,
    pub server: ServerPlan,
    pub measurements: Option<MeasurementPlan>,
}

#[derive(Debug, Serialize)]
pub struct DryRunPlan<'a> {
    pub workflow: Workflow,
    pub dry_run: bool,
    pub workspace: &'a WorkspaceSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe: &'a Option<RecipePlan>,
    pub stack: &'a StackPlan,
    pub server: &'a ServerPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurements: &'a Option<MeasurementPlan>,
}

impl ResolvedExecution {
    pub fn dry_run_plan(&self) -> DryRunPlan<'_> {
        DryRunPlan {
            workflow: self.workflow,
            dry_run: true,
            workspace: &self.workspace,
            recipe: &self.recipe,
            stack: &self.stack,
            server: &self.server,
            measurements: &self.measurements,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RecipePlan {
    pub id: String,
    pub workload_suite: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CasePlan {
    pub id: String,
    pub selection: CaseSelectionSource,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseSelectionSource {
    Explicit,
    Default,
    Sole,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StackPlan {
    pub id: String,
    pub integration: String,
    pub pixi_environment: String,
    pub source_paths: Vec<PathBuf>,
    pub realization: crate::environment::CheckRealization,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<crate::environment::PlannedEnvironmentCheck>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServerPlan {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub case: Option<CasePlan>,
    pub explicit_overrides: Vec<String>,
    pub declarations: Vec<ServerDeclarationPlan>,
    pub topology: ServeTopology,
    pub profiling: bool,
    pub readiness_timeout_seconds: u64,
    pub readiness_attempt_timeout_seconds: u64,
    pub capture_arm_deadline_seconds: u64,
    pub capture_control_deadline_seconds: u64,
    pub capture_finalization_deadline_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer: Option<KvTransferMechanism>,
    /// The closed frontend boundary: logical components, their explicit
    /// process bindings, and every concrete process realizing those
    /// components. A routed topology has this section; a direct Engine does
    /// not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frontend: Option<FrontendPlan>,
    /// The raw profiler escape declaration as written on the server and
    /// its roles ([[RFC-0004:C-WORKLOAD-PROFILING]]); the merged, effective
    /// inputs ride each capture target.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiler_escapes: Option<ProfilerEscapesPlan>,
    pub model: ModelPlan,
    /// The image substitution consuming this launch, when the operator
    /// selected an image build record ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<crate::image::launch::ImageLaunchPlan>,
    /// The external-image substitution consuming this launch: a serving
    /// image this workspace did not build, explicitly not qualified
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_image: Option<crate::image::launch::ExternalImagePlan>,
    pub integration: IntegrationPlan,
    pub resources: ResourcePlan,
    pub placement: PlacementPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkPlan>,
    pub roles: Vec<RolePlan>,
    pub links: Vec<ServeRoleLink>,
    pub endpoint: EndpointPlan,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FrontendPlan {
    pub gateway: GatewayComponentPlan,
    /// Explicit `null` records the absence of a P/D Router for routed-single
    /// serving instead of making consumers infer it from a missing field.
    pub pd_router: Option<PdRouterComponentPlan>,
    /// Concrete frontend processes are owned exactly once here. Components
    /// refer to them by `process_id`, allowing fused components to share one
    /// process and split components to bind independent processes.
    pub processes: Vec<ProcessPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GatewayComponentPlan {
    #[serde(flatten)]
    pub plan: GatewayPlan,
    pub process_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PdRouterComponentPlan {
    #[serde(flatten)]
    pub plan: PdRouterPlan,
    pub process_id: String,
}

/// One ordered behavior declaration consumed while resolving a server.
/// Framework settings remain adapter-owned structured data; each role's
/// declared and effective values remain the execution authorities.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServerDeclarationPlan {
    pub source: DeclarationSource,
    pub common: CommonDeclarationPlan,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub roles: BTreeMap<String, RoleDeclarationPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeclarationSource {
    Server { id: String },
    Case { id: String },
    Invocation { index: usize },
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CommonDeclarationPlan {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readiness_timeout_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readiness_attempt_timeout_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_arm_deadline_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gateway_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pd_router_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer: Option<KvTransferMechanism>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_control_deadline_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_finalization_deadline_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "parallelism_is_empty")]
    pub parallelism: Parallelism,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub settings: BTreeMap<String, SettingValue>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RoleDeclarationPlan {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replicas: Option<u32>,
    #[serde(default, skip_serializing_if = "parallelism_is_empty")]
    pub parallelism: Parallelism,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub settings: BTreeMap<String, SettingValue>,
}

fn parallelism_is_empty(parallelism: &Parallelism) -> bool {
    parallelism == &Parallelism::default()
}

impl ServerPlan {
    pub fn processes(&self) -> impl Iterator<Item = &ProcessPlan> {
        self.roles
            .iter()
            .flat_map(|role| &role.replicas)
            .flat_map(|replica| &replica.ranks)
            .chain(
                self.frontend
                    .iter()
                    .flat_map(|frontend| &frontend.processes),
            )
    }

    pub fn process_count(&self) -> usize {
        self.processes().count()
    }

    pub(crate) fn processes_mut(&mut self) -> impl Iterator<Item = &mut ProcessPlan> {
        let roles = &mut self.roles;
        let frontend = &mut self.frontend;
        roles
            .iter_mut()
            .flat_map(|role| &mut role.replicas)
            .flat_map(|replica| &mut replica.ranks)
            .chain(
                frontend
                    .iter_mut()
                    .flat_map(|frontend| &mut frontend.processes),
            )
    }

    pub fn process_contexts(&self) -> impl Iterator<Item = ProcessContext<'_>> {
        let model_ranks = self.roles.iter().flat_map(|role| {
            role.replicas.iter().flat_map(move |replica| {
                replica.ranks.iter().map(move |process| ProcessContext {
                    role_id: &role.id,
                    replica_id: &replica.id,
                    replica_index: replica.index,
                    process,
                })
            })
        });
        let frontend = self.frontend.iter().flat_map(|frontend| {
            frontend.processes.iter().map(|process| ProcessContext {
                role_id: process.id.as_str(),
                replica_id: process.id.as_str(),
                replica_index: 0,
                process,
            })
        });
        model_ranks.chain(frontend)
    }
}

#[derive(Clone, Copy)]
pub struct ProcessContext<'a> {
    pub role_id: &'a str,
    pub replica_id: &'a str,
    pub replica_index: u32,
    pub process: &'a ProcessPlan,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RolePlan {
    pub id: String,
    pub kind: ServeRoleKind,
    pub declared_replica_count: u32,
    pub effective_replica_count: u32,
    pub declared_parallelism: Parallelism,
    pub effective_parallelism: Parallelism,
    pub declared_settings: BTreeMap<String, SettingValue>,
    pub effective_settings: BTreeMap<String, SettingValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_endpoint: Option<EndpointRequirement>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub render_inputs: Vec<RenderInputDeclaration>,
    pub replicas: Vec<RoleReplicaPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RoleReplicaPlan {
    pub id: String,
    pub index: u32,
    pub device_count: u32,
    pub ports: Vec<String>,
    pub primary_ports: Vec<String>,
    pub primary_readiness: ReadinessProbe,
    pub worker_readiness: ReadinessProbe,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_target: Option<CaptureTargetRequirement>,
    pub entry_process: String,
    pub ranks: Vec<ProcessPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelPlan {
    pub id: String,
    pub served_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IntegrationPlan {
    pub id: String,
    pub adapter_id: String,
    pub adapter_version: String,
    pub framework: String,
    pub framework_version: String,
    pub executable: String,
    pub protocol_version: ProtocolVersion,
    pub plan_request_sha256: String,
    pub plan_response_sha256: String,
    pub render_request_sha256: String,
    pub render_response_sha256: String,
    #[serde(skip)]
    pub plan_timing: Option<inferlab_runtime::operation_bound::OperationTimingEvidence>,
    #[serde(skip)]
    pub render_timing: Option<inferlab_runtime::operation_bound::OperationTimingEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResourcePlan {
    pub device_count: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlacementPlan {
    pub id: String,
    pub selection: PlacementSelectionSource,
    pub machines: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub remote_workspaces: BTreeMap<String, RemoteWorkspacePlan>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub remote_containers: BTreeMap<String, RemoteContainerFacts>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementSelectionSource {
    Explicit,
    Default,
    Sole,
}
