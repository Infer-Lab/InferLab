use crate::InferlabError;
use crate::adapter::{AdapterClient, AdapterLowering, executable_name};
use crate::toml_override::ExactTomlOverride;
use crate::workload::{MeasurementPlan, MeasurementResolveContext, resolve_measurements};
use crate::workspace::{
    DEFAULT_CAPTURE_CONTROL_DEADLINE_SECONDS, LaunchBinding, LoadedWorkspace, ModelDefinition,
    ModelWeightBinding, NsysEscapes, PlacementBinding, PlacementRoleBinding, RecipeDefinition,
    ServerCaseDefinition, ServerDefinition, StackDefinition, WorkloadSuiteDefinition,
    WorkspaceSnapshot,
};
use inferlab_protocol::{
    AllocationLaunch, CaptureTargetRequirement, CaptureWindowControlEndpoint, EndpointAssignment,
    EndpointProtocol, EndpointRequirement, FrontendComponents, FrontendProcessRole, GatewayPlan,
    GatewayTarget, KvTransferMechanism, LaunchFileDeclaration, Parallelism, PdRouterPlan,
    PlanServeInput, PlanServeResult, ProcessSpec, ProtocolVersion, ReadinessProbe,
    RenderInputDeclaration, RenderServeInput, RenderSource, RenderedServeProcess, ServeModelInput,
    ServeProcessAllocation, ServeReplicaRequirement, ServeRoleInput, ServeRoleKind, ServeRoleLink,
    ServeTopology, SettingValue, SuppliedRenderInput, TargetEndpointScheme,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Workflow {
    ServeStart,
    RecipeRun,
}

#[derive(Clone, Copy, Debug)]
pub enum ExecutionTarget<'a> {
    Server(&'a str),
    Recipe(&'a str),
}

pub struct ResolveRequest<'a> {
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
    pub capture_control_deadline_seconds: u64,
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
    pub gateway_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pd_router_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer: Option<KvTransferMechanism>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_control_deadline_seconds: Option<u64>,
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

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ServerOverridePatch {
    topology: Option<ServeTopology>,
    readiness_timeout_seconds: Option<u64>,
    gateway_backend: Option<String>,
    pd_router_backend: Option<String>,
    kv_transfer: Option<KvTransferMechanism>,
    profiling: Option<bool>,
    parallelism: Parallelism,
    roles: BTreeMap<String, ServerRoleOverridePatch>,
    settings: BTreeMap<String, toml::Value>,
}

struct IndexedServerOverride {
    index: usize,
    raw: String,
    patch: ServerOverridePatch,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ServerRoleOverridePatch {
    replicas: Option<u32>,
    parallelism: Parallelism,
    settings: BTreeMap<String, toml::Value>,
}

struct ResolvedRoleInput {
    input: ServeRoleInput,
}

/// Selected public definitions and local bindings. `LoadedWorkspace` already
/// owns loading, semantic validation, and source identity; this stage only
/// selects the exact workflow inputs and never reconstructs those facts.
struct WorkflowSelection<'a> {
    server_id: String,
    recipe: Option<RecipePlan>,
    server: &'a ServerDefinition,
    model: &'a ModelDefinition,
    stack: &'a StackDefinition,
    stack_checks: Vec<crate::environment::PlannedEnvironmentCheck>,
    suite: Option<&'a WorkloadSuiteDefinition>,
    case_id: Option<String>,
    case: Option<&'a ServerCaseDefinition>,
    case_selection: Option<CaseSelectionSource>,
    weight: &'a ModelWeightBinding,
    placement_id: String,
    placement_selection: PlacementSelectionSource,
    placement: &'a PlacementBinding,
}

/// Effective server input after case and invocation precedence, before the
/// integration is allowed to plan framework-specific roles and processes.
struct EffectiveServerInput {
    topology: ServeTopology,
    readiness_timeout_seconds: u64,
    gateway_backend: Option<String>,
    pd_router_backend: Option<String>,
    kv_transfer: Option<KvTransferMechanism>,
    profiling: bool,
    capture_control_deadline_seconds: u64,
    override_patches: Vec<IndexedServerOverride>,
    role_resolutions: Vec<ResolvedRoleInput>,
    declarations: Vec<ServerDeclarationPlan>,
    role_inputs: Vec<ServeRoleInput>,
}

struct LoweringEvidence {
    request_sha256: String,
    response_sha256: String,
    timing: crate::time_bound::OperationTimingEvidence,
}

struct PlannedServeStage {
    planned: PlanServeResult,
    evidence: LoweringEvidence,
    requirements: Vec<ProcessRequirement>,
    integration_rendered_process_ids: BTreeSet<String>,
    public_process: String,
}

struct RenderedServeStage {
    evidence: LoweringEvidence,
    allocations: Vec<ResolvedProcessAllocation>,
    rendered_processes: Vec<RenderedServeProcess>,
}

struct RuntimeRealizationStage {
    processes: Vec<ProcessPlan>,
    public_endpoint: EndpointPlan,
    device_count: u32,
    selected_machines: Vec<String>,
    network: Option<NetworkPlan>,
    remote_workspaces: BTreeMap<String, RemoteWorkspacePlan>,
    remote_containers: BTreeMap<String, RemoteContainerFacts>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadinessPlan {
    Http {
        path: String,
        /// `None` when the server is capture-armed: the readiness wait is
        /// unbounded per [[RFC-0004:C-WORKLOAD-PROFILING]].
        timeout_seconds: Option<u64>,
    },
    HttpTargetRegistry {
        readiness_path: String,
        registry_path: String,
        targets_field: String,
        target_url_field: String,
        target_role_field: String,
        target_healthy_field: String,
        target_bootstrap_port_field: String,
        expected_targets: Vec<TargetRegistryExpectedTarget>,
        timeout_seconds: Option<u64>,
    },
    ProcessAlive,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TargetRegistryExpectedTarget {
    pub url: String,
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
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
    pub plan_timing: Option<crate::time_bound::OperationTimingEvidence>,
    #[serde(skip)]
    pub render_timing: Option<crate::time_bound::OperationTimingEvidence>,
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

/// Machine-scoped launch facts a remote containerized launch consumed
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): the external image was observed
/// present, and the container user identity comes from that machine's
/// realization rather than controller filesystem metadata. No workspace
/// realization is checked — the image replaces the serving environment.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RemoteContainerFacts {
    pub target: String,
    pub user: String,
    pub uid: u32,
    pub gid: u32,
    /// Declared pass-through names observed set in that machine's launching
    /// environment; a declared name absent from this set was reported.
    pub present_pass_env: BTreeSet<String>,
    /// The machine's own PATH and HOME: remote processes launch under a
    /// clean environment, and the docker client resolves on that machine,
    /// not on the controller.
    pub environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NetworkPlan {
    pub selected_interface: String,
    pub reason: NetworkSelectionReason,
    pub machines: BTreeMap<String, NetworkMachinePlan>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum NetworkSelectionReason {
    #[serde(rename = "common-default-route-rdma-interface")]
    RdmaDefaultRoute,
    #[serde(rename = "common-rdma-interface")]
    Rdma,
    #[serde(rename = "common-default-route-interface")]
    DefaultRoute,
    #[serde(rename = "common-routable-interface")]
    Routable,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NetworkMachinePlan {
    pub default_route_interface: Option<String>,
    pub addresses: BTreeMap<String, Vec<String>>,
    pub active_rdma_interfaces: Vec<ActiveRdmaInterfacePlan>,
    pub candidates: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ActiveRdmaInterfacePlan {
    pub interface: String,
    pub device: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProcessPlan {
    pub id: String,
    #[serde(flatten)]
    pub identity: ProcessIdentityPlan,
    pub machine: String,
    pub launch: LaunchPlan,
    #[serde(rename = "dependencies")]
    pub launch_dependencies: Vec<String>,
    #[serde(flatten)]
    pub allocation: AllocationPlan,
    pub command: CommandPlan,
    pub launch_files: Vec<LaunchFilePlan>,
    pub readiness: ReadinessPlan,
    /// The endpoint allocation owned by this process. Public workload-route
    /// semantics live only on `ServerPlan::endpoint`.
    pub endpoint: ProcessEndpointPlan,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<ContainerPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_target: Option<CaptureTargetPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProcessIdentityPlan {
    ModelRank {
        rank: u32,
        rank_count: u32,
    },
    Frontend {
        process_role: FrontendProcessRole,
        components: FrontendComponents,
    },
}

impl ProcessPlan {
    #[must_use]
    pub const fn rank(&self) -> Option<u32> {
        match &self.identity {
            ProcessIdentityPlan::ModelRank { rank, .. } => Some(*rank),
            ProcessIdentityPlan::Frontend { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ContainerPlan {
    pub name: String,
    pub image: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LaunchFilePlan {
    pub relative_path: String,
    pub resolved_path: PathBuf,
    pub text: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProfilerEscapesPlan {
    #[serde(default, skip_serializing_if = "NsysEscapes::is_empty")]
    pub common: NsysEscapes,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub roles: BTreeMap<String, NsysEscapes>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CaptureTargetPlan {
    pub window_control_endpoint: CaptureWindowControlEndpointPlan,
    pub control_process_id: String,
    pub start: CaptureWindowActionPlan,
    pub stop: CaptureWindowActionPlan,
    /// The merged escape inputs for this target's role
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); the raw common and role
    /// declarations live on the server plan.
    #[serde(default, skip_serializing_if = "NsysEscapes::is_empty")]
    pub escapes: NsysEscapes,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureWindowControlEndpointPlan {
    ReplicaEntry,
    Gateway,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureWindowActionPlan {
    pub method: CaptureWindowHttpMethodPlan,
    pub path: String,
    pub effective_url: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureWindowHttpMethodPlan {
    Post,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RemoteWorkspacePlan {
    pub target: String,
    pub path: PathBuf,
    pub revision: String,
    pub dirty: bool,
    pub source_digest: String,
    pub pixi_manifest_sha256: String,
    pub pixi_lock_sha256: String,
    pub pixi_environment: String,
    pub pixi_executable: PathBuf,
    pub environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum LaunchPlan {
    Local,
    Ssh { target: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AllocationPlan {
    pub devices: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_locator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_locator_source: Option<ModelLocatorSource>,
    pub ports: BTreeMap<String, EndpointAssignment>,
    pub runtime_cache: RuntimeCachePlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub communication_interface: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLocatorSource {
    Machine,
    Fallback,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RuntimeCachePlan {
    pub storage_root: PathBuf,
    pub storage_root_source: RuntimeCacheRootSource,
    pub namespace: RuntimeCacheNamespacePlan,
    pub path: PathBuf,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeCacheRootSource {
    WorkspaceDefault,
    MachineBinding,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RuntimeCacheNamespacePlan {
    pub workspace_source_digest: String,
    pub pixi_environment: String,
    /// Set for image-backed launches: the immutable image identity keys the
    /// namespace in place of the invoking workspace source state and Pixi
    /// environment identity ([[RFC-0002:C-RUNTIME-CACHE]]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_id: Option<String>,
    pub machine: String,
    pub process: String,
}

struct ResolvedProcessAllocation {
    wire: ServeProcessAllocation,
    runtime_cache: RuntimeCachePlan,
    model_locator_source: Option<ModelLocatorSource>,
}

impl ResolvedProcessAllocation {
    fn process(&self) -> &str {
        match &self.wire {
            ServeProcessAllocation::ModelRank { process, .. }
            | ServeProcessAllocation::Frontend { process, .. } => process,
        }
    }

    fn machine(&self) -> &str {
        match &self.wire {
            ServeProcessAllocation::ModelRank { machine, .. }
            | ServeProcessAllocation::Frontend { machine, .. } => machine,
        }
    }

    fn devices(&self) -> &[u32] {
        match &self.wire {
            ServeProcessAllocation::ModelRank { devices, .. }
            | ServeProcessAllocation::Frontend { devices, .. } => devices,
        }
    }

    fn endpoint(&self) -> Option<&EndpointAssignment> {
        match &self.wire {
            ServeProcessAllocation::ModelRank { endpoint, .. } => endpoint.as_ref(),
            ServeProcessAllocation::Frontend { endpoint, .. } => Some(endpoint),
        }
    }

    fn ports(&self) -> &BTreeMap<String, EndpointAssignment> {
        match &self.wire {
            ServeProcessAllocation::ModelRank { ports, .. }
            | ServeProcessAllocation::Frontend { ports, .. } => ports,
        }
    }

    fn model_locator(&self) -> Option<&str> {
        match &self.wire {
            ServeProcessAllocation::ModelRank { model_locator, .. } => Some(model_locator),
            ServeProcessAllocation::Frontend { .. } => None,
        }
    }
}

#[derive(Clone)]
struct ProcessRequirement {
    id: String,
    identity: ProcessRequirementIdentity,
    device_count: u32,
    ports: Vec<String>,
    readiness: ReadinessProbe,
    launch_dependencies: Vec<String>,
    capture_target: Option<PendingCaptureTargetPlan>,
    fixed_devices: Option<FixedDeviceAssignment>,
}

#[derive(Clone)]
struct PendingCaptureTargetPlan {
    window_control_endpoint: CaptureWindowControlEndpointPlan,
    replica_entry_process_id: String,
    start: PendingCaptureWindowActionPlan,
    stop: PendingCaptureWindowActionPlan,
    escapes: NsysEscapes,
}

#[derive(Clone)]
struct PendingCaptureWindowActionPlan {
    method: CaptureWindowHttpMethodPlan,
    path: String,
}

#[derive(Clone)]
enum ProcessRequirementIdentity {
    ModelRank {
        role_id: String,
        role_kind: ServeRoleKind,
        replica_id: String,
        replica_index: u32,
        rank: u32,
        effective_settings: BTreeMap<String, SettingValue>,
        effective_parallelism: Parallelism,
        links: Vec<ServeRoleLink>,
        render_inputs: Vec<SuppliedRenderInput>,
    },
    Frontend {
        process_role: FrontendProcessRole,
        components: FrontendComponents,
        gateway: Box<GatewayPlan>,
        pd_router: Option<Box<PdRouterPlan>>,
        links: Vec<ServeRoleLink>,
        render_inputs: Vec<SuppliedRenderInput>,
    },
}

impl ProcessRequirement {
    fn placement_role(&self) -> &str {
        match &self.identity {
            ProcessRequirementIdentity::ModelRank { role_id, .. } => role_id,
            ProcessRequirementIdentity::Frontend { .. } => "gateway",
        }
    }
}

#[derive(Clone)]
struct FixedDeviceAssignment {
    machine: String,
    devices: Vec<u32>,
    endpoint_port: Option<u16>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CommandPlan {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// The variables the resolver and integration set explicitly, as opposed
    /// to the ambient environment composed into `env` for local launches.
    /// The containerized substitution consumes exactly this set
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub explicit_env: Vec<String>,
    /// Declared pass-through names for a containerized launch: the value
    /// flows into the docker client from the launching machine's
    /// environment at spawn. On a remote launch it never enters this plan
    /// or the record; a local launch composes the invoking environment into
    /// `env` above under the standing unredacted-records posture, so the
    /// reference channel keeps the value out of the plan only where no
    /// ambient composition happens ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pass_env: Vec<String>,
    pub cwd: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EndpointPlan {
    pub host: String,
    pub port: u16,
    pub protocol: EndpointProtocol,
    pub completions_path: String,
    pub chat_completions_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<inferlab_protocol::HttpActionSpec>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProcessEndpointPlan {
    pub host: String,
    pub port: u16,
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
    case_id: Option<&str>,
    case: Option<&ServerCaseDefinition>,
    overrides: &[IndexedServerOverride],
    topology: ServeTopology,
) -> Result<Vec<ResolvedRoleInput>, InferlabError> {
    let declarations = role_declarations(server, topology)?;
    let selected = declarations
        .iter()
        .map(|(id, _)| id.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(case) = case {
        for id in case.roles.keys() {
            if !selected.contains(id.as_str()) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "server case {:?} configures role {id:?}, which is not part of the selected topology",
                        case_id.unwrap_or("")
                    ),
                });
            }
        }
    }
    for item in overrides {
        let patch = &item.patch;
        for id in patch.roles.keys() {
            if !selected.contains(id.as_str()) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "invocation configures role {id:?}, which is not part of the selected topology"
                    ),
                });
            }
        }
    }
    let empty_case = ServerCaseDefinition::default();
    let case = case.unwrap_or(&empty_case);
    declarations
        .into_iter()
        .map(|(id, kind)| {
            let mut parallelism = Parallelism::default();
            merge_parallelism(&mut parallelism, &server.parallelism);
            if let Some(role) = server.roles.get(&id) {
                merge_parallelism(&mut parallelism, &role.parallelism);
            }
            merge_parallelism(&mut parallelism, &case.parallelism);
            if let Some(role) = case.roles.get(&id) {
                merge_parallelism(&mut parallelism, &role.parallelism);
            }
            for item in overrides {
                let patch = &item.patch;
                merge_parallelism(&mut parallelism, &patch.parallelism);
                if let Some(role) = patch.roles.get(&id) {
                    merge_parallelism(&mut parallelism, &role.parallelism);
                }
            }

            let mut settings = BTreeMap::new();
            merge_toml_settings(&mut settings, &server.settings)?;
            if let Some(role) = server.roles.get(&id) {
                merge_toml_settings(&mut settings, &role.settings)?;
            }
            merge_toml_settings(&mut settings, &case.settings)?;
            if let Some(role) = case.roles.get(&id) {
                merge_toml_settings(&mut settings, &role.settings)?;
            }
            for item in overrides {
                let patch = &item.patch;
                merge_toml_settings(&mut settings, &patch.settings)?;
                if let Some(role) = patch.roles.get(&id) {
                    merge_toml_settings(&mut settings, &role.settings)?;
                }
            }
            let server_role = server.roles.get(&id);
            let mut replica_count = server_role.and_then(|role| role.replicas).unwrap_or(1);
            if let Some(role) = case.roles.get(&id)
                && let Some(replicas) = role.replicas
            {
                replica_count = replicas;
            }
            for item in overrides {
                let patch = &item.patch;
                if let Some(role) = patch.roles.get(&id)
                    && let Some(replicas) = role.replicas
                {
                    replica_count = replicas;
                }
            }
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
            gateway_backend: server.gateway_backend.clone(),
            pd_router_backend: server.pd_router_backend.clone(),
            kv_transfer: server.kv_transfer,
            profiling: server.profiling,
            capture_control_deadline_seconds: server.capture_control_deadline_seconds,
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
                gateway_backend: case.gateway_backend.clone(),
                pd_router_backend: case.pd_router_backend.clone(),
                kv_transfer: case.kv_transfer,
                profiling: case.profiling,
                capture_control_deadline_seconds: None,
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
        let index = item.index;
        let patch = &item.patch;
        declarations.push(ServerDeclarationPlan {
            source: DeclarationSource::Invocation { index },
            common: CommonDeclarationPlan {
                readiness_timeout_seconds: patch.readiness_timeout_seconds,
                gateway_backend: patch.gateway_backend.clone(),
                pd_router_backend: patch.pd_router_backend.clone(),
                kv_transfer: patch.kv_transfer,
                profiling: patch.profiling,
                capture_control_deadline_seconds: None,
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
    settings: &BTreeMap<String, toml::Value>,
) -> Result<BTreeMap<String, SettingValue>, InferlabError> {
    settings
        .iter()
        .map(|(key, value)| {
            setting_value(value, &format!("{scope}.{key}")).map(|value| (key.clone(), value))
        })
        .collect()
}

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
                &allocation.wire,
                ServeProcessAllocation::ModelRank { role, rank: 0, .. }
                    if role == &pd_router.prefill_role
            )
        })
        .collect::<Vec<_>>();
    let decode = allocations
        .iter()
        .filter(|allocation| {
            matches!(
                &allocation.wire,
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

fn endpoint_url(endpoint: &EndpointAssignment) -> String {
    format!("http://{}:{}", endpoint.host, endpoint.port)
}

fn load_render_inputs(
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

fn validate_launch_file_declarations(
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

fn select_workflow<'a>(
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

fn resolve_effective_server_input(
    selection: &WorkflowSelection<'_>,
    request: &ResolveRequest<'_>,
) -> Result<EffectiveServerInput, InferlabError> {
    let server = selection.server;
    let case = selection.case;
    let topology = server.topology;
    let mut readiness_timeout_seconds = server.readiness_timeout_seconds;
    let mut gateway_backend = server.gateway_backend.clone();
    let mut pd_router_backend = server.pd_router_backend.clone();
    let mut kv_transfer = server.kv_transfer;
    let mut profiling = server.profiling.unwrap_or(false);
    let capture_control_deadline_seconds = server
        .capture_control_deadline_seconds
        .unwrap_or(DEFAULT_CAPTURE_CONTROL_DEADLINE_SECONDS);
    if let Some(value) = case.and_then(|case| case.readiness_timeout_seconds) {
        readiness_timeout_seconds = value;
    }
    if let Some(value) = case.and_then(|case| case.gateway_backend.as_ref()) {
        gateway_backend = Some(value.clone());
    }
    if let Some(value) = case.and_then(|case| case.pd_router_backend.as_ref()) {
        pd_router_backend = Some(value.clone());
    }
    if let Some(value) = case.and_then(|case| case.kv_transfer) {
        kv_transfer = Some(value);
    }
    if let Some(value) = case.and_then(|case| case.profiling) {
        profiling = value;
    }

    let mut override_patches = Vec::new();
    for (index, value) in request.overrides.iter().enumerate() {
        if !value.starts_with("server.")
            && matches!(request.workflow, Workflow::RecipeRun)
            && (value.starts_with("evals.") || value.starts_with("benches."))
        {
            continue;
        }
        let patch = parse_override(value)?;
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
        if let Some(value) = &patch.gateway_backend {
            gateway_backend = Some(value.clone());
        }
        if let Some(value) = &patch.pd_router_backend {
            pd_router_backend = Some(value.clone());
        }
        if let Some(value) = patch.kv_transfer {
            kv_transfer = Some(value);
        }
        if let Some(value) = patch.profiling {
            profiling = value;
        }
        if let Some(value) = patch.readiness_timeout_seconds {
            if value == 0 {
                return Err(InferlabError::InvalidConfig {
                    message: "readiness_timeout_seconds must be nonzero".to_owned(),
                });
            }
            readiness_timeout_seconds = value;
        }
        override_patches.push(IndexedServerOverride {
            index,
            raw: value.clone(),
            patch,
        });
    }
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
    let role_resolutions = resolve_role_inputs(server, case_id, case, &override_patches, topology)?;
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
        gateway_backend,
        pd_router_backend,
        kv_transfer,
        profiling,
        capture_control_deadline_seconds,
        override_patches,
        role_resolutions,
        declarations,
        role_inputs,
    })
}

fn split_lowering<T>(lowering: AdapterLowering<T>) -> (T, LoweringEvidence) {
    (
        lowering.output,
        LoweringEvidence {
            request_sha256: lowering.request_sha256,
            response_sha256: lowering.response_sha256,
            timing: lowering.timing,
        },
    )
}

fn plan_integration<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    selection: &WorkflowSelection<'_>,
    effective: &mut EffectiveServerInput,
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
        .map(|requirement| requirement.id.clone())
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
            Some(FixedDeviceAssignment {
                machine: rank.machine.clone(),
                devices: Vec::new(),
                endpoint_port: rank.endpoint_port,
            })
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
            .map(|requirement| requirement.id.clone())
            .collect();
        let mut links = links_for_node(&planned.links, "gateway");
        for link in links_for_node(&planned.links, "pd_router") {
            if !links.contains(&link) {
                links.push(link);
            }
        }
        requirements.push(ProcessRequirement {
            id: "gateway".to_owned(),
            identity: ProcessRequirementIdentity::Frontend {
                process_role: FrontendProcessRole::Gateway,
                components,
                gateway: Box::new(gateway.clone()),
                pd_router: planned.pd_router.clone().map(Box::new),
                links,
                render_inputs,
            },
            device_count: 0,
            ports,
            readiness,
            launch_dependencies: dependencies,
            capture_target: None,
            fixed_devices: fixed_gateway,
        });
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
                    &requirement.identity,
                    ProcessRequirementIdentity::ModelRank {
                        role_id,
                        replica_index: 0,
                        rank: 0,
                        ..
                    } if role_id == &role.id
                )
            })
            .map(|requirement| requirement.id.clone())
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} selected unknown direct public Engine role {:?}",
                    stack.integration, role.id
                ),
            })?
    };
    validate_launch_dependencies(&stack.integration, &requirements)?;
    Ok(PlannedServeStage {
        planned,
        evidence,
        requirements,
        integration_rendered_process_ids,
        public_process,
    })
}

fn rendered_process_id(process: &RenderedServeProcess) -> &str {
    match process {
        RenderedServeProcess::ModelRank { process, .. }
        | RenderedServeProcess::Frontend { process, .. } => process,
    }
}

fn render_integration<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    selection: &WorkflowSelection<'_>,
    effective: &EffectiveServerInput,
    planned_stage: &PlannedServeStage,
    adapter: &C,
) -> Result<RenderedServeStage, InferlabError> {
    let stack = selection.stack;
    let planned = &planned_stage.planned;
    let control_plane_frontend = planned
        .gateway
        .as_ref()
        .is_some_and(|gateway| gateway.render_source == RenderSource::ControlPlane);
    let allocations = allocate_processes(
        workspace,
        &selection.placement_id,
        selection.placement,
        selection.weight,
        &stack.pixi_environment,
        request
            .image
            .map(|image| image.image_id.as_str())
            .or_else(|| request.external.map(|external| external.digest.as_str())),
        &planned_stage.requirements,
        control_plane_frontend.then_some(planned_stage.public_process.as_str()),
    )?;
    let render_allocations = allocations
        .iter()
        .filter(|allocation| {
            planned_stage
                .integration_rendered_process_ids
                .contains(allocation.process())
        })
        .map(|allocation| allocation.wire.clone())
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
    if rendered.processes.len() != planned_stage.integration_rendered_process_ids.len() {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {:?} rendered {} processes for {} integration-rendered allocations",
                stack.integration,
                rendered.processes.len(),
                planned_stage.integration_rendered_process_ids.len()
            ),
        });
    }
    let mut rendered_by_id = BTreeMap::new();
    for process in rendered.processes {
        let id = rendered_process_id(&process).to_owned();
        if !planned_stage.integration_rendered_process_ids.contains(&id)
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
        let process = render_builtin_frontend(
            gateway,
            pd_router,
            &planned.integration.framework,
            effective.kv_transfer,
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
        .requirements
        .iter()
        .map(|requirement| {
            rendered_by_id
                .remove(&requirement.id)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "resolved topology has no rendered process for allocation {:?}",
                        requirement.id
                    ),
                })
        })
        .collect::<Result<Vec<_>, InferlabError>>()?;
    if !rendered_by_id.is_empty() {
        return Err(InferlabError::InvalidConfig {
            message: "rendering returned processes outside the resolved topology".to_owned(),
        });
    }
    Ok(RenderedServeStage {
        evidence,
        allocations,
        rendered_processes,
    })
}

fn realize_runtime(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    selection: &WorkflowSelection<'_>,
    effective: &EffectiveServerInput,
    planned_stage: &PlannedServeStage,
    rendered_stage: &RenderedServeStage,
) -> Result<RuntimeRealizationStage, InferlabError> {
    let planned = &planned_stage.planned;
    let requirements = &planned_stage.requirements;
    let public_process = &planned_stage.public_process;
    let allocations = &rendered_stage.allocations;
    let endpoint_requirement =
        public_endpoint_requirement(&selection.stack.integration, effective.topology, planned)?;
    let mut processes = Vec::with_capacity(requirements.len());
    let mut public_endpoint = None;
    let mut device_count = 0_u32;
    let gateway_process_id = resolved_gateway_process_id(requirements)?;
    for ((requirement, allocation), rendered_process) in requirements
        .iter()
        .zip(allocations)
        .zip(&rendered_stage.rendered_processes)
    {
        let (identity, command, launch_file_declarations, integration_rendered) = match (
            &requirement.identity,
            &allocation.wire,
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
            ) if process == &requirement.id
                && rendered_id == &requirement.id
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
            ) if process == &requirement.id
                && rendered_id == &requirement.id
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
                        requirement.id
                    ),
                });
            }
        };
        if command.argv.is_empty() {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} rendered an empty argv for process {:?}",
                    selection.stack.integration, requirement.id
                ),
            });
        }
        if command.env.contains_key("CUDA_VISIBLE_DEVICES") {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} attempted to select devices for process {:?}",
                    selection.stack.integration, requirement.id
                ),
            });
        }
        let launch_files = validate_launch_file_declarations(
            &selection.stack.integration,
            &requirement.id,
            &allocation.runtime_cache.path,
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
        if requirement.id == *public_process {
            public_endpoint = Some(EndpointPlan {
                host: endpoint.host.clone(),
                port: endpoint.port,
                protocol: endpoint_requirement.protocol,
                completions_path: endpoint_requirement.completions_path.clone(),
                chat_completions_path: endpoint_requirement.chat_completions_path.clone(),
                prefix_cache_reset: endpoint_requirement.prefix_cache_reset.clone(),
            });
        }
        device_count += requirement.device_count;
        processes.push(ProcessPlan {
            id: requirement.id.clone(),
            identity,
            machine: machine_id.to_owned(),
            launch: launch_plan(&machine.launch),
            launch_dependencies: requirement.launch_dependencies.clone(),
            allocation: AllocationPlan {
                devices: allocation.devices().to_vec(),
                model_locator: allocation.model_locator().map(str::to_owned),
                model_locator_source: allocation.model_locator_source,
                ports: allocation.ports().clone(),
                runtime_cache: allocation.runtime_cache.clone(),
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
                &requirement.readiness,
                effective.readiness_timeout_seconds,
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
    Ok(RuntimeRealizationStage {
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
    })
}

fn compose_measurements(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
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
                        &allocation.wire,
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
        request.overrides,
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

fn assemble_process_hierarchy(
    integration: &str,
    effective: &EffectiveServerInput,
    planned_stage: &PlannedServeStage,
    processes: Vec<ProcessPlan>,
) -> Result<(Vec<RolePlan>, Option<FrontendPlan>), InferlabError> {
    let planned = &planned_stage.planned;
    let requirements = &planned_stage.requirements;
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
                            } = &requirement.identity
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
                        .map(|(requirement, _)| requirement.id.clone())
                        .ok_or_else(|| InferlabError::InvalidConfig {
                            message: format!("resolved replica {:?} contains no ranks", replica.id),
                        })?;
                    let ranks = ranks
                        .into_iter()
                        .map(|(requirement, _)| {
                            processes_by_id.remove(&requirement.id).ok_or_else(|| {
                                InferlabError::InvalidConfig {
                                    message: format!(
                                        "resolved replica {:?} references missing process {:?}",
                                        replica.id, requirement.id
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
                requirement.identity,
                ProcessRequirementIdentity::Frontend { .. }
            )
            .then_some(requirement.id.as_str())
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

pub fn resolve<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    adapter: &C,
) -> Result<ResolvedExecution, InferlabError> {
    let selection = select_workflow(workspace, request)?;
    let mut effective = resolve_effective_server_input(&selection, request)?;
    let server = selection.server;
    let stack = selection.stack;
    let server_id = selection.server_id.as_str();
    let case_id = selection.case_id.as_deref();
    let topology = effective.topology;
    let profiling = effective.profiling;

    let served_name = selection.model.served_name.clone();
    let planned_stage = plan_integration(workspace, &selection, &mut effective, adapter)?;
    let rendered_stage = render_integration(
        workspace,
        request,
        &selection,
        &effective,
        &planned_stage,
        adapter,
    )?;
    let planned = &planned_stage.planned;
    let RuntimeRealizationStage {
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
    let measurements = compose_measurements(
        workspace,
        request,
        &selection,
        &public_endpoint,
        &rendered_stage.allocations,
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
                .map(|item| item.raw.clone())
                .collect(),
            declarations: effective.declarations,
            topology,
            profiling,
            readiness_timeout_seconds: effective.readiness_timeout_seconds,
            capture_control_deadline_seconds: effective.capture_control_deadline_seconds,
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
                plan_request_sha256: planned_stage.evidence.request_sha256,
                plan_response_sha256: planned_stage.evidence.response_sha256,
                render_request_sha256: rendered_stage.evidence.request_sha256,
                render_response_sha256: rendered_stage.evidence.response_sha256,
                plan_timing: Some(planned_stage.evidence.timing),
                render_timing: Some(rendered_stage.evidence.timing),
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

fn validate_workload_endpoint(
    integration: &str,
    endpoint: &inferlab_protocol::EndpointRequirement,
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
    Ok(())
}

fn validate_capture_targets(
    integration: &str,
    profiling: bool,
    has_gateway: bool,
    replicas: &[ServeReplicaRequirement],
) -> Result<(), InferlabError> {
    for replica in replicas {
        if replica.capture_target.as_ref().is_some_and(|target| {
            target.window_control.endpoint == CaptureWindowControlEndpoint::Gateway && !has_gateway
        }) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} selected Gateway profiling window control without planning a Gateway"
                ),
            });
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

fn resolve_capture_target(
    requirement: &ProcessRequirement,
    gateway_process_id: Option<&str>,
    allocations: &[ResolvedProcessAllocation],
) -> Result<Option<CaptureTargetPlan>, InferlabError> {
    requirement
        .capture_target
        .as_ref()
        .map(|target| {
            let control_process_id = match target.window_control_endpoint {
                CaptureWindowControlEndpointPlan::ReplicaEntry => {
                    target.replica_entry_process_id.clone()
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
            Ok(CaptureTargetPlan {
                window_control_endpoint: target.window_control_endpoint,
                control_process_id,
                start: resolve_capture_window_action(&target.start, control_endpoint),
                stop: resolve_capture_window_action(&target.stop, control_endpoint),
                escapes: target.escapes.clone(),
            })
        })
        .transpose()
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
    action: &inferlab_protocol::HttpActionSpec,
) -> PendingCaptureWindowActionPlan {
    PendingCaptureWindowActionPlan {
        method: match action.method {
            inferlab_protocol::HttpMethod::Post => CaptureWindowHttpMethodPlan::Post,
        },
        path: action.path.clone(),
    }
}

fn resolve_capture_window_action(
    action: &PendingCaptureWindowActionPlan,
    endpoint: &EndpointAssignment,
) -> CaptureWindowActionPlan {
    CaptureWindowActionPlan {
        method: action.method,
        path: action.path.clone(),
        effective_url: format!("{}{}", endpoint_url(endpoint), action.path),
    }
}

fn resolved_gateway_process_id(
    requirements: &[ProcessRequirement],
) -> Result<Option<&str>, InferlabError> {
    let mut gateway_process_ids = requirements.iter().filter_map(|requirement| {
        let ProcessRequirementIdentity::Frontend { components, .. } = &requirement.identity else {
            return None;
        };
        match components {
            FrontendComponents::Gateway(_) | FrontendComponents::GatewayPdRouter(_) => {
                Some(requirement.id.as_str())
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

fn profiler_escapes_plan(server: &ServerDefinition) -> Option<ProfilerEscapesPlan> {
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

fn uses_explicit_replica_placement(placement: &PlacementBinding) -> bool {
    placement
        .roles
        .values()
        .any(PlacementRoleBinding::uses_explicit_replicas)
}

fn links_for_node(links: &[ServeRoleLink], node: &str) -> Vec<ServeRoleLink> {
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

fn expand_replica_requirements(
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
            let fixed_devices = explicit_ranks
                .get(rank)
                .map(|assignment| FixedDeviceAssignment {
                    machine: assignment.machine.clone(),
                    devices: assignment.devices.clone(),
                    endpoint_port: assignment.endpoint_port,
                });
            let device_count = fixed_devices.as_ref().map_or_else(
                || Ok(replica.device_count),
                |fixed| {
                    u32::try_from(fixed.devices.len()).map_err(|_| InferlabError::InvalidConfig {
                        message: format!("replica {:?} rank has too many devices", replica.id),
                    })
                },
            )?;
            let mut ports = replica.ports.clone();
            if rank == 0 && rank_count > 1 {
                ports.extend(replica.primary_ports.iter().cloned());
            }
            let capture_target =
                replica
                    .capture_target
                    .as_ref()
                    .map(|target| PendingCaptureTargetPlan {
                        window_control_endpoint: capture_window_control_endpoint_plan(
                            target.window_control.endpoint,
                        ),
                        replica_entry_process_id: primary_id.clone(),
                        start: capture_window_action_plan(&target.window_control.start),
                        stop: capture_window_action_plan(&target.window_control.stop),
                        escapes: server.roles.get(&replica.role_id).map_or_else(
                            || server.profiler.nsys.clone(),
                            |role| server.profiler.nsys.merged_with(&role.profiler.nsys),
                        ),
                    });
            processes.push(ProcessRequirement {
                id: process_id(&replica.id, rank_index, rank_count),
                identity: ProcessRequirementIdentity::ModelRank {
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
                readiness: if rank == 0 {
                    replica.primary_readiness.clone()
                } else {
                    replica.worker_readiness.clone()
                },
                launch_dependencies: if rank == 0 {
                    Vec::new()
                } else {
                    vec![primary_id.clone()]
                },
                capture_target,
                fixed_devices,
            });
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

fn public_endpoint_requirement<'a>(
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

fn validate_serve_graph(
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

fn validate_launch_dependencies(
    integration: &str,
    processes: &[ProcessRequirement],
) -> Result<(), InferlabError> {
    let mut prior = BTreeSet::new();
    for process in processes {
        if process.id.is_empty() || !prior.insert(process.id.as_str()) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned a duplicate or empty process id"
                ),
            });
        }
        let mut dependencies = BTreeSet::new();
        for dependency in &process.launch_dependencies {
            if !dependencies.insert(dependency) || !prior.contains(dependency.as_str()) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} process {:?} has an invalid or unordered launch dependency {dependency:?}",
                        process.id
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_integration_identity(expected: &str, actual: &str) -> Result<(), InferlabError> {
    if actual == expected {
        Ok(())
    } else {
        Err(InferlabError::InvalidConfig {
            message: format!("integration {expected:?} returned framework identity {actual:?}"),
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn allocate_processes(
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
        if requirement.id.is_empty() || !process_ids.insert(requirement.id.clone()) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration returned invalid or duplicate process id {:?}",
                    requirement.id
                ),
            });
        }
        let placement_role = requirement.placement_role();
        if placement_role.is_empty() {
            return Err(InferlabError::InvalidConfig {
                message: format!("process {:?} has an empty placement role", requirement.id),
            });
        }
        let mut port_names = BTreeSet::new();
        if requirement
            .ports
            .iter()
            .any(|name| name.is_empty() || !port_names.insert(name))
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration returned invalid or duplicate port requirements for process {:?}",
                    requirement.id
                ),
            });
        }

        let mut candidates = if let Some(fixed) = &requirement.fixed_devices {
            vec![fixed.machine.clone()]
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
        if local_process == Some(requirement.id.as_str()) {
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
                        requirement.id
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
                    requirement.device_count as usize,
                    requirement.ports.len() + 1,
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
                if available < requirement.device_count as usize {
                    return Err(InferlabError::InsufficientDevices {
                        machine: candidate.clone(),
                        required: requirement.device_count,
                        available,
                    });
                }
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "machine {candidate:?} has insufficient free ports for process {:?}",
                        requirement.id
                    ),
                });
            }
            None => {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement {placement_id:?} has no machine with {} free devices and {} free ports for process {:?} in role {placement_role:?}",
                        requirement.device_count,
                        requirement.ports.len() + 1,
                        requirement.id
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
            if let Some(fixed) = &requirement.fixed_devices {
                if fixed.devices.iter().any(|device| {
                    !machine.devices.contains(device) || used.devices.contains(device)
                }) {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "placement assigns unavailable or overlapping devices to process {:?}",
                            requirement.id
                        ),
                    });
                }
                fixed.devices.clone()
            } else {
                machine
                    .devices
                    .iter()
                    .filter(|device| !used.devices.contains(device))
                    .take(requirement.device_count as usize)
                    .copied()
                    .collect::<Vec<_>>()
            };
        used.devices.extend(&devices);
        let endpoint_port = requirement
            .fixed_devices
            .as_ref()
            .and_then(|fixed| fixed.endpoint_port);
        let endpoint_port = match endpoint_port {
            Some(port) => {
                if !machine.ports.contains(&port) || used.ports.contains(&port) {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "placement assigns unavailable endpoint port {port} to process {:?}",
                            requirement.id
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
                        requirement.id
                    ),
                })?,
        };
        used.ports.insert(endpoint_port);
        let selected_ports = machine
            .ports
            .iter()
            .filter(|port| !used.ports.contains(port))
            .take(requirement.ports.len())
            .copied()
            .collect::<Vec<_>>();
        if selected_ports.len() != requirement.ports.len() {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "machine {machine_id:?} has insufficient free named ports for process {:?}",
                    requirement.id
                ),
            });
        }
        used.ports.extend(&selected_ports);
        let named_ports = requirement
            .ports
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
            &requirement.id,
            pixi_environment,
            image_identity,
        );
        let cache = runtime_cache
            .path
            .to_str()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "runtime cache path for process {:?} is not valid UTF-8",
                    requirement.id
                ),
            })?
            .to_owned();
        let launch = match &machine.launch {
            LaunchBinding::Local => AllocationLaunch::Local,
            LaunchBinding::Ssh { target } => AllocationLaunch::Ssh {
                target: target.clone(),
            },
        };
        let (wire, model_locator_source) = match &requirement.identity {
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
                            requirement.id
                        ),
                    });
                };
                let rank_count = u32::try_from(
                    requirements
                        .iter()
                        .filter(|candidate| {
                            matches!(
                                &candidate.identity,
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
                        process: requirement.id.clone(),
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
                        dependencies: requirement.launch_dependencies.clone(),
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
                if requirement.device_count != 0 || !devices.is_empty() {
                    return Err(InferlabError::InvalidConfig {
                        message: "frontend process must not allocate model devices".to_owned(),
                    });
                }
                (
                    ServeProcessAllocation::Frontend {
                        process: requirement.id.clone(),
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
                        dependencies: requirement.launch_dependencies.clone(),
                        render_inputs: render_inputs.clone(),
                    },
                    None,
                )
            }
        };
        allocations.push(ResolvedProcessAllocation {
            wire,
            runtime_cache,
            model_locator_source,
        });
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

fn launch_plan(binding: &LaunchBinding) -> LaunchPlan {
    match binding {
        LaunchBinding::Local => LaunchPlan::Local,
        LaunchBinding::Ssh { target } => LaunchPlan::Ssh {
            target: target.clone(),
        },
    }
}

fn pixi_command(environment: &str, process: Vec<String>) -> Vec<String> {
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

fn readiness_plan(
    probe: &ReadinessProbe,
    timeout: u64,
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
        }),
        ReadinessProbe::HttpTargetRegistry(registry) => {
            let expected_targets = allocations
                .iter()
                .filter_map(|allocation| {
                    let ServeProcessAllocation::ModelRank {
                        role_kind,
                        rank: 0,
                        ..
                    } = &allocation.wire
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
            })
        }
        ReadinessProbe::ProcessAlive => Ok(ReadinessPlan::ProcessAlive),
    }
}

fn target_endpoint_url(endpoint: &EndpointAssignment, scheme: TargetEndpointScheme) -> String {
    let scheme = match scheme {
        TargetEndpointScheme::Http => "http",
        TargetEndpointScheme::Grpc => "grpc",
    };
    format!("{scheme}://{}:{}", endpoint.host, endpoint.port)
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

fn merge_toml_settings(
    settings: &mut BTreeMap<String, SettingValue>,
    patch: &BTreeMap<String, toml::Value>,
) -> Result<(), InferlabError> {
    for (key, value) in patch {
        let converted = setting_value(value, key)?;
        merge_value(settings, std::slice::from_ref(key), converted);
    }
    Ok(())
}

fn merge_parallelism(parallelism: &mut Parallelism, patch: &Parallelism) {
    parallelism.merge_from(patch);
}

fn validate_effective_parallelism(
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

fn parse_override(value: &str) -> Result<ServerOverridePatch, InferlabError> {
    let (path, raw_value) =
        value
            .split_once('=')
            .ok_or_else(|| InferlabError::InvalidOverride {
                value: value.to_owned(),
                message: "expected server.<path>=<TOML-value>".to_owned(),
            })?;
    let setting_path =
        path.strip_prefix("server.")
            .ok_or_else(|| InferlabError::InvalidOverride {
                value: value.to_owned(),
                message: "only paths under server. may be overridden".to_owned(),
            })?;
    let nested = ExactTomlOverride::parse(setting_path, raw_value, value)?.into_patch();
    let patch: ServerOverridePatch =
        nested
            .try_into()
            .map_err(|error| InferlabError::InvalidOverride {
                value: value.to_owned(),
                message: format!("invalid server setting: {error}"),
            })?;

    Ok(patch)
}

fn setting_value(value: &toml::Value, path: &str) -> Result<SettingValue, InferlabError> {
    match value {
        toml::Value::String(value) => Ok(SettingValue::String(value.clone())),
        toml::Value::Integer(value) => Ok(SettingValue::Integer(*value)),
        toml::Value::Float(value) => Ok(SettingValue::Float(*value)),
        toml::Value::Boolean(value) => Ok(SettingValue::Bool(*value)),
        toml::Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| setting_value(value, &format!("{path}[{index}]")))
            .collect::<Result<Vec<_>, _>>()
            .map(SettingValue::Array),
        toml::Value::Table(values) => values
            .iter()
            .map(|(key, value)| {
                setting_value(value, &format!("{path}.{key}")).map(|value| (key.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map(SettingValue::Object),
        toml::Value::Datetime(_) => Err(InferlabError::InvalidConfig {
            message: format!("server setting {path:?} cannot be a TOML date or time"),
        }),
    }
}

fn merge_value(
    settings: &mut BTreeMap<String, SettingValue>,
    path: &[String],
    value: SettingValue,
) {
    if let SettingValue::Object(values) = value {
        if values.is_empty() {
            set_map_value(settings, path, SettingValue::Object(values));
        } else {
            for (key, value) in values {
                let mut child_path = path.to_vec();
                child_path.push(key);
                merge_value(settings, &child_path, value);
            }
        }
    } else {
        set_map_value(settings, path, value);
    }
}

fn set_map_value(
    settings: &mut BTreeMap<String, SettingValue>,
    path: &[String],
    value: SettingValue,
) {
    let Some((head, tail)) = path.split_first() else {
        return;
    };
    if tail.is_empty() {
        settings.insert(head.clone(), value);
    } else {
        let entry = settings
            .entry(head.clone())
            .or_insert_with(|| SettingValue::Object(BTreeMap::new()));
        if !matches!(entry, SettingValue::Object(_)) {
            *entry = SettingValue::Object(BTreeMap::new());
        }
        if let SettingValue::Object(children) = entry {
            set_map_value(children, tail, value);
        }
    }
}

fn validate_effective_settings(
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

#[cfg(test)]
mod tests {
    use super::*;
    use inferlab_protocol::{
        EndpointRequirement, FrontendCoRendering, FrontendHandoff, GatewayTarget,
        HttpTargetRegistryReadiness, IntegrationIdentity, LaunchFileDeclaration,
        ParallelismAttention, ParallelismExperts, ParallelismOuter, PdRoutingPolicies,
        RenderInputDeclaration, ServeRoleResult,
    };
    use std::error::Error;

    #[test]
    fn rejects_an_integration_that_rebinds_a_named_workload_path() -> Result<(), Box<dyn Error>> {
        let endpoint = EndpointRequirement {
            protocol: EndpointProtocol::Http,
            completions_path: "/v1/completions".to_owned(),
            chat_completions_path: "/v1/completions".to_owned(),
            prefix_cache_reset: None,
        };

        let error = validate_workload_endpoint("fixture", &endpoint)
            .err()
            .ok_or("rebound chat-completions path was accepted")?;

        assert!(error.to_string().contains("chat_completions_path"));
        assert!(error.to_string().contains("/v1/chat/completions"));
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
                ResolvedProcessAllocation {
                    wire: ServeProcessAllocation::ModelRank {
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
                    runtime_cache: RuntimeCachePlan {
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
                    model_locator_source: Some(ModelLocatorSource::Fallback),
                }
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

        let readiness = readiness_plan(&probe, 900, false, &allocations)?;
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
