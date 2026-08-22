//! Validated serve-planning stages and frozen process execution facts.

use inferlab_profiler::plan::{
    CaptureWindowControlEndpointPlan, CaptureWindowHttpMethodPlan, NsysEscapes, ProcessCapturePlan,
};
use inferlab_protocol::{
    CaptureMechanism, EndpointAssignment, EndpointProtocol, FrontendComponents,
    FrontendProcessRole, GatewayPlan, Parallelism, PdRouterPlan, PlanServeResult, ReadinessProbe,
    RenderedServeProcess, ServeProcessAllocation, ServeRoleKind, ServeRoleLink, SettingValue,
    SuppliedRenderInput,
};
use inferlab_runtime::operation_bound::OperationTimingEvidence;
use inferlab_runtime::plan::{
    CommandPlan, LaunchFilePlan, LaunchPlan, ProcessEndpointPlan, ReadinessPlan,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RemoteContainerFacts {
    pub target: String,
    pub user: String,
    pub uid: u32,
    pub gid: u32,
    pub present_pass_env: BTreeSet<String>,
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
    pub command_source: ProcessCommandSource,
    pub machine: String,
    pub launch: LaunchPlan,
    #[serde(rename = "dependencies")]
    pub launch_dependencies: Vec<String>,
    #[serde(flatten)]
    pub allocation: AllocationPlan,
    pub command: CommandPlan,
    pub launch_files: Vec<LaunchFilePlan>,
    pub readiness: ReadinessPlan,
    pub endpoint: ProcessEndpointPlan,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<ContainerPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_target: Option<ProcessCapturePlan>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessCommandSource {
    Integration,
    ControlPlane,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProfilerEscapesPlan {
    /// The raw mechanism declaration on the server, when present; omission
    /// resolves to managed collection ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mechanism: Option<CaptureMechanism>,
    #[serde(default, skip_serializing_if = "NsysEscapes::is_empty")]
    pub common: NsysEscapes,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub roles: BTreeMap<String, NsysEscapes>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_id: Option<String>,
    pub machine: String,
    pub process: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EndpointPlan {
    pub host: String,
    pub port: u16,
    pub protocol: EndpointProtocol,
    pub completions_path: String,
    pub chat_completions_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ServerMetricsEndpointPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<inferlab_protocol::HttpActionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_cache_conditioning: Option<inferlab_protocol::HttpActionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_read_zero_representation:
        Option<inferlab_protocol::PromptCacheReadZeroRepresentation>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServerMetricsEndpointPlan {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port_name: Option<String>,
    pub url: String,
}

#[derive(Clone)]
pub enum ProcessRequirementIdentity {
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

#[derive(Clone)]
pub struct PendingCaptureTargetPlan {
    mechanism: CaptureMechanism,
    window_control_endpoint: CaptureWindowControlEndpointPlan,
    replica_entry_process_id: String,
    /// The target replica's declared whole-replica device count: engine-trace
    /// coverage verification expects one new trace artifact per device
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    device_count: u32,
    start: PendingCaptureWindowActionPlan,
    stop: PendingCaptureWindowActionPlan,
    escapes: NsysEscapes,
}

#[derive(Clone)]
pub struct PendingCaptureWindowActionPlan {
    method: CaptureWindowHttpMethodPlan,
    path: String,
    body: Option<BTreeMap<String, SettingValue>>,
}

#[derive(Clone)]
pub struct FixedDeviceAssignment {
    machine: String,
    devices: Vec<u32>,
    endpoint_port: Option<u16>,
}

#[derive(Clone)]
pub struct ProcessRequirement {
    id: String,
    identity: ProcessRequirementIdentity,
    device_count: u32,
    ports: Vec<String>,
    readiness: ReadinessProbe,
    launch_dependencies: Vec<String>,
    capture_target: Option<PendingCaptureTargetPlan>,
    fixed_devices: Option<FixedDeviceAssignment>,
}

impl ProcessRequirement {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        identity: ProcessRequirementIdentity,
        device_count: u32,
        ports: Vec<String>,
        readiness: ReadinessProbe,
        launch_dependencies: Vec<String>,
        capture_target: Option<PendingCaptureTargetPlan>,
        fixed_devices: Option<FixedDeviceAssignment>,
    ) -> Self {
        Self {
            id,
            identity,
            device_count,
            ports,
            readiness,
            launch_dependencies,
            capture_target,
            fixed_devices,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn identity(&self) -> &ProcessRequirementIdentity {
        &self.identity
    }
    pub const fn device_count(&self) -> u32 {
        self.device_count
    }
    pub fn ports(&self) -> &[String] {
        &self.ports
    }
    pub fn readiness(&self) -> &ReadinessProbe {
        &self.readiness
    }
    pub fn launch_dependencies(&self) -> &[String] {
        &self.launch_dependencies
    }
    pub fn capture_target(&self) -> Option<&PendingCaptureTargetPlan> {
        self.capture_target.as_ref()
    }
    pub fn fixed_devices(&self) -> Option<&FixedDeviceAssignment> {
        self.fixed_devices.as_ref()
    }

    pub fn placement_role(&self) -> &str {
        match &self.identity {
            ProcessRequirementIdentity::ModelRank { role_id, .. } => role_id,
            ProcessRequirementIdentity::Frontend { .. } => "gateway",
        }
    }
}

impl PendingCaptureTargetPlan {
    pub fn new(
        mechanism: CaptureMechanism,
        window_control_endpoint: CaptureWindowControlEndpointPlan,
        replica_entry_process_id: String,
        device_count: u32,
        start: PendingCaptureWindowActionPlan,
        stop: PendingCaptureWindowActionPlan,
        escapes: NsysEscapes,
    ) -> Self {
        Self {
            mechanism,
            window_control_endpoint,
            replica_entry_process_id,
            device_count,
            start,
            stop,
            escapes,
        }
    }
    pub const fn mechanism(&self) -> CaptureMechanism {
        self.mechanism
    }
    pub const fn window_control_endpoint(&self) -> CaptureWindowControlEndpointPlan {
        self.window_control_endpoint
    }
    pub fn replica_entry_process_id(&self) -> &str {
        &self.replica_entry_process_id
    }
    pub const fn device_count(&self) -> u32 {
        self.device_count
    }
    pub fn start(&self) -> &PendingCaptureWindowActionPlan {
        &self.start
    }
    pub fn stop(&self) -> &PendingCaptureWindowActionPlan {
        &self.stop
    }
    pub fn escapes(&self) -> &NsysEscapes {
        &self.escapes
    }
}

impl PendingCaptureWindowActionPlan {
    pub fn new(
        method: CaptureWindowHttpMethodPlan,
        path: String,
        body: Option<BTreeMap<String, SettingValue>>,
    ) -> Self {
        Self { method, path, body }
    }
    pub const fn method(&self) -> CaptureWindowHttpMethodPlan {
        self.method
    }
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn body(&self) -> Option<&BTreeMap<String, SettingValue>> {
        self.body.as_ref()
    }
}

impl FixedDeviceAssignment {
    pub const fn new(machine: String, devices: Vec<u32>, endpoint_port: Option<u16>) -> Self {
        Self {
            machine,
            devices,
            endpoint_port,
        }
    }
    pub fn machine(&self) -> &str {
        &self.machine
    }
    pub fn devices(&self) -> &[u32] {
        &self.devices
    }
    pub const fn endpoint_port(&self) -> Option<u16> {
        self.endpoint_port
    }
}

pub struct ResolvedProcessAllocation {
    wire: ServeProcessAllocation,
    runtime_cache: RuntimeCachePlan,
    model_locator_source: Option<ModelLocatorSource>,
}

impl ResolvedProcessAllocation {
    pub const fn new(
        wire: ServeProcessAllocation,
        runtime_cache: RuntimeCachePlan,
        model_locator_source: Option<ModelLocatorSource>,
    ) -> Self {
        Self {
            wire,
            runtime_cache,
            model_locator_source,
        }
    }
    pub fn wire(&self) -> &ServeProcessAllocation {
        &self.wire
    }
    pub fn runtime_cache(&self) -> &RuntimeCachePlan {
        &self.runtime_cache
    }
    pub const fn model_locator_source(&self) -> Option<ModelLocatorSource> {
        self.model_locator_source
    }
    pub fn process(&self) -> &str {
        match &self.wire {
            ServeProcessAllocation::ModelRank { process, .. }
            | ServeProcessAllocation::Frontend { process, .. } => process,
        }
    }
    pub fn machine(&self) -> &str {
        match &self.wire {
            ServeProcessAllocation::ModelRank { machine, .. }
            | ServeProcessAllocation::Frontend { machine, .. } => machine,
        }
    }
    pub fn devices(&self) -> &[u32] {
        match &self.wire {
            ServeProcessAllocation::ModelRank { devices, .. }
            | ServeProcessAllocation::Frontend { devices, .. } => devices,
        }
    }
    pub fn endpoint(&self) -> Option<&EndpointAssignment> {
        match &self.wire {
            ServeProcessAllocation::ModelRank { endpoint, .. } => endpoint.as_ref(),
            ServeProcessAllocation::Frontend { endpoint, .. } => Some(endpoint),
        }
    }
    pub fn ports(&self) -> &BTreeMap<String, EndpointAssignment> {
        match &self.wire {
            ServeProcessAllocation::ModelRank { ports, .. }
            | ServeProcessAllocation::Frontend { ports, .. } => ports,
        }
    }
    pub fn model_locator(&self) -> Option<&str> {
        match &self.wire {
            ServeProcessAllocation::ModelRank { model_locator, .. } => Some(model_locator),
            ServeProcessAllocation::Frontend { .. } => None,
        }
    }
}

#[derive(Clone)]
pub struct LoweringEvidence {
    request_sha256: String,
    response_sha256: String,
    timing: OperationTimingEvidence,
}

impl LoweringEvidence {
    pub const fn new(
        request_sha256: String,
        response_sha256: String,
        timing: OperationTimingEvidence,
    ) -> Self {
        Self {
            request_sha256,
            response_sha256,
            timing,
        }
    }
    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }
    pub fn response_sha256(&self) -> &str {
        &self.response_sha256
    }
    pub const fn timing(&self) -> &OperationTimingEvidence {
        &self.timing
    }
}

pub struct PlannedServeStage {
    planned: PlanServeResult,
    evidence: LoweringEvidence,
    requirements: Vec<ProcessRequirement>,
    integration_rendered_process_ids: BTreeSet<String>,
    public_process: String,
}

impl PlannedServeStage {
    pub const fn new(
        planned: PlanServeResult,
        evidence: LoweringEvidence,
        requirements: Vec<ProcessRequirement>,
        integration_rendered_process_ids: BTreeSet<String>,
        public_process: String,
    ) -> Self {
        Self {
            planned,
            evidence,
            requirements,
            integration_rendered_process_ids,
            public_process,
        }
    }
    pub const fn planned(&self) -> &PlanServeResult {
        &self.planned
    }
    pub const fn evidence(&self) -> &LoweringEvidence {
        &self.evidence
    }
    pub fn requirements(&self) -> &[ProcessRequirement] {
        &self.requirements
    }
    pub const fn integration_rendered_process_ids(&self) -> &BTreeSet<String> {
        &self.integration_rendered_process_ids
    }
    pub fn public_process(&self) -> &str {
        &self.public_process
    }
}

pub struct RenderedServeStage {
    evidence: LoweringEvidence,
    allocations: Vec<ResolvedProcessAllocation>,
    rendered_processes: Vec<RenderedServeProcess>,
}

impl RenderedServeStage {
    pub const fn new(
        evidence: LoweringEvidence,
        allocations: Vec<ResolvedProcessAllocation>,
        rendered_processes: Vec<RenderedServeProcess>,
    ) -> Self {
        Self {
            evidence,
            allocations,
            rendered_processes,
        }
    }
    pub const fn evidence(&self) -> &LoweringEvidence {
        &self.evidence
    }
    pub fn allocations(&self) -> &[ResolvedProcessAllocation] {
        &self.allocations
    }
    pub fn rendered_processes(&self) -> &[RenderedServeProcess] {
        &self.rendered_processes
    }
}

pub struct RuntimeRealizationStage {
    processes: Vec<ProcessPlan>,
    public_endpoint: EndpointPlan,
    device_count: u32,
    selected_machines: Vec<String>,
    network: Option<NetworkPlan>,
    remote_workspaces: BTreeMap<String, RemoteWorkspacePlan>,
    remote_containers: BTreeMap<String, RemoteContainerFacts>,
}

pub struct RuntimeRealizationParts {
    pub processes: Vec<ProcessPlan>,
    pub public_endpoint: EndpointPlan,
    pub device_count: u32,
    pub selected_machines: Vec<String>,
    pub network: Option<NetworkPlan>,
    pub remote_workspaces: BTreeMap<String, RemoteWorkspacePlan>,
    pub remote_containers: BTreeMap<String, RemoteContainerFacts>,
}

impl RuntimeRealizationStage {
    pub fn new(parts: RuntimeRealizationParts) -> Self {
        Self {
            processes: parts.processes,
            public_endpoint: parts.public_endpoint,
            device_count: parts.device_count,
            selected_machines: parts.selected_machines,
            network: parts.network,
            remote_workspaces: parts.remote_workspaces,
            remote_containers: parts.remote_containers,
        }
    }
    pub fn into_parts(self) -> RuntimeRealizationParts {
        RuntimeRealizationParts {
            processes: self.processes,
            public_endpoint: self.public_endpoint,
            device_count: self.device_count,
            selected_machines: self.selected_machines,
            network: self.network,
            remote_workspaces: self.remote_workspaces,
            remote_containers: self.remote_containers,
        }
    }
}
