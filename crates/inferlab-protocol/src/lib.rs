mod wire;

use schemars::generate::SchemaSettings;

pub use wire::{
    AdapterError, AdapterErrorCode, AdapterProtocol, AdapterRequest, AdapterResponse,
    AdapterResult, AllocationLaunch, BenchCaseInput, BenchClientRequest, BenchClientResult,
    BenchDatasetCacheState, BenchDatasetCatalogInput, BenchDatasetInput,
    BenchDatasetPreparationRequest, BenchDatasetPreparationResult, BenchDefinitionInput,
    BenchLoadInput, BenchPopulationInput, BenchRequestSloInput, BenchRequestSloResult,
    BenchRequestSourceInput, BenchTokenCountSummary, CaptureTargetRequirement,
    CaptureWindowControlEndpoint, CaptureWindowControlRequirement, ClientEndpointInput,
    ClientStatus, EndpointAssignment, EndpointProtocol, EndpointRequirement, EvalClientRequest,
    EvalClientResult, EvalDefinitionInput, EvalFailureKind, EvalMetricComparison, EvalMetricGate,
    EvalMetricGateConclusion, EvalNormalizedMetric, EvalTaskSourceInput, EvalTrialSummary,
    FrontendCoRendering, FrontendComponents, FrontendGatewayComponent, FrontendHandoff,
    FrontendPdRouterComponent, FrontendProcessRole, GatewayFrontendBinding,
    GatewayPdRouterFrontendBinding, GatewayPlan, GatewayTarget, HttpActionSpec, HttpMethod,
    HttpTargetRegistryReadiness, IntegrationIdentity, KvTransferMechanism, LaunchFileDeclaration,
    MeasurementModelInput, Parallelism, ParallelismAttention, ParallelismExperts, ParallelismOuter,
    PdRouterPlan, PdRoutingPolicies, PlanServeInput, PlanServeResult, ProcessSpec, ProtocolVersion,
    RawArtifact, ReadinessProbe, RenderInputDeclaration, RenderServeInput, RenderServeResult,
    RenderSource, RenderedServeProcess, ServeModelInput, ServeProcessAllocation,
    ServeReplicaRequirement, ServeRoleInput, ServeRoleKind, ServeRoleLink, ServeRoleResult,
    ServeTopology, SettingValue, SuppliedRenderInput, TargetEndpointScheme,
};

pub const PROTOCOL_SCHEMA_ID: &str = "https://inferlab.dev/schema/adapter-protocol/v7";
pub const PROTOCOL_WIRE_SOURCE: &str = "crates/inferlab-protocol/src/wire.rs";

#[must_use]
pub fn protocol_schema() -> schemars::Schema {
    let mut schema = SchemaSettings::draft2020_12()
        .for_deserialize()
        .into_generator()
        .into_root_schema_for::<AdapterProtocol>();
    schema
        .ensure_object()
        .insert("$id".to_owned(), PROTOCOL_SCHEMA_ID.into());
    schema.ensure_object().insert(
        "$comment".to_owned(),
        format!("Generated from {PROTOCOL_WIRE_SOURCE}; do not edit.").into(),
    );
    schema
}
