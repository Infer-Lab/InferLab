mod wire;

use schemars::generate::SchemaSettings;

pub use wire::{
    AdapterError, AdapterErrorCode, AdapterProtocol, AdapterRequest, AdapterResponse,
    AdapterResult, AllocationLaunch, BenchAgenticAcquisitionOutcome, BenchAgenticBranchStats,
    BenchAgenticCatalogInput, BenchAgenticResultEvidence, BenchAgenticRunEvidence,
    BenchAgenticSourceInput, BenchAgenticSourceVerification, BenchCaseInput, BenchClientRequest,
    BenchClientResult, BenchDatasetCacheState, BenchDatasetCatalogInput, BenchDatasetFilterInput,
    BenchDefinitionInput, BenchInclusiveUniformInput, BenchLoadInput, BenchNativeInvocation,
    BenchPopulationInput, BenchPopulationPreparationRequest, BenchPopulationPreparationResult,
    BenchPrefixGeometrySummary, BenchPrefixSharingInput, BenchPromptInput, BenchPromptRouteInput,
    BenchPromptTemplateProjection, BenchPromptTemplateSource, BenchPromptTokenReconciliation,
    BenchPromptTokenTargetingSummary, BenchRandomShapeInput, BenchRenderingAuthorityInput,
    BenchRequestRepresentationInput, BenchRequestSloInput, BenchRequestSloResult,
    BenchRequestSourceInput, BenchRuntimeSessionResult, BenchSessionDatasetCatalogInput,
    BenchSessionPhaseSummary, BenchSessionResultEvidence, BenchSessionSourceInput,
    BenchSessionTemplateInput, BenchSessionTurnResult, BenchSharedSystemContentInput,
    BenchSharedSystemContentSummary, BenchTokenCountSummary, BenchTokenDistributionKindInput,
    BenchTokenSelectorInput, CaptureTargetRequirement, CaptureWindowControlEndpoint,
    CaptureWindowControlRequirement, CaptureWindowHttpActionSpec, ClientEndpointInput,
    ClientStatus, EndpointAssignment, EndpointProtocol, EndpointRequirement, EvalClientRequest,
    EvalClientResult, EvalDefinitionInput, EvalFailureKind, EvalMetricComparison, EvalMetricGate,
    EvalMetricGateConclusion, EvalNormalizedMetric, EvalTaskSourceInput, EvalTrialSummary,
    FrontendCoRendering, FrontendComponents, FrontendGatewayComponent, FrontendHandoff,
    FrontendPdRouterComponent, FrontendProcessRole, GatewayFrontendBinding,
    GatewayPdRouterFrontendBinding, GatewayPlan, GatewayTarget, HttpActionSpec, HttpMethod,
    HttpTargetRegistryReadiness, IntegrationIdentity, KvTransferMechanism, LaunchFileDeclaration,
    MeasurementModelInput, MeasurementProtocol, Parallelism, ParallelismAttention,
    ParallelismExperts, ParallelismOuter, PdRouterPlan, PdRoutingPolicies, PlanServeInput,
    PlanServeResult, ProcessSpec, ProtocolVersion, RawArtifact, ReadinessProbe,
    RenderInputDeclaration, RenderServeInput, RenderServeResult, RenderSource,
    RenderedServeProcess, ServeModelInput, ServeProcessAllocation, ServeReplicaRequirement,
    ServeRoleInput, ServeRoleKind, ServeRoleLink, ServeRoleResult, ServeTopology,
    ServerMetricsEndpointInput, ServerMetricsEndpointRequirement, SettingValue,
    SuppliedRenderInput, TargetEndpointScheme,
};

pub const PROTOCOL_SCHEMA_ID: &str = "https://inferlab.dev/schema/adapter-protocol/v7";
pub const MEASUREMENT_SCHEMA_ID: &str = "https://inferlab.dev/schema/measurement-protocol/v1";
pub const PROTOCOL_WIRE_SOURCE: &str = "crates/inferlab-protocol/src/wire.rs";

#[must_use]
pub fn protocol_schema() -> schemars::Schema {
    schema_for::<AdapterProtocol>(PROTOCOL_SCHEMA_ID)
}

#[must_use]
pub fn measurement_schema() -> schemars::Schema {
    schema_for::<MeasurementProtocol>(MEASUREMENT_SCHEMA_ID)
}

fn schema_for<T: schemars::JsonSchema>(id: &str) -> schemars::Schema {
    let mut schema = SchemaSettings::draft2020_12()
        .for_deserialize()
        .into_generator()
        .into_root_schema_for::<T>();
    schema.ensure_object().insert("$id".to_owned(), id.into());
    schema.ensure_object().insert(
        "$comment".to_owned(),
        format!("Generated from {PROTOCOL_WIRE_SOURCE}; do not edit.").into(),
    );
    schema
}
