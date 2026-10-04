"""Planning and rendering for the shared token-only Specialized Engine contract."""

from typing import Annotated

import inferlab_gateway_smg
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    CaptureMechanism,
    CaptureTargetRequirement,
    CaptureWindowControlEndpoint,
    CaptureWindowControlRequirement,
    CaptureWindowHttpActionSpec,
    EndpointDeclaration,
    EndpointProtocol,
    FrontendCoRendering,
    FrontendGatewayComponent,
    FrontendProcessRole,
    GatewayFrontendBinding,
    GatewayPlan,
    GatewayTarget,
    GatewayTargetEngine,
    HttpActionSpec,
    HttpMethod,
    IntegrationIdentity,
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PlanServeInput,
    PlanServeResult,
    ProcessSpec,
    PromptCacheReadZeroRepresentation,
    ReadinessProbe,
    ReadinessProbeProcessAlive,
    RenderedServeProcess,
    RenderServeInput,
    RenderServeResult,
    RenderSource,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
    ServeReplicaRequirement,
    ServerMetricsEndpointRequirement,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkRequestRouting,
    ServeRoleResult,
    ServeTopology,
    SettingValue,
    effective_settings,
    integration_identity,
    merge_serve_args,
    rendered_frontend,
    rendered_model_rank,
    replica_id,
    require_role,
    split_serve_allocations,
    validate_extra_args,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .auxiliary import reject_auxiliary_locators, validate_auxiliary_models

_ADAPTER_DISTRIBUTION = "inferlab-integration-specialized-engine"

# Flags the managed Engine argv owns; the escape hatch must not restate them.
_INFERLAB_OWNED_OPTIONS: set[str] = {
    "--default-max-output-tokens",
    "--gpu-memory-utilization-percent",
    "--listen",
    "--max-num-batched-tokens",
    "--model",
    "--prefix-cache-cpu-bytes-per-rank",
    "--prefix-cache-gpu-entries",
    "--prefix-cache-host-memory-percent",
    "--prefix-cache-numa-node-per-rank",
    "--served-model-name",
    "--tensor-parallel-size",
    "--workspace-reserve-mib",
}


class PrefixCacheRank(BaseModel):
    """Explicit host prefix-cache placement for one tensor-parallel rank.

    Pairing bytes with their NUMA node in one entry keeps the worker's two
    positionally matched argument lists from drifting apart in configuration.
    """

    model_config = ConfigDict(extra="forbid")

    cpu_bytes: Annotated[int, Field(ge=1)]
    numa_node: Annotated[int, Field(ge=0)]


class EngineContractSettings(BaseModel):
    """Settings shared by every implementation of the token Engine contract.

    An omitted optional setting renders no worker argument, so the worker's own
    default governs and InferLab does not restate it. `extra_args` and
    `extra_env` carry implementation-specific worker knobs whose exact
    effective contents are returned and recorded ([[ADR-0048]]).
    """

    model_config = ConfigDict(extra="forbid")

    default_max_output_tokens: Annotated[int, Field(ge=1)] | None = None
    max_num_batched_tokens: Annotated[int, Field(ge=1)] | None = None
    gpu_memory_utilization_percent: Annotated[int, Field(ge=1, le=100)] | None = None
    workspace_reserve_mib: Annotated[int, Field(ge=0)] | None = None
    prefix_cache_gpu_entries: Annotated[int, Field(ge=1)] | None = None
    prefix_cache_host_memory_percent: Annotated[int, Field(ge=1, le=100)] | None = None
    prefix_cache_ranks: list[PrefixCacheRank] | None = None
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None

    @model_validator(mode="after")
    def _one_host_prefix_cache_authority(self) -> "EngineContractSettings":
        if self.prefix_cache_ranks is None:
            return self
        if not self.prefix_cache_ranks:
            raise ValueError("prefix_cache_ranks must declare at least one rank when present")
        if self.prefix_cache_host_memory_percent is not None:
            # The worker ignores the percent once an explicit list sizes the
            # host cache, so accepting both would record a value that did not
            # participate in the capacity it appears to describe.
            raise ValueError(
                "prefix_cache_host_memory_percent does not size the host cache when "
                "prefix_cache_ranks is declared; declare exactly one host sizing authority"
            )
        return self


def _settings(values: dict[str, SettingValue]) -> EngineContractSettings:
    settings = validate_settings(EngineContractSettings, values)
    validate_extra_args(settings.extra_args or [], _INFERLAB_OWNED_OPTIONS)
    return settings


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-specialized-engine",
        adapter_distribution=_ADAPTER_DISTRIBUTION,
        framework="specialized-engine",
        framework_distribution=_ADAPTER_DISTRIBUTION,
    )


def _pure_tp_parallelism(
    parallelism: Parallelism,
    error_code: AdapterErrorCode = AdapterErrorCode.invalid_settings,
) -> tuple[Parallelism, int]:
    outer = parallelism.outer
    tensor_parallel_size = (
        outer.tensor_parallel_size
        if outer is not None and outer.tensor_parallel_size is not None
        else 1
    )
    non_tp_values = [
        outer.pipeline_parallel_size if outer is not None else None,
        (parallelism.attention.data_parallel_size if parallelism.attention is not None else None),
        (
            parallelism.attention.context_parallel_size
            if parallelism.attention is not None
            else None
        ),
        (parallelism.experts.data_parallel_size if parallelism.experts is not None else None),
        (parallelism.experts.expert_parallel_size if parallelism.experts is not None else None),
    ]
    if any(value not in {None, 1} for value in non_tp_values):
        raise AdapterOperationError(
            error_code,
            "the Specialized Engine contract supports only tensor parallelism",
        )

    component_tp_values = [
        (parallelism.attention.tensor_parallel_size if parallelism.attention is not None else None),
        (parallelism.experts.tensor_parallel_size if parallelism.experts is not None else None),
        (
            parallelism.experts.dense_tensor_parallel_size
            if parallelism.experts is not None
            else None
        ),
    ]
    if any(value is not None and value != tensor_parallel_size for value in component_tp_values):
        raise AdapterOperationError(
            error_code,
            "attention and expert tensor parallelism must match outer tensor parallelism",
        )

    effective = Parallelism(
        outer=ParallelismOuter(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
        ),
        attention=ParallelismAttention(
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=1,
            context_parallel_size=1,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=1,
            expert_parallel_size=1,
            dense_tensor_parallel_size=tensor_parallel_size,
        ),
    )
    return effective, tensor_parallel_size


def _public_endpoint() -> EndpointDeclaration:
    return EndpointDeclaration(
        protocol=EndpointProtocol(),
        server_metrics=ServerMetricsEndpointRequirement(path="/metrics", port="prometheus"),
        prefix_cache_reset=HttpActionSpec(method=HttpMethod(), path="/flush_cache"),
        # SMG serves the conditioning fan-out at the admin router next to
        # flush_cache, so a multi-target primed start can plan through the
        # declared capability.
        prefix_cache_conditioning=HttpActionSpec(method=HttpMethod(), path="/prime_prefix_cache"),
        # The worker protocol carries an unconditional cached-token count, so a
        # zero cache read is reported rather than omitted.
        prompt_cache_read_zero_representation=PromptCacheReadZeroRepresentation.explicit,
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    """Plan one token Engine behind one SMG Gateway."""
    validate_auxiliary_models(input)
    if input.topology != ServeTopology.single:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration supports only single topology",
        )
    inferlab_gateway_smg.require_backend(input.gateway_backend, component="Gateway")
    if input.pd_router_backend is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine routed-single workflow must not select a P/D Router",
        )
    if input.kv_transfer is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "single topology does not use KV transfer",
        )
    role = require_role(input, ServeRoleKind.serve)
    if role.replica_count != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration supports exactly one replica",
        )
    settings = _settings(role.settings)
    parallelism, tensor_parallel_size = _pure_tp_parallelism(role.parallelism)
    mechanism = input.profiling
    if mechanism == CaptureMechanism.engine_trace:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration does not support engine-trace capture",
        )
    if input.synthetic_acceptance is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration cannot apply the synthetic acceptance overlay",
        )
    role_result = ServeRoleResult(
        id=role.id,
        kind=role.kind,
        declared_replica_count=role.replica_count,
        effective_replica_count=role.replica_count,
        effective_settings=effective_settings(settings),
        effective_parallelism=parallelism,
        public_endpoint=None,
    )
    gateway = GatewayPlan(
        backend=inferlab_gateway_smg.BACKEND,
        implementation=inferlab_gateway_smg.IMPLEMENTATION,
        implementation_version=inferlab_gateway_smg.installed_version(),
        effective_settings={
            "worker_protocol": SettingValue(root="tokenspeed.grpc.scheduler.TokenSpeedScheduler"),
            "policy": SettingValue(root="least_load"),
            **inferlab_gateway_smg.gateway_settings(),
        },
        endpoint=_public_endpoint(),
        readiness=inferlab_gateway_smg.gateway_readiness(),
        ports=["prometheus"],
        targets=[GatewayTarget(root=GatewayTargetEngine(role=role.id))],
        render_inputs=[],
        render_source=RenderSource.integration,
        co_rendering=FrontendCoRendering(process_role=FrontendProcessRole()),
    )
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=[
            ServeReplicaRequirement(
                id=replica_id(role, 0),
                role_id=role.id,
                replica_index=0,
                device_count=tensor_parallel_size,
                ports=[],
                primary_ports=[],
                primary_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
                worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
                capture_target=(
                    CaptureTargetRequirement(
                        mechanism=CaptureMechanism.managed_collection,
                        window_control=CaptureWindowControlRequirement(
                            endpoint=CaptureWindowControlEndpoint.gateway,
                            start=CaptureWindowHttpActionSpec(
                                method=HttpMethod(), path="/start_profile"
                            ),
                            stop=CaptureWindowHttpActionSpec(
                                method=HttpMethod(), path="/stop_profile"
                            ),
                        ),
                    )
                    if mechanism is not None
                    else None
                ),
            )
        ],
        links=[
            ServeRoleLink(root=ServeRoleLinkRequestRouting(source="gateway", targets=[role.id]))
        ],
        gateway=gateway,
        pd_router=None,
    )


def _require_engine(
    allocations: list[ServeProcessAllocationModelRank],
) -> ServeProcessAllocationModelRank:
    if len(allocations) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Specialized Engine integration requires one model-rank allocation",
        )
    engine = allocations[0]
    if (
        engine.role_kind != ServeRoleKind.serve
        or engine.replica != 0
        or engine.rank != 0
        or engine.rank_count != 1
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine allocation must be serve replica 0 in one rank process",
        )
    effective_parallelism, tensor_parallel_size = _pure_tp_parallelism(
        engine.effective_parallelism,
        AdapterErrorCode.invalid_request,
    )
    if (
        engine.effective_parallelism != effective_parallelism
        or len(engine.devices) != tensor_parallel_size
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine rank process must own one device per effective tensor-parallel rank",
        )
    if engine.endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine allocation is missing its endpoint",
        )
    return engine


def _require_gateway(allocation: object, engine_role: str) -> ServeProcessAllocationFrontend:
    if not isinstance(allocation, ServeProcessAllocationFrontend):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the routed-single workflow requires one Gateway frontend allocation",
        )
    if not isinstance(allocation.components.root, GatewayFrontendBinding):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the routed-single frontend must bind only [gateway]",
        )
    if allocation.components.root.root != [FrontendGatewayComponent()]:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the routed-single frontend must bind only [gateway]",
        )
    gateway = allocation.gateway
    if (
        gateway.backend != inferlab_gateway_smg.BACKEND
        or gateway.implementation != inferlab_gateway_smg.IMPLEMENTATION
        or gateway.implementation_version != inferlab_gateway_smg.installed_version()
        or gateway.render_source != RenderSource.integration
        or allocation.pd_router is not None
        or allocation.process_role != gateway.co_rendering.process_role
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the frontend allocation does not preserve the planned SMG Gateway",
        )
    targets = gateway.targets
    if (
        len(targets) != 1
        or not isinstance(targets[0].root, GatewayTargetEngine)
        or targets[0].root.role != engine_role
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the SMG Gateway must target the sole Engine role",
        )
    return allocation


def _render_engine(
    input: RenderServeInput,
    allocation: ServeProcessAllocationModelRank,
) -> RenderedServeProcess:
    reject_auxiliary_locators(allocation)
    endpoint = allocation.endpoint
    if endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine allocation is missing its endpoint",
        )
    settings = _settings(allocation.effective_settings)
    _, tensor_parallel_size = _pure_tp_parallelism(
        allocation.effective_parallelism,
        AdapterErrorCode.invalid_request,
    )
    if (
        settings.prefix_cache_ranks is not None
        and len(settings.prefix_cache_ranks) != tensor_parallel_size
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"prefix_cache_ranks declares {len(settings.prefix_cache_ranks)} ranks "
            f"but the resolved tensor-parallel size is {tensor_parallel_size}",
        )
    inferlab_args = [
        "--listen",
        f"{endpoint.host}:{endpoint.port}",
        "--model",
        allocation.model_locator,
        "--served-model-name",
        input.model.served_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    for option, value in (
        ("--default-max-output-tokens", settings.default_max_output_tokens),
        ("--max-num-batched-tokens", settings.max_num_batched_tokens),
        ("--gpu-memory-utilization-percent", settings.gpu_memory_utilization_percent),
        ("--workspace-reserve-mib", settings.workspace_reserve_mib),
        ("--prefix-cache-gpu-entries", settings.prefix_cache_gpu_entries),
        ("--prefix-cache-host-memory-percent", settings.prefix_cache_host_memory_percent),
    ):
        if value is not None:
            inferlab_args.extend([option, str(value)])
    # The worker pairs these two lists by occurrence order, so each is emitted
    # once per rank in rank order.
    for rank in settings.prefix_cache_ranks or ():
        inferlab_args.extend(["--prefix-cache-cpu-bytes-per-rank", str(rank.cpu_bytes)])
    for rank in settings.prefix_cache_ranks or ():
        inferlab_args.extend(["--prefix-cache-numa-node-per-rank", str(rank.numa_node)])
    argv = ["inferlab-token-engine", "smg-worker"]
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, _INFERLAB_OWNED_OPTIONS))
    return rendered_model_rank(
        allocation, ProcessSpec(argv=argv, env=dict(settings.extra_env or {}))
    )


def _render_gateway(
    allocation: ServeProcessAllocationFrontend,
    engine: ServeProcessAllocationModelRank,
) -> RenderedServeProcess:
    engine_endpoint = engine.endpoint
    if engine_endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine allocation is missing its endpoint",
        )
    prometheus = allocation.ports.get("prometheus")
    if prometheus is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the SMG Gateway allocation is missing its Prometheus port",
        )
    return rendered_frontend(
        allocation,
        inferlab_gateway_smg.routed_single_command(
            endpoint=allocation.endpoint,
            prometheus=prometheus,
            model_locator=engine.model_locator,
            worker=engine_endpoint,
            policy="least_load",
        ),
    )


def render_serve(input: RenderServeInput) -> RenderServeResult:
    if (
        input.topology != ServeTopology.single
        or input.gateway_backend != inferlab_gateway_smg.BACKEND
        or input.pd_router_backend is not None
        or input.kv_transfer is not None
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "render input is not the planned routed-single SMG workflow",
        )
    allocations, model_allocations = split_serve_allocations(input.allocations)
    if len(allocations) != 2:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the routed-single workflow requires one Engine and one Gateway allocation",
        )
    engine = _require_engine(model_allocations)
    frontend_candidates = [
        allocation
        for allocation in allocations
        if isinstance(allocation, ServeProcessAllocationFrontend)
    ]
    if len(frontend_candidates) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the routed-single workflow requires one Gateway allocation",
        )
    _require_gateway(frontend_candidates[0], engine.role)

    processes: list[RenderedServeProcess] = []
    for allocation in allocations:
        if isinstance(allocation, ServeProcessAllocationModelRank):
            processes.append(_render_engine(input, allocation))
        elif isinstance(allocation, ServeProcessAllocationFrontend):
            processes.append(_render_gateway(allocation, engine))
    return RenderServeResult(integration=_identity(), processes=processes)
