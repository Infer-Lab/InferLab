"""Planning and rendering for the shared token-only Specialized Engine contract."""

from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    CaptureTargetRequirement,
    CaptureWindowControlEndpoint,
    CaptureWindowControlRequirement,
    CaptureWindowHttpActionSpec,
    EndpointProtocol,
    EndpointRequirement,
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
    ReadinessProbeHttp,
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
    rendered_frontend,
    rendered_model_rank,
    replica_id,
    require_role,
    split_serve_allocations,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ADAPTER_DISTRIBUTION = "inferlab-integration-specialized-engine"
_GATEWAY_BACKEND = "smg"
_GATEWAY_IMPLEMENTATION = "tokenspeed-smg"
_DEFERRED_WORKER_STARTUP_TIMEOUT_SECS = 2_147_483_647


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
    default governs and InferLab does not restate it.
    """

    model_config = ConfigDict(extra="forbid")

    default_max_output_tokens: Annotated[int, Field(ge=1)] = 16
    max_num_batched_tokens: Annotated[int, Field(ge=1)] = 12_288
    gpu_memory_utilization_percent: Annotated[int, Field(ge=1, le=100)] | None = None
    workspace_reserve_mib: Annotated[int, Field(ge=0)] | None = None
    prefix_cache_gpu_entries: Annotated[int, Field(ge=1)] | None = None
    prefix_cache_host_memory_percent: Annotated[int, Field(ge=1, le=100)] | None = None
    prefix_cache_ranks: list[PrefixCacheRank] | None = None

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


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-specialized-engine",
        adapter_distribution=_ADAPTER_DISTRIBUTION,
        framework="specialized-engine",
        framework_distribution=_ADAPTER_DISTRIBUTION,
    )


def _smg_version() -> str:
    try:
        return version("tokenspeed-smg")
    except PackageNotFoundError:
        return "unavailable"


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


def _public_endpoint() -> EndpointRequirement:
    return EndpointRequirement(
        protocol=EndpointProtocol(),
        completions_path="/v1/completions",
        chat_completions_path="/v1/chat/completions",
        server_metrics=ServerMetricsEndpointRequirement(path="/metrics", port="prometheus"),
        prefix_cache_reset=HttpActionSpec(method=HttpMethod(), path="/flush_cache"),
        # The worker protocol carries an unconditional cached-token count, so a
        # zero cache read is reported rather than omitted.
        prompt_cache_read_zero_representation=PromptCacheReadZeroRepresentation.explicit,
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    """Plan one token Engine behind one SMG Gateway."""
    if input.topology != ServeTopology.single:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration supports only single topology",
        )
    if input.gateway_backend != _GATEWAY_BACKEND:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration requires Gateway backend smg",
        )
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
    settings = validate_settings(EngineContractSettings, role.settings)
    parallelism, tensor_parallel_size = _pure_tp_parallelism(role.parallelism)
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
        backend=_GATEWAY_BACKEND,
        implementation=_GATEWAY_IMPLEMENTATION,
        implementation_version=_smg_version(),
        effective_settings={
            "worker_protocol": SettingValue(root="tokenspeed_scheduler_v1"),
            "policy": SettingValue(root="least_load"),
            "retries": SettingValue(root=False),
            "circuit_breaker": SettingValue(root=False),
        },
        endpoint=_public_endpoint(),
        readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/readiness")),
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
                        window_control=CaptureWindowControlRequirement(
                            endpoint=CaptureWindowControlEndpoint.gateway,
                            start=CaptureWindowHttpActionSpec(
                                method=HttpMethod(), path="/start_profile"
                            ),
                            stop=CaptureWindowHttpActionSpec(
                                method=HttpMethod(), path="/stop_profile"
                            ),
                        )
                    )
                    if input.profiling
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
        gateway.backend != _GATEWAY_BACKEND
        or gateway.implementation != _GATEWAY_IMPLEMENTATION
        or gateway.implementation_version != _smg_version()
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
    endpoint = allocation.endpoint
    if endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine allocation is missing its endpoint",
        )
    settings = validate_settings(EngineContractSettings, allocation.effective_settings)
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
    argv = [
        "inferlab-token-engine",
        "smg-worker",
        "--listen",
        f"{endpoint.host}:{endpoint.port}",
        "--model",
        allocation.model_locator,
        "--served-model-name",
        input.model.served_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--default-max-output-tokens",
        str(settings.default_max_output_tokens),
        "--max-num-batched-tokens",
        str(settings.max_num_batched_tokens),
    ]
    for option, value in (
        ("--gpu-memory-utilization-percent", settings.gpu_memory_utilization_percent),
        ("--workspace-reserve-mib", settings.workspace_reserve_mib),
        ("--prefix-cache-gpu-entries", settings.prefix_cache_gpu_entries),
        ("--prefix-cache-host-memory-percent", settings.prefix_cache_host_memory_percent),
    ):
        if value is not None:
            argv.extend([option, str(value)])
    # The worker pairs these two lists by occurrence order, so each is emitted
    # once per rank in rank order.
    for rank in settings.prefix_cache_ranks or ():
        argv.extend(["--prefix-cache-cpu-bytes-per-rank", str(rank.cpu_bytes)])
    for rank in settings.prefix_cache_ranks or ():
        argv.extend(["--prefix-cache-numa-node-per-rank", str(rank.numa_node)])
    return rendered_model_rank(allocation, ProcessSpec(argv=argv, env={}))


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
        ProcessSpec(
            argv=[
                "smg",
                "launch",
                "--host",
                "0.0.0.0",
                "--port",
                str(allocation.endpoint.port),
                "--prometheus-port",
                str(prometheus.port),
                "--worker-startup-timeout-secs",
                str(_DEFERRED_WORKER_STARTUP_TIMEOUT_SECS),
                "--worker-urls",
                f"grpc://{engine_endpoint.host}:{engine_endpoint.port}",
                "--model-path",
                engine.model_locator,
                "--tokenizer-path",
                engine.model_locator,
                "--policy",
                "least_load",
                "--disable-retries",
                "--disable-circuit-breaker",
            ],
            env={},
        ),
    )


def render_serve(input: RenderServeInput) -> RenderServeResult:
    if (
        input.topology != ServeTopology.single
        or input.gateway_backend != _GATEWAY_BACKEND
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
