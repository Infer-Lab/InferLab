"""Planning and rendering for the shared token-only Specialized Engine contract."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Annotated

import inferlab_gateway_smg
import inferlab_kvstore
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
    HttpMethod,
    IntegrationIdentity,
    KvStoreAccess,
    KvStorePlan,
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PdRoutingPolicies,
    PlanServeInput,
    PlanServeResult,
    ProbeServeInput,
    ProbeServeProcessState,
    ProbeServeResult,
    ProcessSpec,
    PromptCacheReadZeroRepresentation,
    ReadinessProbe,
    ReadinessProbeProcessAlive,
    RenderedServeProcess,
    RenderServeInput,
    RenderServeResult,
    RenderSource,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationKvStore,
    ServeProcessAllocationModelRank,
    ServeReplicaRequirement,
    ServerMetricsEndpointRequirement,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkKvTransfer,
    ServeRoleLinkRequestRouting,
    ServeRoleResult,
    ServeTopology,
    SettingValue,
    effective_settings,
    fill_outer_tensor_parallel,
    integration_identity,
    merge_serve_args,
    rendered_frontend,
    rendered_model_rank,
    replica_id,
    require_declared_devices,
    require_role,
    split_serve_allocations,
    validate_extra_args,
    validate_settings,
)
from inferlab_kvstore import mooncake
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .auxiliary import reject_auxiliary_locators, validate_auxiliary_models

_ADAPTER_DISTRIBUTION = "inferlab-integration-specialized-engine"

# Flags the managed Engine argv owns; the escape hatch must not restate them.
_INFERLAB_OWNED_OPTIONS: set[str] = {
    "--context-parallel-size",
    "--data-parallel-size",
    "--default-max-output-tokens",
    "--disaggregation-bootstrap-port",
    "--disaggregation-mode",
    "--disaggregation-transfer-backend",
    "--expert-parallel-size",
    "--gpu-memory-utilization-percent",
    "--kv-tier-capacity",
    "--kv-tier-snapshot-interval",
    "--kv-tier-store",
    "--kv-tier-timeout",
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
    # Tier settings apply only with a declared KV store
    # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    kv_tier_capacity_bytes: Annotated[int, Field(ge=0)] | None = None
    kv_tier_snapshot_interval: Annotated[int, Field(ge=2)] | None = None
    kv_tier_timeout_seconds: Annotated[float, Field(gt=0)] | None = None
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


class GatewaySettings(BaseModel):
    """The typed SMG Gateway settings a server may declare
    ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""

    model_config = ConfigDict(extra="forbid")

    policy: inferlab_gateway_smg.Policy = "least_load"


class PdRouterSettings(BaseModel):
    """The typed SMG P/D Router settings a prefill/decode server may declare
    ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""

    model_config = ConfigDict(extra="forbid")

    prefill_policy: inferlab_gateway_smg.Policy = "round_robin"
    decode_policy: inferlab_gateway_smg.Policy = "round_robin"


_WORKER_PROTOCOL = "tokenspeed.grpc.scheduler.TokenSpeedScheduler"


class StackSettings(BaseModel):
    """Facts of the Engine implementation the stack realizes
    ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""

    model_config = ConfigDict(extra="forbid")

    # The leading argv the implementation defines: a bare executable name on
    # the stack environment's PATH, then its fixed arguments. The contract
    # arguments follow it. The default is the former fixed command, so stacks
    # written for it keep working.
    engine_command: list[str] = Field(
        default_factory=lambda: ["inferlab-token-engine", "smg-worker"]
    )
    # How long after a request completes a prefix the Engine wrote becomes
    # readable from the KV tier; a primed start waits it after conditioning
    # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    kv_tier_publication_seconds: Annotated[float, Field(ge=0)] = 0.0

    @model_validator(mode="after")
    def _bare_executable(self) -> "StackSettings":
        if not self.engine_command or not self.engine_command[0]:
            raise ValueError("engine_command must name the Engine executable")
        if "/" in self.engine_command[0]:
            # A path is a machine-local fact; the executable resolves on the
            # stack environment's PATH.
            raise ValueError(
                f"engine_command executable {self.engine_command[0]!r} must be a bare "
                "command name resolved on the stack environment's PATH, not a path"
            )
        return self


def stack_settings(values: dict[str, SettingValue]) -> dict[str, SettingValue]:
    settings = validate_settings(StackSettings, values)
    _reject_owned_options("engine_command", settings.engine_command)
    return effective_settings(settings)


def _reject_owned_options(field: str, arguments: list[str]) -> None:
    for argument in arguments:
        name, _separator, _value = argument.partition("=")
        if name in _INFERLAB_OWNED_OPTIONS:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                f"{field} entry {argument!r} names contract-owned option {name!r}",
            )


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


@dataclass(frozen=True)
class _Layout:
    """The rank process's world and how the contract splits it."""

    world: int
    attention_tensor_parallel_size: int
    data_parallel_size: int
    context_parallel_size: int
    expert_parallel_size: int


def _layout(
    parallelism: Parallelism,
    error_code: AdapterErrorCode = AdapterErrorCode.invalid_settings,
) -> tuple[Parallelism, _Layout]:
    """The contract's reading: the device count is outer tensor parallelism;
    attention tensor x data x context parallelism multiply to it and expert
    parallelism divides it ([[RFC-0003:C-SPECIALIZED-ENGINE]]). Only this
    arithmetic is checked; the Engine rejects layouts it does not support."""
    outer = parallelism.outer or ParallelismOuter()
    attention = parallelism.attention or ParallelismAttention()
    experts = parallelism.experts or ParallelismExperts()
    world = outer.tensor_parallel_size or 1
    if (outer.pipeline_parallel_size or 1) != 1 or (experts.data_parallel_size or 1) != 1:
        raise AdapterOperationError(
            error_code,
            "the Specialized Engine contract has no pipeline or expert data parallelism",
        )
    data = attention.data_parallel_size or 1
    context = attention.context_parallel_size or 1
    if world % (data * context) != 0:
        raise AdapterOperationError(
            error_code,
            f"attention data x context parallelism ({data * context}) must divide "
            f"the device count ({world})",
        )
    attention_tp = world // (data * context)
    if attention.tensor_parallel_size not in (None, attention_tp):
        raise AdapterOperationError(
            error_code,
            f"attention tensor x data x context parallelism must equal the device count "
            f"({world}); attention tensor parallelism is {attention_tp}",
        )
    expert = experts.expert_parallel_size or 1
    if world % expert != 0:
        raise AdapterOperationError(
            error_code,
            f"expert parallelism ({expert}) must divide the device count ({world})",
        )
    expert_tp = world // expert
    if experts.tensor_parallel_size not in (None, expert_tp):
        raise AdapterOperationError(
            error_code,
            f"expert tensor parallelism is the device count divided by expert "
            f"parallelism ({expert_tp})",
        )
    if experts.dense_tensor_parallel_size not in (None, attention_tp):
        raise AdapterOperationError(
            error_code,
            f"dense expert tensor parallelism equals attention tensor parallelism ({attention_tp})",
        )

    effective = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=world, pipeline_parallel_size=1),
        attention=ParallelismAttention(
            tensor_parallel_size=attention_tp,
            data_parallel_size=data,
            context_parallel_size=context,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=expert_tp,
            data_parallel_size=1,
            expert_parallel_size=expert,
            dense_tensor_parallel_size=attention_tp,
        ),
    )
    return effective, _Layout(
        world=world,
        attention_tensor_parallel_size=attention_tp,
        data_parallel_size=data,
        context_parallel_size=context,
        expert_parallel_size=expert,
    )


def _public_endpoint(worker_count: int) -> EndpointDeclaration:
    return EndpointDeclaration(
        protocol=EndpointProtocol(),
        server_metrics=ServerMetricsEndpointRequirement(path="/metrics", port="prometheus"),
        prefix_cache_reset=inferlab_gateway_smg.flush_cache_reset(worker_count),
        # No conditioning fan-out: SMG Gateways answer POST /prime_prefix_cache
        # with 404, so a primed start conditions through the ordinary serving
        # flow when one cache-owning target exists and is rejected otherwise.
        # The worker protocol carries an unconditional cached-token count, so a
        # zero cache read is reported rather than omitted.
        prompt_cache_read_zero_representation=PromptCacheReadZeroRepresentation.explicit,
    )


def _kv_store_plan(
    input: PlanServeInput, role_access: dict[str, KvStoreAccess]
) -> KvStorePlan | None:
    """The declared store, with how each Engine role uses it and how long the
    Engine takes to publish to it ([[RFC-0003:C-SERVE-KV-STORE]])."""
    if input.kv_store is None:
        return None
    engine = validate_settings(StackSettings, input.stack_settings or {})
    return inferlab_kvstore.plan(input.kv_store).model_copy(
        update={
            "role_access": role_access,
            "publication_bound_ms": round(engine.kv_tier_publication_seconds * 1000),
        }
    )


def _plan_engine_role(
    input: PlanServeInput,
    role: ServeRoleInput,
    ports: list[str],
    profiled: bool,
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement]]:
    """One Engine role: each replica is one rank process owning its devices."""
    settings = _settings(role.settings)
    if input.kv_store is None and any(
        value is not None
        for value in (
            settings.kv_tier_capacity_bytes,
            settings.kv_tier_snapshot_interval,
            settings.kv_tier_timeout_seconds,
        )
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "kv_tier_* settings apply only when the server declares a kv_store",
        )
    parallelism, layout = _layout(
        fill_outer_tensor_parallel(role.parallelism, role.devices, 1, "no other parallelism")
    )
    require_declared_devices(
        layout.world,
        role.devices,
        "the Specialized Engine contract (outer tensor parallelism is the device count)",
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
    replicas = [
        ServeReplicaRequirement(
            id=replica_id(role, replica_index),
            role_id=role.id,
            replica_index=replica_index,
            device_count=layout.world,
            ports=list(ports),
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
                        stop=CaptureWindowHttpActionSpec(method=HttpMethod(), path="/stop_profile"),
                    ),
                )
                if profiled
                else None
            ),
        )
        for replica_index in range(role.replica_count)
    ]
    return role_result, replicas


def _plan_prefill_decode(input: PlanServeInput) -> PlanServeResult:
    """Prefill and decode Engines behind one fused SMG Gateway and P/D Router
    ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""
    inferlab_gateway_smg.require_backend(input.gateway_backend, component="Gateway")
    inferlab_gateway_smg.require_backend(input.pd_router_backend, component="P/D Router")
    if input.kv_transfer is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "a prefill/decode server must declare kv_transfer",
        )
    if input.profiling is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine profiles only a routed single server",
        )
    if input.gateway_settings:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "a prefill/decode server routes through P/D Router settings; declare "
            "pd_router_settings instead of gateway_settings",
        )
    routing = validate_settings(PdRouterSettings, input.pd_router_settings or {})
    prefill = require_role(input, ServeRoleKind.prefill)
    decode = require_role(input, ServeRoleKind.decode)
    prefill_result, prefill_replicas = _plan_engine_role(
        input, prefill, [inferlab_gateway_smg.BOOTSTRAP_PORT], profiled=False
    )
    decode_result, decode_replicas = _plan_engine_role(input, decode, [], profiled=False)
    gateway, pd_router, frontend_links = inferlab_gateway_smg.pd_frontend(
        prefill_role=prefill.id,
        decode_role=decode.id,
        prefill_workers=prefill.replica_count,
        decode_workers=decode.replica_count,
        endpoint=_public_endpoint(prefill.replica_count + decode.replica_count),
        policies=PdRoutingPolicies(prefill=routing.prefill_policy, decode=routing.decode_policy),
        pd_router_settings={
            "worker_protocol": SettingValue(root=_WORKER_PROTOCOL),
            **effective_settings(routing),
        },
    )
    return PlanServeResult(
        # Prefill reads and writes the tier; decode writes the blocks it
        # generates so a later request can reuse them
        # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
        kv_store=_kv_store_plan(
            input, {prefill.id: KvStoreAccess.read_write, decode.id: KvStoreAccess.write}
        ),
        serve_state_probe=input.kv_store is not None,
        integration=_identity(),
        roles=[prefill_result, decode_result],
        replicas=[*prefill_replicas, *decode_replicas],
        links=[
            *frontend_links,
            ServeRoleLink(
                root=ServeRoleLinkKvTransfer(
                    source=prefill.id, target=decode.id, mechanism=input.kv_transfer
                )
            ),
        ],
        gateway=gateway,
        pd_router=pd_router,
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    """Plan token Engines behind one SMG Gateway, as routed single replicas or
    as a prefill/decode pair."""
    validate_auxiliary_models(input)
    if input.profiling == CaptureMechanism.engine_trace:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration does not support engine-trace capture",
        )
    if input.synthetic_acceptance is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration cannot apply the synthetic acceptance overlay",
        )
    if input.topology == ServeTopology.prefill_decode:
        return _plan_prefill_decode(input)
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
    gateway_settings = validate_settings(GatewaySettings, input.gateway_settings or {})
    inferlab_gateway_smg.require_policy_workers(
        gateway_settings.policy, role.replica_count, role.id
    )
    if input.profiling is not None and role.replica_count != 1:
        # Every capture target opens its window through the one Gateway, so
        # several replicas would start and stop each worker's window once per
        # target.
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine integration profiles only a single replica",
        )
    role_result, replicas = _plan_engine_role(input, role, [], input.profiling is not None)
    gateway = GatewayPlan(
        backend=inferlab_gateway_smg.BACKEND,
        implementation=inferlab_gateway_smg.IMPLEMENTATION,
        implementation_version=inferlab_gateway_smg.installed_version(),
        effective_settings={
            "worker_protocol": SettingValue(root=_WORKER_PROTOCOL),
            "policy": SettingValue(root=gateway_settings.policy),
            **inferlab_gateway_smg.gateway_settings(),
        },
        endpoint=_public_endpoint(role.replica_count),
        readiness=inferlab_gateway_smg.routed_single_readiness(),
        ports=[inferlab_gateway_smg.PROMETHEUS_PORT],
        targets=[GatewayTarget(root=GatewayTargetEngine(role=role.id))],
        render_inputs=[],
        render_source=RenderSource.integration,
        co_rendering=FrontendCoRendering(process_role=FrontendProcessRole()),
    )
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[
            ServeRoleLink(root=ServeRoleLinkRequestRouting(source="gateway", targets=[role.id]))
        ],
        gateway=gateway,
        pd_router=None,
        # The contract's Engine reaches a declared store through
        # `--kv-tier-store` ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
        # Every Engine writes computed prefixes to the tier and reads from it,
        # so one priming request covers the server.
        kv_store=_kv_store_plan(input, {role.id: KvStoreAccess.read_write}),
        # A worker keeps serving without its store, so readiness and every
        # profiling window read each worker's tier state
        # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
        serve_state_probe=input.kv_store is not None,
    )


def _require_engines(
    allocations: list[ServeProcessAllocationModelRank],
) -> list[ServeProcessAllocationModelRank]:
    if not allocations or sorted(engine.replica for engine in allocations) != list(
        range(len(allocations))
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Specialized Engine integration requires one model-rank allocation per replica",
        )
    return [_require_engine(engine) for engine in allocations]


def _require_engine(
    engine: ServeProcessAllocationModelRank,
    role_kinds: tuple[ServeRoleKind, ...] = (ServeRoleKind.serve,),
) -> ServeProcessAllocationModelRank:
    if engine.role_kind not in role_kinds or engine.rank != 0 or engine.rank_count != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "each Engine allocation must be a serve replica in one rank process",
        )
    effective_parallelism, layout = _layout(
        engine.effective_parallelism,
        AdapterErrorCode.invalid_request,
    )
    if engine.effective_parallelism != effective_parallelism or len(engine.devices) != layout.world:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine rank process must own one device per rank of its world",
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
    _, layout = _layout(
        allocation.effective_parallelism,
        AdapterErrorCode.invalid_request,
    )
    if settings.prefix_cache_ranks is not None and len(settings.prefix_cache_ranks) != layout.world:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"prefix_cache_ranks declares {len(settings.prefix_cache_ranks)} ranks "
            f"but the Engine's world has {layout.world}",
        )
    inferlab_args = [
        "--listen",
        f"{endpoint.host}:{endpoint.port}",
        "--model",
        allocation.model_locator,
        "--served-model-name",
        input.model.served_name,
        "--tensor-parallel-size",
        str(layout.attention_tensor_parallel_size),
    ]
    if allocation.kv_store is not None:
        inferlab_args.extend(["--kv-tier-store", mooncake.client_address(allocation.kv_store)])
        for option, value in (
            ("--kv-tier-capacity", settings.kv_tier_capacity_bytes),
            ("--kv-tier-snapshot-interval", settings.kv_tier_snapshot_interval),
            ("--kv-tier-timeout", settings.kv_tier_timeout_seconds),
        ):
            if value is not None:
                inferlab_args.extend([option, str(value)])
    # A prefill/decode Engine learns its role, the declared transfer
    # mechanism, and, on prefill, the bootstrap port its decode counterparts
    # connect to ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    if input.topology == ServeTopology.prefill_decode:
        if input.kv_transfer is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "the prefill/decode render input carries no KV-transfer mechanism",
            )
        mode = "prefill" if allocation.role_kind == ServeRoleKind.prefill else "decode"
        inferlab_args.extend(
            [
                "--disaggregation-mode",
                mode,
                "--disaggregation-transfer-backend",
                input.kv_transfer.value,
            ]
        )
        if allocation.role_kind == ServeRoleKind.prefill:
            bootstrap = allocation.ports.get(inferlab_gateway_smg.BOOTSTRAP_PORT)
            if bootstrap is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"prefill allocation {allocation.process!r} is missing its bootstrap port",
                )
            inferlab_args.extend(["--disaggregation-bootstrap-port", str(bootstrap.port)])
    # Layout arguments render only when not one, so pure-TP Engines see the
    # same command as before ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    for option, size in (
        ("--data-parallel-size", layout.data_parallel_size),
        ("--context-parallel-size", layout.context_parallel_size),
        ("--expert-parallel-size", layout.expert_parallel_size),
    ):
        if size != 1:
            inferlab_args.extend([option, str(size)])
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
    argv = list(validate_settings(StackSettings, input.stack_settings or {}).engine_command)
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, _INFERLAB_OWNED_OPTIONS))
    return rendered_model_rank(
        allocation, ProcessSpec(argv=argv, env=dict(settings.extra_env or {}))
    )


def _render_gateway(
    allocation: ServeProcessAllocationFrontend,
    engines: list[ServeProcessAllocationModelRank],
) -> RenderedServeProcess:
    workers = [engine.endpoint for engine in sorted(engines, key=lambda engine: engine.replica)]
    if any(worker is None for worker in workers):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine allocation is missing its endpoint",
        )
    policy = allocation.gateway.effective_settings.get("policy")
    if policy is None or not isinstance(policy.root, str):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Gateway plan is missing its effective policy",
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
            model_locator=engines[0].model_locator,
            workers=[worker for worker in workers if worker is not None],
            policy=policy.root,
        ),
    )


def _render_prefill_decode(input: RenderServeInput) -> RenderServeResult:
    if (
        input.gateway_backend != inferlab_gateway_smg.BACKEND
        or input.pd_router_backend != inferlab_gateway_smg.BACKEND
        or input.kv_transfer is None
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "render input is not the planned prefill/decode SMG workflow",
        )
    allocations, model_allocations = split_serve_allocations(input.allocations)
    engines = [
        _require_engine(engine, (ServeRoleKind.prefill, ServeRoleKind.decode))
        for engine in model_allocations
    ]
    frontends = [
        allocation
        for allocation in allocations
        if isinstance(allocation, ServeProcessAllocationFrontend)
    ]
    stores = [
        allocation
        for allocation in allocations
        if isinstance(allocation, ServeProcessAllocationKvStore)
    ]
    if len(frontends) != 1 or len(stores) > 1 or len(allocations) != len(engines) + len(stores) + 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the prefill/decode workflow requires Engine allocations, one fused frontend, "
            "and at most one KV store",
        )
    processes = [inferlab_kvstore.render(store) for store in stores]
    processes.extend(_render_engine(input, engine) for engine in engines)
    processes.append(inferlab_gateway_smg.render_pd_frontend(frontends[0], engines))
    return RenderServeResult(integration=_identity(), processes=processes)


def render_serve(input: RenderServeInput) -> RenderServeResult:
    if input.topology == ServeTopology.prefill_decode:
        return _render_prefill_decode(input)
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
    engines = _require_engines(model_allocations)
    stores = [
        allocation
        for allocation in allocations
        if isinstance(allocation, ServeProcessAllocationKvStore)
    ]
    if len(allocations) != len(engines) + len(stores) + 1 or len(stores) > 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the routed-single workflow requires Engine allocations and one Gateway allocation",
        )
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
    _require_gateway(frontend_candidates[0], engines[0].role)

    processes: list[RenderedServeProcess] = []
    for allocation in allocations:
        if isinstance(allocation, ServeProcessAllocationModelRank):
            processes.append(_render_engine(input, allocation))
        elif isinstance(allocation, ServeProcessAllocationFrontend):
            processes.append(_render_gateway(allocation, engines))
        elif isinstance(allocation, ServeProcessAllocationKvStore):
            processes.append(inferlab_kvstore.render(allocation))
    return RenderServeResult(integration=_identity(), processes=processes)


def probe_serve(input: ProbeServeInput) -> ProbeServeResult:
    """Read each Engine's `kv_tier_state` and `kv_tier_store` from its
    `GetServerInfo` `scheduler_info`: engaged only when the tier is
    `available` at the store address InferLab rendered
    ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""
    if input.kv_store is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Specialized Engine probes serve state only for a server with a KV store",
        )
    expected_store = mooncake.client_address(input.kv_store)
    with ThreadPoolExecutor(max_workers=max(1, len(input.processes))) as pool:
        readings = list(
            pool.map(
                lambda target: inferlab_gateway_smg.worker_server_info(
                    target.endpoint, float(input.attempt_bound_seconds)
                ),
                input.processes,
            )
        )
    states: list[ProbeServeProcessState] = []
    for target, (info, error) in zip(input.processes, readings, strict=True):
        if info is None:
            states.append(
                ProbeServeProcessState(process=target.process, engaged=False, state={}, error=error)
            )
            continue
        scheduler_info = info.get("scheduler_info")
        reported = scheduler_info if isinstance(scheduler_info, dict) else {}
        state = {
            key: SettingValue(root=value)
            for key in ("kv_tier_state", "kv_tier_store")
            if isinstance(value := reported.get(key), str)
        }
        engaged = (
            reported.get("kv_tier_state") == "available"
            and reported.get("kv_tier_store") == expected_store
        )
        states.append(ProbeServeProcessState(process=target.process, engaged=engaged, state=state))
    return ProbeServeResult(integration=_identity(), processes=states)
