"""Planning and rendering for the shared token-only Specialized Engine contract."""

from concurrent.futures import ThreadPoolExecutor

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
    EndpointAssignment,
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
    PdRoutingPolicies,
    PlanServeInput,
    PlanServeResult,
    ProbeServeInput,
    ProbeServeProcessState,
    ProbeServeResult,
    ProbeServeTarget,
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
    authority,
    effective_settings,
    integration_identity,
    merge_serve_args,
    rendered_frontend,
    rendered_model_rank,
    replica_id,
    require_role,
    split_serve_allocations,
    validate_settings,
)
from inferlab_kvstore import filesystem, mooncake
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .auxiliary import reject_auxiliary_locators, validate_auxiliary_models

_ADAPTER_DISTRIBUTION = "inferlab-integration-specialized-engine"

# The contract arguments: the allocation facts InferLab renders, which an
# Engine argument must not restate, before or after the override sentinel
# ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
_CONTRACT_ARGUMENTS: frozenset[str] = frozenset(
    {
        "--disaggregation-bootstrap-port",
        "--disaggregation-mode",
        "--disaggregation-transfer-backend",
        "--dist-init-addr",
        "--kv-tier-backend",
        "--kv-tier-store",
        "--listen",
        "--model",
        "--nnodes",
        "--node-rank",
        "--served-model-name",
    }
)

# The port rank zero of a multi-host replica admits its members on; the
# control plane allocates a primary port only to rank zero of a multi-rank
# replica ([[ADR-0073]]).
_RENDEZVOUS_PORT = "rendezvous"

# Environment the allocation owns: the Engine's devices are its visibility,
# enumerated in PCI bus order ([[RFC-0002:C-LOCAL-PLACEMENT]]).
_DEVICE_VISIBILITY = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")

# Typed settings this contract no longer renders: each is an Engine option,
# spelled by the Engine, in `extra_args` ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
_RETIRED_ROLE_SETTINGS: frozenset[str] = frozenset(
    {
        "default_max_output_tokens",
        "gpu_memory_utilization_percent",
        "kv_tier_capacity_bytes",
        "kv_tier_snapshot_interval",
        "kv_tier_timeout_seconds",
        "max_num_batched_tokens",
        "prefix_cache_gpu_entries",
        "prefix_cache_host_memory_percent",
        "prefix_cache_ranks",
        "workspace_reserve_mib",
    }
)


class EngineContractSettings(BaseModel):
    """A role's Engine arguments: every Engine option that is not a contract
    argument, in the Engine's own spelling, rendered and recorded verbatim
    ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""

    model_config = ConfigDict(extra="forbid")

    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


class GatewaySettings(BaseModel):
    """The typed SMG Gateway settings a server may declare
    ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""

    model_config = ConfigDict(extra="forbid")

    policy: inferlab_gateway_smg.Policy = "least_load"
    # SMG options of the operator's own, appended to the rendered command.
    extra_args: list[str] | None = None


class PdRouterSettings(BaseModel):
    """The typed SMG P/D Router settings a prefill/decode server may declare
    ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""

    model_config = ConfigDict(extra="forbid")

    prefill_policy: inferlab_gateway_smg.Policy = "round_robin"
    decode_policy: inferlab_gateway_smg.Policy = "round_robin"
    # SMG options of the operator's own, appended to the rendered command.
    extra_args: list[str] | None = None


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


# Stack settings this contract no longer takes: the Engine reports its tier
# publication bound at readiness ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
_RETIRED_STACK_SETTINGS: frozenset[str] = frozenset({"kv_tier_publication_seconds"})


def stack_settings(values: dict[str, SettingValue]) -> dict[str, SettingValue]:
    retired = sorted(_RETIRED_STACK_SETTINGS.intersection(values))
    if retired:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"stack setting {retired[0]!r} is no longer part of the Specialized Engine "
            "contract; the Engine reports kv_tier_publication_ms at readiness",
        )
    settings = validate_settings(StackSettings, values)
    _reject_owned_options("engine_command", settings.engine_command)
    return effective_settings(settings)


def _reject_owned_options(field: str, arguments: list[str]) -> None:
    for argument in arguments:
        name, _separator, _value = argument.partition("=")
        if name in _CONTRACT_ARGUMENTS:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                f"{field} entry {argument!r} names contract argument {name!r}, which "
                "InferLab renders from the allocation",
            )


def _settings(values: dict[str, SettingValue]) -> EngineContractSettings:
    retired = sorted(_RETIRED_ROLE_SETTINGS.intersection(values))
    if retired:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"setting {retired[0]!r} is no longer part of the Specialized Engine contract; "
            "pass the Engine's own option for it in extra_args",
        )
    settings = validate_settings(EngineContractSettings, values)
    # The sentinel overrides Engine options only; no declaration may redirect
    # an allocation ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    _reject_owned_options("extra_args", settings.extra_args or [])
    for name in _DEVICE_VISIBILITY:
        if name in (settings.extra_env or {}):
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                f"extra_env must not set {name}: the Engine's devices are its allocation",
            )
    return settings


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-specialized-engine",
        adapter_distribution=_ADAPTER_DISTRIBUTION,
        framework="specialized-engine",
        framework_distribution=_ADAPTER_DISTRIBUTION,
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


def _kv_store_plan(input: PlanServeInput) -> KvStorePlan | None:
    """The declared tier — a `mooncake` store InferLab runs, or a `filesystem`
    directory each machine binds — whose access and publication bound every
    Engine reports at readiness ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""
    if input.kv_store is None:
        return None
    return inferlab_kvstore.plan(input.kv_store).model_copy(update={"access_observed": True})


def _plan_engine_role(
    input: PlanServeInput,
    role: ServeRoleInput,
    ports: list[str],
    profiled: bool,
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement]]:
    """One Engine role: each replica is one rank process owning its devices on
    each machine its placement binds ([[ADR-0073]])."""
    settings = _settings(role.settings)
    # The Engine owns its layout: a role declares only its device count, and
    # the Engine reports its data-parallel rank count at readiness
    # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    if role.parallelism != Parallelism():
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} declares parallelism components; a Specialized Engine "
            "takes its layout from Engine arguments in extra_args, so declare only devices",
        )
    if role.devices is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} must declare devices, the Engine's device count",
        )
    role_result = ServeRoleResult(
        id=role.id,
        kind=role.kind,
        declared_replica_count=role.replica_count,
        effective_replica_count=role.replica_count,
        effective_settings=effective_settings(settings),
        effective_parallelism=Parallelism(),
        public_endpoint=None,
    )
    replicas = [
        ServeReplicaRequirement(
            id=replica_id(role, replica_index),
            role_id=role.id,
            replica_index=replica_index,
            device_count=role.devices,
            ports=list(ports),
            primary_ports=[_RENDEZVOUS_PORT] if role.kind == ServeRoleKind.serve else [],
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
    inferlab_gateway_smg.validate_extra_args(routing.extra_args or [], component="P/D Router")
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
        observe_data_parallel_size=True,
    )
    return PlanServeResult(
        kv_store=_kv_store_plan(input),
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
    inferlab_gateway_smg.validate_extra_args(gateway_settings.extra_args or [], component="Gateway")
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
            **effective_settings(gateway_settings),
            **inferlab_gateway_smg.gateway_settings(),
        },
        endpoint=_public_endpoint(role.replica_count),
        readiness=inferlab_gateway_smg.routed_single_readiness(observe_data_parallel_size=True),
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
        # The contract's Engine reaches a declared tier through
        # `--kv-tier-store` and reports how it uses it
        # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
        kv_store=_kv_store_plan(input),
        # A worker keeps serving without its store, so readiness and every
        # profiling window read each worker's tier state
        # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
        serve_state_probe=input.kv_store is not None,
    )


def _hosts(
    engines: list[ServeProcessAllocationModelRank],
) -> list[list[ServeProcessAllocationModelRank]]:
    """Each replica's rank processes in rank order: one per host, rank zero
    being host 0 ([[ADR-0073]]). Ranks of one replica hold equal device
    counts, since the Engine's world is its visible devices times the host
    count."""
    replicas: dict[tuple[str, int], list[ServeProcessAllocationModelRank]] = {}
    for engine in engines:
        replicas.setdefault((engine.role, engine.replica), []).append(engine)
    hosts = []
    for (role, replica), ranks in sorted(replicas.items()):
        ranks.sort(key=lambda engine: engine.rank)
        if [engine.rank for engine in ranks] != list(range(ranks[0].rank_count)):
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"replica {replica} of role {role!r} does not carry every rank of its allocation",
            )
        if len({len(engine.devices) for engine in ranks}) > 1:
            held = ", ".join(f"{engine.machine} {len(engine.devices)}" for engine in ranks)
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                f"replica {replica} of role {role!r} spans hosts with unequal device counts "
                f"({held}); every host of a Specialized Engine replica holds the same count",
            )
        hosts.append(ranks)
    return hosts


def _require_engines(
    allocations: list[ServeProcessAllocationModelRank],
    role_kinds: tuple[ServeRoleKind, ...] = (ServeRoleKind.serve,),
) -> list[list[ServeProcessAllocationModelRank]]:
    if not allocations:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Specialized Engine integration requires model-rank allocations",
        )
    return _hosts([_require_engine(engine, role_kinds) for engine in allocations])


def _require_engine(
    engine: ServeProcessAllocationModelRank,
    role_kinds: tuple[ServeRoleKind, ...],
) -> ServeProcessAllocationModelRank:
    if engine.role_kind not in role_kinds:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"Engine allocation {engine.process!r} belongs to an unplanned role kind",
        )
    if engine.effective_parallelism != Parallelism() or not engine.devices:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine rank process must own its allocated devices and no parallelism shape",
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


def _tier_location(
    store: EndpointAssignment | None, directory: str | None
) -> tuple[str, str] | None:
    """The tier backend and location rendered for an Engine: the `mooncake`
    store's address, or the machine's bound `filesystem` directory."""
    if store is not None:
        return mooncake.BACKEND, mooncake.client_address(store)
    if directory is not None:
        return filesystem.BACKEND, directory
    return None


def _rendezvous(host_zero: ServeProcessAllocationModelRank) -> str:
    port = host_zero.ports.get(_RENDEZVOUS_PORT)
    if port is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"multi-host rank zero {host_zero.process!r} is missing its rendezvous port",
        )
    return authority(port.host, port.port)


def _render_engine(
    input: RenderServeInput,
    allocation: ServeProcessAllocationModelRank,
    rendezvous: str | None,
) -> RenderedServeProcess:
    """One host's Engine. Rank zero is host 0 and the replica's endpoint; on a
    replica spanning machines, every rank receives rank zero's `rendezvous`
    address, which rank zero binds and the others join, and only rank zero
    listens ([[ADR-0073]], [[ADR-0077]])."""
    reject_auxiliary_locators(allocation)
    endpoint = allocation.endpoint
    if endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Engine allocation is missing its endpoint",
        )
    settings = _settings(allocation.effective_settings)
    contract_args = (
        ["--listen", authority(endpoint.host, endpoint.port)] if allocation.rank == 0 else []
    )
    contract_args.extend(
        ["--model", allocation.model_locator, "--served-model-name", input.model.served_name]
    )
    if rendezvous is not None:
        # Every rank names rank zero's address; its node rank decides whether
        # it binds or connects ([[ADR-0077]]).
        contract_args.extend(
            [
                "--nnodes",
                str(allocation.rank_count),
                "--node-rank",
                str(allocation.rank),
                "--dist-init-addr",
                rendezvous,
            ]
        )
    tier = _tier_location(allocation.kv_store, allocation.kv_tier_directory)
    if tier is not None:
        backend, location = tier
        contract_args.extend(["--kv-tier-backend", backend, "--kv-tier-store", location])
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
        contract_args.extend(
            [
                "--disaggregation-mode",
                mode,
                "--disaggregation-transfer-backend",
                input.kv_transfer.root,
            ]
        )
        if allocation.role_kind == ServeRoleKind.prefill:
            bootstrap = allocation.ports.get(inferlab_gateway_smg.BOOTSTRAP_PORT)
            if bootstrap is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"prefill allocation {allocation.process!r} is missing its bootstrap port",
                )
            contract_args.extend(["--disaggregation-bootstrap-port", str(bootstrap.port)])
    argv = list(validate_settings(StackSettings, input.stack_settings or {}).engine_command)
    argv.extend(merge_serve_args(settings.extra_args or [], contract_args, _CONTRACT_ARGUMENTS))
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
            extra_args=inferlab_gateway_smg.extra_args_setting(
                allocation.gateway.effective_settings
            ),
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
    hosts = _require_engines(model_allocations, (ServeRoleKind.prefill, ServeRoleKind.decode))
    if any(len(ranks) > 1 for ranks in hosts):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "a prefill or decode Specialized Engine replica cannot span machines; "
            "multi-host replicas are supported for the serve role",
        )
    engines = [ranks[0] for ranks in hosts]
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
    processes.extend(_render_engine(input, engine, None) for engine in engines)
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
    hosts = _require_engines(model_allocations)
    if input.profiling is not None and any(len(ranks) > 1 for ranks in hosts):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the Specialized Engine cannot profile a replica that spans machines: the "
            "Gateway's profiling window reaches only host 0",
        )
    # The Gateway routes to each replica's host 0, its only endpoint.
    engines = [ranks[0] for ranks in hosts]
    rendezvous = {
        engine.process: _rendezvous(ranks[0]) if len(ranks) > 1 else None
        for ranks in hosts
        for engine in ranks
    }
    stores = [
        allocation
        for allocation in allocations
        if isinstance(allocation, ServeProcessAllocationKvStore)
    ]
    if len(allocations) != len(model_allocations) + len(stores) + 1 or len(stores) > 1:
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
            processes.append(_render_engine(input, allocation, rendezvous[allocation.process]))
        elif isinstance(allocation, ServeProcessAllocationFrontend):
            processes.append(_render_gateway(allocation, engines))
        elif isinstance(allocation, ServeProcessAllocationKvStore):
            processes.append(inferlab_kvstore.render(allocation))
    return RenderServeResult(integration=_identity(), processes=processes)


def _reported_access(value: object) -> KvStoreAccess | None:
    if isinstance(value, str) and value in {access.value for access in KvStoreAccess}:
        return KvStoreAccess(value)
    return None


def _reported_publication_ms(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _probe_state(
    target: ProbeServeTarget,
    location: str,
    info: dict[str, object] | None,
    error: str | None,
) -> ProbeServeProcessState:
    if info is None:
        return ProbeServeProcessState(process=target.process, engaged=False, state={}, error=error)
    scheduler_info = info.get("scheduler_info")
    reported: dict[str, object] = scheduler_info if isinstance(scheduler_info, dict) else {}
    state = {
        key: SettingValue(root=value)
        for key in ("kv_tier_state", "kv_tier_store", "kv_tier_access")
        if isinstance(value := reported.get(key), str)
    }
    publication_ms = _reported_publication_ms(reported.get("kv_tier_publication_ms"))
    if publication_ms is not None:
        state["kv_tier_publication_ms"] = SettingValue(root=publication_ms)
    engaged = (
        reported.get("kv_tier_state") == "available" and reported.get("kv_tier_store") == location
    )
    return ProbeServeProcessState(
        process=target.process,
        engaged=engaged,
        state=state,
        access=_reported_access(reported.get("kv_tier_access")),
        publication_bound_ms=publication_ms,
    )


def probe_serve(input: ProbeServeInput) -> ProbeServeResult:
    """Read each Engine's tier report from its `GetServerInfo`
    `scheduler_info`: engaged only when the tier is `available` at the
    location InferLab rendered for it, with the Engine's own access and
    publication bound ([[RFC-0003:C-SPECIALIZED-ENGINE]])."""
    tiers = [_tier_location(input.kv_store, target.kv_tier_directory) for target in input.processes]
    locations = [tier[1] for tier in tiers if tier is not None]
    if not input.processes or len(locations) != len(input.processes):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Specialized Engine probes serve state only for Engines rendered with a KV tier",
        )
    with ThreadPoolExecutor(max_workers=len(input.processes)) as pool:
        readings = list(
            pool.map(
                lambda target: inferlab_gateway_smg.worker_server_info(
                    target.endpoint, float(input.attempt_bound_seconds)
                ),
                input.processes,
            )
        )
    return ProbeServeResult(
        integration=_identity(),
        processes=[
            _probe_state(target, location, info, error)
            for target, location, (info, error) in zip(
                input.processes, locations, readings, strict=True
            )
        ],
    )
