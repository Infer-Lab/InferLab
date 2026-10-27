import inferlab_gateway_smg
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    EndpointDeclaration,
    EndpointProtocol,
    HttpActionSpec,
    HttpMethod,
    IntegrationIdentity,
    KvTransferMechanism,
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PdRoutingPolicies,
    PlanServeInput,
    PlanServeResult,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeProcessAlive,
    ServeReplicaRequirement,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkKvTransfer,
    ServeRoleResult,
    ServeTopology,
    effective_settings,
    fill_outer_tensor_parallel,
    integration_identity,
    replica_id,
    require_declared_devices,
    require_role,
    validate_settings,
)

from .auxiliary import validate_auxiliary_models
from .settings import PdRouterSettings, _settings


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-tokenspeed",
        adapter_distribution="inferlab-integration-tokenspeed",
        framework="tokenspeed",
        framework_distribution="tokenspeed",
    )


def _effective_parallelism(declared: Parallelism) -> Parallelism:
    """Resolve TokenSpeed's component parallelism over one process world."""
    outer = declared.outer or ParallelismOuter()
    attention = declared.attention or ParallelismAttention()
    experts = declared.experts or ParallelismExperts()
    world_size = outer.tensor_parallel_size or 1

    if (outer.pipeline_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration does not support pipeline parallelism",
        )
    if (attention.context_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration does not support attention context parallelism",
        )

    attention_dp = attention.data_parallel_size or 1
    if world_size % attention_dp != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TokenSpeed attention.data_parallel_size ({attention_dp}) must divide "
            f"outer.tensor_parallel_size ({world_size})",
        )
    effective_attention_tp = world_size // attention_dp
    if (
        attention.tensor_parallel_size is not None
        and attention.tensor_parallel_size != effective_attention_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed effective attention.tensor_parallel_size is "
            "outer.tensor_parallel_size / attention.data_parallel_size "
            f"({effective_attention_tp})",
        )

    expert_ep = experts.expert_parallel_size or 1
    expert_dp = experts.data_parallel_size or 1
    expert_divisor = expert_ep * expert_dp
    if world_size % expert_divisor != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed experts.expert_parallel_size * experts.data_parallel_size "
            f"({expert_divisor}) must divide outer.tensor_parallel_size ({world_size})",
        )
    effective_expert_tp = world_size // expert_divisor
    if (
        experts.tensor_parallel_size is not None
        and experts.tensor_parallel_size != effective_expert_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed effective experts.tensor_parallel_size is "
            "outer.tensor_parallel_size / experts.expert_parallel_size / "
            f"experts.data_parallel_size ({effective_expert_tp})",
        )
    if effective_expert_tp > 1 and expert_ep > 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed does not support MoE tensor and expert parallelism "
            "greater than one at the same time",
        )

    dense_tp = experts.dense_tensor_parallel_size or world_size
    if world_size % dense_tp != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TokenSpeed experts.dense_tensor_parallel_size ({dense_tp}) must divide "
            f"outer.tensor_parallel_size ({world_size})",
        )

    return Parallelism(
        outer=ParallelismOuter(
            tensor_parallel_size=world_size,
            pipeline_parallel_size=1,
        ),
        attention=ParallelismAttention(
            tensor_parallel_size=effective_attention_tp,
            data_parallel_size=attention_dp,
            context_parallel_size=1,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=effective_expert_tp,
            data_parallel_size=expert_dp,
            expert_parallel_size=expert_ep,
            dense_tensor_parallel_size=dense_tp,
        ),
    )


def _plan_role(
    input: PlanServeInput,
    role: ServeRoleInput,
    ports: list[str],
    primary_readiness: ReadinessProbe,
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement]]:
    if role.replica_count < 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} replica count must be positive",
        )
    settings = _settings(role.settings)
    parallelism = _effective_parallelism(
        fill_outer_tensor_parallel(
            role.parallelism,
            role.devices,
            1,
            "no other TokenSpeed parallelism",
        )
    )
    outer = parallelism.outer or ParallelismOuter()
    require_declared_devices(
        outer.tensor_parallel_size or 1,
        role.devices,
        "TokenSpeed's reading (outer tensor parallelism)",
    )
    replicas = [
        ServeReplicaRequirement(
            id=replica_id(role, replica_index),
            role_id=role.id,
            replica_index=replica_index,
            device_count=outer.tensor_parallel_size or 1,
            ports=list(ports),
            primary_ports=[],
            primary_readiness=primary_readiness,
            worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
        )
        for replica_index in range(role.replica_count)
    ]
    return (
        ServeRoleResult(
            id=role.id,
            kind=role.kind,
            declared_replica_count=role.replica_count,
            effective_replica_count=role.replica_count,
            effective_settings=effective_settings(settings),
            effective_parallelism=parallelism,
        ),
        replicas,
    )


def _endpoint_declaration() -> EndpointDeclaration:
    return EndpointDeclaration(
        protocol=EndpointProtocol(),
        prefix_cache_reset=HttpActionSpec(
            method=HttpMethod(),
            path="/flush_cache",
        ),
    )


def _plan_single(input: PlanServeInput) -> PlanServeResult:
    if input.kv_transfer is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "single topology does not use a KV-transfer mechanism",
        )
    if input.gateway_backend is not None or input.pd_router_backend is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed single topology does not have a qualified Gateway backend",
        )
    role = require_role(input, ServeRoleKind.serve)
    if role.replica_count != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration supports exactly one serve replica",
        )
    role_result, replicas = _plan_role(
        input,
        role,
        ["control", "dist_init"],
        ReadinessProbe(root=ReadinessProbeHttp(path="/readiness")),
    )
    role_result.public_endpoint = _endpoint_declaration()
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[],
    )


def _plan_prefill_decode(input: PlanServeInput) -> PlanServeResult:
    if input.kv_store is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration does not support a KV store on a prefill/decode server",
        )
    if input.kv_transfer != KvTransferMechanism.mooncake:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed prefill/decode only supports Mooncake KV transfer",
        )
    inferlab_gateway_smg.require_backend(input.gateway_backend, component="Gateway")
    inferlab_gateway_smg.require_backend(input.pd_router_backend, component="P/D Router")
    prefill = require_role(input, ServeRoleKind.prefill)
    decode = require_role(input, ServeRoleKind.decode)
    routing = validate_settings(PdRouterSettings, input.pd_router_settings or {})
    process_alive = ReadinessProbe(root=ReadinessProbeProcessAlive())
    prefill_result, prefill_replicas = _plan_role(
        input,
        prefill,
        ["dist_init", inferlab_gateway_smg.BOOTSTRAP_PORT],
        process_alive,
    )
    decode_result, decode_replicas = _plan_role(
        input,
        decode,
        ["dist_init"],
        process_alive,
    )
    gateway, pd_router, frontend_links = inferlab_gateway_smg.pd_frontend(
        prefill_role=prefill.id,
        decode_role=decode.id,
        prefill_workers=prefill.replica_count,
        decode_workers=decode.replica_count,
        endpoint=EndpointDeclaration(
            protocol=EndpointProtocol(),
            # The SMG Gateway flushes every registered prefill and decode
            # worker and counts them.
            prefix_cache_reset=inferlab_gateway_smg.flush_cache_reset(
                prefill.replica_count + decode.replica_count
            ),
        ),
        policies=PdRoutingPolicies(prefill=routing.prefill_policy, decode=routing.decode_policy),
        pd_router_settings=effective_settings(routing),
    )
    # The worker-to-worker KV transfer link sits between the frontend's
    # routing and bootstrap links, as the qualified plan recorded it.
    *routing_links, bootstrap_link = frontend_links
    links = [
        *routing_links,
        ServeRoleLink(
            root=ServeRoleLinkKvTransfer(
                source=prefill.id,
                target=decode.id,
                mechanism=KvTransferMechanism.mooncake,
            )
        ),
        bootstrap_link,
    ]
    return PlanServeResult(
        integration=_identity(),
        roles=[prefill_result, decode_result],
        replicas=[*prefill_replicas, *decode_replicas],
        links=links,
        gateway=gateway,
        pd_router=pd_router,
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    validate_auxiliary_models(input)
    if input.profiling is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration does not support profiling capture yet",
        )
    if input.synthetic_acceptance is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration cannot apply the synthetic acceptance overlay",
        )
    if input.topology == ServeTopology.single:
        return _plan_single(input)
    return _plan_prefill_decode(input)
