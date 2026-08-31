from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    EndpointProtocol,
    EndpointRequirement,
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
    ReadinessProbeHttpTargetRegistry,
    ReadinessProbeProcessAlive,
    RenderSource,
    ServeReplicaRequirement,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkBootstrap,
    ServeRoleLinkKvTransfer,
    ServeRoleLinkRequestRouting,
    ServeRoleResult,
    ServeTopology,
    TargetEndpointScheme,
    effective_settings,
    fused_pd_frontend_plans,
    integration_identity,
    replica_id,
    require_role,
)

from .settings import _settings


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
    parallelism = _effective_parallelism(role.parallelism)
    outer = parallelism.outer or ParallelismOuter()
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


def _endpoint_requirement() -> EndpointRequirement:
    return EndpointRequirement(
        protocol=EndpointProtocol(),
        completions_path="/v1/completions",
        chat_completions_path="/v1/chat/completions",
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
    role_result.public_endpoint = _endpoint_requirement()
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[],
    )


def _plan_prefill_decode(input: PlanServeInput) -> PlanServeResult:
    if input.kv_transfer != KvTransferMechanism.mooncake:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed prefill/decode only supports Mooncake KV transfer",
        )
    backend_pair = (input.gateway_backend, input.pd_router_backend)
    if backend_pair != ("tokenspeed-smg", "tokenspeed-smg"):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TokenSpeed does not support Gateway/P/D Router pair {backend_pair!r}",
        )
    prefill = require_role(input, ServeRoleKind.prefill)
    decode = require_role(input, ServeRoleKind.decode)
    process_alive = ReadinessProbe(root=ReadinessProbeProcessAlive())
    prefill_result, prefill_replicas = _plan_role(
        input,
        prefill,
        ["dist_init", "bootstrap"],
        process_alive,
    )
    decode_result, decode_replicas = _plan_role(
        input,
        decode,
        ["dist_init"],
        process_alive,
    )
    links = [
        ServeRoleLink(
            root=ServeRoleLinkRequestRouting(
                source="gateway",
                targets=["pd_router"],
            )
        ),
        ServeRoleLink(
            root=ServeRoleLinkRequestRouting(
                source="pd_router",
                targets=[prefill.id, decode.id],
            )
        ),
        ServeRoleLink(
            root=ServeRoleLinkKvTransfer(
                source=prefill.id,
                target=decode.id,
                mechanism=KvTransferMechanism.mooncake,
            )
        ),
        ServeRoleLink(
            root=ServeRoleLinkBootstrap(
                source="pd_router",
                target=prefill.id,
                port="bootstrap",
            )
        ),
    ]
    identity = _identity()
    gateway, pd_router = fused_pd_frontend_plans(
        gateway_backend="tokenspeed-smg",
        pd_router_backend="tokenspeed-smg",
        implementation="tokenspeed-smg",
        implementation_version=identity.adapter_version,
        render_source=RenderSource.integration,
        endpoint=_endpoint_requirement(),
        gateway_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/readiness")),
        pd_router_readiness=ReadinessProbe(
            root=ReadinessProbeHttpTargetRegistry(
                readiness_path="/readiness",
                registry_path="/workers",
                targets_field="workers",
                target_url_field="url",
                target_role_field="worker_type",
                target_healthy_field="is_healthy",
                target_bootstrap_port_field="bootstrap_port",
                target_scheme=TargetEndpointScheme.grpc,
                prefill_role_value="prefill",
                decode_role_value="decode",
                prefill_bootstrap_port="bootstrap",
            )
        ),
        policies=PdRoutingPolicies(prefill="round_robin", decode="round_robin"),
        prefill_role=prefill.id,
        decode_role=decode.id,
        target_scheme=TargetEndpointScheme.grpc,
        pd_router_ports=["prometheus"],
    )
    return PlanServeResult(
        integration=_identity(),
        roles=[prefill_result, decode_result],
        replicas=[*prefill_replicas, *decode_replicas],
        links=links,
        gateway=gateway,
        pd_router=pd_router,
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
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
