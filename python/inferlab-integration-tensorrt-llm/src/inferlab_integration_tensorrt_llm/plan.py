import os
from pathlib import Path

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    EndpointProtocol,
    EndpointRequirement,
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
    RenderInputDeclaration,
    RenderSource,
    ServeReplicaRequirement,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLink,
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
    validate_extra_args,
)

from .settings import _INFERLAB_OWNED_OPTIONS, _settings

_NATIVE_ROUTING_BACKEND = "trtllm-disaggregated"
_PREFILL_DECODE_OWNED_OPTIONS = _INFERLAB_OWNED_OPTIONS | {"--backend"}


def _render_source_path(path: str) -> str:
    if Path(path).is_absolute():
        return path
    return os.path.normpath(Path(".inferlab") / path)


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-tensorrt-llm",
        adapter_distribution="inferlab-integration-tensorrt-llm",
        framework="tensorrt-llm",
        framework_distribution="tensorrt_llm",
    )


def _effective_parallelism(declared: Parallelism) -> Parallelism:
    """The TensorRT-LLM 1.3 algebra: `outer.tensor_parallel_size` is the
    tensor-parallel world (`--tp_size`), which attention data parallelism
    divides all-or-nothing (`--enable_attention_dp` replicates attention on
    every rank) and expert parallelism divides freely (the framework derives
    the MoE tensor split from `moe_tp * moe_ep == tp`). Context parallelism
    multiplies the TensorRT-LLM world instead of dividing it, and MoE data
    and dense-tensor parallelism have no TensorRT-LLM equivalent, so those
    components reject rather than silently reshape the deployment."""
    outer = declared.outer or ParallelismOuter()
    attention = declared.attention or ParallelismAttention()
    experts = declared.experts or ParallelismExperts()
    outer_tp = outer.tensor_parallel_size or 1
    outer_pp = outer.pipeline_parallel_size or 1
    attention_dp = attention.data_parallel_size or 1
    if (attention.context_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support attention context parallelism",
        )
    if attention_dp not in (1, outer_tp):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM attention data parallelism is all-or-nothing: "
            f"attention.data_parallel_size must be 1 or equal "
            f"outer.tensor_parallel_size ({outer_tp}), got {attention_dp}",
        )
    effective_attention_tp = outer_tp // attention_dp
    if (
        attention.tensor_parallel_size is not None
        and attention.tensor_parallel_size != effective_attention_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TensorRT-LLM effective attention.tensor_parallel_size is "
            f"outer.tensor_parallel_size / attention.data_parallel_size "
            f"({effective_attention_tp})",
        )
    if (experts.data_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support MoE data parallelism",
        )
    if (experts.dense_tensor_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support dense tensor parallelism",
        )
    expert_ep = experts.expert_parallel_size or 1
    if outer_tp % expert_ep != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM experts.expert_parallel_size ({expert_ep}) "
            f"must divide outer.tensor_parallel_size ({outer_tp})",
        )
    effective_expert_tp = outer_tp // expert_ep
    if (
        experts.tensor_parallel_size is not None
        and experts.tensor_parallel_size != effective_expert_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TensorRT-LLM effective experts.tensor_parallel_size is "
            f"outer.tensor_parallel_size / experts.expert_parallel_size "
            f"({effective_expert_tp})",
        )
    return Parallelism(
        outer=ParallelismOuter(
            tensor_parallel_size=outer_tp,
            pipeline_parallel_size=outer_pp,
        ),
        attention=ParallelismAttention(
            tensor_parallel_size=effective_attention_tp,
            data_parallel_size=attention_dp,
            context_parallel_size=1,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=effective_expert_tp,
            data_parallel_size=1,
            expert_parallel_size=expert_ep,
            dense_tensor_parallel_size=1,
        ),
    )


def _device_count(parallelism: Parallelism) -> int:
    outer = parallelism.outer or ParallelismOuter()
    return (outer.tensor_parallel_size or 1) * (outer.pipeline_parallel_size or 1)


def _plan_role(
    input: PlanServeInput, role: ServeRoleInput
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement]]:
    if role.replica_count < 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} replica count must be positive",
        )
    settings = _settings(role.settings)
    parallelism = _effective_parallelism(role.parallelism)
    replicas = [
        ServeReplicaRequirement(
            id=replica_id(role, replica_index),
            role_id=role.id,
            replica_index=replica_index,
            device_count=_device_count(parallelism),
            ports=[],
            primary_ports=["master"],
            primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/health")),
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
            "TensorRT-LLM single topology does not have a qualified Gateway backend",
        )
    role = require_role(input, ServeRoleKind.serve)
    role_result, replicas = _plan_role(input, role)
    settings = _settings(role_result.effective_settings)
    render_inputs: list[RenderInputDeclaration] = []
    if (
        settings.extra_llm_api_options_patch is not None
        and settings.extra_llm_api_options is not None
    ):
        render_inputs.append(
            RenderInputDeclaration(source_path=_render_source_path(settings.extra_llm_api_options))
        )
    role_result.public_endpoint = _endpoint_requirement()
    role_result.render_inputs = render_inputs
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[],
    )


def _plan_prefill_decode(input: PlanServeInput) -> PlanServeResult:
    if input.kv_transfer != KvTransferMechanism.nixl:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TensorRT-LLM prefill_decode requires NIXL KV transfer",
        )
    backend_pair = (input.gateway_backend, input.pd_router_backend)
    if backend_pair not in {
        ("builtin", "builtin"),
        (_NATIVE_ROUTING_BACKEND, _NATIVE_ROUTING_BACKEND),
    }:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM does not support Gateway/P/D Router pair {backend_pair!r}",
        )
    gateway_backend = input.gateway_backend
    pd_router_backend = input.pd_router_backend
    if gateway_backend is None or pd_router_backend is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TensorRT-LLM prefill_decode requires both frontend backends",
        )
    prefill = require_role(input, ServeRoleKind.prefill)
    decode = require_role(input, ServeRoleKind.decode)
    prefill_result, prefill_replicas = _plan_role(input, prefill)
    decode_result, decode_replicas = _plan_role(input, decode)
    roles = [prefill_result, decode_result]
    replicas = [*prefill_replicas, *decode_replicas]
    for role in roles:
        settings = _settings(role.effective_settings)
        # Engine roles in this topology own additional render flags (e.g.
        # --backend), so the escape hatch is validated against the extended
        # prefill/decode table rather than only the base table.
        validate_extra_args(settings.extra_args or [], _PREFILL_DECODE_OWNED_OPTIONS)
        path = settings.extra_llm_api_options
        if path is not None:
            role.render_inputs = [RenderInputDeclaration(source_path=_render_source_path(path))]
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
                mechanism=KvTransferMechanism.nixl,
            )
        ),
    ]
    if backend_pair == ("builtin", "builtin"):
        implementation = "trtllm"
        implementation_version = "2"
        render_source = RenderSource.control_plane
        readiness = ReadinessProbe(root=ReadinessProbeHttp(path="/healthcheck"))
    else:
        implementation = _NATIVE_ROUTING_BACKEND
        implementation_version = _identity().adapter_version
        render_source = RenderSource.integration
        readiness = ReadinessProbe(root=ReadinessProbeHttp(path="/health"))
    gateway, pd_router = fused_pd_frontend_plans(
        gateway_backend=gateway_backend,
        pd_router_backend=pd_router_backend,
        implementation=implementation,
        implementation_version=implementation_version,
        render_source=render_source,
        endpoint=_endpoint_requirement(),
        gateway_readiness=readiness,
        pd_router_readiness=readiness,
        policies=PdRoutingPolicies(prefill="round_robin", decode="context_first"),
        prefill_role=prefill.id,
        decode_role=decode.id,
        target_scheme=TargetEndpointScheme.http,
    )
    return PlanServeResult(
        integration=_identity(),
        roles=roles,
        replicas=replicas,
        links=links,
        gateway=gateway,
        pd_router=pd_router,
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    if input.profiling is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support profiling capture yet",
        )
    if input.topology == ServeTopology.single:
        return _plan_single(input)
    return _plan_prefill_decode(input)
