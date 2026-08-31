from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    CaptureTargetRequirement,
    CaptureWindowControlEndpoint,
    CaptureWindowControlRequirement,
    CaptureWindowHttpActionSpec,
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
    PromptCacheReadZeroRepresentation,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeProcessAlive,
    RenderSource,
    ServeReplicaRequirement,
    ServerMetricsEndpointRequirement,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkBootstrap,
    ServeRoleLinkKvTransfer,
    ServeRoleLinkRequestRouting,
    ServeRoleLinkSideChannel,
    ServeRoleResult,
    ServeTopology,
    SyntheticAcceptanceOutcome,
    TargetEndpointScheme,
    effective_settings,
    fused_pd_frontend_plans,
    integration_identity,
    replica_id,
    require_role,
)

from .settings import _settings
from .synthetic import apply_synthetic_acceptance


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-vllm",
        adapter_distribution="inferlab-integration-vllm",
        framework="vllm",
        framework_distribution="vllm",
    )


def _effective_parallelism(declared: Parallelism, role_kind: ServeRoleKind) -> Parallelism:
    """The vLLM algebra: attention runs tensor-parallel across
    `outer.tensor_parallel_size` and data-parallel across
    `attention.data_parallel_size`, so the MoE layers span the product of both
    (`moe_world_size = outer.tensor_parallel_size * attention.data_parallel_size`)
    — every attention rank hosts experts. That world is decomposed one of two
    ways: with expert parallelism the experts shard across it
    (expert_ep = moe_world_size, expert_tp = 1); otherwise they are
    tensor-parallel across it (expert_tp = moe_world_size, expert_ep = 1).
    vLLM supports neither independent expert data parallelism nor a separate
    dense tensor-parallel size, so both stay 1.

    `attention.context_parallel_size` lowers per role. For the `prefill` role
    it is prefill context parallelism (`--prefill-context-parallel-size`):
    it multiplies the role's device count and joins the MoE expert world
    (moe_world_size gains a context-parallel factor), and it excludes attention
    data parallelism. For the `serve` and `decode` roles it is decode context
    parallelism (`--decode-context-parallel-size`): it splits KV inside the
    existing TP group, so it must divide `outer.tensor_parallel_size`, and it
    neither changes the device count nor enters the expert world. Model-level
    applicability (attention architecture, head counts, backend selection) is
    owned by vLLM itself and surfaces as a launch failure."""
    outer = declared.outer or ParallelismOuter()
    attention = declared.attention or ParallelismAttention()
    experts = declared.experts or ParallelismExperts()
    outer_tp = outer.tensor_parallel_size or 1
    outer_pp = outer.pipeline_parallel_size or 1
    attention_dp = attention.data_parallel_size or 1
    attention_cp = attention.context_parallel_size or 1
    is_prefill = role_kind == ServeRoleKind.prefill
    expert_cp = attention_cp if is_prefill else 1
    moe_world_size = outer_tp * expert_cp * attention_dp
    requested_ep = experts.expert_parallel_size or 1
    uses_ep = requested_ep > 1
    effective_expert_tp = 1 if uses_ep else moe_world_size
    effective_expert_ep = moe_world_size if uses_ep else 1

    if attention.tensor_parallel_size is not None and attention.tensor_parallel_size != outer_tp:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM attention.tensor_parallel_size must equal outer.tensor_parallel_size",
        )
    if attention_cp > 1:
        if is_prefill:
            if attention_dp != 1:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_settings,
                    "vLLM prefill context parallelism excludes "
                    "attention.data_parallel_size greater than 1 "
                    f"(declared attention.context_parallel_size={attention_cp}, "
                    f"attention.data_parallel_size={attention_dp})",
                )
        elif outer_tp % attention_cp != 0:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                "vLLM attention.context_parallel_size "
                f"({attention_cp}) must divide outer.tensor_parallel_size ({outer_tp})",
            )
    if (
        experts.tensor_parallel_size is not None
        and experts.tensor_parallel_size != effective_expert_tp
    ):
        effective_expert_tp_derivation = (
            "1 under expert parallelism"
            if uses_ep
            else "outer.tensor_parallel_size * "
            + ("attention.context_parallel_size * " if is_prefill else "")
            + "attention.data_parallel_size"
        )
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"vLLM effective experts.tensor_parallel_size is "
            f"{effective_expert_tp_derivation} ({effective_expert_tp}), but "
            f"declared {experts.tensor_parallel_size}",
        )
    if experts.data_parallel_size is not None and experts.data_parallel_size != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM does not support independent experts.data_parallel_size",
        )
    if experts.dense_tensor_parallel_size is not None and experts.dense_tensor_parallel_size != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM does not support experts.dense_tensor_parallel_size greater than 1",
        )
    if requested_ep not in (1, effective_expert_ep):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM experts.expert_parallel_size must equal "
            "outer.tensor_parallel_size * "
            + ("attention.context_parallel_size * " if is_prefill else "")
            + f"attention.data_parallel_size ({effective_expert_ep})",
        )

    return Parallelism(
        outer=ParallelismOuter(
            tensor_parallel_size=outer_tp,
            pipeline_parallel_size=outer_pp,
        ),
        attention=ParallelismAttention(
            tensor_parallel_size=outer_tp,
            data_parallel_size=attention_dp,
            context_parallel_size=attention_cp,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=effective_expert_tp,
            data_parallel_size=1,
            expert_parallel_size=effective_expert_ep,
            dense_tensor_parallel_size=1,
        ),
    )


def _device_count(parallelism: Parallelism, role_kind: ServeRoleKind) -> int:
    outer = parallelism.outer or ParallelismOuter()
    attention = parallelism.attention or ParallelismAttention()
    # Decode context parallelism splits KV inside the existing TP group, so
    # only prefill context parallelism multiplies the whole-replica devices.
    context = attention.context_parallel_size or 1 if role_kind == ServeRoleKind.prefill else 1
    return (
        (outer.tensor_parallel_size or 1)
        * (outer.pipeline_parallel_size or 1)
        * (attention.data_parallel_size or 1)
        * context
    )


def _plan_role(
    input: PlanServeInput,
    role: ServeRoleInput,
    role_ports: list[str],
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement], SyntheticAcceptanceOutcome | None]:
    if role.replica_count < 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} replica count must be positive",
        )
    settings = _settings(role.settings)
    outcome: SyntheticAcceptanceOutcome | None = None
    if input.synthetic_acceptance is not None:
        outcome = apply_synthetic_acceptance(settings, input.synthetic_acceptance, role.id)
    parallelism = _effective_parallelism(role.parallelism, role.kind)
    device_count = _device_count(parallelism, role.kind)
    replicas = []
    mechanism = input.profiling
    for replica_index in range(role.replica_count):
        planned_replica_id = replica_id(role, replica_index)
        capture_target = (
            CaptureTargetRequirement(
                mechanism=mechanism,
                window_control=CaptureWindowControlRequirement(
                    endpoint=CaptureWindowControlEndpoint.replica_entry,
                    start=CaptureWindowHttpActionSpec(method=HttpMethod(), path="/start_profile"),
                    stop=CaptureWindowHttpActionSpec(method=HttpMethod(), path="/stop_profile"),
                ),
            )
            if mechanism is not None
            else None
        )
        replicas.append(
            ServeReplicaRequirement(
                id=planned_replica_id,
                role_id=role.id,
                replica_index=replica_index,
                device_count=device_count,
                ports=list(role_ports),
                primary_ports=["master"],
                primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/v1/models")),
                worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
                capture_target=capture_target,
            )
        )
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
        outcome,
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
            "vLLM single topology does not have a qualified Gateway backend",
        )
    role = require_role(input, ServeRoleKind.serve)
    role_result, replicas, outcome = _plan_role(input, role, [])
    settings = _settings(role_result.effective_settings)
    role_result.public_endpoint = EndpointRequirement(
        protocol=EndpointProtocol(),
        completions_path="/v1/completions",
        chat_completions_path="/v1/chat/completions",
        server_metrics=ServerMetricsEndpointRequirement(path="/metrics"),
        prefix_cache_reset=HttpActionSpec(
            method=HttpMethod(),
            path="/reset_prefix_cache",
        ),
        prompt_cache_read_zero_representation=(
            PromptCacheReadZeroRepresentation.explicit
            if settings.enable_prompt_tokens_details
            else None
        ),
    )
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[],
        synthetic_acceptance=outcome,
    )


def _plan_prefill_decode(input: PlanServeInput) -> PlanServeResult:
    transport = input.kv_transfer
    if transport is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "prefill_decode topology requires a KV-transfer mechanism",
        )
    backend_pair = (input.gateway_backend, input.pd_router_backend)
    if backend_pair not in {
        ("builtin", "builtin"),
        ("vllm-router", "vllm-router"),
    }:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"vLLM does not support Gateway/P/D Router pair {backend_pair!r}",
        )
    gateway_backend = input.gateway_backend
    pd_router_backend = input.pd_router_backend
    if gateway_backend is None or pd_router_backend is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM prefill_decode requires both frontend backends",
        )
    prefill = require_role(input, ServeRoleKind.prefill)
    decode = require_role(input, ServeRoleKind.decode)
    prefill_ports = ["bootstrap" if transport == KvTransferMechanism.mooncake else "side_channel"]
    decode_ports = [] if transport == KvTransferMechanism.mooncake else ["side_channel"]
    prefill_result, prefill_replicas, prefill_outcome = _plan_role(input, prefill, prefill_ports)
    decode_result, decode_replicas, decode_outcome = _plan_role(input, decode, decode_ports)
    if prefill_outcome != decode_outcome:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the prefill and decode roles resolve different synthetic acceptance "
            f"outcomes ({prefill_outcome} vs {decode_outcome}); the plan response "
            "carries one effective acceptance length, so both roles must determine "
            "the same draft count",
        )
    roles = [prefill_result, decode_result]
    replicas = [*prefill_replicas, *decode_replicas]
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
                mechanism=transport,
            )
        ),
    ]
    if transport == KvTransferMechanism.mooncake:
        links.append(
            ServeRoleLink(
                root=ServeRoleLinkBootstrap(
                    source="pd_router",
                    target=prefill.id,
                    port="bootstrap",
                )
            )
        )
    else:
        links.append(
            ServeRoleLink(
                root=ServeRoleLinkSideChannel(
                    source=prefill.id,
                    target=decode.id,
                    port="side_channel",
                )
            )
        )

    if backend_pair == ("builtin", "builtin"):
        implementation = (
            "vllm_mooncake" if transport == KvTransferMechanism.mooncake else "vllm_nixl"
        )
        implementation_version = "1"
        render_source = RenderSource.control_plane
        frontend_readiness = ReadinessProbe(root=ReadinessProbeHttp(path="/healthcheck"))
        # The built-in proxies forward engine responses verbatim, so the
        # gateway endpoint exposes backend cache-read usage whenever both
        # serving roles report prompt-tokens details.
        cache_read = (
            PromptCacheReadZeroRepresentation.explicit
            if _settings(prefill_result.effective_settings).enable_prompt_tokens_details
            and _settings(decode_result.effective_settings).enable_prompt_tokens_details
            else None
        )
        frontend_endpoint = EndpointRequirement(
            protocol=EndpointProtocol(),
            completions_path="/v1/completions",
            chat_completions_path="/v1/chat/completions",
            prefix_cache_reset=HttpActionSpec(
                method=HttpMethod(),
                path="/reset_prefix_cache",
            ),
            prefix_cache_conditioning=HttpActionSpec(
                method=HttpMethod(),
                path="/prime_prefix_cache",
            ),
            prompt_cache_read_zero_representation=cache_read,
        )
    else:
        implementation = "vllm-router"
        implementation_version = _identity().adapter_version
        render_source = RenderSource.integration
        frontend_readiness = ReadinessProbe(root=ReadinessProbeHttp(path="/v1/models"))
        frontend_endpoint = EndpointRequirement(
            protocol=EndpointProtocol(),
            completions_path="/v1/completions",
            chat_completions_path="/v1/chat/completions",
        )

    gateway, pd_router = fused_pd_frontend_plans(
        gateway_backend=gateway_backend,
        pd_router_backend=pd_router_backend,
        implementation=implementation,
        implementation_version=implementation_version,
        render_source=render_source,
        endpoint=frontend_endpoint,
        gateway_readiness=frontend_readiness,
        pd_router_readiness=frontend_readiness,
        policies=PdRoutingPolicies(prefill="round_robin", decode="round_robin"),
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
        synthetic_acceptance=prefill_outcome,
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    if input.topology == ServeTopology.single:
        return _plan_single(input)
    return _plan_prefill_decode(input)
