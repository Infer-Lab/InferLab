from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    CaptureMechanism,
    CaptureTargetRequirement,
    CaptureWindowControlEndpoint,
    CaptureWindowControlRequirement,
    CaptureWindowHttpActionSpec,
    EndpointProtocol,
    EndpointRequirement,
    HttpActionSpec,
    HttpMethod,
    IntegrationIdentity,
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
    ReadinessProbeHttpTargetRegistry,
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
    ServeRoleResult,
    ServeTopology,
    SettingValue,
    SyntheticAcceptanceOutcome,
    TargetEndpointScheme,
    effective_settings,
    fused_pd_frontend_plans,
    integration_identity,
    replica_id,
    require_role,
)

from .settings import _settings
from .synthetic import resolve_synthetic_acceptance


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-sglang",
        adapter_distribution="inferlab-integration-sglang",
        framework="sglang",
        framework_distribution="sglang",
    )


def _effective_parallelism(declared: Parallelism, role_kind: ServeRoleKind) -> Parallelism:
    """The v1-proven SGLang algebra: `outer.tensor_parallel_size` is the
    total world size (`--tensor-parallel-size`), which attention data and
    context parallelism divide (`--enable-dp-attention`), and which expert
    and expert-data parallelism divide independently.

    Context parallelism is role-dependent: serve and prefill roles lower it
    to prefill CP (`--attention-context-parallel-size` with
    `--enable-prefill-cp`), which divides the attention world together with
    attention DP; the decode role lowers it to decode CP (`--dcp-size`),
    which only has to divide the world and leaves attention TP untouched.
    Applicability beyond this arithmetic (attention architecture, hardware,
    kernel backends) is owned by the engine at launch."""
    outer = declared.outer or ParallelismOuter()
    attention = declared.attention or ParallelismAttention()
    experts = declared.experts or ParallelismExperts()
    outer_tp = outer.tensor_parallel_size or 1
    outer_pp = outer.pipeline_parallel_size or 1
    attention_dp = attention.data_parallel_size or 1
    attention_cp = attention.context_parallel_size or 1
    decode_role = role_kind == ServeRoleKind.decode
    if decode_role:
        attention_divisor = attention_dp
        attention_divisor_description = (
            f"SGLang decode-role attention.data_parallel_size ({attention_dp})"
        )
        attention_tp_derivation = "outer.tensor_parallel_size / attention.data_parallel_size"
    else:
        attention_divisor = attention_dp * attention_cp
        attention_divisor_description = (
            "SGLang attention.data_parallel_size * attention.context_parallel_size "
            f"({attention_divisor})"
        )
        attention_tp_derivation = (
            "outer.tensor_parallel_size / attention.data_parallel_size / "
            "attention.context_parallel_size"
        )
    if outer_tp % attention_divisor != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"{attention_divisor_description} must divide outer.tensor_parallel_size ({outer_tp})",
        )
    if decode_role and outer_tp % attention_cp != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"SGLang decode-role attention.context_parallel_size ({attention_cp}) "
            f"must divide outer.tensor_parallel_size ({outer_tp})",
        )
    effective_attention_tp = outer_tp // attention_divisor
    if (
        attention.tensor_parallel_size is not None
        and attention.tensor_parallel_size != effective_attention_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"SGLang effective attention.tensor_parallel_size is "
            f"{attention_tp_derivation} ({effective_attention_tp})",
        )
    expert_ep = experts.expert_parallel_size or 1
    expert_dp = experts.data_parallel_size or 1
    expert_divisor = expert_ep * expert_dp
    if outer_tp % expert_divisor != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"SGLang experts.expert_parallel_size * experts.data_parallel_size "
            f"({expert_divisor}) must divide outer.tensor_parallel_size ({outer_tp})",
        )
    effective_expert_tp = outer_tp // expert_divisor
    if (
        experts.tensor_parallel_size is not None
        and experts.tensor_parallel_size != effective_expert_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "SGLang effective experts.tensor_parallel_size is outer.tensor_parallel_size / "
            f"experts.expert_parallel_size / experts.data_parallel_size "
            f"({effective_expert_tp})",
        )
    # The MoE-DP combination limits below mirror the asserts the vendored
    # SGLang enforces at server start (server_args.py), expressed here so an
    # impossible shape rejects at planning instead of dying in the server. They
    # were verified against the SGLang version pinned by the workspace serving
    # environment (the committed workspace pixi.toml is the pin authority) and
    # must be re-verified against server_args.py whenever that pin moves.
    if expert_dp > 1:
        if outer_pp > 1:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                "SGLang does not support pipeline parallelism with "
                f"experts.data_parallel_size > 1 (pp={outer_pp}, moe-dp={expert_dp})",
            )
        # The decode role lowers CP to --dcp-size, so its launch-time
        # attention CP is 1 regardless of the declared degree.
        lowered_attention_cp = 1 if decode_role else attention_cp
        if lowered_attention_cp != expert_dp:
            message = (
                "SGLang requires attention.context_parallel_size to equal "
                f"experts.data_parallel_size when the latter exceeds 1 "
                f"(declared attention.context_parallel_size={attention_cp}, "
                f"moe-dp={expert_dp})"
            )
            if decode_role:
                message += (
                    "; the decode role lowers CP to --dcp-size, so launch-time "
                    f"attention CP is {lowered_attention_cp}"
                )
            raise AdapterOperationError(AdapterErrorCode.invalid_settings, message)
        if expert_ep > 1 and expert_divisor != outer_tp:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                "SGLang requires experts.expert_parallel_size * experts.data_parallel_size "
                f"to equal outer.tensor_parallel_size when both exceed 1 "
                f"(ep={expert_ep} * moe-dp={expert_dp} != tp={outer_tp})",
            )
    return Parallelism(
        outer=ParallelismOuter(
            tensor_parallel_size=outer_tp,
            pipeline_parallel_size=outer_pp,
        ),
        attention=ParallelismAttention(
            tensor_parallel_size=effective_attention_tp,
            data_parallel_size=attention_dp,
            context_parallel_size=attention_cp,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=effective_expert_tp,
            data_parallel_size=expert_dp,
            expert_parallel_size=expert_ep,
            dense_tensor_parallel_size=experts.dense_tensor_parallel_size or 1,
        ),
    )


def _device_count(parallelism: Parallelism) -> int:
    outer = parallelism.outer or ParallelismOuter()
    return (outer.tensor_parallel_size or 1) * (outer.pipeline_parallel_size or 1)


def _capture_target(profiling: CaptureMechanism | None) -> CaptureTargetRequirement | None:
    if profiling is None:
        return None
    # Managed collection rides SGLang's CUDA_PROFILER activity so the managed
    # Nsight Systems range opens and closes with the window; engine trace
    # profiles through the framework's own torch profiler, whose output
    # directory is rendered into the process environment (the plan-time
    # action body is static and cannot carry it).
    activities = "CUDA_PROFILER" if profiling == CaptureMechanism.managed_collection else "GPU"
    return CaptureTargetRequirement(
        mechanism=profiling,
        window_control=CaptureWindowControlRequirement(
            endpoint=CaptureWindowControlEndpoint.replica_entry,
            start=CaptureWindowHttpActionSpec(
                method=HttpMethod(),
                path="/start_profile",
                body={"activities": SettingValue(root=[SettingValue(root=activities)])},
            ),
            stop=CaptureWindowHttpActionSpec(method=HttpMethod(), path="/stop_profile"),
        ),
    )


def _plan_role(
    input: PlanServeInput,
    role: ServeRoleInput,
    ports: list[str],
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement], SyntheticAcceptanceOutcome | None]:
    if role.replica_count < 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} replica count must be positive",
        )
    settings = _settings(role.settings)
    outcome: SyntheticAcceptanceOutcome | None = None
    if input.synthetic_acceptance is not None:
        outcome = resolve_synthetic_acceptance(settings, input.synthetic_acceptance, role.id)
    parallelism = _effective_parallelism(role.parallelism, role.kind)
    replicas = [
        ServeReplicaRequirement(
            id=replica_id(role, replica_index),
            role_id=role.id,
            replica_index=replica_index,
            device_count=_device_count(parallelism),
            ports=list(ports),
            primary_ports=["master"],
            primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/v1/models")),
            worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
            capture_target=_capture_target(input.profiling),
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
        outcome,
    )


def _endpoint_requirement(
    *,
    include_server_metrics: bool,
    include_cache_reporting: bool,
    include_conditioning_fanout: bool = False,
) -> EndpointRequirement:
    return EndpointRequirement(
        protocol=EndpointProtocol(),
        completions_path="/v1/completions",
        chat_completions_path="/v1/chat/completions",
        server_metrics=(
            ServerMetricsEndpointRequirement(path="/metrics") if include_server_metrics else None
        ),
        prefix_cache_reset=HttpActionSpec(method=HttpMethod(), path="/flush_cache"),
        prefix_cache_conditioning=(
            HttpActionSpec(method=HttpMethod(), path="/prime_prefix_cache")
            if include_conditioning_fanout
            else None
        ),
        prompt_cache_read_zero_representation=(
            PromptCacheReadZeroRepresentation.omitted if include_cache_reporting else None
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
            "SGLang single topology does not have a qualified Gateway backend",
        )
    role = require_role(input, ServeRoleKind.serve)
    role_result, replicas, outcome = _plan_role(input, role, [])
    settings = _settings(role_result.effective_settings)
    role_result.public_endpoint = _endpoint_requirement(
        include_server_metrics=settings.enable_metrics,
        include_cache_reporting=settings.enable_cache_report,
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
        ("sglang-router", "sglang-router"),
    }:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"SGLang does not support Gateway/P/D Router pair {backend_pair!r}",
        )
    gateway_backend = input.gateway_backend
    pd_router_backend = input.pd_router_backend
    if gateway_backend is None or pd_router_backend is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "SGLang prefill_decode requires both frontend backends",
        )
    prefill = require_role(input, ServeRoleKind.prefill)
    decode = require_role(input, ServeRoleKind.decode)
    prefill_result, prefill_replicas, prefill_outcome = _plan_role(input, prefill, ["bootstrap"])
    decode_result, decode_replicas, decode_outcome = _plan_role(input, decode, [])
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
        ServeRoleLink(
            root=ServeRoleLinkBootstrap(
                source="pd_router",
                target=prefill.id,
                port="bootstrap",
            )
        ),
    ]
    if backend_pair == ("builtin", "builtin"):
        implementation = "sglang"
        implementation_version = "2"
        render_source = RenderSource.control_plane
        gateway_readiness = ReadinessProbe(root=ReadinessProbeHttp(path="/healthcheck"))
        pd_router_readiness = gateway_readiness
        # The built-in proxies forward engine responses verbatim, so the
        # gateway endpoint exposes backend cache-read usage whenever both
        # serving roles enable cache report.
        cache_reporting = (
            _settings(prefill_result.effective_settings).enable_cache_report
            and _settings(decode_result.effective_settings).enable_cache_report
        )
    else:
        implementation = "sglang-router"
        implementation_version = _identity().adapter_version
        render_source = RenderSource.integration
        gateway_readiness = ReadinessProbe(root=ReadinessProbeHttp(path="/readiness"))
        pd_router_readiness = ReadinessProbe(
            root=ReadinessProbeHttpTargetRegistry(
                target_scheme=TargetEndpointScheme.http,
                readiness_path="/readiness",
                registry_path="/workers",
                targets_field="workers",
                target_url_field="url",
                target_role_field="worker_type",
                target_healthy_field="is_healthy",
                target_bootstrap_port_field="bootstrap_port",
                prefill_role_value="prefill",
                decode_role_value="decode",
                prefill_bootstrap_port="bootstrap",
            )
        )
        cache_reporting = False
    gateway, pd_router = fused_pd_frontend_plans(
        gateway_backend=gateway_backend,
        pd_router_backend=pd_router_backend,
        implementation=implementation,
        implementation_version=implementation_version,
        render_source=render_source,
        endpoint=_endpoint_requirement(
            include_server_metrics=False,
            include_cache_reporting=cache_reporting,
            include_conditioning_fanout=backend_pair == ("builtin", "builtin"),
        ),
        gateway_readiness=gateway_readiness,
        pd_router_readiness=pd_router_readiness,
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
