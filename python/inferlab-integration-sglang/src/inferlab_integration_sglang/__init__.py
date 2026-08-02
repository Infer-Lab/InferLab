from pathlib import Path

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
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PdRoutingPolicies,
    PlanServeInput,
    PlanServeResult,
    ProcessSpec,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    ReadinessProbeProcessAlive,
    RenderedServeProcess,
    RenderServeInput,
    RenderServeResult,
    RenderSource,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
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
    TargetEndpointScheme,
    append_option,
    effective_settings,
    fused_pd_frontend_plans,
    integration_identity,
    merge_serve_args,
    rendered_frontend,
    rendered_model_rank,
    replica_id,
    require_integration_fused_frontend,
    require_role,
    split_serve_allocations,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field

_ROUTER_WORKER_STARTUP_TIMEOUT_SECS = 2_147_483_647

_INFERLAB_OPTION_ARITY: dict[str, int | None] = {
    "--attention-context-parallel-size": 1,
    "--context-length": 1,
    "--cuda-graph-max-bs-decode": 1,
    "--data-parallel-size": 1,
    "--disaggregation-bootstrap-port": 1,
    "--disaggregation-mode": 1,
    "--disaggregation-transfer-backend": 1,
    "--dp-size": 1,
    "--enable-dp-attention": 0,
    "--enable-metrics": 0,
    "--ep-size": 1,
    "--expert-parallel-size": 1,
    "--host": 1,
    "--kv-cache-dtype": 1,
    "--mem-fraction-static": 1,
    "--model-path": 1,
    "--moe-data-parallel-size": 1,
    "--moe-dense-tp-size": 1,
    "--moe-runner-backend": 1,
    "--pipeline-parallel-size": 1,
    "--port": 1,
    "--pp-size": 1,
    "--served-model-name": 1,
    "--tensor-parallel-size": 1,
    "--tp-size": 1,
    "--trust-remote-code": 0,
}

_RUNTIME_CACHE_SUBDIRS = {
    "DG_JIT_CACHE_DIR": "deep_gemm_jit",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASHINFER_CUBIN_DIR": "flashinfer_cubin",
    "TILELANG_CACHE_DIR": "tilelang",
    "TILELANG_TMP_DIR": "tilelang/tmp",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}


class SglangServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_length: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    mem_fraction_static: float | None = Field(default=None, gt=0.0, le=1.0)
    cuda_graph_max_bs_decode: int | None = Field(default=None, ge=1)
    moe_runner_backend: str | None = None
    trust_remote_code: bool = False
    enable_metrics: bool = False
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


def _runtime_cache_env(root: str) -> dict[str, str]:
    cache_root = Path(root)
    return {
        name: str(cache_root / subdirectory)
        for name, subdirectory in _RUNTIME_CACHE_SUBDIRS.items()
    }


def _settings(values: dict[str, SettingValue]) -> SglangServeSettings:
    return validate_settings(SglangServeSettings, values)


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-sglang",
        adapter_distribution="inferlab-integration-sglang",
        framework="sglang",
        framework_distribution="sglang",
    )


def _effective_parallelism(declared: Parallelism) -> Parallelism:
    """The v1-proven SGLang algebra: `outer.tensor_parallel_size` is the
    total world size (`--tensor-parallel-size`), which attention data and
    context parallelism divide (`--enable-dp-attention`), and which expert
    and expert-data parallelism divide independently."""
    outer = declared.outer or ParallelismOuter()
    attention = declared.attention or ParallelismAttention()
    experts = declared.experts or ParallelismExperts()
    outer_tp = outer.tensor_parallel_size or 1
    outer_pp = outer.pipeline_parallel_size or 1
    attention_dp = attention.data_parallel_size or 1
    attention_cp = attention.context_parallel_size or 1
    attention_divisor = attention_dp * attention_cp
    if outer_tp % attention_divisor != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"SGLang attention.data_parallel_size * attention.context_parallel_size "
            f"({attention_divisor}) must divide outer.tensor_parallel_size ({outer_tp})",
        )
    effective_attention_tp = outer_tp // attention_divisor
    if (
        attention.tensor_parallel_size is not None
        and attention.tensor_parallel_size != effective_attention_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "SGLang effective attention.tensor_parallel_size is outer.tensor_parallel_size / "
            f"attention.data_parallel_size / attention.context_parallel_size "
            f"({effective_attention_tp})",
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
        if attention_cp != expert_dp:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                "SGLang requires attention.context_parallel_size to equal "
                f"experts.data_parallel_size when the latter exceeds 1 "
                f"(cp={attention_cp}, moe-dp={expert_dp})",
            )
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


def _capture_target(profiling: bool) -> CaptureTargetRequirement | None:
    if not profiling:
        return None
    return CaptureTargetRequirement(
        window_control=CaptureWindowControlRequirement(
            endpoint=CaptureWindowControlEndpoint.replica_entry,
            start=CaptureWindowHttpActionSpec(
                method=HttpMethod(),
                path="/start_profile",
                body={"activities": SettingValue(root=[SettingValue(root="CUDA_PROFILER")])},
            ),
            stop=CaptureWindowHttpActionSpec(method=HttpMethod(), path="/stop_profile"),
        )
    )


def _plan_role(
    input: PlanServeInput,
    role: ServeRoleInput,
    ports: list[str],
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
    )


def _endpoint_requirement(*, include_server_metrics: bool) -> EndpointRequirement:
    return EndpointRequirement(
        protocol=EndpointProtocol(),
        completions_path="/v1/completions",
        chat_completions_path="/v1/chat/completions",
        server_metrics=(
            ServerMetricsEndpointRequirement(path="/metrics") if include_server_metrics else None
        ),
        prefix_cache_reset=HttpActionSpec(method=HttpMethod(), path="/flush_cache"),
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
    role_result, replicas = _plan_role(input, role, [])
    settings = _settings(role_result.effective_settings)
    role_result.public_endpoint = _endpoint_requirement(
        include_server_metrics=settings.enable_metrics
    )
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[],
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
    prefill_result, prefill_replicas = _plan_role(input, prefill, ["bootstrap"])
    decode_result, decode_replicas = _plan_role(input, decode, [])
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
    gateway, pd_router = fused_pd_frontend_plans(
        gateway_backend=gateway_backend,
        pd_router_backend=pd_router_backend,
        implementation=implementation,
        implementation_version=implementation_version,
        render_source=render_source,
        endpoint=_endpoint_requirement(include_server_metrics=False),
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
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    if input.topology == ServeTopology.single:
        return _plan_single(input)
    return _plan_prefill_decode(input)


def _render_process(
    input: RenderServeInput,
    allocation: ServeProcessAllocationModelRank,
) -> RenderedServeProcess:
    settings = _settings(allocation.effective_settings)
    outer = allocation.effective_parallelism.outer or ParallelismOuter()
    attention = allocation.effective_parallelism.attention or ParallelismAttention()
    experts = allocation.effective_parallelism.experts or ParallelismExperts()
    if allocation.endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"serving allocation {allocation.process!r} is missing its endpoint",
        )
    endpoint = allocation.endpoint
    argv = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        allocation.model_locator,
    ]
    inferlab_args = [
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        "--served-model-name",
        input.model.served_name,
        "--tensor-parallel-size",
        str(outer.tensor_parallel_size or 1),
    ]
    if (outer.pipeline_parallel_size or 1) != 1:
        inferlab_args.extend(["--pipeline-parallel-size", str(outer.pipeline_parallel_size)])
    attention_dp = attention.data_parallel_size or 1
    attention_cp = attention.context_parallel_size or 1
    if attention_dp != 1:
        inferlab_args.extend(["--data-parallel-size", str(attention_dp)])
    if attention_cp != 1:
        inferlab_args.extend(["--attention-context-parallel-size", str(attention_cp)])
    if attention_dp != 1 or attention_cp != 1:
        inferlab_args.append("--enable-dp-attention")
    if (experts.expert_parallel_size or 1) != 1:
        inferlab_args.extend(["--expert-parallel-size", str(experts.expert_parallel_size)])
    if (experts.data_parallel_size or 1) != 1:
        inferlab_args.extend(["--moe-data-parallel-size", str(experts.data_parallel_size)])
    if (experts.dense_tensor_parallel_size or 1) != 1:
        inferlab_args.extend(["--moe-dense-tp-size", str(experts.dense_tensor_parallel_size)])
    append_option(inferlab_args, "--context-length", settings.context_length)
    append_option(inferlab_args, "--kv-cache-dtype", settings.kv_cache_dtype)
    append_option(inferlab_args, "--mem-fraction-static", settings.mem_fraction_static)
    append_option(
        inferlab_args,
        "--cuda-graph-max-bs-decode",
        settings.cuda_graph_max_bs_decode,
    )
    append_option(inferlab_args, "--moe-runner-backend", settings.moe_runner_backend)
    if settings.trust_remote_code:
        inferlab_args.append("--trust-remote-code")
    if settings.enable_metrics:
        inferlab_args.append("--enable-metrics")
    if input.topology == ServeTopology.prefill_decode:
        transport = input.kv_transfer
        if transport is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "prefill_decode render is missing its KV-transfer mechanism",
            )
        if allocation.role_kind == ServeRoleKind.prefill:
            mode = "prefill"
        elif allocation.role_kind == ServeRoleKind.decode:
            mode = "decode"
        else:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill_decode allocation has unsupported role {allocation.role!r}",
            )
        inferlab_args.extend(
            [
                "--disaggregation-mode",
                mode,
                "--disaggregation-transfer-backend",
                transport.value,
            ]
        )
        if allocation.role_kind == ServeRoleKind.prefill:
            bootstrap = allocation.ports.get("bootstrap")
            if bootstrap is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"prefill process {allocation.process!r} is missing its bootstrap port",
                )
            inferlab_args.extend(["--disaggregation-bootstrap-port", str(bootstrap.port)])
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, _INFERLAB_OPTION_ARITY))
    process_env = _runtime_cache_env(allocation.cache)
    process_env.update(settings.extra_env or {})
    return rendered_model_rank(
        allocation,
        ProcessSpec(argv=argv, env=process_env),
    )


def _render_router(
    allocation: ServeProcessAllocationFrontend,
    model_allocations: list[ServeProcessAllocationModelRank],
) -> RenderedServeProcess:
    prefill = [
        item
        for item in model_allocations
        if item.role_kind == ServeRoleKind.prefill and item.rank == 0
    ]
    decode = [
        item
        for item in model_allocations
        if item.role_kind == ServeRoleKind.decode and item.rank == 0
    ]
    endpoint = allocation.endpoint
    argv = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        "--worker-startup-timeout-secs",
        str(_ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
        "--pd-disaggregation",
    ]
    for item in prefill:
        bootstrap = item.ports.get("bootstrap")
        if bootstrap is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill replica {item.replica!r} is missing its bootstrap port",
            )
        if item.endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill allocation {item.process!r} has no endpoint",
            )
        argv.extend(
            [
                "--prefill",
                f"http://{item.endpoint.host}:{item.endpoint.port}",
                str(bootstrap.port),
            ]
        )
    for item in decode:
        if item.endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"decode allocation {item.process!r} has no endpoint",
            )
        argv.extend(["--decode", f"http://{item.endpoint.host}:{item.endpoint.port}"])
    argv.extend(["--policy", "round_robin"])
    return rendered_frontend(allocation, ProcessSpec(argv=argv, env={}))


def render_serve(input: RenderServeInput) -> RenderServeResult:
    if not input.allocations:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request, "serve allocation must not be empty"
        )
    allocations, model_allocations = split_serve_allocations(input.allocations)
    processes: list[RenderedServeProcess] = []
    for allocation in allocations:
        if isinstance(allocation, ServeProcessAllocationModelRank):
            if allocation.rank_count > 1:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    "the SGLang integration does not support multi-node serving yet",
                )
            processes.append(_render_process(input, allocation))
        elif isinstance(allocation, ServeProcessAllocationFrontend):
            require_integration_fused_frontend(
                allocation,
                gateway_backend="sglang-router",
                pd_router_backend="sglang-router",
            )
            processes.append(_render_router(allocation, model_allocations))
    return RenderServeResult(integration=_identity(), processes=processes)


__all__ = ["SglangServeSettings", "plan_serve", "render_serve"]
