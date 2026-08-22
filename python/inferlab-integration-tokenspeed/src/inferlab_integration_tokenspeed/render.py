from inferlab_adapter_sdk import (
    ROUTER_WORKER_STARTUP_TIMEOUT_SECS,
    AdapterErrorCode,
    AdapterOperationError,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    ProcessSpec,
    RenderedServeProcess,
    RenderServeInput,
    RenderServeResult,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
    ServeRoleKind,
    ServeTopology,
    append_option,
    merge_serve_args,
    rank_zero_allocations,
    rendered_frontend,
    rendered_model_rank,
    require_integration_fused_frontend,
    runtime_cache_env,
    split_serve_allocations,
)

from .plan import _identity
from .settings import _INFERLAB_OWNED_OPTIONS, _settings


def _render_worker(
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
    world_size = outer.tensor_parallel_size or 1
    if input.topology == ServeTopology.single:
        control_endpoint = allocation.ports.get("control")
        if control_endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "TokenSpeed serve allocation is missing its control port",
            )
        argv = [
            "python3",
            "-m",
            "tokenspeed.cli",
            "serve",
            allocation.model_locator,
        ]
        endpoint_args = ["--control-port", str(control_endpoint.port)]
    elif allocation.role_kind in {ServeRoleKind.prefill, ServeRoleKind.decode}:
        argv = [
            "python3",
            "-m",
            "smg_grpc_servicer.tokenspeed",
            "--model",
            allocation.model_locator,
        ]
        endpoint_args = []
    else:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"prefill_decode allocation has unsupported role {allocation.role!r}",
        )
    dist_init_endpoint = allocation.ports.get("dist_init")
    if dist_init_endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "TokenSpeed serve allocation is missing its distributed initialization port",
        )
    inferlab_args = [
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        *endpoint_args,
        "--dist-init-addr",
        f"{dist_init_endpoint.host}:{dist_init_endpoint.port}",
        "--served-model-name",
        input.model.served_name,
        "--world-size",
        str(world_size),
        "--nprocs-per-node",
        str(world_size),
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--attn-tp-size",
        str(attention.tensor_parallel_size or 1),
        "--data-parallel-size",
        str(attention.data_parallel_size or 1),
        "--dense-tp-size",
        str(experts.dense_tensor_parallel_size or world_size),
        "--moe-tp-size",
        str(experts.tensor_parallel_size or 1),
        "--expert-parallel-size",
        str(experts.expert_parallel_size or 1),
    ]
    if input.topology == ServeTopology.prefill_decode:
        inferlab_args.extend(
            [
                "--disaggregation-mode",
                allocation.role_kind.value,
                "--disaggregation-transfer-backend",
                "mooncake",
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
    append_option(inferlab_args, "--max-model-len", settings.max_model_len)
    append_option(inferlab_args, "--kv-cache-dtype", settings.kv_cache_dtype)
    append_option(inferlab_args, "--gpu-memory-utilization", settings.gpu_memory_utilization)
    append_option(inferlab_args, "--max-num-seqs", settings.max_num_seqs)
    append_option(inferlab_args, "--max-total-tokens", settings.max_total_tokens)
    append_option(inferlab_args, "--chunked-prefill-size", settings.chunked_prefill_size)
    append_option(inferlab_args, "--block-size", settings.block_size)
    append_option(inferlab_args, "--moe-backend", settings.moe_backend)
    append_option(inferlab_args, "--attention-backend", settings.attention_backend)
    append_option(inferlab_args, "--sampling-backend", settings.sampling_backend)
    if settings.attention_use_fp4_indexer_cache:
        inferlab_args.append("--attention-use-fp4-indexer-cache")
    if settings.enable_mixed_batch:
        inferlab_args.append("--enable-mixed-batch")
    inferlab_args.append(
        "--enable-prefix-caching"
        if settings.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
    if settings.disable_kvstore:
        inferlab_args.append("--disable-kvstore")
    if settings.trust_remote_code:
        inferlab_args.append("--trust-remote-code")
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, _INFERLAB_OWNED_OPTIONS))

    process_env = runtime_cache_env(allocation.cache)
    process_env.update(settings.extra_env or {})
    if input.topology == ServeTopology.prefill_decode:
        process_env["TOKENSPEED_SKIP_GRPC_WARMUP"] = "1"
    return rendered_model_rank(
        allocation,
        ProcessSpec(argv=argv, env=process_env),
    )


def _render_router(
    allocation: ServeProcessAllocationFrontend,
    model_allocations: list[ServeProcessAllocationModelRank],
) -> RenderedServeProcess:
    prometheus = allocation.ports.get("prometheus")
    if prometheus is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "TokenSpeed SMG allocation is missing its Prometheus port",
        )
    prefill = rank_zero_allocations(model_allocations, ServeRoleKind.prefill)
    decode = rank_zero_allocations(model_allocations, ServeRoleKind.decode)
    model_locator = next(
        (item.model_locator for item in [*prefill, *decode]),
        None,
    )
    if model_locator is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "TokenSpeed SMG requires a public endpoint and one serving model locator",
        )
    endpoint = allocation.endpoint
    argv = [
        "python3",
        "-m",
        "smg",
        "launch",
        "--host",
        "0.0.0.0",
        "--port",
        str(endpoint.port),
        "--prometheus-port",
        str(prometheus.port),
        "--worker-startup-timeout-secs",
        str(ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
        "--model-path",
        model_locator,
        "--tokenizer-path",
        model_locator,
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
                f"grpc://{item.endpoint.host}:{item.endpoint.port}",
                str(bootstrap.port),
            ]
        )
    for item in decode:
        if item.endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"decode allocation {item.process!r} has no endpoint",
            )
        argv.extend(["--decode", f"grpc://{item.endpoint.host}:{item.endpoint.port}"])
    argv.extend(
        [
            "--policy",
            "round_robin",
            "--prefill-policy",
            "round_robin",
            "--decode-policy",
            "round_robin",
            "--disable-retries",
            "--disable-circuit-breaker",
        ]
    )
    return rendered_frontend(allocation, ProcessSpec(argv=argv, env={}))


def render_serve(input: RenderServeInput) -> RenderServeResult:
    if not input.allocations:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request, "serve allocation must not be empty"
        )
    allocations, model_allocations = split_serve_allocations(input.allocations)
    if input.topology == ServeTopology.single and len(allocations) != 1:
        message = (
            "the TokenSpeed integration does not support multi-node serving yet"
            if any(allocation.rank_count > 1 for allocation in model_allocations)
            else "the TokenSpeed single topology supports exactly one process"
        )
        raise AdapterOperationError(AdapterErrorCode.invalid_request, message)

    processes: list[RenderedServeProcess] = []
    for allocation in allocations:
        if isinstance(allocation, ServeProcessAllocationModelRank):
            if allocation.rank_count > 1:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    "the TokenSpeed integration does not support multi-node serving yet",
                )
            processes.append(_render_worker(input, allocation))
        elif isinstance(allocation, ServeProcessAllocationFrontend):
            require_integration_fused_frontend(
                allocation,
                gateway_backend="tokenspeed-smg",
                pd_router_backend="tokenspeed-smg",
            )
            processes.append(_render_router(allocation, model_allocations))
    return RenderServeResult(integration=_identity(), processes=processes)
