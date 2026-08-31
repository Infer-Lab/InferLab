from inferlab_adapter_sdk import (
    ROUTER_WORKER_STARTUP_TIMEOUT_SECS,
    AdapterErrorCode,
    AdapterOperationError,
    CaptureMechanism,
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
from .synthetic import resolve_synthetic_acceptance, synthetic_acceptance_env

_RUNTIME_CACHE_EXTRA_SUBDIRS = {
    "TILELANG_CACHE_DIR": "tilelang",
    "TILELANG_TMP_DIR": "tilelang/tmp",
}

# The managed prefill-CP flag group is verbatim-replaceable
# ([[RFC-0003:C-RESOLUTION]]): last-wins parsing cannot retract the
# store-true `--enable-prefill-cp`, and DSA/NSA-family models need their own
# spellings, so a post-sentinel mention of any member hands the whole group
# to the verbatim block.
_PREFILL_CP_FLAG_GROUP = {
    "--attention-context-parallel-size",
    "--enable-prefill-cp",
    "--cp-strategy",
    "--enable-dsa-prefill-context-parallel",
    "--enable-nsa-prefill-context-parallel",
    "--prefill-cp-mode",
    "--dsa-prefill-cp-mode",
    "--nsa-prefill-cp-mode",
}


def _verbatim_block_owns_prefill_cp(extra_args: list[str]) -> bool:
    if "--" not in extra_args:
        return False
    tail = extra_args[extra_args.index("--") + 1 :]
    return any(token.partition("=")[0] in _PREFILL_CP_FLAG_GROUP for token in tail)


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
    if allocation.role_kind == ServeRoleKind.decode:
        # Decode CP lowers to --dcp-size: it does not subdivide the
        # attention TP world and does not by itself require DP attention.
        if attention_cp != 1:
            inferlab_args.extend(["--dcp-size", str(attention_cp)])
        if attention_dp != 1:
            inferlab_args.append("--enable-dp-attention")
    else:
        # Serve and prefill roles lower CP to prefill CP, which needs the
        # explicit enable flag and a strategy or the engine ignores it. A
        # verbatim post-sentinel mention of the group owns the spelling
        # instead ([[RFC-0003:C-RESOLUTION]]).
        if attention_cp != 1 and not _verbatim_block_owns_prefill_cp(settings.extra_args or []):
            inferlab_args.extend(
                [
                    "--attention-context-parallel-size",
                    str(attention_cp),
                    "--enable-prefill-cp",
                    "--cp-strategy",
                    "zigzag",
                ]
            )
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
    if settings.enable_cache_report:
        inferlab_args.append("--enable-cache-report")
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
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, _INFERLAB_OWNED_OPTIONS))
    process_env = runtime_cache_env(allocation.cache, _RUNTIME_CACHE_EXTRA_SUBDIRS)
    process_env.update(settings.extra_env or {})
    if input.synthetic_acceptance is not None:
        # Re-resolve from the same inputs planning saw: the wire member plus
        # this role's effective settings. Deterministic by construction.
        outcome = resolve_synthetic_acceptance(
            settings, input.synthetic_acceptance, allocation.role
        )
        process_env.update(synthetic_acceptance_env(outcome.acceptance_length))
    if input.profiling == CaptureMechanism.engine_trace:
        # SGLang's torch profiler reads its output directory from the
        # environment when the start action body omits it
        # ([[RFC-0004:C-WORKLOAD-PROFILING]]).
        capture_storage = allocation.capture_storage
        if capture_storage is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"engine-trace allocation {allocation.process!r} is missing its "
                "assigned trace directory",
            )
        process_env["SGLANG_TORCH_PROFILER_DIR"] = capture_storage
    return rendered_model_rank(
        allocation,
        ProcessSpec(argv=argv, env=process_env),
    )


def _render_router(
    allocation: ServeProcessAllocationFrontend,
    model_allocations: list[ServeProcessAllocationModelRank],
) -> RenderedServeProcess:
    prefill = rank_zero_allocations(model_allocations, ServeRoleKind.prefill)
    decode = rank_zero_allocations(model_allocations, ServeRoleKind.decode)
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
        str(ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
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
