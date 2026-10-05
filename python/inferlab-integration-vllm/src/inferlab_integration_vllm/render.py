import json

import inferlab_gateway_dynamo
from inferlab_adapter_sdk import (
    ROUTER_WORKER_STARTUP_TIMEOUT_SECS,
    AdapterErrorCode,
    AdapterOperationError,
    CaptureMechanism,
    JsonValue,
    KvTransferMechanism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    ProcessSpec,
    RenderedServeProcess,
    RenderServeInput,
    RenderServeResult,
    ServeAllocation,
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

from .auxiliary import splice_draft_model
from .plan import _identity
from .settings import (
    _INFERLAB_OWNED_OPTIONS,
    VllmServeSettings,
    _settings,
)

_RUNTIME_CACHE_EXTRA_SUBDIRS = {
    "VLLM_CACHE_ROOT": "vllm",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "flashinfer_autotune",
    "TILELANG_CACHE_DIR": "tilelang",
    "TILELANG_TMP_DIR": "tilelang/tmp",
}

# The Dynamo worker parses vLLM engine arguments only; the API-server options
# below have no Dynamo spelling or are realized by the Dynamo frontend.
_DYNAMO_OWNED_OPTIONS = {
    *_INFERLAB_OWNED_OPTIONS,
    *inferlab_gateway_dynamo.OWNED_OPTIONS,
    "--disaggregation-mode",
    "--model",
}


def _render_process(
    input: RenderServeInput,
    role_allocations: list[ServeProcessAllocationModelRank],
    allocation: ServeProcessAllocationModelRank,
    rank: int,
) -> RenderedServeProcess:
    settings = _settings(allocation.effective_settings)
    splice_draft_model(settings, allocation, input.auxiliary_models)
    if allocation.endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"serving allocation {allocation.process!r} is missing its endpoint",
        )
    endpoint = allocation.endpoint
    argv = [
        # python3, not python: conda-family realizations carry both, while
        # Debian-family external serving images ship no bare `python`.
        "python3",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        allocation.model_locator,
    ]
    inferlab_args = [
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        *_engine_args(input, allocation, settings, api_server=True),
    ]

    if input.topology == ServeTopology.prefill_decode:
        role_name = (
            "kv_producer" if allocation.role_kind == ServeRoleKind.prefill else "kv_consumer"
        )
        inferlab_args.extend(_kv_transfer_args(input.kv_transfer, role_name, settings))

    process_env = {
        "VLLM_SERVER_DEV_MODE": "1",
        **runtime_cache_env(allocation.cache, _RUNTIME_CACHE_EXTRA_SUBDIRS),
    }
    process_env.update(settings.extra_env or {})
    if input.topology == ServeTopology.prefill_decode:
        process_env.update(_kv_transfer_env(input.kv_transfer, settings))
        if (
            input.kv_transfer == KvTransferMechanism.mooncake
            and allocation.role_kind == ServeRoleKind.prefill
        ):
            bootstrap = allocation.ports.get("bootstrap")
            if bootstrap is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"prefill process {allocation.process!r} is missing its bootstrap port",
                )
            process_env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(bootstrap.port)
        if input.kv_transfer == KvTransferMechanism.nixl:
            process_env.update(_nixl_side_channel_env(allocation))
    node_count = len(role_allocations)
    if node_count != allocation.rank_count:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"allocation {allocation.process!r} declares rank count "
            f"{allocation.rank_count} but the render input carries {node_count} "
            "rank allocations for its replica",
        )
    if node_count > 1:
        primary = next((candidate for candidate in role_allocations if candidate.rank == 0), None)
        master = None if primary is None else primary.ports.get("master")
        if master is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "multi-node allocation is missing the master endpoint",
            )
        inferlab_args.extend(
            [
                "--nnodes",
                str(node_count),
                "--node-rank",
                str(rank),
                "--master-addr",
                master.host,
                "--master-port",
                str(master.port),
            ]
        )
        if rank != 0:
            inferlab_args.append("--headless")
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, _INFERLAB_OWNED_OPTIONS))
    return rendered_model_rank(
        allocation,
        ProcessSpec(argv=argv, env=process_env),
    )


def _render_dynamo_worker(
    input: RenderServeInput,
    allocation: ServeProcessAllocationModelRank,
    allocations: list[ServeAllocation],
) -> RenderedServeProcess:
    """One Dynamo vLLM worker: vLLM engine arguments, the disaggregation role,
    and the Dynamo discovery and listener environment."""
    if allocation.rank_count != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"Dynamo serves single-rank vLLM replicas only; {allocation.process!r} "
            f"was decomposed into {allocation.rank_count} ranks",
        )
    settings = _settings(allocation.effective_settings)
    splice_draft_model(settings, allocation, input.auxiliary_models)
    if input.topology == ServeTopology.prefill_decode:
        mode = "prefill" if allocation.role_kind == ServeRoleKind.prefill else "decode"
    else:
        mode = "agg"
    inferlab_args = [
        *_engine_args(input, allocation, settings, api_server=False),
        "--disaggregation-mode",
        mode,
        *inferlab_gateway_dynamo.reasoning_parser_args(settings.reasoning_parser),
    ]
    process_env = runtime_cache_env(allocation.cache, _RUNTIME_CACHE_EXTRA_SUBDIRS)
    process_env.update(settings.extra_env or {})
    if input.topology == ServeTopology.prefill_decode:
        inferlab_args.extend(
            _kv_transfer_args(
                input.kv_transfer,
                "kv_producer" if mode == "prefill" else "kv_consumer",
                settings,
            )
        )
        process_env.update(_kv_transfer_env(input.kv_transfer, settings))
        process_env.update(_nixl_side_channel_env(allocation))
    process_env.update(inferlab_gateway_dynamo.worker_env(allocation, allocations))
    argv = [
        "python3",
        "-m",
        "dynamo.vllm",
        "--model",
        allocation.model_locator,
        *merge_serve_args(settings.extra_args or [], inferlab_args, _DYNAMO_OWNED_OPTIONS),
    ]
    return rendered_model_rank(allocation, ProcessSpec(argv=argv, env=process_env))


def _nixl_side_channel_env(allocation: ServeProcessAllocationModelRank) -> dict[str, str]:
    side_channel = allocation.ports.get("side_channel")
    if side_channel is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"process {allocation.process!r} is missing its NIXL side-channel port",
        )
    return {
        "VLLM_NIXL_SIDE_CHANNEL_HOST": side_channel.host,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel.port),
    }


def _engine_args(
    input: RenderServeInput,
    allocation: ServeProcessAllocationModelRank,
    settings: VllmServeSettings,
    *,
    api_server: bool,
) -> list[str]:
    """vLLM engine arguments shared by `vllm serve` and the Dynamo worker; the
    API-server options apply only where vLLM's own server answers requests."""
    outer = allocation.effective_parallelism.outer or ParallelismOuter()
    attention = allocation.effective_parallelism.attention or ParallelismAttention()
    experts = allocation.effective_parallelism.experts or ParallelismExperts()
    inferlab_args = [
        "--served-model-name",
        input.model.served_name,
        "--tensor-parallel-size",
        str(outer.tensor_parallel_size or 1),
    ]
    if (outer.pipeline_parallel_size or 1) != 1:
        inferlab_args.extend(
            [
                "--pipeline-parallel-size",
                str(outer.pipeline_parallel_size),
            ]
        )
    if (attention.data_parallel_size or 1) != 1:
        inferlab_args.extend(
            [
                "--data-parallel-size",
                str(attention.data_parallel_size),
            ]
        )
    attention_cp = attention.context_parallel_size or 1
    if attention_cp != 1:
        if allocation.role_kind == ServeRoleKind.prefill:
            inferlab_args.extend(["--prefill-context-parallel-size", str(attention_cp)])
        else:
            inferlab_args.extend(["--decode-context-parallel-size", str(attention_cp)])
    append_option(inferlab_args, "--max-model-len", settings.max_model_len)
    append_option(inferlab_args, "--kv-cache-dtype", settings.kv_cache_dtype)
    append_option(inferlab_args, "--gpu-memory-utilization", settings.gpu_memory_utilization)
    append_option(inferlab_args, "--block-size", settings.block_size)
    append_option(inferlab_args, "--tokenizer-mode", settings.tokenizer_mode)
    if api_server:
        append_option(inferlab_args, "--tool-call-parser", settings.tool_call_parser)
        append_option(inferlab_args, "--reasoning-parser", settings.reasoning_parser)
        if settings.enable_auto_tool_choice:
            inferlab_args.append("--enable-auto-tool-choice")
    if settings.reasoning_config is not None:
        inferlab_args.extend(
            [
                "--reasoning-config",
                json.dumps(settings.reasoning_config, sort_keys=True, separators=(",", ":")),
            ]
        )
    if settings.enable_flashinfer_autotune is not None:
        inferlab_args.append(
            "--enable-flashinfer-autotune"
            if settings.enable_flashinfer_autotune
            else "--no-enable-flashinfer-autotune"
        )
    if settings.enable_prompt_tokens_details:
        inferlab_args.append("--enable-prompt-tokens-details")
    if settings.trust_remote_code:
        inferlab_args.append("--trust-remote-code")
    if (experts.expert_parallel_size or 1) > 1:
        inferlab_args.append("--enable-expert-parallel")
    if settings.compilation_config is not None:
        inferlab_args.extend(
            [
                "--compilation-config",
                json.dumps(settings.compilation_config, sort_keys=True, separators=(",", ":")),
            ]
        )
    if input.profiling == CaptureMechanism.managed_collection:
        inferlab_args.extend(["--profiler-config", '{"profiler":"cuda"}'])
    elif input.profiling == CaptureMechanism.engine_trace:
        # Engine trace: the framework's own torch profiler writes one trace
        # artifact per rank into the control-plane-assigned directory
        # ([[RFC-0004:C-WORKLOAD-PROFILING]]); the window stays on the
        # /start_profile//stop_profile control pair. vLLM's default
        # key-averages table dump runs inside the EngineCore on window close
        # and blocks serving for minutes; it is not trace evidence, so it
        # stays off.
        capture_storage = allocation.capture_storage
        if capture_storage is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"engine-trace allocation {allocation.process!r} is missing its "
                "assigned trace directory",
            )
        inferlab_args.extend(
            [
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": capture_storage,
                        "torch_profiler_dump_cuda_time_total": False,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )

    return inferlab_args


def _kv_transfer_args(
    transport: KvTransferMechanism | None,
    role: str,
    settings: VllmServeSettings,
) -> list[str]:
    if transport is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "prefill_decode render is missing its KV-transfer mechanism",
        )
    if transport == KvTransferMechanism.mooncake:
        extra: dict[str, JsonValue] = {
            "num_workers": settings.mooncake_num_workers or 1,
        }
        if settings.kv_transfer_protocol is not None:
            extra["mooncake_protocol"] = settings.kv_transfer_protocol
        config: dict[str, JsonValue] = {
            "kv_connector": "MooncakeConnector",
            "kv_role": role,
            "kv_connector_extra_config": extra,
        }
    else:
        config = {
            "kv_connector": "NixlConnector",
            "kv_role": role,
            "kv_load_failure_policy": "fail",
        }
        if settings.kv_transfer_protocol is not None:
            backends: list[JsonValue] = (
                ["UCX", "GDS"] if settings.kv_transfer_protocol.lower() == "gds" else ["UCX"]
            )
            config["kv_connector_extra_config"] = {"backends": backends}
    return [
        "--kv-transfer-config",
        json.dumps(config, sort_keys=True, separators=(",", ":")),
    ]


def _kv_transfer_env(
    transport: KvTransferMechanism | None, settings: VllmServeSettings
) -> dict[str, str]:
    if settings.kv_transfer_protocol is None:
        return {}
    if settings.kv_transfer_protocol.lower() != "tcp":
        return {}
    if transport == KvTransferMechanism.mooncake:
        return {"MC_FORCE_TCP": "1"}
    if transport == KvTransferMechanism.nixl:
        # tcp names the wire; cuda_copy is the orthogonal staging lane UCX
        # needs to register and move GPU memory through host bounce buffers
        # (a bare "tcp" fails KV-buffer registration with NIXL_ERR_BACKEND,
        # observed on real hardware), and self serves agent-local transfers.
        return {"UCX_TLS": "tcp,cuda_copy,self"}
    return {}


def _render_router(
    input: RenderServeInput,
    allocation: ServeProcessAllocationFrontend,
    model_allocations: list[ServeProcessAllocationModelRank],
) -> RenderedServeProcess:
    prefill = rank_zero_allocations(model_allocations, ServeRoleKind.prefill)
    decode_allocations = rank_zero_allocations(model_allocations, ServeRoleKind.decode)
    if any(item.endpoint is None for item in [*prefill, *decode_allocations]):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "vLLM Router allocations require public endpoints",
        )
    endpoint = allocation.endpoint
    decode = [
        f"http://{item.endpoint.host}:{item.endpoint.port}"
        for item in decode_allocations
        if item.endpoint is not None
    ]
    argv = [
        "vllm-router",
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        "--worker-startup-timeout-secs",
        str(ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
        "--vllm-pd-disaggregation",
    ]
    if input.kv_transfer is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "vLLM Router render is missing its KV-transfer mechanism",
        )
    for item in prefill:
        item_endpoint = item.endpoint
        if item_endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill allocation {item.process!r} has no endpoint",
            )
        argv.extend(["--prefill", f"http://{item_endpoint.host}:{item_endpoint.port}"])
        if input.kv_transfer == KvTransferMechanism.mooncake:
            bootstrap = item.ports.get("bootstrap")
            if bootstrap is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"prefill replica {item.replica!r} is missing its bootstrap port",
                )
            argv.append(str(bootstrap.port))
    for decode_endpoint in decode:
        argv.extend(["--decode", decode_endpoint])
    argv.extend(
        [
            "--kv-connector",
            input.kv_transfer.value,
            "--policy",
            "round_robin",
        ]
    )
    return rendered_frontend(allocation, ProcessSpec(argv=argv, env={}))


def render_serve(input: RenderServeInput) -> RenderServeResult:
    if not input.allocations:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request, "serve allocation must not be empty"
        )
    allocations, model_allocations = split_serve_allocations(input.allocations)
    allocations_by_replica = {
        (allocation.role, allocation.replica): [
            candidate
            for candidate in model_allocations
            if candidate.role == allocation.role and candidate.replica == allocation.replica
        ]
        for allocation in model_allocations
    }
    processes: list[RenderedServeProcess] = []
    dynamo = input.gateway_backend == inferlab_gateway_dynamo.BACKEND
    for allocation in allocations:
        if dynamo:
            processes.append(_render_dynamo_process(input, allocation, allocations))
        elif isinstance(allocation, ServeProcessAllocationModelRank):
            role_allocations = allocations_by_replica[(allocation.role, allocation.replica)]
            processes.append(
                _render_process(
                    input,
                    role_allocations,
                    allocation,
                    allocation.rank,
                )
            )
        elif isinstance(allocation, ServeProcessAllocationFrontend):
            require_integration_fused_frontend(
                allocation,
                gateway_backend="vllm-router",
                pd_router_backend="vllm-router",
            )
            processes.append(_render_router(input, allocation, model_allocations))
        else:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"vLLM renders no discovery process for {input.gateway_backend!r}",
            )
    return RenderServeResult(integration=_identity(), processes=processes)


def _render_dynamo_process(
    input: RenderServeInput, allocation: ServeAllocation, allocations: list[ServeAllocation]
) -> RenderedServeProcess:
    if isinstance(allocation, ServeProcessAllocationModelRank):
        return _render_dynamo_worker(input, allocation, allocations)
    if isinstance(allocation, ServeProcessAllocationFrontend):
        return inferlab_gateway_dynamo.render_frontend(allocation, allocations)
    return inferlab_gateway_dynamo.render_discovery(allocation)
