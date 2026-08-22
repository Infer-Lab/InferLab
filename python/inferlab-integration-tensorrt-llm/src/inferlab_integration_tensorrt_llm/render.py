import hashlib
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]
from inferlab_adapter_sdk import (
    ROUTER_WORKER_STARTUP_TIMEOUT_SECS,
    AdapterErrorCode,
    AdapterOperationError,
    LaunchFileDeclaration,
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
    SuppliedRenderInput,
    append_option,
    merge_serve_args,
    rank_zero_allocations,
    rendered_frontend,
    rendered_model_rank,
    require_integration_fused_frontend,
    runtime_cache_env,
    split_serve_allocations,
)

from .plan import (
    _NATIVE_ROUTING_BACKEND,
    _PREFILL_DECODE_OWNED_OPTIONS,
    _identity,
    _render_source_path,
)
from .settings import (
    _INFERLAB_OWNED_OPTIONS,
    TrtllmServeSettings,
    YamlValue,
    _settings,
)


def _yaml_mapping(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM YAML {source} must be a mapping",
        )
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM YAML {source} must use string keys",
        )
    return cast(dict[str, object], mapping)


def _load_worker_config(
    render_inputs: list[SuppliedRenderInput], path: str | None
) -> dict[str, object]:
    if path is None:
        return {}
    supplied = next(
        (item for item in render_inputs if item.source_path == _render_source_path(path)),
        None,
    )
    if supplied is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"TensorRT-LLM render input {path!r} was not supplied",
        )
    try:
        value: object = yaml.safe_load(supplied.text)
    except yaml.YAMLError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"cannot parse TensorRT-LLM extra_llm_api_options {path!r}: {error}",
        ) from error
    if value is None:
        return {}
    return dict(_yaml_mapping(value, repr(path)))


def _nested_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if value is None:
        nested: dict[str, object] = {}
    else:
        nested = dict(_yaml_mapping(value, key))
    config[key] = nested
    return nested


def _merge_yaml_patch(config: dict[str, object], patch: dict[str, YamlValue]) -> None:
    for key, value in patch.items():
        current = config.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_yaml_patch(_yaml_mapping(current, key), value)
        else:
            config[key] = value


def _worker_launch_text(
    input: RenderServeInput,
    render_inputs: list[SuppliedRenderInput],
    settings: TrtllmServeSettings,
    kind: ServeRoleKind,
) -> str:
    config = _load_worker_config(render_inputs, settings.extra_llm_api_options)
    _merge_yaml_patch(config, settings.extra_llm_api_options_patch or {})
    if input.topology == ServeTopology.prefill_decode:
        config["backend"] = "pytorch"
        transceiver = _nested_mapping(config, "cache_transceiver_config")
        transceiver["backend"] = "NIXL"
        transceiver["transceiver_runtime"] = "PYTHON"
        kv_cache = _nested_mapping(config, "kv_cache_config")
        kv_cache["enable_block_reuse"] = False
        if kind == ServeRoleKind.prefill:
            config["disable_overlap_scheduler"] = True
    return cast(str, yaml.safe_dump(config, sort_keys=False))


def _launch_file(
    runtime_cache_root: str, name: str, text: str
) -> tuple[LaunchFileDeclaration, str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    relative_path = f"launch-files/{digest}/{name}"
    declaration = LaunchFileDeclaration(
        relative_path=relative_path,
        sha256=digest,
        text=text,
    )
    return declaration, str(Path(runtime_cache_root) / relative_path)


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
    argv = [
        "python3",
        "-m",
        "tensorrt_llm.commands.serve",
        allocation.model_locator,
    ]
    inferlab_args = [
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        "--served_model_name",
        input.model.served_name,
        "--tensor_parallel_size",
        str(outer.tensor_parallel_size or 1),
    ]
    if (outer.pipeline_parallel_size or 1) != 1:
        inferlab_args.extend(["--pipeline_parallel_size", str(outer.pipeline_parallel_size)])
    if (attention.data_parallel_size or 1) != 1:
        inferlab_args.append("--enable_attention_dp")
    if (experts.expert_parallel_size or 1) != 1:
        inferlab_args.extend(["--moe_expert_parallel_size", str(experts.expert_parallel_size)])
    append_option(inferlab_args, "--max_batch_size", settings.max_batch_size)
    append_option(inferlab_args, "--max_num_tokens", settings.max_num_tokens)
    append_option(inferlab_args, "--max_seq_len", settings.max_seq_len)
    append_option(inferlab_args, "--kv_cache_dtype", settings.kv_cache_dtype)
    append_option(inferlab_args, "--free_gpu_memory_fraction", settings.free_gpu_memory_fraction)
    append_option(inferlab_args, "--custom_tokenizer", settings.custom_tokenizer)
    append_option(inferlab_args, "--tool_parser", settings.tool_parser)
    append_option(inferlab_args, "--reasoning_parser", settings.reasoning_parser)
    launch_files: list[LaunchFileDeclaration] = []
    if (
        input.topology == ServeTopology.prefill_decode
        or settings.extra_llm_api_options_patch is not None
    ):
        if input.topology == ServeTopology.prefill_decode and allocation.role_kind not in {
            ServeRoleKind.prefill,
            ServeRoleKind.decode,
        }:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill_decode allocation has unsupported role {allocation.role!r}",
            )
        launch_text = _worker_launch_text(
            input,
            allocation.render_inputs,
            settings,
            allocation.role_kind,
        )
        launch_file, resolved_path = _launch_file(
            allocation.cache,
            "extra-llm-api-options.yaml",
            launch_text,
        )
        launch_files.append(launch_file)
        inferlab_args.extend(["--extra_llm_api_options", resolved_path])
        if input.topology == ServeTopology.prefill_decode:
            inferlab_args.extend(["--backend", "pytorch"])
    else:
        append_option(inferlab_args, "--extra_llm_api_options", settings.extra_llm_api_options)
    if settings.enable_chunked_prefill:
        inferlab_args.append("--enable_chunked_prefill")
    if settings.trust_remote_code:
        inferlab_args.append("--trust_remote_code")
    owned_options = (
        _PREFILL_DECODE_OWNED_OPTIONS
        if input.topology == ServeTopology.prefill_decode
        else _INFERLAB_OWNED_OPTIONS
    )
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, owned_options))
    process_env = runtime_cache_env(allocation.cache)
    process_env.update(settings.extra_env or {})
    return rendered_model_rank(
        allocation,
        ProcessSpec(argv=argv, env=process_env),
        launch_files=launch_files,
    )


def _render_native_router(
    allocation: ServeProcessAllocationFrontend,
    model_allocations: list[ServeProcessAllocationModelRank],
) -> RenderedServeProcess:
    prefill = rank_zero_allocations(model_allocations, ServeRoleKind.prefill)
    decode = rank_zero_allocations(model_allocations, ServeRoleKind.decode)
    if any(item.endpoint is None for item in [*prefill, *decode]):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "TensorRT-LLM disaggregated allocations require public endpoints",
        )
    endpoint = allocation.endpoint
    config = {
        "hostname": endpoint.host,
        "port": endpoint.port,
        "schedule_style": "context_first",
        "context_servers": {
            "num_instances": len(prefill),
            "urls": [
                f"{item.endpoint.host}:{item.endpoint.port}"
                for item in prefill
                if item.endpoint is not None
            ],
            "router": {"type": "round_robin"},
        },
        "generation_servers": {
            "num_instances": len(decode),
            "urls": [
                f"{item.endpoint.host}:{item.endpoint.port}"
                for item in decode
                if item.endpoint is not None
            ],
            "router": {"type": "round_robin"},
        },
    }
    text = cast(str, yaml.safe_dump(config, sort_keys=False))
    launch_file, resolved_path = _launch_file(
        allocation.cache,
        "disaggregated.yaml",
        text,
    )
    process_env = runtime_cache_env(allocation.cache)
    process_env.update(_settings(allocation.gateway.effective_settings).extra_env or {})
    return rendered_frontend(
        allocation,
        ProcessSpec(
            argv=[
                "python3",
                "-m",
                "tensorrt_llm.commands.serve",
                "disaggregated",
                "--config",
                resolved_path,
                "--server_start_timeout",
                str(ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
            ],
            env=process_env,
        ),
        launch_files=[launch_file],
    )


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
                    "the TensorRT-LLM integration does not support multi-node serving yet",
                )
            processes.append(_render_worker(input, allocation))
        elif isinstance(allocation, ServeProcessAllocationFrontend):
            require_integration_fused_frontend(
                allocation,
                gateway_backend=_NATIVE_ROUTING_BACKEND,
                pd_router_backend=_NATIVE_ROUTING_BACKEND,
            )
            processes.append(_render_native_router(allocation, model_allocations))
    return RenderServeResult(integration=_identity(), processes=processes)
