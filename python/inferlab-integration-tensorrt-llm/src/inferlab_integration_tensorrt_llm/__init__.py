import hashlib
import os
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    EndpointProtocol,
    EndpointRequirement,
    IntegrationIdentity,
    KvTransferMechanism,
    LaunchFileDeclaration,
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
    ReadinessProbeProcessAlive,
    RenderedServeProcess,
    RenderInputDeclaration,
    RenderServeInput,
    RenderServeResult,
    RenderSource,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
    ServeReplicaRequirement,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkKvTransfer,
    ServeRoleLinkRequestRouting,
    ServeRoleResult,
    ServeTopology,
    SettingValue,
    SuppliedRenderInput,
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

# TensorRT-LLM declares its click options in underscore spellings plus short
# aliases and does no hyphen/underscore normalization, so the claim list must
# name every accepted spelling of every inferlab- or settings-owned option.
_INFERLAB_OPTION_ARITY: dict[str, int | None] = {
    "--cluster_size": 1,
    "--config": 1,
    "--context_parallel_size": 1,
    "--cp_size": 1,
    "--custom_tokenizer": 1,
    "--enable_attention_dp": 0,
    "--enable_chunked_prefill": 0,
    "--ep_size": 1,
    "--extra_llm_api_options": 1,
    "--free_gpu_memory_fraction": 1,
    "--host": 1,
    "--kv_cache_dtype": 1,
    "--kv_cache_free_gpu_memory_fraction": 1,
    "--max_batch_size": 1,
    "--max_num_tokens": 1,
    "--max_seq_len": 1,
    "--moe_cluster_parallel_size": 1,
    "--moe_expert_parallel_size": 1,
    "--pipeline_parallel_size": 1,
    "--port": 1,
    "--pp_size": 1,
    "--served_model_name": 1,
    "--tensor_parallel_size": 1,
    "--tp_size": 1,
    "--trust_remote_code": 0,
    "--tool_parser": 1,
    "--reasoning_parser": 1,
}

_RUNTIME_CACHE_SUBDIRS = {
    "DG_JIT_CACHE_DIR": "deep_gemm_jit",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASHINFER_CUBIN_DIR": "flashinfer_cubin",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}

_NATIVE_ROUTING_BACKEND = "trtllm-disaggregated"
_PREFILL_DECODE_OPTION_ARITY = {**_INFERLAB_OPTION_ARITY, "--backend": 1}
# Inferlab owns readiness; the router's internal guard must not expire first.
_ROUTER_WORKER_STARTUP_TIMEOUT_SECS = 2_147_483_647

type YamlValue = bool | int | float | str | list[YamlValue] | dict[str, YamlValue]


class TrtllmServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_batch_size: int | None = Field(default=None, ge=1)
    max_num_tokens: int | None = Field(default=None, ge=1)
    max_seq_len: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    free_gpu_memory_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    enable_chunked_prefill: bool = False
    trust_remote_code: bool = False
    custom_tokenizer: str | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    # Source YAML; P/D composition overrides its transport and cache invariants.
    extra_llm_api_options: str | None = None
    extra_llm_api_options_patch: dict[str, YamlValue] | None = None
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


def _runtime_cache_env(root: str) -> dict[str, str]:
    cache_root = Path(root)
    return {
        name: str(cache_root / subdirectory)
        for name, subdirectory in _RUNTIME_CACHE_SUBDIRS.items()
    }


def _settings(values: dict[str, SettingValue]) -> TrtllmServeSettings:
    return validate_settings(TrtllmServeSettings, values)


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


def _render_source_path(path: str) -> str:
    if Path(path).is_absolute():
        return path
    return os.path.normpath(Path(".inferlab") / path)


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


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-tensorrt-llm",
        adapter_distribution="inferlab-integration-tensorrt-llm",
        framework="tensorrt-llm",
        framework_distribution="tensorrt_llm",
        module_file=__file__,
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
        path = _settings(role.effective_settings).extra_llm_api_options
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
    if input.profiling:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support profiling capture yet",
        )
    if input.topology == ServeTopology.single:
        return _plan_single(input)
    return _plan_prefill_decode(input)


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
    option_arity = (
        _PREFILL_DECODE_OPTION_ARITY
        if input.topology == ServeTopology.prefill_decode
        else _INFERLAB_OPTION_ARITY
    )
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, option_arity))
    process_env = _runtime_cache_env(allocation.cache)
    process_env.update(settings.extra_env or {})
    return rendered_model_rank(
        allocation,
        ProcessSpec(argv=argv, env=process_env),
        launch_files=launch_files,
    )


def _rank_zero_allocations(
    allocations: list[ServeProcessAllocationModelRank], kind: ServeRoleKind
) -> list[ServeProcessAllocationModelRank]:
    return sorted(
        [
            allocation
            for allocation in allocations
            if allocation.role_kind == kind and allocation.rank == 0
        ],
        key=lambda allocation: allocation.replica,
    )


def _render_native_router(
    allocation: ServeProcessAllocationFrontend,
    model_allocations: list[ServeProcessAllocationModelRank],
) -> RenderedServeProcess:
    prefill = _rank_zero_allocations(model_allocations, ServeRoleKind.prefill)
    decode = _rank_zero_allocations(model_allocations, ServeRoleKind.decode)
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
    process_env = _runtime_cache_env(allocation.cache)
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
                str(_ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
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


__all__ = ["TrtllmServeSettings", "plan_serve", "render_serve"]
