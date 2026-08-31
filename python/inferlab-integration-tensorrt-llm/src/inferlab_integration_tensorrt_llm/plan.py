import os
from pathlib import Path

import yaml  # type: ignore[import-untyped]
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
    SuppliedRenderInput,
    SyntheticAcceptanceInput,
    SyntheticAcceptanceInput2,
    SyntheticAcceptanceOutcome,
    TargetEndpointScheme,
    effective_settings,
    fused_pd_frontend_plans,
    integration_identity,
    replica_id,
    require_role,
    resolve_golden_acceptance_length,
    validate_extra_args,
)

from .settings import (
    _INFERLAB_OWNED_OPTIONS,
    TrtllmServeSettings,
    _merge_yaml_patch,
    _settings,
    _yaml_mapping,
)
from .synthetic import FORCE_ACCEPTED_TOKENS_ENV

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


def _read_operator_config(path: str) -> str:
    """Plan-time read of the operator's source YAML through the workspace filesystem."""
    try:
        return Path(_render_source_path(path)).read_text(encoding="utf-8")
    except OSError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"cannot read TensorRT-LLM extra_llm_api_options {path!r}: {error}",
        ) from error


def _parse_operator_config(text: str, path: str) -> dict[str, object]:
    try:
        value: object = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"cannot parse TensorRT-LLM extra_llm_api_options {path!r}: {error}",
        ) from error
    if value is None:
        return {}
    return dict(_yaml_mapping(value, repr(path)))


def _resolve_synthetic_acceptance(
    settings: TrtllmServeSettings,
    synthetic: SyntheticAcceptanceInput,
    role_id: str,
    render_inputs: list[SuppliedRenderInput] | None = None,
) -> SyntheticAcceptanceOutcome:
    """Validate the overlay target and resolve the effective acceptance length.

    The forced-acceptance variable is rendered per engine process, so the
    overlay target is validated against the merged `extra_llm_api_options`
    content (source YAML plus patch) and an operator restatement of the owned
    variable is rejected ([[RFC-0003:C-SERVE-SYNTHETIC-ACCEPTANCE]]). For the
    curve form the draft count comes from that merged configuration's
    `speculative_config.max_draft_len` ([[ADR-0043]]); the same resolution
    runs at plan and at render, so both see one effective value. At plan no
    supplied render inputs exist, so the source YAML is read through the
    workspace filesystem; at render the control-plane-supplied frozen text is
    consumed instead ([[RFC-0006:C-LAUNCH-FILES]]).
    """
    if settings.extra_env and FORCE_ACCEPTED_TOKENS_ENV in settings.extra_env:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role_id!r} extra_env restates {FORCE_ACCEPTED_TOKENS_ENV}; the "
            "synthetic acceptance declaration is the single authority for that key",
        )
    config: dict[str, object] = {}
    path = settings.extra_llm_api_options
    if path is not None:
        if render_inputs is None:
            text = _read_operator_config(path)
        else:
            supplied = next(
                (item for item in render_inputs if item.source_path == _render_source_path(path)),
                None,
            )
            if supplied is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"TensorRT-LLM render input {path!r} was not supplied",
                )
            text = supplied.text
        config = _parse_operator_config(text, path)
    _merge_yaml_patch(config, settings.extra_llm_api_options_patch or {})
    speculative = config.get("speculative_config")
    if speculative is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role_id!r} declares no speculative_config in its "
            "extra_llm_api_options source YAML or patch; the synthetic acceptance "
            "overlay requires the operator's speculative configuration as its target",
        )
    form = synthetic.root
    if not isinstance(form, SyntheticAcceptanceInput2):
        return SyntheticAcceptanceOutcome(acceptance_length=form.explicit.acceptance_length)
    # The golden curve's lookup key is the draft length, exposed by the
    # operator's speculative_config as max_draft_len.
    draft_length: object = None
    if isinstance(speculative, dict):
        draft_length = _yaml_mapping(speculative, "speculative_config").get("max_draft_len")
    if isinstance(draft_length, bool):
        draft_length = None
    if isinstance(draft_length, float) and draft_length.is_integer():
        draft_length = int(draft_length)
    if not isinstance(draft_length, int):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role_id!r} extra_llm_api_options speculative_config does not "
            "determine an integer max_draft_len; the curve form of the synthetic "
            "acceptance declaration needs it as the curve lookup coordinate "
            "(use the explicit form otherwise)",
        )
    acceptance_length = resolve_golden_acceptance_length(
        curve_text=form.curve.text,
        model_key=form.curve.model_key,
        thinking_mode=form.curve.thinking_mode,
        draft_count=draft_length,
    )
    return SyntheticAcceptanceOutcome(
        acceptance_length=acceptance_length,
        draft_count=draft_length,
    )


def _plan_role(
    input: PlanServeInput, role: ServeRoleInput
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement], SyntheticAcceptanceOutcome | None]:
    if role.replica_count < 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} replica count must be positive",
        )
    settings = _settings(role.settings)
    outcome: SyntheticAcceptanceOutcome | None = None
    if input.synthetic_acceptance is not None:
        outcome = _resolve_synthetic_acceptance(settings, input.synthetic_acceptance, role.id)
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
        outcome,
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
    role_result, replicas, outcome = _plan_role(input, role)
    settings = _settings(role_result.effective_settings)
    render_inputs: list[RenderInputDeclaration] = []
    # The source YAML crosses as a supplied render input when rendering
    # re-reads its content: for the launch-file merge (patch present) or for
    # the synthetic acceptance overlay's render-time re-resolution
    # ([[RFC-0006:C-LAUNCH-FILES]]).
    if settings.extra_llm_api_options is not None and (
        settings.extra_llm_api_options_patch is not None or input.synthetic_acceptance is not None
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
        synthetic_acceptance=outcome,
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
    prefill_result, prefill_replicas, prefill_outcome = _plan_role(input, prefill)
    decode_result, decode_replicas, decode_outcome = _plan_role(input, decode)
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
        synthetic_acceptance=prefill_outcome,
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
