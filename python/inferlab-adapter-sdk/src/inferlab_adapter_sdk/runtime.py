import json
import sys
import traceback
from collections.abc import Callable, Collection, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ValidationError

from ._generated import (
    AdapterError,
    AdapterErrorCode,
    AdapterRequest,
    AdapterRequestPlanServe,
    AdapterRequestRenderServe,
    AdapterResponse,
    AdapterResponseError,
    AdapterResponseOk,
    AdapterResult,
    AdapterResultPlanServe,
    AdapterResultRenderServe,
    EndpointRequirement,
    FrontendCoRendering,
    FrontendHandoff,
    FrontendProcessRole,
    GatewayPdRouterFrontendBinding,
    GatewayPlan,
    GatewayTarget,
    GatewayTargetPdRouter,
    IntegrationIdentity,
    LaunchFileDeclaration,
    PdRouterPlan,
    PdRoutingPolicies,
    PlanServeInput,
    PlanServeResult,
    ProcessSpec,
    ProtocolVersion,
    ReadinessProbe,
    RenderedServeProcess,
    RenderedServeProcessFrontend,
    RenderedServeProcessModelRank,
    RenderInputDeclaration,
    RenderServeInput,
    RenderServeResult,
    RenderSource,
    ServeProcessAllocation,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
    ServeRoleInput,
    ServeRoleKind,
    SettingValue,
    TargetEndpointScheme,
)

type PlanServeHandler = Callable[[PlanServeInput], PlanServeResult]
type RenderServeHandler = Callable[[RenderServeInput], RenderServeResult]
type ServeAllocation = ServeProcessAllocationModelRank | ServeProcessAllocationFrontend

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
PROTOCOL_V8 = ProtocolVersion()

# Inferlab owns readiness; the router's internal guard must not expire first.
ROUTER_WORKER_STARTUP_TIMEOUT_SECS = 2_147_483_647

_RUNTIME_CACHE_SUBDIRS = {
    "DG_JIT_CACHE_DIR": "deep_gemm_jit",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASHINFER_CUBIN_DIR": "flashinfer_cubin",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}


def runtime_cache_env(root: str, extra_subdirs: Mapping[str, str] | None = None) -> dict[str, str]:
    """Map the shared JIT/runtime cache variables under one cache root."""
    cache_root = Path(root)
    subdirs = dict(_RUNTIME_CACHE_SUBDIRS)
    subdirs.update(extra_subdirs or {})
    return {name: str(cache_root / subdirectory) for name, subdirectory in subdirs.items()}


def rank_zero_allocations(
    allocations: list[ServeProcessAllocationModelRank], kind: ServeRoleKind
) -> list[ServeProcessAllocationModelRank]:
    """Select one role's rank-0 allocations in replica order."""
    return sorted(
        [
            allocation
            for allocation in allocations
            if allocation.role_kind == kind and allocation.rank == 0
        ],
        key=lambda allocation: allocation.replica,
    )


def plain_setting(value: SettingValue) -> JsonValue:
    root = value.root
    if isinstance(root, list):
        return [plain_setting(item) for item in root]
    if isinstance(root, dict):
        return {key: plain_setting(item) for key, item in root.items()}
    return root


def validate_settings[SettingsModel: BaseModel](
    model: type[SettingsModel], values: dict[str, SettingValue]
) -> SettingsModel:
    try:
        return model.model_validate({key: plain_setting(value) for key, value in values.items()})
    except ValidationError as error:
        raise AdapterOperationError(AdapterErrorCode.invalid_settings, str(error)) from error


def effective_settings(settings: BaseModel) -> dict[str, SettingValue]:
    return {
        key: SettingValue.model_validate(value)
        for key, value in settings.model_dump(exclude_none=True).items()
    }


def integration_identity(
    *,
    adapter_id: str,
    adapter_distribution: str,
    framework: str,
    framework_distribution: str,
) -> IntegrationIdentity:
    try:
        adapter_version = version(adapter_distribution)
    except PackageNotFoundError:
        adapter_version = "unavailable"
    try:
        framework_version = version(framework_distribution)
    except PackageNotFoundError:
        framework_version = "unavailable"
    return IntegrationIdentity(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        framework=framework,
        framework_version=framework_version,
    )


def require_role(input: PlanServeInput, kind: ServeRoleKind) -> ServeRoleInput:
    matches = [role for role in input.roles if role.kind == kind]
    if len(matches) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"{input.topology.value} topology requires exactly one {kind.value} role",
        )
    return matches[0]


def replica_id(role: ServeRoleInput, replica_index: int) -> str:
    base = "server" if role.kind == ServeRoleKind.serve else role.id
    if role.replica_count == 1:
        return base
    return f"{base}-{replica_index:03d}"


def fused_pd_frontend_plans(
    *,
    gateway_backend: str,
    pd_router_backend: str,
    implementation: str,
    implementation_version: str,
    render_source: RenderSource,
    endpoint: EndpointRequirement,
    gateway_readiness: ReadinessProbe,
    pd_router_readiness: ReadinessProbe,
    policies: PdRoutingPolicies,
    prefill_role: str,
    decode_role: str,
    target_scheme: TargetEndpointScheme,
    gateway_settings: dict[str, SettingValue] | None = None,
    pd_router_settings: dict[str, SettingValue] | None = None,
    gateway_ports: list[str] | None = None,
    pd_router_ports: list[str] | None = None,
    gateway_render_inputs: list[RenderInputDeclaration] | None = None,
    pd_router_render_inputs: list[RenderInputDeclaration] | None = None,
) -> tuple[GatewayPlan, PdRouterPlan]:
    """Build the two logical component plans for one fused P/D frontend."""
    co_rendering = FrontendCoRendering(process_role=FrontendProcessRole())
    gateway = GatewayPlan(
        backend=gateway_backend,
        implementation=implementation,
        implementation_version=implementation_version,
        render_source=render_source,
        effective_settings=dict(gateway_settings or {}),
        ports=list(gateway_ports or []),
        endpoint=endpoint,
        readiness=gateway_readiness,
        targets=[GatewayTarget(root=GatewayTargetPdRouter())],
        co_rendering=co_rendering,
        render_inputs=list(gateway_render_inputs or []),
    )
    pd_router = PdRouterPlan(
        backend=pd_router_backend,
        implementation=implementation,
        implementation_version=implementation_version,
        render_source=render_source,
        effective_settings=dict(pd_router_settings or {}),
        ports=list(pd_router_ports or []),
        readiness=pd_router_readiness,
        policies=policies,
        prefill_role=prefill_role,
        decode_role=decode_role,
        target_scheme=target_scheme,
        handoff=FrontendHandoff(),
        co_rendering=co_rendering,
        render_inputs=list(pd_router_render_inputs or []),
    )
    return gateway, pd_router


def split_serve_allocations(
    allocations: list[ServeProcessAllocation],
) -> tuple[list[ServeAllocation], list[ServeProcessAllocationModelRank]]:
    """Unwrap allocations once and retain the ordered model-rank subset."""
    unwrapped: list[ServeAllocation] = [allocation.root for allocation in allocations]
    model_allocations = [
        allocation
        for allocation in unwrapped
        if isinstance(allocation, ServeProcessAllocationModelRank)
    ]
    return unwrapped, model_allocations


def require_integration_fused_frontend(
    allocation: ServeAllocation,
    *,
    gateway_backend: str,
    pd_router_backend: str,
) -> PdRouterPlan:
    """Validate the protocol invariants shared by integration-rendered P/D frontends."""
    if not isinstance(allocation, ServeProcessAllocationFrontend):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "integration-rendered P/D serving requires one frontend allocation",
        )
    if allocation.gateway.backend != gateway_backend:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"frontend Gateway backend must be {gateway_backend}",
        )
    pd_router = allocation.pd_router
    if pd_router is None or pd_router.backend != pd_router_backend:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"frontend P/D Router backend must be {pd_router_backend}",
        )
    if (
        allocation.gateway.render_source != RenderSource.integration
        or pd_router.render_source != RenderSource.integration
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "integration-rendered frontend components must use render_source=integration",
        )
    if not isinstance(allocation.components.root, GatewayPdRouterFrontendBinding):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "P/D frontend allocation must bind [gateway, pd_router]",
        )
    if (
        allocation.process_role != allocation.gateway.co_rendering.process_role
        or allocation.process_role != pd_router.co_rendering.process_role
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "frontend components must co-render in the allocated Gateway process",
        )
    targets = allocation.gateway.targets
    if len(targets) != 1 or not isinstance(targets[0].root, GatewayTargetPdRouter):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "P/D Gateway must target its co-rendered P/D Router",
        )
    return pd_router


def rendered_model_rank(
    allocation: ServeProcessAllocationModelRank,
    command: ProcessSpec,
    *,
    launch_files: list[LaunchFileDeclaration] | None = None,
) -> RenderedServeProcess:
    """Preserve a model-rank allocation identity in an adapter render result."""
    return RenderedServeProcess(
        root=RenderedServeProcessModelRank(
            process=allocation.process,
            role=allocation.role,
            replica=allocation.replica,
            rank=allocation.rank,
            rank_count=allocation.rank_count,
            command=command,
            launch_files=list(launch_files or []),
        )
    )


def rendered_frontend(
    allocation: ServeAllocation,
    command: ProcessSpec,
    *,
    launch_files: list[LaunchFileDeclaration] | None = None,
) -> RenderedServeProcess:
    """Preserve a process-only frontend allocation identity in a render result."""
    if not isinstance(allocation, ServeProcessAllocationFrontend):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "a frontend render result requires a frontend allocation",
        )
    return RenderedServeProcess(
        root=RenderedServeProcessFrontend(
            process=allocation.process,
            process_role=allocation.process_role,
            components=allocation.components,
            command=command,
            launch_files=list(launch_files or []),
        )
    )


def append_option(argv: list[str], name: str, value: str | int | float | None) -> None:
    if value is not None:
        argv.extend([name, str(value)])


def validate_extra_args(extra_args: list[str], owned_options: Collection[str]) -> None:
    """Reject InferLab-owned options in the extra-args escape hatch.

    Only tokens before the `--` passthrough sentinel are checked; anything
    after it is a deliberate verbatim override and passes through untouched.
    """
    for argument in extra_args:
        if argument == "--":
            return
        name, _separator, _value = argument.partition("=")
        if name in owned_options:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                f"extra_args entry {argument!r} names InferLab-owned option {name!r}; "
                "set it through the typed serve settings instead, or place it "
                "after a '--' sentinel for a deliberate verbatim override",
            )


def merge_serve_args(
    extra_args: list[str],
    inferlab_args: list[str],
    owned_options: Collection[str],
) -> list[str]:
    """Splice operator extra args around the managed argv tail.

    The bare ``--`` sentinel is an InferLab-side composition marker and is
    stripped here: argparse-based engine launchers reject a literal ``--``,
    so only the tokens after it are appended verbatim after the managed tail
    (engine last-wins parsing then applies the deliberate override).
    """
    validate_extra_args(extra_args, owned_options)
    merged = []
    remainder = []
    index = 0
    while index < len(extra_args):
        argument = extra_args[index]
        if argument == "--":
            remainder = extra_args[index + 1 :]
            break
        merged.append(argument)
        index += 1

    merged.extend(inferlab_args)
    merged.extend(remainder)
    return merged


class AdapterOperationError(Exception):
    def __init__(self, code: AdapterErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def error_response(code: AdapterErrorCode, message: str) -> AdapterResponse:
    return AdapterResponse(
        root=AdapterResponseError(
            protocol_version=PROTOCOL_V8,
            error=AdapterError(code=code, message=message),
        )
    )


SUPPORTED_PROTOCOL_VERSION: str = PROTOCOL_V8.root


def handle_request(
    payload: str,
    plan_serve: PlanServeHandler | None = None,
    *,
    render_serve: RenderServeHandler | None = None,
) -> AdapterResponse:
    # Validate the request protocol version before request-shape validation
    # ([[RFC-0006:C-INTEGRATIONS]]): a cross-version request also fails shape
    # checks, and a field-error flood would bury the one actionable fact. A
    # declared-but-unsupported version is the mismatch; malformed JSON and an
    # absent version fall through to ordinary shape validation.
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, ValueError) as error:
        return error_response(AdapterErrorCode.invalid_request, str(error))
    if isinstance(raw, dict) and "protocol_version" in raw:
        version = raw["protocol_version"]
        if version != SUPPORTED_PROTOCOL_VERSION:
            return error_response(
                AdapterErrorCode.unsupported_protocol_version,
                f"received protocol version {version}; this integration supports protocol "
                f"version {SUPPORTED_PROTOCOL_VERSION}",
            )

    try:
        request = AdapterRequest.model_validate_json(payload)
    except ValidationError as error:
        return error_response(AdapterErrorCode.invalid_request, str(error))

    try:
        root = request.root
        result: AdapterResultPlanServe | AdapterResultRenderServe
        if isinstance(root, AdapterRequestPlanServe) and plan_serve is not None:
            result = AdapterResultPlanServe(output=plan_serve(root.input))
        elif isinstance(root, AdapterRequestRenderServe) and render_serve is not None:
            result = AdapterResultRenderServe(output=render_serve(root.input))
        else:
            return error_response(
                AdapterErrorCode.unsupported_operation,
                "adapter does not support the requested operation",
            )
    except AdapterOperationError as error:
        return error_response(error.code, error.message)
    except ValidationError as error:
        return error_response(AdapterErrorCode.invalid_settings, str(error))

    return AdapterResponse(
        root=AdapterResponseOk(
            protocol_version=PROTOCOL_V8,
            result=AdapterResult(root=result),
        )
    )


def run_adapter(
    plan_serve: PlanServeHandler | None = None,
    *,
    render_serve: RenderServeHandler | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    diagnostics_stream: TextIO | None = None,
) -> int:
    source = input_stream if input_stream is not None else sys.stdin
    destination = output_stream if output_stream is not None else sys.stdout
    diagnostics = diagnostics_stream if diagnostics_stream is not None else sys.stderr
    try:
        response = handle_request(
            source.read(),
            plan_serve,
            render_serve=render_serve,
        )
    except Exception:
        traceback.print_exc(file=diagnostics)
        response = error_response(
            AdapterErrorCode.internal,
            "adapter operation failed; diagnostics were written to stderr",
        )
    destination.write(response.model_dump_json())
    destination.write("\n")
    return 0
