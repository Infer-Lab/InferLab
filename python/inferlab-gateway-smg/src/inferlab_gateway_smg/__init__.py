"""SMG frontend lowering shared by the integrations that run SMG ([[ADR-0058]]).

SMG serves as a routed-single Gateway or as a fused Gateway and P/D Router.
This package owns what is a fact of SMG itself: its backend name, its recorded
identity, its launch command, and its readiness. Integrations keep what their
qualification decides, such as routing policies and declared endpoint
capabilities.
"""

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

from inferlab_adapter_sdk import (
    ROUTER_WORKER_STARTUP_TIMEOUT_SECS,
    AdapterErrorCode,
    AdapterOperationError,
    EndpointAssignment,
    ProcessSpec,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    SettingValue,
    TargetEndpointScheme,
)

__all__ = [
    "BACKEND",
    "DISTRIBUTION",
    "IMPLEMENTATION",
    "PackageNotFoundError",
    "gateway_readiness",
    "gateway_settings",
    "installed_version",
    "pd_router_readiness",
    "prefill_decode_command",
    "require_backend",
    "routed_single_command",
]

BACKEND = "smg"
"""The backend name a workspace selects for an SMG Gateway or P/D Router."""

IMPLEMENTATION = "smg"
"""The implementation identity recorded for every SMG frontend component."""

DISTRIBUTION = "tokenspeed-smg"
"""The distribution that installs the SMG Gateway and its `smg` command."""

_FORMER_BACKEND = "tokenspeed-smg"


def require_backend(selected: str | None, *, component: str) -> None:
    """Accept only `smg`; the former name gets the rename instruction."""
    if selected == BACKEND:
        return
    if selected == _FORMER_BACKEND:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f'the SMG {component} backend "tokenspeed-smg" is now named "smg"; '
            f'select "smg" in the workspace',
        )
    if selected is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f'no {component} backend is selected; select "smg"',
        )
    raise AdapterOperationError(
        AdapterErrorCode.invalid_settings,
        f'{component} backend {selected!r} is not SMG; select "smg"',
    )


def installed_version() -> str:
    """The installed SMG Gateway distribution version, or `unavailable`."""
    try:
        return version(DISTRIBUTION)
    except PackageNotFoundError:
        return "unavailable"


def gateway_settings() -> dict[str, SettingValue]:
    """Settings every SMG frontend launches with."""
    return {
        "retries": SettingValue(root=False),
        "circuit_breaker": SettingValue(root=False),
    }


def gateway_readiness() -> ReadinessProbe:
    return ReadinessProbe(root=ReadinessProbeHttp(path="/readiness"))


def pd_router_readiness(*, prefill_bootstrap_port: str) -> ReadinessProbe:
    """Readiness through SMG's `/workers` registry of gRPC targets."""
    return ReadinessProbe(
        root=ReadinessProbeHttpTargetRegistry(
            readiness_path="/readiness",
            registry_path="/workers",
            targets_field="workers",
            target_url_field="url",
            target_role_field="worker_type",
            target_healthy_field="is_healthy",
            target_bootstrap_port_field="bootstrap_port",
            target_scheme=TargetEndpointScheme.grpc,
            prefill_role_value="prefill",
            decode_role_value="decode",
            prefill_bootstrap_port=prefill_bootstrap_port,
        )
    )


def _grpc(target: EndpointAssignment) -> str:
    return f"grpc://{target.host}:{target.port}"


def _launch(
    endpoint: EndpointAssignment, prometheus: EndpointAssignment, model_locator: str
) -> list[str]:
    return [
        "smg",
        "launch",
        "--host",
        endpoint.host,
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
    ]


_FAILURE_HANDLING = ["--disable-retries", "--disable-circuit-breaker"]


def routed_single_command(
    *,
    endpoint: EndpointAssignment,
    prometheus: EndpointAssignment,
    model_locator: str,
    worker: EndpointAssignment,
    policy: str,
) -> ProcessSpec:
    """SMG as the Gateway in front of one gRPC worker."""
    argv = _launch(endpoint, prometheus, model_locator)
    argv.extend(["--worker-urls", _grpc(worker), "--policy", policy, *_FAILURE_HANDLING])
    return ProcessSpec(argv=argv, env={})


def prefill_decode_command(
    *,
    endpoint: EndpointAssignment,
    prometheus: EndpointAssignment,
    model_locator: str,
    prefill: Sequence[tuple[EndpointAssignment, int]],
    decode: Sequence[EndpointAssignment],
    prefill_policy: str,
    decode_policy: str,
) -> ProcessSpec:
    """SMG as the fused Gateway and P/D Router over gRPC workers; each
    prefill target carries its bootstrap port."""
    argv = _launch(endpoint, prometheus, model_locator)
    argv.append("--pd-disaggregation")
    for target, bootstrap_port in prefill:
        argv.extend(["--prefill", _grpc(target), str(bootstrap_port)])
    for target in decode:
        argv.extend(["--decode", _grpc(target)])
    argv.extend(
        [
            "--policy",
            prefill_policy,
            "--prefill-policy",
            prefill_policy,
            "--decode-policy",
            decode_policy,
            *_FAILURE_HANDLING,
        ]
    )
    return ProcessSpec(argv=argv, env={})
