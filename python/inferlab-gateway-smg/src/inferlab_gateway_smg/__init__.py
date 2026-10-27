"""SMG frontend lowering shared by the integrations that run SMG ([[ADR-0058]]).

SMG serves as a routed-single Gateway or as a fused Gateway and P/D Router.
This package owns what is a fact of SMG itself: its backend name, its recorded
identity, its launch command, and its readiness. Integrations keep what their
qualification decides, such as routing policies and declared endpoint
capabilities.
"""

import ipaddress
import socket
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from inferlab_adapter_sdk import (
    ROUTER_WORKER_STARTUP_TIMEOUT_SECS,
    AdapterErrorCode,
    AdapterOperationError,
    EndpointAssignment,
    EndpointDeclaration,
    GatewayPlan,
    HttpActionSpec,
    HttpMethod,
    JsonScalar,
    PdRouterPlan,
    PdRoutingPolicies,
    ProcessSpec,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    RenderedServeProcess,
    RenderSource,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkBootstrap,
    ServeRoleLinkRequestRouting,
    SettingValue,
    SuccessMatch,
    TargetEndpointScheme,
    fused_pd_frontend_plans,
    rank_zero_allocations,
    rendered_frontend,
)

__all__ = [
    "BACKEND",
    "BOOTSTRAP_PORT",
    "DISTRIBUTION",
    "IMPLEMENTATION",
    "PROMETHEUS_PORT",
    "PackageNotFoundError",
    "Policy",
    "flush_cache_reset",
    "gateway_readiness",
    "gateway_settings",
    "installed_version",
    "pd_frontend",
    "pd_router_readiness",
    "prefill_decode_command",
    "render_pd_frontend",
    "require_backend",
    "require_policy_workers",
    "routed_single_command",
    "routed_single_readiness",
    "worker_server_info",
]

BACKEND = "smg"
"""The backend name a workspace selects for an SMG Gateway or P/D Router."""

IMPLEMENTATION = "smg"
"""The implementation identity recorded for every SMG frontend component."""

DISTRIBUTION = "tokenspeed-smg"
"""The distribution that installs the SMG Gateway and its `smg` command."""

Policy = Literal[
    "least_load",
    "power_of_two",
    "round_robin",
    "random",
    "cache_aware",
    "prefix_hash",
    "consistent_hashing",
]
"""The load-balancing policies SMG accepts for `--policy`, `--prefill-policy`,
and `--decode-policy` that an integration may offer. Which ones it offers, and
its default, stay the integration's choice ([[ADR-0058]])."""

BOOTSTRAP_PORT = "bootstrap"
"""The logical port every prefill worker must serve for SMG's P/D Router,
which hands it to each paired decode worker."""

PROMETHEUS_PORT = "prometheus"
"""The logical port SMG serves Prometheus metrics on."""

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


def flush_cache_reset(worker_count: int) -> HttpActionSpec:
    """`POST /flush_cache` fans out to every registered worker. SMG answers a
    partial flush with 206 `partial_success`, and a flush with no workers with
    200; requiring the flushed count to equal the worker count makes the reset
    record show that every worker flushed."""
    return HttpActionSpec(
        method=HttpMethod(),
        path="/flush_cache",
        success=SuccessMatch(pointer="/workers_flushed", value=JsonScalar(worker_count)),
    )


def worker_server_info(
    endpoint: EndpointAssignment, timeout_seconds: float
) -> tuple[dict[str, object] | None, str | None]:
    """A TokenSpeed worker's `GetServerInfo` `server_args` and `scheduler_info`
    as JSON, or why it could not be read. SMG relays neither for gRPC workers,
    so the integration reads the worker itself ([[RFC-0006:C-INTEGRATIONS]])."""
    import grpc  # type: ignore[import-untyped]
    from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
    from smg_grpc_proto import (  # type: ignore[import-untyped]
        tokenspeed_scheduler_pb2,
        tokenspeed_scheduler_pb2_grpc,
    )

    try:
        with grpc.insecure_channel(f"{endpoint.host}:{endpoint.port}") as channel:
            stub = tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerStub(channel)
            response = stub.GetServerInfo(
                tokenspeed_scheduler_pb2.GetServerInfoRequest(), timeout=timeout_seconds
            )
    except grpc.RpcError as error:
        return None, f"GetServerInfo at {endpoint.host}:{endpoint.port} failed: {error}"
    return {
        "server_args": MessageToDict(response.server_args),
        "scheduler_info": MessageToDict(response.scheduler_info),
    }, None


def routed_single_readiness() -> ReadinessProbe:
    """Readiness through SMG's `/workers` registry: SMG's `/readiness` turns
    ready once any one worker is healthy, so only the registry shows that
    every regular worker has registered and is healthy."""
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
            serve_role_value="regular",
        )
    )


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


def _bind_address(host: str) -> str:
    """SMG parses `--host` as an IP literal and exits on a hostname, which
    other frameworks resolve themselves; a hostname binds its first resolved
    address, so the rendered argv records what SMG listens on."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return host
    try:
        resolved = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"SMG Gateway host {host!r} does not resolve to an IP address to bind: {error}; "
            "declare a resolvable hostname or an IP address as the gateway machine's host",
        ) from error
    return str(resolved[0][4][0])


def _launch(
    endpoint: EndpointAssignment, prometheus: EndpointAssignment, model_locator: str
) -> list[str]:
    return [
        "smg",
        "launch",
        "--host",
        _bind_address(endpoint.host),
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
    workers: Sequence[EndpointAssignment],
    policy: str,
) -> ProcessSpec:
    """SMG as the Gateway in front of one or more gRPC workers."""
    argv = _launch(endpoint, prometheus, model_locator)
    argv.extend(["--worker-urls", *(_grpc(worker) for worker in workers), "--policy", policy])
    argv.extend(_FAILURE_HANDLING)
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


def require_policy_workers(policy: str, workers: int, pool: str) -> None:
    """SMG refuses to start `power_of_two` over a pool of fewer than two
    workers, so such a server is rejected while planning instead of failing
    at launch."""
    if policy == "power_of_two" and workers < 2:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"SMG's power_of_two policy needs at least 2 {pool} workers, but the server "
            f"has {workers}; choose another policy or more {pool} replicas",
        )


def pd_frontend(
    *,
    prefill_role: str,
    decode_role: str,
    prefill_workers: int,
    decode_workers: int,
    endpoint: EndpointDeclaration,
    policies: PdRoutingPolicies,
    pd_router_settings: dict[str, SettingValue],
) -> tuple[GatewayPlan, PdRouterPlan, list[ServeRoleLink]]:
    """SMG as the fused Gateway and P/D Router over gRPC prefill and decode
    workers: both component plans and the frontend links. Each prefill worker
    must serve `BOOTSTRAP_PORT`; the integration declares its endpoint
    capabilities, policies, and settings."""
    require_policy_workers(policies.prefill, prefill_workers, "prefill")
    require_policy_workers(policies.decode, decode_workers, "decode")
    gateway, pd_router = fused_pd_frontend_plans(
        gateway_backend=BACKEND,
        pd_router_backend=BACKEND,
        implementation=IMPLEMENTATION,
        implementation_version=installed_version(),
        render_source=RenderSource.integration,
        endpoint=endpoint,
        gateway_readiness=gateway_readiness(),
        pd_router_readiness=pd_router_readiness(prefill_bootstrap_port=BOOTSTRAP_PORT),
        gateway_settings=gateway_settings(),
        pd_router_settings=pd_router_settings,
        policies=policies,
        prefill_role=prefill_role,
        decode_role=decode_role,
        target_scheme=TargetEndpointScheme.grpc,
        pd_router_ports=[PROMETHEUS_PORT],
    )
    links = [
        ServeRoleLink(root=ServeRoleLinkRequestRouting(source="gateway", targets=["pd_router"])),
        ServeRoleLink(
            root=ServeRoleLinkRequestRouting(
                source="pd_router", targets=[prefill_role, decode_role]
            )
        ),
        ServeRoleLink(
            root=ServeRoleLinkBootstrap(
                source="pd_router", target=prefill_role, port=BOOTSTRAP_PORT
            )
        ),
    ]
    return gateway, pd_router, links


def render_pd_frontend(
    allocation: ServeProcessAllocationFrontend,
    model_allocations: Sequence[ServeProcessAllocationModelRank],
) -> RenderedServeProcess:
    """The fused SMG process over every prefill and decode replica's rank-zero
    worker, routed by the P/D Router plan's policies."""
    if allocation.pd_router is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the SMG P/D frontend allocation carries no P/D Router plan",
        )
    prometheus = allocation.ports.get(PROMETHEUS_PORT)
    if prometheus is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the SMG P/D frontend allocation is missing its Prometheus port",
        )
    prefill = rank_zero_allocations(list(model_allocations), ServeRoleKind.prefill)
    decode = rank_zero_allocations(list(model_allocations), ServeRoleKind.decode)
    model_locator = next((item.model_locator for item in [*prefill, *decode]), None)
    if model_locator is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the SMG P/D frontend requires one serving model locator",
        )
    prefill_targets: list[tuple[EndpointAssignment, int]] = []
    for item in prefill:
        bootstrap = item.ports.get(BOOTSTRAP_PORT)
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
        prefill_targets.append((item.endpoint, bootstrap.port))
    decode_targets: list[EndpointAssignment] = []
    for item in decode:
        if item.endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"decode allocation {item.process!r} has no endpoint",
            )
        decode_targets.append(item.endpoint)
    policies = allocation.pd_router.policies
    return rendered_frontend(
        allocation,
        prefill_decode_command(
            endpoint=allocation.endpoint,
            prometheus=prometheus,
            model_locator=model_locator,
            prefill=prefill_targets,
            decode=decode_targets,
            prefill_policy=policies.prefill,
            decode_policy=policies.decode,
        ),
    )
