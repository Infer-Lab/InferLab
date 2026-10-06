"""Dynamo lowering shared by the integrations that run Dynamo ([[ADR-0060]]).

Dynamo serves as a routed-single Gateway or as a fused Gateway and P/D Router,
and its frontend finds workers through a discovery service InferLab starts for
each server. This package owns what is a fact of Dynamo itself: its backend
name, its recorded identity, the frontend and etcd commands, the discovery and
listener environment every Dynamo process needs, its readiness, and the worker
control routes. Integrations keep their framework's worker module and engine
arguments.
"""

from collections.abc import Collection, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from ipaddress import ip_address
from pathlib import Path

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    CaptureWindowControlEndpoint,
    CaptureWindowControlRequirement,
    CaptureWindowHttpActionSpec,
    DiscoveryRequirement,
    EndpointDeclaration,
    EndpointProtocol,
    FrontendCoRendering,
    FrontendProcessRole,
    GatewayFrontendBinding,
    GatewayPlan,
    GatewayTarget,
    GatewayTargetEngine,
    HttpActionSpec,
    HttpMethod,
    JsonScalar,
    JsonValueMatch,
    ModelListRequirement,
    PdRouterPlan,
    PdRoutingPolicies,
    ProcessSpec,
    PromptCacheReadZeroRepresentation,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeRegistryMembership,
    RegistryRoleValues,
    RenderedServeProcess,
    RenderSource,
    ServeAllocation,
    ServeProcessAllocationDiscovery,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
    SettingValue,
    SuccessMatch,
    TargetEndpointScheme,
    fused_pd_frontend_plans,
    rendered_discovery,
    rendered_frontend,
    require_integration_fused_frontend,
)

__all__ = [
    "BACKEND",
    "DISTRIBUTION",
    "IMPLEMENTATION",
    "NAMESPACE",
    "OWNED_ENV",
    "OWNED_OPTIONS",
    "REQUEST_PORT",
    "RESPONSE_STREAM_PORT",
    "SYSTEM_PORT_LIMIT",
    "PackageNotFoundError",
    "capture_window_control",
    "discovery_requirement",
    "gateway_readiness",
    "installed_version",
    "prefill_decode_frontends",
    "prefix_cache_reset",
    "public_endpoint",
    "reasoning_parser_args",
    "render_discovery",
    "render_frontend",
    "require_backend",
    "routed_single_gateway",
    "validate_escape_hatch",
    "worker_env",
    "worker_readiness",
]

BACKEND = "dynamo"
"""The backend name a workspace selects for a Dynamo Gateway or P/D Router."""

IMPLEMENTATION = "dynamo"
"""The implementation identity recorded for every Dynamo frontend component."""

DISTRIBUTION = "ai-dynamo"
"""The distribution that installs the Dynamo frontend and worker modules."""

NAMESPACE = "inferlab"
"""The Dynamo namespace of every process of one server. Each server owns its
discovery service, so one fixed name is unique to the server."""

REQUEST_PORT = "request"
"""The worker's request-plane listener: Gateway targets and registry entries
carry its allocated address."""

RESPONSE_STREAM_PORT = "response_stream"
"""The frontend's response-stream listener, rendered as a port only."""

_PEER_PORT = "peer"
_ROUTER_MODE = "round-robin"
_DISCOVERY_MEMBER = "inferlab"

SYSTEM_PORT_LIMIT = 32_767
"""Dynamo 1.5 parses its worker system port as a signed 16-bit integer, so a
larger allocated port aborts the worker at startup."""

# Every Dynamo runtime option also reads an environment variable, so the
# option and its variable are owned together.
_OWNED_RUNTIME_OPTIONS = {
    "--discovery-backend": "DYN_DISCOVERY_BACKEND",
    "--dyn-reasoning-parser": "DYN_REASONING_PARSER",
    "--endpoint": "DYN_ENDPOINT",
    "--event-plane": "DYN_EVENT_PLANE",
    "--kv-state-endpoint": "DYN_KV_STATE_ENDPOINT",
    "--namespace": "DYN_NAMESPACE",
    "--request-plane": "DYN_REQUEST_PLANE",
}

OWNED_OPTIONS: frozenset[str] = frozenset(_OWNED_RUNTIME_OPTIONS)
"""Dynamo runtime options that carry the server's discovery wiring, process
identity, or the rendered reasoning parser; no escape hatch may set them."""

OWNED_ENV: frozenset[str] = frozenset(
    {
        *_OWNED_RUNTIME_OPTIONS.values(),
        "DYN_FORWARDPASS_METRIC_PORT",
        "DYN_NAMESPACE_WORKER_SUFFIX",
        "DYN_SYSTEM_HOST",
        "DYN_SYSTEM_PORT",
        "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
        "DYN_TCP_RESPONSE_STREAM_HOST",
        "DYN_TCP_RESPONSE_STREAM_PORT",
        "DYN_TCP_RPC_HOST",
        "DYN_TCP_RPC_PORT",
        "ETCD_ENDPOINTS",
    }
)
"""The environment form of every owned option, the listener and discovery
variables InferLab renders from allocations, and the variables that would
rename or add listeners to a Dynamo process."""

# Dynamo's server-control routes answer HTTP 200 even when the engine reports
# a failure; only the body status distinguishes success.
_ACTION_SUCCESS = SuccessMatch(pointer="/status", value=JsonScalar("ok"))


def require_backend(selected: str | None, *, component: str) -> None:
    if selected == BACKEND:
        return
    if selected is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f'no {component} backend is selected; select "{BACKEND}"',
        )
    raise AdapterOperationError(
        AdapterErrorCode.invalid_settings,
        f'{component} backend {selected!r} is not Dynamo; select "{BACKEND}"',
    )


def installed_version() -> str:
    """The installed Dynamo distribution version, or `unavailable`."""
    try:
        return version(DISTRIBUTION)
    except PackageNotFoundError:
        return "unavailable"


def validate_escape_hatch(
    extra_args: Sequence[str],
    extra_env: Mapping[str, str],
    *,
    framework_options: Mapping[str, str] | None = None,
) -> None:
    """Reject Dynamo-owned options and environment anywhere in an escape hatch,
    including after the `--` sentinel: they carry the server's discovery wiring,
    which readiness and cleanup depend on. `framework_options` maps each
    framework option the Dynamo worker cannot accept to the reason."""
    for argument in extra_args:
        name = argument.partition("=")[0]
        if name in OWNED_OPTIONS:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                f"extra_args entry {argument!r} names Dynamo option {name!r}, "
                "which InferLab renders for every Dynamo process",
            )
        reason = (framework_options or {}).get(name)
        if reason is not None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_settings,
                f"extra_args entry {argument!r} cannot be used when Dynamo serves the "
                f"framework: {reason}",
            )
    owned = sorted(OWNED_ENV.intersection(extra_env))
    if owned:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"extra_env names Dynamo variables {owned}, "
            "which InferLab renders for every Dynamo process",
        )


def reasoning_parser_args(reasoning_parser: str | None) -> list[str]:
    """The Dynamo frontend, not the framework, post-processes responses."""
    return [] if reasoning_parser is None else ["--dyn-reasoning-parser", reasoning_parser]


def discovery_requirement() -> DiscoveryRequirement:
    return DiscoveryRequirement(
        ports=[_PEER_PORT],
        readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/health")),
        render_inputs=[],
    )


def gateway_readiness() -> ReadinessProbe:
    """Ready when the frontend registry lists every worker's generation
    endpoint at its allocated request-plane address and the model is listed."""
    return ReadinessProbe(
        root=ReadinessProbeRegistryMembership(
            registry_path="/health",
            target_port=REQUEST_PORT,
            entries_pointer="/instances",
            entry_filter=JsonValueMatch(pointer="/endpoint", value="generate"),
            role_pointer="/component",
            role_values=RegistryRoleValues(serve="backend", prefill="prefill", decode="backend"),
            address_pointer="/transport/tcp",
            model_list=ModelListRequirement(
                path="/v1/models", models_pointer="/data", name_pointer="/id"
            ),
        )
    )


def worker_readiness() -> ReadinessProbe:
    """The worker system port answers 200 only once its generation endpoint
    reports ready (see `worker_env`)."""
    return ReadinessProbe(root=ReadinessProbeHttp(path="/health"))


def prefix_cache_reset() -> HttpActionSpec:
    return HttpActionSpec(method=HttpMethod(), path="/engine/flush_cache", success=_ACTION_SUCCESS)


def capture_window_control() -> CaptureWindowControlRequirement:
    return CaptureWindowControlRequirement(
        endpoint=CaptureWindowControlEndpoint.replica_entry,
        start=CaptureWindowHttpActionSpec(
            method=HttpMethod(), path="/engine/control/start_profile", success=_ACTION_SUCCESS
        ),
        stop=CaptureWindowHttpActionSpec(
            method=HttpMethod(), path="/engine/control/stop_profile", success=_ACTION_SUCCESS
        ),
    )


def public_endpoint() -> EndpointDeclaration:
    """The frontend reports prompt cache-read usage by default; the cache
    reset is per worker, so the public endpoint declares none."""
    return EndpointDeclaration(
        protocol=EndpointProtocol(),
        prompt_cache_read_zero_representation=PromptCacheReadZeroRepresentation.explicit,
    )


def _gateway_settings() -> dict[str, SettingValue]:
    return {"router_mode": SettingValue(root=_ROUTER_MODE)}


def routed_single_gateway(role: str) -> GatewayPlan:
    return GatewayPlan(
        backend=BACKEND,
        implementation=IMPLEMENTATION,
        implementation_version=installed_version(),
        render_source=RenderSource.integration,
        effective_settings=_gateway_settings(),
        ports=[RESPONSE_STREAM_PORT],
        endpoint=public_endpoint(),
        readiness=gateway_readiness(),
        targets=[GatewayTarget(root=GatewayTargetEngine(role=role))],
        co_rendering=FrontendCoRendering(process_role=FrontendProcessRole()),
        discovery=discovery_requirement(),
        render_inputs=[],
    )


def prefill_decode_frontends(
    prefill_role: str, decode_role: str
) -> tuple[GatewayPlan, PdRouterPlan]:
    gateway, pd_router = fused_pd_frontend_plans(
        gateway_backend=BACKEND,
        pd_router_backend=BACKEND,
        implementation=IMPLEMENTATION,
        implementation_version=installed_version(),
        render_source=RenderSource.integration,
        endpoint=public_endpoint(),
        gateway_readiness=gateway_readiness(),
        pd_router_readiness=gateway_readiness(),
        policies=PdRoutingPolicies(prefill=_ROUTER_MODE, decode=_ROUTER_MODE),
        prefill_role=prefill_role,
        decode_role=decode_role,
        # Workers are reached over Dynamo's TCP request plane.
        target_scheme=TargetEndpointScheme.tcp,
        gateway_settings=_gateway_settings(),
        gateway_ports=[RESPONSE_STREAM_PORT],
    )
    gateway.discovery = discovery_requirement()
    return gateway, pd_router


def _discovery_url(process: str, link: str | None, allocations: Collection[ServeAllocation]) -> str:
    discovery = next(
        (
            allocation
            for allocation in allocations
            if isinstance(allocation, ServeProcessAllocationDiscovery)
            and allocation.process == link
        ),
        None,
    )
    if discovery is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"Dynamo process {process!r} has no discovery allocation to link to",
        )
    return f"http://{discovery.endpoint.host}:{discovery.endpoint.port}"


def _discovery_env(url: str) -> dict[str, str]:
    return {
        "ETCD_ENDPOINTS": url,
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_NAMESPACE": NAMESPACE,
        "DYN_REQUEST_PLANE": "tcp",
    }


def _require_ip(process: str, host: str) -> str:
    """etcd binds and Dynamo's request plane listens only on IP literals."""
    try:
        ip_address(host)
    except ValueError:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"Dynamo process {process!r} was allocated host {host!r}; Dynamo and its "
            "etcd discovery listen only on IP addresses, so bind its machine by IP",
        ) from None
    return host


def worker_env(
    allocation: ServeProcessAllocationModelRank, allocations: Collection[ServeAllocation]
) -> dict[str, str]:
    """Discovery, request-plane, and control-listener environment of one
    worker; its allocated endpoint is the control listener."""
    if allocation.endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"Dynamo worker {allocation.process!r} is missing its control endpoint",
        )
    if allocation.endpoint.port > SYSTEM_PORT_LIMIT:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"Dynamo worker {allocation.process!r} was allocated control port "
            f"{allocation.endpoint.port}, but Dynamo accepts control ports only up to "
            f"{SYSTEM_PORT_LIMIT}; bind the machine to lower ports",
        )
    request = allocation.ports.get(REQUEST_PORT)
    if request is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"Dynamo worker {allocation.process!r} is missing its {REQUEST_PORT} port",
        )
    return {
        **_discovery_env(_discovery_url(allocation.process, allocation.discovery, allocations)),
        "DYN_SYSTEM_HOST": _require_ip(allocation.process, allocation.endpoint.host),
        "DYN_SYSTEM_PORT": str(allocation.endpoint.port),
        "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": '["generate"]',
        "DYN_TCP_RPC_HOST": _require_ip(allocation.process, request.host),
        "DYN_TCP_RPC_PORT": str(request.port),
    }


def render_discovery(allocation: ServeProcessAllocationDiscovery) -> RenderedServeProcess:
    """One single-member etcd on the allocated client and peer listeners."""
    host = _require_ip(allocation.process, allocation.endpoint.host)
    client = f"http://{host}:{allocation.endpoint.port}"
    peer_port = allocation.ports.get(_PEER_PORT)
    if peer_port is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"discovery process {allocation.process!r} is missing its {_PEER_PORT} port",
        )
    peer = f"http://{_require_ip(allocation.process, peer_port.host)}:{peer_port.port}"
    argv = [
        "etcd",
        "--name",
        _DISCOVERY_MEMBER,
        "--data-dir",
        allocation.data_directory,
        "--listen-client-urls",
        client,
        "--advertise-client-urls",
        client,
        "--listen-peer-urls",
        peer,
        "--initial-advertise-peer-urls",
        peer,
        "--initial-cluster",
        f"{_DISCOVERY_MEMBER}={peer}",
    ]
    return rendered_discovery(allocation, ProcessSpec(argv=argv, env={}))


def render_frontend(
    allocation: ServeProcessAllocationFrontend, allocations: Collection[ServeAllocation]
) -> RenderedServeProcess:
    """The Dynamo frontend realizing the Gateway, and the P/D Router when the
    allocation binds both."""
    if isinstance(allocation.components.root, GatewayFrontendBinding):
        if (
            allocation.gateway.backend != BACKEND
            or allocation.gateway.render_source != RenderSource.integration
        ):
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "a Dynamo Gateway allocation requires an integration-rendered dynamo Gateway",
            )
    else:
        require_integration_fused_frontend(
            allocation, gateway_backend=BACKEND, pd_router_backend=BACKEND
        )
    response_stream = allocation.ports.get(RESPONSE_STREAM_PORT)
    if response_stream is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"Dynamo frontend {allocation.process!r} is missing its {RESPONSE_STREAM_PORT} port",
        )
    setting = allocation.gateway.effective_settings.get("router_mode")
    router_mode = None if setting is None else setting.root
    if not isinstance(router_mode, str):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the Dynamo Gateway plan carries no router_mode setting",
        )
    argv = [
        "python3",
        "-m",
        "dynamo.frontend",
        "--http-host",
        allocation.endpoint.host,
        "--http-port",
        str(allocation.endpoint.port),
        "--router-mode",
        router_mode,
    ]
    env = {
        **_discovery_env(_discovery_url(allocation.process, allocation.discovery, allocations)),
        # Only the port: Dynamo picks the interface its workers can reach.
        "DYN_TCP_RESPONSE_STREAM_PORT": str(response_stream.port),
        # The frontend caches model cards under the user cache directory.
        "XDG_CACHE_HOME": str(Path(allocation.cache) / "xdg"),
    }
    return rendered_frontend(allocation, ProcessSpec(argv=argv, env=env))
