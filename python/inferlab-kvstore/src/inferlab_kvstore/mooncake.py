"""The Mooncake Store backend ([[RFC-0003:C-SERVE-KV-STORE]]).

This package owns what is a fact of the Mooncake master: its backend name, its
recorded identity, its typed settings, its launch command, and its readiness.
Integrations keep each framework's connection to the store and the memory each
Engine contributes, because those differ by framework.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    EndpointAssignment,
    KvStoreInput,
    KvStorePlan,
    ProcessSpec,
    ReadinessProbe,
    ReadinessProbeHttp,
    RenderedServeProcess,
    ServeProcessAllocationKvStore,
    effective_settings,
    rendered_kv_store,
    validate_extra_args,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ADMIN_PORT",
    "BACKEND",
    "DISTRIBUTIONS",
    "IMPLEMENTATION",
    "METADATA_SERVER",
    "StoreSettings",
    "client_address",
    "installed_version",
    "master_command",
    "plan",
    "render",
    "require_backend",
]

BACKEND = "mooncake"
"""The backend name a workspace selects for a Mooncake Store."""

IMPLEMENTATION = "mooncake"
"""The implementation identity recorded for the Mooncake master."""

DISTRIBUTIONS = (
    "mooncake-transfer-engine-cuda13",
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-non-cuda",
)
"""The distributions that install `mooncake_master`, in lookup order."""

ADMIN_PORT = "admin"
"""The logical port of the master's admin server, which serves `/health`."""

METADATA_SERVER = "P2PHANDSHAKE"
"""Every consumer exchanges transfer metadata peer to peer, so the master runs
no metadata service."""

_OWNED_OPTIONS = {
    "-rpc_address",
    "-rpc_port",
    "-port",
    "-metrics_host",
    "-metrics_port",
    "-enable_http_metadata_server",
    "-eviction_high_watermark_ratio",
    "-eviction_ratio",
    "-default_kv_lease_ttl",
}


class StoreSettings(BaseModel):
    """The typed Mooncake master settings a server may declare. An omitted
    setting renders no flag, so the master's own default governs."""

    model_config = ConfigDict(extra="forbid")

    eviction_high_watermark_ratio: Annotated[float, Field(gt=0, le=1)] | None = None
    eviction_ratio: Annotated[float, Field(gt=0, lt=1)] | None = None
    default_kv_lease_ttl_ms: Annotated[int, Field(ge=1)] | None = None
    extra_args: list[str] | None = None


def require_backend(store: KvStoreInput) -> None:
    if store.backend != BACKEND:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f'KV store backend {store.backend!r} is not supported; select "{BACKEND}"',
        )


def installed_version() -> str:
    for distribution in DISTRIBUTIONS:
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    raise AdapterOperationError(
        AdapterErrorCode.invalid_settings,
        "no Mooncake transfer-engine distribution is installed; install one of "
        + ", ".join(DISTRIBUTIONS),
    )


def _settings(store: KvStoreInput) -> StoreSettings:
    settings = validate_settings(StoreSettings, store.settings or {})
    validate_extra_args(settings.extra_args or [], _OWNED_OPTIONS)
    return settings


def plan(store: KvStoreInput) -> KvStorePlan:
    """The KV store plan for a declared Mooncake Store."""
    require_backend(store)
    settings = _settings(store)
    return KvStorePlan(
        backend=BACKEND,
        implementation=IMPLEMENTATION,
        implementation_version=installed_version(),
        effective_settings=effective_settings(settings),
        ports=[ADMIN_PORT],
        readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/health", port=ADMIN_PORT)),
        render_inputs=[],
    )


def master_command(
    *,
    endpoint: EndpointAssignment,
    admin: EndpointAssignment,
    store_plan: KvStorePlan,
) -> ProcessSpec:
    """`mooncake_master` bound to its allocated RPC and admin addresses."""
    settings = validate_settings(StoreSettings, store_plan.effective_settings)
    argv = [
        "mooncake_master",
        f"-rpc_address={endpoint.host}",
        f"-rpc_port={endpoint.port}",
        f"-metrics_host={admin.host}",
        f"-metrics_port={admin.port}",
        "-enable_http_metadata_server=false",
    ]
    for flag, value in (
        ("-eviction_high_watermark_ratio", settings.eviction_high_watermark_ratio),
        ("-eviction_ratio", settings.eviction_ratio),
        ("-default_kv_lease_ttl", settings.default_kv_lease_ttl_ms),
    ):
        if value is not None:
            argv.append(f"{flag}={value}")
    argv.extend(settings.extra_args or [])
    return ProcessSpec(argv=argv, env={})


def render(allocation: ServeProcessAllocationKvStore) -> RenderedServeProcess:
    """Render the master for its allocation; the admin port carries `/health`."""
    admin = allocation.ports.get(ADMIN_PORT)
    if admin is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "the KV store allocation is missing its admin port",
        )
    return rendered_kv_store(
        allocation,
        master_command(endpoint=allocation.endpoint, admin=admin, store_plan=allocation.kv_store),
    )


def client_address(endpoint: EndpointAssignment) -> str:
    """The `host:port` every consumer passes as the master address."""
    return f"{endpoint.host}:{endpoint.port}"
