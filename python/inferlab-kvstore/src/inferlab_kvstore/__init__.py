"""External KV store lowering shared by the integrations that consume a store
([[RFC-0003:C-SERVE-KV-STORE]]).

Each backend lives in its own module and owns what is a fact of its store
process: its name, recorded identity, typed settings, command, and readiness.
`plan` and `render` select the backend a server declared, so an integration
needs a backend module only for the client side it renders itself.
"""

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    KvStoreInput,
    KvStorePlan,
    RenderedServeProcess,
    ServeProcessAllocationKvStore,
)

from . import mooncake

__all__ = ["BACKENDS", "mooncake", "plan", "render"]

BACKENDS = (mooncake.BACKEND,)
"""Every KV store backend a server may declare."""


def plan(store: KvStoreInput) -> KvStorePlan:
    """The KV store plan for the declared backend."""
    if store.backend == mooncake.BACKEND:
        return mooncake.plan(store)
    raise AdapterOperationError(
        AdapterErrorCode.invalid_settings,
        f"KV store backend {store.backend!r} is not supported; select one of "
        + ", ".join(repr(backend) for backend in BACKENDS),
    )


def render(allocation: ServeProcessAllocationKvStore) -> RenderedServeProcess:
    """Render the store process for the backend its plan names."""
    if allocation.kv_store.backend == mooncake.BACKEND:
        return mooncake.render(allocation)
    raise AdapterOperationError(
        AdapterErrorCode.invalid_request,
        f"KV store allocation names unsupported backend {allocation.kv_store.backend!r}",
    )
