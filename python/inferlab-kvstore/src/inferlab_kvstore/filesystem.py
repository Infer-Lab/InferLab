"""The `filesystem` backend ([[RFC-0003:C-SERVE-KV-STORE]]).

A `filesystem` store is a directory the file system's own infrastructure
operates. Each machine binds it in its local binding, and the control plane
passes it unchanged to every Engine on that machine. InferLab runs no process
for it and does not create, check, maintain, or clean it. Each consuming
Engine reports its own access and publication bound at readiness, so the plan
declares neither.
"""

from importlib.metadata import version

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    KvStoreInput,
    KvStorePlan,
    ReadinessProbe,
    ReadinessProbeProcessAlive,
)

__all__ = ["BACKEND", "IMPLEMENTATION", "plan"]

BACKEND = "filesystem"

IMPLEMENTATION = "filesystem"
"""The implementation identity recorded for a bound tier directory. The
directory runs no software of its own, so its recorded version is that of the
package defining the backend."""


def plan(store: KvStoreInput) -> KvStorePlan:
    if store.settings:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f'KV store backend "{BACKEND}" takes no settings; the directory is each '
            "machine's kv_tier_directory binding",
        )
    return KvStorePlan(
        backend=BACKEND,
        implementation=IMPLEMENTATION,
        implementation_version=version("inferlab-kvstore"),
        effective_settings={},
        externally_operated=True,
        access_observed=True,
        readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
    )
