# inferlab-kvstore

The external KV store lowering that InferLab integrations share when a server
declares a `kv_store`: for each backend, its name, implementation identity and
installed version, typed store settings, store-process command, and readiness.
Each backend is a module; `mooncake` is the first.

Integrations keep what belongs to their framework: how each Engine connects to
the store and how much memory it contributes. The store process itself comes
from the workspace environment, whose installed version this package records.
