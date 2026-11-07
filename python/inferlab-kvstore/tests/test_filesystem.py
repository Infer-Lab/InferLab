import inferlab_kvstore
from inferlab_adapter_sdk import KvStoreInput, ReadinessProbeProcessAlive


def test_a_filesystem_store_runs_no_process_and_observes_access() -> None:
    plan = inferlab_kvstore.plan(KvStoreInput(backend="filesystem", settings={}))

    assert plan.backend == "filesystem"
    assert plan.externally_operated is True
    assert plan.access_observed is True
    assert not plan.role_access
    assert plan.publication_bound_ms is None
    assert plan.ports == []
    assert isinstance(plan.readiness.root, ReadinessProbeProcessAlive)
