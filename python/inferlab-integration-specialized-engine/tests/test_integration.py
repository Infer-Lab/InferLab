import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    AuxiliaryModelKind,
    AuxiliaryModelLocator,
    CaptureMechanism,
    EndpointAssignment,
    GatewayTargetEngine,
    KvStoreAccess,
    KvStoreInput,
    Parallelism,
    PlanServeInput,
    PlanServeResult,
    ProbeServeInput,
    ProbeServeTarget,
    PromptCacheReadZeroRepresentation,
    ReadinessProbeHttpTargetRegistry,
    RenderServeInput,
    ServeModelInput,
    ServeProcessAllocation,
    ServeProcessAllocationFrontend,
    ServeProcessAllocationModelRank,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLinkRequestRouting,
    ServeTopology,
    SettingValue,
    handle_request,
)
from inferlab_adapter_sdk.runtime import SUPPORTED_PROTOCOL_VERSION
from inferlab_integration_specialized_engine import (
    plan_serve,
    probe_serve,
    render_serve,
    stack_settings,
)

ROOT = Path(__file__).parents[3]


def _plan_input(
    parallelism: Parallelism | None = None,
    devices: int | None = 1,
    **overrides: object,
) -> PlanServeInput:
    base: dict[str, object] = {
        "model": ServeModelInput(id="fixture-model", served_name="fixture-model"),
        "topology": ServeTopology.single,
        "state_dir": ".inferlab",
        "gateway_backend": "smg",
        "pd_router_backend": None,
        "kv_transfer": None,
        "roles": [
            ServeRoleInput(
                id="serve",
                kind=ServeRoleKind.serve,
                replica_count=1,
                devices=devices,
                parallelism=parallelism or Parallelism(),
                settings={},
            )
        ],
        "profiling": None,
    }
    base.update(overrides)
    return PlanServeInput.model_validate(base)


def test_plan_models_one_smg_gateway_in_front_of_one_token_engine() -> None:
    result = plan_serve(_plan_input())

    assert result.integration.adapter_id == "inferlab-specialized-engine"
    assert result.integration.framework == "specialized-engine"
    assert result.pd_router is None
    assert result.gateway is not None
    assert result.gateway.backend == "smg"
    assert result.gateway.implementation == "smg"
    assert result.gateway.render_source.value == "integration"
    assert result.gateway.endpoint.server_metrics is not None
    assert result.gateway.endpoint.server_metrics.model_dump(mode="json") == {
        "path": "/metrics",
        "port": "prometheus",
    }
    # A primed or prefix-sharing Bench needs cache-read usage, so the endpoint
    # states the reporting capability rather than leaving it unsaid.
    assert result.gateway.endpoint.prefix_cache_reset is not None
    # Current SMG Gateways answer POST /prime_prefix_cache with 404, so the
    # endpoint declares no conditioning fan-out: a single-target primed Bench
    # conditions through the ordinary serving flow, and a multi-target one is
    # rejected while planning ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    assert result.gateway.endpoint.prefix_cache_conditioning is None
    assert (
        result.gateway.endpoint.prompt_cache_read_zero_representation
        is PromptCacheReadZeroRepresentation.explicit
    )
    assert result.gateway.effective_settings["policy"].root == "least_load"
    # The worker protocol is the published service, not an InferLab label
    # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    assert (
        result.gateway.effective_settings["worker_protocol"].root
        == "tokenspeed.grpc.scheduler.TokenSpeedScheduler"
    )
    target = result.gateway.targets[0].root
    assert isinstance(target, GatewayTargetEngine)
    assert target.role == "serve"
    assert result.roles[0].public_endpoint is None
    assert result.replicas[0].device_count == 1
    # The Engine owns its layout: no shape is claimed, and readiness records
    # each Engine's reported data-parallel rank count
    # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    assert result.roles[0].effective_parallelism == Parallelism()
    readiness = result.gateway.readiness.root
    assert isinstance(readiness, ReadinessProbeHttpTargetRegistry)
    assert readiness.target_data_parallel_size_pointer == "/labels/dp_size"
    link = result.links[0].root
    assert isinstance(link, ServeRoleLinkRequestRouting)
    assert link.source == "gateway"
    assert link.targets == ["serve"]


def test_plan_rejects_engine_trace_capture() -> None:
    with pytest.raises(AdapterOperationError, match="engine-trace"):
        plan_serve(_plan_input(profiling=CaptureMechanism.engine_trace))


def test_plan_rejects_the_synthetic_acceptance_overlay() -> None:
    with pytest.raises(AdapterOperationError, match="synthetic acceptance"):
        plan_serve(_plan_input(synthetic_acceptance={"explicit": {"acceptance_length": 2.49}}))
    with pytest.raises(AdapterOperationError, match="synthetic acceptance"):
        plan_serve(
            _plan_input(
                synthetic_acceptance={
                    "curve": {
                        "model_key": "deepseek-v4-flash",
                        "text": "deepseek-v4-flash:\n  - 3: 2.49\n",
                        "sha256": "f" * 64,
                    }
                }
            )
        )


def test_plan_profiles_the_engine_through_the_smg_gateway_window() -> None:
    result = plan_serve(_plan_input(profiling=CaptureMechanism.managed_collection))

    target = result.replicas[0].capture_target
    assert target is not None
    assert target.model_dump(mode="json") == {
        "mechanism": "managed_collection",
        "window_control": {
            "endpoint": "gateway",
            "start": {"method": "post", "path": "/start_profile", "body": None, "success": None},
            "stop": {"method": "post", "path": "/stop_profile", "body": None, "success": None},
        },
    }


@pytest.mark.parametrize("devices", [2, 8])
def test_plan_places_the_declared_devices_and_claims_no_layout(devices: int) -> None:
    result = plan_serve(_plan_input(devices=devices))

    assert result.replicas[0].device_count == devices
    assert result.roles[0].effective_parallelism == Parallelism()


def test_plan_rejects_parallelism_components_and_a_missing_device_count() -> None:
    with pytest.raises(AdapterOperationError, match="declare only devices"):
        plan_serve(
            _plan_input(Parallelism.model_validate({"attention": {"data_parallel_size": 8}}))
        )
    with pytest.raises(AdapterOperationError, match="must declare devices"):
        plan_serve(_plan_input(devices=None))


def test_plan_reads_the_tokenspeed_smg_distribution_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def package_version(distribution: str) -> str:
        requested.append(distribution)
        return "1.7.0"

    import inferlab_gateway_smg

    monkeypatch.setattr(inferlab_gateway_smg, "version", package_version)

    result = plan_serve(_plan_input())

    assert requested == ["tokenspeed-smg"]
    assert result.gateway is not None
    assert result.gateway.implementation_version == "1.7.0"


def test_plan_rejects_non_contract_topologies_and_engine_specific_settings() -> None:
    with pytest.raises(AdapterOperationError, match='select "smg"'):
        plan_serve(_plan_input(gateway_backend=None))
    with pytest.raises(AdapterOperationError, match="P/D Router"):
        plan_serve(_plan_input(pd_router_backend="smg"))

    request = _plan_input()
    request.roles[0].settings["engine_profile"] = SettingValue(root="qwen3-4b-sm120")
    with pytest.raises(AdapterOperationError, match="engine_profile"):
        plan_serve(request)


ENGINE_STACK_SETTINGS = {
    "engine_command": SettingValue.model_validate(["example-token-engine", "serve"])
}


def _render_input(
    devices: int = 1,
    engine_settings: dict[str, object] | None = None,
) -> RenderServeInput:
    plan_input = _plan_input(devices=devices)
    for name, value in (engine_settings or {}).items():
        plan_input.roles[0].settings[name] = SettingValue.model_validate(value)
    plan = plan_serve(plan_input)
    role = plan.roles[0]
    assert plan.gateway is not None
    allocations = [
        ServeProcessAllocation.model_validate(
            {
                "kind": "model_rank",
                "process": "server",
                "role": "serve",
                "role_kind": "serve",
                "replica": 0,
                "rank": 0,
                "rank_count": 1,
                "machine": "engine-node",
                "devices": list(range(devices)),
                "model_locator": "/models/fixture-model",
                "endpoint": {"host": "engine.example", "port": 50051},
                "ports": {},
                "cache": "/cache/server",
                "launch": {"kind": "local"},
                "effective_settings": role.effective_settings,
                "effective_parallelism": role.effective_parallelism,
                "links": plan.links,
                "dependencies": [],
                "render_inputs": [],
            }
        ),
        ServeProcessAllocation.model_validate(
            {
                "kind": "frontend",
                "process": "gateway",
                "process_role": "gateway",
                "components": ["gateway"],
                "machine": "gateway-node",
                "devices": [],
                "endpoint": {"host": "192.0.2.10", "port": 30000},
                "ports": {"prometheus": {"host": "192.0.2.10", "port": 30001}},
                "cache": "/cache/gateway",
                "launch": {"kind": "local"},
                "gateway": plan.gateway,
                "pd_router": None,
                "links": plan.links,
                "dependencies": ["server"],
                "render_inputs": [],
            }
        ),
    ]
    return RenderServeInput(
        model=plan_input.model,
        topology=plan_input.topology,
        state_dir=plan_input.state_dir,
        gateway_backend=plan_input.gateway_backend,
        pd_router_backend=plan_input.pd_router_backend,
        kv_transfer=plan_input.kv_transfer,
        stack_settings=ENGINE_STACK_SETTINGS,
        profiling=None,
        allocations=allocations,
    )


def test_render_uses_only_the_contract_arguments_and_smg() -> None:
    render_input = _render_input(devices=2)
    result = render_serve(render_input)

    assert [process.root.process for process in result.processes] == ["server", "gateway"]
    engine = result.processes[0].root.command.argv
    assert engine == [
        "example-token-engine",
        "serve",
        "--listen",
        "engine.example:50051",
        "--model",
        "/models/fixture-model",
        "--served-model-name",
        "fixture-model",
    ]
    gateway = result.processes[1].root.command.argv
    assert gateway[:4] == ["smg", "launch", "--host", "192.0.2.10"]
    assert gateway[gateway.index("--worker-urls") + 1] == "grpc://engine.example:50051"
    assert gateway[gateway.index("--worker-startup-timeout-secs") + 1] == "2147483647"
    assert gateway[gateway.index("--tokenizer-path") + 1] == "/models/fixture-model"
    assert gateway[gateway.index("--policy") + 1] == "least_load"
    assert "--pd-disaggregation" not in gateway


@pytest.mark.parametrize(
    "retired",
    ["max_num_batched_tokens", "gpu_memory_utilization_percent", "kv_tier_capacity_bytes"],
)
def test_a_retired_typed_setting_points_to_engine_arguments(retired: str) -> None:
    plan_input = _plan_input()
    plan_input.roles[0].settings[retired] = SettingValue(root=1)

    with pytest.raises(AdapterOperationError, match=f"{retired}.*extra_args"):
        plan_serve(plan_input)


def test_render_splices_extra_args_around_the_managed_worker_argv() -> None:
    render_input = _render_input(
        devices=2,
        engine_settings={
            "extra_args": [
                "--speculative=dspark",
                "--speculative-block-size",
                "4",
                "--",
                "--speculative-block-size",
                "8",
            ],
            "extra_env": {"INFERLAB_FORM_D": "1"},
        },
    )

    command = render_serve(render_input).processes[0].root.command

    # Pre-sentinel Engine arguments precede the contract arguments;
    # post-sentinel tokens follow them verbatim as a deliberate override of an
    # Engine option. The sentinel itself is never rendered.
    assert command.argv == [
        "example-token-engine",
        "serve",
        "--speculative=dspark",
        "--speculative-block-size",
        "4",
        "--listen",
        "engine.example:50051",
        "--model",
        "/models/fixture-model",
        "--served-model-name",
        "fixture-model",
        "--speculative-block-size",
        "8",
    ]
    assert command.env == {"INFERLAB_FORM_D": "1"}


@pytest.mark.parametrize(
    "extra_args",
    [["--listen", "0.0.0.0:1"], ["--", "--kv-tier-store=/elsewhere"]],
)
def test_no_engine_argument_redirects_an_allocation(extra_args: list[str]) -> None:
    # A contract argument carries an allocation, so it is rejected before or
    # after the sentinel ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    with pytest.raises(AdapterOperationError, match="contract argument"):
        _render_input(engine_settings={"extra_args": extra_args})
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        with pytest.raises(AdapterOperationError, match=name):
            _render_input(engine_settings={"extra_env": {name: "0"}})


def test_the_engine_layout_is_an_engine_argument() -> None:
    render_input = _render_input(
        devices=8,
        engine_settings={
            "extra_args": ["--data-parallel-size", "8", "--expert-parallel-size", "8"]
        },
    )

    argv = render_serve(render_input).processes[0].root.command.argv

    assert argv[2:6] == ["--data-parallel-size", "8", "--expert-parallel-size", "8"]
    assert "--tensor-parallel-size" not in argv


def test_plan_returns_the_exact_escape_hatch_contents_in_effective_settings() -> None:
    plan_input = _plan_input()
    plan_input.roles[0].settings["extra_args"] = SettingValue.model_validate(
        ["--speculative=dspark", "--speculative-block-size", "4"]
    )
    plan_input.roles[0].settings["extra_env"] = SettingValue.model_validate(
        {"INFERLAB_FORM_D": "1"}
    )

    effective = plan_serve(plan_input).roles[0].effective_settings

    args = effective["extra_args"].root
    assert isinstance(args, list)
    assert [entry.root for entry in args] == [
        "--speculative=dspark",
        "--speculative-block-size",
        "4",
    ]
    env = effective["extra_env"].root
    assert isinstance(env, dict)
    assert env["INFERLAB_FORM_D"].root == "1"


def _two_host_render_input(
    devices: tuple[int, int] = (2, 2), profiling: CaptureMechanism | None = None
) -> RenderServeInput:
    """One serve replica placed as two rank bindings, one per machine."""
    render_input = _render_input(devices=devices[0])
    host_zero = render_input.allocations[0].root
    assert isinstance(host_zero, ServeProcessAllocationModelRank)
    host_zero.rank_count = 2
    host_zero.process = "server-rank-000"
    host_zero.machine = "host-a"
    host_zero.ports = {"rendezvous": EndpointAssignment(host="host-a.example", port=50060)}
    member = host_zero.model_copy(
        update={
            "process": "server-rank-001",
            "rank": 1,
            "machine": "host-b",
            "devices": list(range(devices[1])),
            "model_locator": "/models/fixture-model-on-host-b",
            "endpoint": EndpointAssignment(host="host-b.example", port=50051),
            "ports": {},
        }
    )
    render_input.allocations.insert(1, ServeProcessAllocation(root=member))
    return render_input.model_copy(update={"profiling": profiling})


def test_a_serve_replica_declares_its_rendezvous_as_a_rank_zero_port() -> None:
    plan = plan_serve(_plan_input())

    assert plan.replicas[0].primary_ports == ["rendezvous"]


def test_a_replica_spanning_machines_renders_one_engine_per_host() -> None:
    rendered = render_serve(_two_host_render_input())

    argv = {process.root.process: process.root.command.argv for process in rendered.processes}
    host_zero = " ".join(argv["server-rank-000"])
    member = argv["server-rank-001"]
    assert "--listen engine.example:50051" in host_zero
    assert "--host-count 2 --rendezvous-listen host-a.example:50060" in host_zero
    assert "--host-index" not in host_zero
    assert "--listen" not in member
    assert member[member.index("--model") + 1] == "/models/fixture-model-on-host-b"
    assert member[member.index("--served-model-name") + 1] == "fixture-model"
    assert " ".join(member).endswith(
        "--host-count 2 --host-index 1 --rendezvous host-a.example:50060"
    )
    # The Gateway routes only to host 0.
    assert "grpc://host-b.example" not in " ".join(argv["gateway"])
    assert "grpc://engine.example:50051" in " ".join(argv["gateway"])


@pytest.mark.parametrize(
    ("render_input", "message"),
    [
        (lambda: _two_host_render_input(devices=(2, 1)), "unequal device counts"),
        (
            lambda: _two_host_render_input(profiling=CaptureMechanism.managed_collection),
            "cannot profile a replica that spans machines",
        ),
    ],
)
def test_a_replica_spanning_machines_rejects_what_host_zero_cannot_speak_for(
    render_input: Callable[[], RenderServeInput], message: str
) -> None:
    with pytest.raises(AdapterOperationError, match=message):
        render_serve(render_input())


def test_a_prefill_replica_cannot_span_machines() -> None:
    plan_input = _pd_plan_input()
    render_input = _pd_render_input(plan_input, plan_serve(plan_input))
    prefill = render_input.allocations[0].root
    assert isinstance(prefill, ServeProcessAllocationModelRank)
    prefill.rank_count = 2
    member = prefill.model_copy(update={"process": "prefill-000-rank-001", "rank": 1})
    render_input.allocations.insert(1, ServeProcessAllocation(root=member))

    with pytest.raises(AdapterOperationError, match="cannot span machines"):
        render_serve(render_input)


def test_render_rejects_an_engine_without_devices() -> None:
    render_input = _render_input(devices=2)
    engine = render_input.allocations[0].root
    assert isinstance(engine, ServeProcessAllocationModelRank)
    engine.devices = []

    with pytest.raises(AdapterOperationError, match="own its allocated devices"):
        render_serve(render_input)


def test_render_rejects_gateway_implementation_identity_drift() -> None:
    render_input = _render_input()
    frontend = render_input.allocations[1].root
    assert isinstance(frontend, ServeProcessAllocationFrontend)
    frontend.gateway.implementation = "different-smg"

    with pytest.raises(AdapterOperationError, match="preserve the planned SMG Gateway"):
        render_serve(render_input)


def test_specialized_engine_is_one_workspace_side_integration_package() -> None:
    inventory = subprocess.run(
        [ROOT / "scripts/python-package-inventory.sh", "workspace-side"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert inventory.count("inferlab-integration-specialized-engine") == 1
    assert not any(package == "inferlab-integration-grout" for package in inventory)


AUXILIARY_PLAN_FIXTURE = (
    ROOT / "protocol" / "fixtures" / "valid" / "plan-serve-request-auxiliary-models.json"
)


def test_plan_rejects_a_declared_draft_model_auxiliary() -> None:
    # The Specialized Engine contract has no splice target for auxiliary
    # weights, so a declaration fails planning with a typed error naming the
    # kind ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]).
    payload = cast(dict[str, object], json.loads(AUXILIARY_PLAN_FIXTURE.read_text()))

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"
    assert "'draft-model'" in response.root.error.message


def test_render_rejects_an_allocation_carrying_a_draft_model_locator() -> None:
    # Planning refuses the declaration, so an allocation that still carries a
    # resolved locator must not render — the command would drop the declared
    # artifact from the record silently.
    render_input = _render_input()
    engine = render_input.allocations[0].root
    assert isinstance(engine, ServeProcessAllocationModelRank)
    engine.auxiliary_model_locators = [
        AuxiliaryModelLocator(
            kind=AuxiliaryModelKind(root="draft-model"), locator="/models/deepseek-v4-flash-draft"
        )
    ]

    with pytest.raises(AdapterOperationError, match="draft-model") as error:
        render_serve(render_input)

    assert error.value.code == AdapterErrorCode.invalid_request


def test_several_replicas_share_one_gateway_with_the_declared_policy() -> None:
    plan_input = _plan_input(gateway_settings={"policy": SettingValue(root="cache_aware")})
    plan_input.roles[0].replica_count = 2

    plan = plan_serve(plan_input)

    assert [replica.replica_index for replica in plan.replicas] == [0, 1]
    assert plan.gateway is not None
    assert plan.gateway.effective_settings["policy"] == SettingValue(root="cache_aware")
    readiness = plan.gateway.readiness.root
    assert isinstance(readiness, ReadinessProbeHttpTargetRegistry)
    assert readiness.serve_role_value == "regular"

    render_input = _render_input()
    first = render_input.allocations[0].root
    gateway = render_input.allocations[1].root
    assert isinstance(first, ServeProcessAllocationModelRank)
    assert isinstance(gateway, ServeProcessAllocationFrontend)
    second = first.model_copy(
        update={
            "process": "server-1",
            "replica": 1,
            "endpoint": first.endpoint.model_copy(update={"port": 50052})
            if first.endpoint
            else None,
        }
    )
    gateway.gateway = plan.gateway
    render_input.allocations.insert(1, ServeProcessAllocation(root=second))

    argv = render_serve(render_input).processes[-1].root.command.argv

    urls = argv.index("--worker-urls")
    assert argv[urls + 1 : urls + 3] == [
        "grpc://engine.example:50051",
        "grpc://engine.example:50052",
    ]
    assert argv[argv.index("--policy") + 1] == "cache_aware"


def test_an_unknown_gateway_setting_is_rejected() -> None:
    with pytest.raises(AdapterOperationError, match="retries"):
        plan_serve(_plan_input(gateway_settings={"retries": SettingValue(root=3)}))


FILESYSTEM_TIER = KvStoreInput(backend="filesystem", settings={})


def test_a_filesystem_tier_reaches_the_engine_as_its_bound_directory() -> None:
    plan = plan_serve(_plan_input(kv_store=FILESYSTEM_TIER))
    assert plan.kv_store is not None
    assert plan.kv_store.externally_operated

    render_input = _render_input()
    engine = render_input.allocations[0].root
    assert isinstance(engine, ServeProcessAllocationModelRank)
    engine.kv_tier_directory = "/fixture/kv-tier"

    argv = render_serve(render_input).processes[0].root.command.argv

    assert argv[argv.index("--kv-tier-backend") + 1] == "filesystem"
    assert argv[argv.index("--kv-tier-store") + 1] == "/fixture/kv-tier"


def test_a_filesystem_tier_leaves_access_to_the_engine_report() -> None:
    plan = plan_serve(_plan_input(kv_store=FILESYSTEM_TIER))
    assert plan.kv_store is not None
    assert plan.kv_store.access_observed
    assert not plan.kv_store.role_access


def test_a_mooncake_tier_runs_its_store_and_reaches_the_engine_at_its_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inferlab_kvstore import mooncake

    monkeypatch.setattr(mooncake, "installed_version", lambda: "9.9.9")
    plan = plan_serve(_plan_input(kv_store=KvStoreInput(backend="mooncake", settings={})))
    assert plan.kv_store is not None
    assert not plan.kv_store.externally_operated
    assert plan.kv_store.access_observed
    assert not plan.kv_store.role_access

    render_input = _render_input()
    engine = render_input.allocations[0].root
    assert isinstance(engine, ServeProcessAllocationModelRank)
    engine.kv_store = EndpointAssignment(host="store.example", port=50100)

    argv = render_serve(render_input).processes[0].root.command.argv

    assert argv[argv.index("--kv-tier-backend") + 1] == "mooncake"
    assert argv[argv.index("--kv-tier-store") + 1] == "store.example:50100"


def test_probe_serve_engages_a_mooncake_tier_only_at_the_store_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inferlab_gateway_smg

    def report(store: str) -> tuple[dict[str, object], None]:
        return (
            {
                "scheduler_info": {
                    "kv_tier_state": "available",
                    "kv_tier_store": store,
                    "kv_tier_access": "read_write",
                    "kv_tier_publication_ms": 0,
                }
            },
            None,
        )

    readings = {50051: report("store.example:50100"), 50052: report("other.example:1")}
    monkeypatch.setattr(
        inferlab_gateway_smg,
        "worker_server_info",
        lambda endpoint, timeout: readings[endpoint.port],
    )
    result = probe_serve(
        ProbeServeInput(
            model=ServeModelInput(id="fixture-model", served_name="fixture-model"),
            topology=ServeTopology.single,
            attempt_bound_seconds=5,
            kv_store=EndpointAssignment(host="store.example", port=50100),
            processes=[
                ProbeServeTarget(
                    process=f"server-{port}",
                    role="serve",
                    replica=index,
                    endpoint=EndpointAssignment(host="engine.example", port=port),
                )
                for index, port in enumerate(readings)
            ],
        )
    )

    assert [state.engaged for state in result.processes] == [True, False]
    assert result.processes[0].access == KvStoreAccess.read_write


def test_a_kv_tier_declares_the_serve_state_probe_and_a_counted_flush() -> None:
    plan_input = _plan_input(kv_store=FILESYSTEM_TIER)
    plan_input.roles[0].replica_count = 2

    plan = plan_serve(plan_input)

    assert plan.serve_state_probe
    assert plan.gateway is not None
    reset = plan.gateway.endpoint.prefix_cache_reset
    assert reset is not None and reset.success is not None
    assert reset.success.value.root == 2


def test_probe_serve_engages_only_an_available_tier_at_the_rendered_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inferlab_gateway_smg

    def report(
        state: str, store: str, access: str = "read_write", publication_ms: int = 1000
    ) -> dict[str, object]:
        return {
            "scheduler_info": {
                "kv_tier_state": state,
                "kv_tier_store": store,
                "kv_tier_access": access,
                "kv_tier_publication_ms": publication_ms,
            }
        }

    readings: dict[int, tuple[dict[str, object] | None, str | None]] = {
        50051: (report("available", "/fixture/kv-tier"), None),
        50052: (report("unavailable", "/fixture/kv-tier"), None),
        50053: (report("available", "/fixture/other-tier"), None),
        50054: (None, "GetServerInfo failed"),
        # An unreadable access or bound is left for the control plane to reject.
        50055: (report("available", "/fixture/kv-tier", "read", -1), None),
    }
    monkeypatch.setattr(
        inferlab_gateway_smg,
        "worker_server_info",
        lambda endpoint, timeout: readings[endpoint.port],
    )
    probe_input = ProbeServeInput(
        model=ServeModelInput(id="fixture-model", served_name="fixture-model"),
        topology=ServeTopology.single,
        attempt_bound_seconds=5,
        processes=[
            ProbeServeTarget(
                process=f"server-{port}",
                role="serve",
                replica=index,
                endpoint=EndpointAssignment(host="engine.example", port=port),
                kv_tier_directory="/fixture/kv-tier",
            )
            for index, port in enumerate(readings)
        ],
    )

    result = probe_serve(probe_input)

    assert [state.engaged for state in result.processes] == [True, False, False, False, True]
    assert result.processes[0].access == KvStoreAccess.read_write
    assert result.processes[0].publication_bound_ms == 1000
    assert result.processes[1].state["kv_tier_state"].root == "unavailable"
    assert result.processes[3].error == "GetServerInfo failed"
    assert result.processes[3].access is None
    assert result.processes[4].access is None
    assert result.processes[4].publication_bound_ms is None


def _plan_payload(stack_settings: dict[str, object]) -> str:
    plan_input = _plan_input()
    payload = {
        "protocol_version": SUPPORTED_PROTOCOL_VERSION,
        "operation": "plan_serve",
        "input": plan_input.model_dump(mode="json", exclude_none=True),
    }
    cast(dict[str, object], payload["input"])["stack_settings"] = stack_settings
    return json.dumps(payload)


def test_plan_returns_the_declared_engine_command_as_an_effective_stack_setting() -> None:
    response = handle_request(
        _plan_payload({"engine_command": ["example-token-engine", "serve"]}),
        plan_serve,
        stack_settings=stack_settings,
    )

    assert response.root.status == "ok"
    planned = response.root.result.root.output
    assert isinstance(planned, PlanServeResult)
    assert planned.stack_settings == ENGINE_STACK_SETTINGS


def test_an_undeclared_engine_command_defaults_to_the_former_fixed_command() -> None:
    response = handle_request(_plan_payload({}), plan_serve, stack_settings=stack_settings)

    assert response.root.status == "ok"
    planned = response.root.result.root.output
    assert isinstance(planned, PlanServeResult)
    default = ["inferlab-token-engine", "smg-worker"]
    assert planned.stack_settings == {
        "engine_command": SettingValue.model_validate(default),
    }
    render_input = _render_input().model_copy(update={"stack_settings": planned.stack_settings})
    assert render_serve(render_input).processes[0].root.command.argv[:2] == default


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ({"engine_command": []}, "engine_command"),
        ({"engine_command": ["/opt/engine/bin/example-token-engine"]}, "not a path"),
        ({"engine_command": ["example-token-engine", "--listen=0.0.0.0:1"]}, "--listen"),
    ],
)
def test_plan_rejects_an_empty_path_or_contract_owned_engine_command(
    declared: dict[str, object], message: str
) -> None:
    # A path would put a machine-local fact in the shareable stack
    # ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    response = handle_request(_plan_payload(declared), plan_serve, stack_settings=stack_settings)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"
    assert message in response.root.error.message


def _pd_plan_input(**overrides: object) -> PlanServeInput:
    role_settings: dict[str, SettingValue] = {}
    base: dict[str, object] = {
        "model": ServeModelInput(id="fixture-model", served_name="fixture-model"),
        "topology": ServeTopology.prefill_decode,
        "state_dir": ".inferlab",
        "gateway_backend": "smg",
        "pd_router_backend": "smg",
        "kv_transfer": "mooncake",
        "pd_router_settings": {"prefill_policy": SettingValue(root="cache_aware")},
        "roles": [
            ServeRoleInput(
                id="prefill",
                kind=ServeRoleKind.prefill,
                replica_count=2,
                devices=2,
                parallelism=Parallelism(),
                settings=dict(role_settings),
            ),
            ServeRoleInput(
                id="decode",
                kind=ServeRoleKind.decode,
                replica_count=1,
                devices=4,
                parallelism=Parallelism(),
                settings=dict(role_settings),
            ),
        ],
        "profiling": None,
    }
    base.update(overrides)
    return PlanServeInput.model_validate(base)


def test_prefill_decode_plans_engines_behind_one_fused_smg_frontend() -> None:
    plan = plan_serve(_pd_plan_input())

    assert [(replica.id, replica.device_count, replica.ports) for replica in plan.replicas] == [
        ("prefill-000", 2, ["bootstrap"]),
        ("prefill-001", 2, ["bootstrap"]),
        ("decode", 4, []),
    ]
    assert plan.pd_router is not None
    assert plan.pd_router.policies.prefill == "cache_aware"
    assert plan.pd_router.policies.decode == "round_robin"
    assert plan.pd_router.effective_settings["prefill_policy"].root == "cache_aware"
    kv_links = [link.root for link in plan.links if link.root.kind == "kv_transfer"]
    assert [(link.source, link.target, link.mechanism.root) for link in kv_links] == [
        ("prefill", "decode", "mooncake")
    ]
    assert plan.gateway is not None
    reset = plan.gateway.endpoint.prefix_cache_reset
    assert reset is not None and reset.success is not None
    assert reset.success.value.root == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"profiling": CaptureMechanism.managed_collection}, "profiles only"),
        ({"gateway_settings": {"policy": SettingValue(root="cache_aware")}}, "pd_router_settings"),
    ],
)
def test_prefill_decode_rejects_profiling_and_gateway_routing_settings(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(AdapterOperationError, match=message):
        plan_serve(_pd_plan_input(**overrides))


def _pd_render_input(
    plan_input: PlanServeInput, plan: PlanServeResult, tier_directory: str | None = None
) -> RenderServeInput:
    assert plan.gateway is not None and plan.pd_router is not None
    roles = {role.id: role for role in plan.roles}

    def engine(process: str, role: str, replica: int, port: int, bootstrap: int | None) -> object:
        return {
            "kind": "model_rank",
            "process": process,
            "role": role,
            "role_kind": role,
            "replica": replica,
            "rank": 0,
            "rank_count": 1,
            "machine": "engine-node",
            "devices": [0, 1] if role == "prefill" else [0, 1, 2, 3],
            "model_locator": "/models/fixture-model",
            "endpoint": {"host": "engine.example", "port": port},
            "ports": (
                {"bootstrap": {"host": "engine.example", "port": bootstrap}}
                if bootstrap is not None
                else {}
            ),
            "kv_tier_directory": tier_directory,
            "cache": f"/cache/{process}",
            "launch": {"kind": "local"},
            "effective_settings": roles[role].effective_settings,
            "effective_parallelism": roles[role].effective_parallelism,
            "links": plan.links,
            "dependencies": [],
            "render_inputs": [],
        }

    items: list[object] = [
        engine("prefill-000", "prefill", 0, 50051, 8998),
        engine("prefill-001", "prefill", 1, 50052, 8999),
        engine("decode", "decode", 0, 50053, None),
        {
            "kind": "frontend",
            "process": "gateway",
            "process_role": "gateway",
            "components": ["gateway", "pd_router"],
            "machine": "gateway-node",
            "devices": [],
            "endpoint": {"host": "192.0.2.10", "port": 30000},
            "ports": {"prometheus": {"host": "192.0.2.10", "port": 30001}},
            "cache": "/cache/gateway",
            "launch": {"kind": "local"},
            "gateway": plan.gateway,
            "pd_router": plan.pd_router,
            "links": plan.links,
            "dependencies": ["prefill-000", "prefill-001", "decode"],
            "render_inputs": [],
        },
    ]
    return RenderServeInput(
        model=plan_input.model,
        topology=plan_input.topology,
        state_dir=plan_input.state_dir,
        gateway_backend=plan_input.gateway_backend,
        pd_router_backend=plan_input.pd_router_backend,
        kv_transfer=plan_input.kv_transfer,
        stack_settings=ENGINE_STACK_SETTINGS,
        profiling=None,
        allocations=[ServeProcessAllocation.model_validate(item) for item in items],
    )


def test_prefill_decode_renders_role_mode_transfer_and_bootstrap_arguments() -> None:
    plan_input = _pd_plan_input()
    rendered = render_serve(_pd_render_input(plan_input, plan_serve(plan_input)))

    argv = {process.root.process: process.root.command.argv for process in rendered.processes}
    assert argv["prefill-001"][:2] == ["example-token-engine", "serve"]
    assert "--disaggregation-bootstrap-port" in argv["prefill-001"]
    prefill = " ".join(argv["prefill-001"])
    assert "--disaggregation-mode prefill --disaggregation-transfer-backend mooncake" in prefill
    assert "--disaggregation-bootstrap-port 8999" in prefill
    decode = " ".join(argv["decode"])
    assert "--disaggregation-mode decode --disaggregation-transfer-backend mooncake" in decode
    assert "--disaggregation-bootstrap-port" not in decode
    gateway = " ".join(argv["gateway"])
    assert "--pd-disaggregation" in gateway
    assert "--prefill grpc://engine.example:50051 8998" in gateway
    assert "--prefill grpc://engine.example:50052 8999" in gateway
    assert "--decode grpc://engine.example:50053" in gateway
    assert "--prefill-policy cache_aware --decode-policy round_robin" in gateway


def test_an_engine_owned_transfer_mechanism_passes_through_unchanged() -> None:
    # Mechanism names are open; the Engine owns its transport and rejects a
    # name it does not implement ([[RFC-0003:C-SPECIALIZED-ENGINE]]).
    plan_input = _pd_plan_input(kv_transfer="example-transport")
    plan = plan_serve(plan_input)
    kv_links = [link.root for link in plan.links if link.root.kind == "kv_transfer"]
    assert [link.mechanism.root for link in kv_links] == ["example-transport"]
    rendered = render_serve(_pd_render_input(plan_input, plan))
    argv = {process.root.process: process.root.command.argv for process in rendered.processes}
    for process in ("prefill-001", "decode"):
        transfer = argv[process][argv[process].index("--disaggregation-transfer-backend") + 1]
        assert transfer == "example-transport"


def test_prefill_decode_with_a_kv_tier_probes_both_roles_for_their_access() -> None:
    plan = plan_serve(_pd_plan_input(kv_store=FILESYSTEM_TIER))

    assert plan.kv_store is not None
    assert plan.kv_store.access_observed
    assert not plan.kv_store.role_access
    assert plan.serve_state_probe


def test_prefill_decode_with_a_kv_tier_renders_the_directory_to_both_roles() -> None:
    plan_input = _pd_plan_input(kv_store=FILESYSTEM_TIER)
    rendered = render_serve(
        _pd_render_input(plan_input, plan_serve(plan_input), "/fixture/kv-tier")
    )

    argv = {process.root.process: process.root.command.argv for process in rendered.processes}
    assert set(argv) == {"prefill-000", "prefill-001", "decode", "gateway"}
    for process in ("prefill-000", "prefill-001", "decode"):
        assert argv[process][argv[process].index("--kv-tier-store") + 1] == "/fixture/kv-tier"


@pytest.mark.parametrize(
    ("plan_input", "message"),
    [
        (
            lambda: _plan_input(gateway_settings={"policy": SettingValue(root="power_of_two")}),
            "at least 2 serve workers",
        ),
        (
            lambda: _pd_plan_input(
                pd_router_settings={"decode_policy": SettingValue(root="power_of_two")}
            ),
            "at least 2 decode workers",
        ),
    ],
)
def test_power_of_two_over_one_worker_is_rejected_during_planning(
    plan_input: Callable[[], PlanServeInput], message: str
) -> None:
    # SMG refuses to start power_of_two over one worker, so the Gateway would
    # exit during startup.
    with pytest.raises(AdapterOperationError, match=message):
        plan_serve(plan_input())


def test_the_retired_publication_setting_points_to_the_engine_report() -> None:
    with pytest.raises(AdapterOperationError, match="kv_tier_publication_ms"):
        stack_settings({"kv_tier_publication_seconds": SettingValue(root=1.0)})


def test_gateway_extra_args_follow_the_managed_smg_options() -> None:
    plan_input = _plan_input(
        gateway_settings={
            "extra_args": SettingValue.model_validate(["--worker-startup-check-interval", "1"])
        }
    )
    plan = plan_serve(plan_input)
    assert plan.gateway is not None
    assert plan.gateway.effective_settings["extra_args"] == SettingValue.model_validate(
        ["--worker-startup-check-interval", "1"]
    )
    render_input = _render_input()
    gateway = render_input.allocations[1].root
    assert isinstance(gateway, ServeProcessAllocationFrontend)
    gateway.gateway = plan.gateway

    argv = render_serve(render_input).processes[1].root.command.argv

    assert argv[-2:] == ["--worker-startup-check-interval", "1"]
    assert argv.index("--disable-circuit-breaker") < argv.index("--worker-startup-check-interval")


def test_pd_router_extra_args_reach_the_fused_smg_command() -> None:
    plan_input = _pd_plan_input(
        pd_router_settings={
            "extra_args": SettingValue.model_validate(["--health-check-interval-secs", "5"])
        }
    )
    rendered = render_serve(_pd_render_input(plan_input, plan_serve(plan_input)))

    gateway = next(p.root for p in rendered.processes if p.root.process == "gateway")
    assert gateway.command.argv[-2:] == ["--health-check-interval-secs", "5"]


@pytest.mark.parametrize("argument", ["--policy", "--worker-startup-timeout-secs=1", "--"])
def test_smg_extra_args_may_not_name_a_rendered_option(argument: str) -> None:
    with pytest.raises(AdapterOperationError, match="extra_args"):
        plan_serve(
            _plan_input(gateway_settings={"extra_args": SettingValue.model_validate([argument])})
        )
