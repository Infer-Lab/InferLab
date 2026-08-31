import subprocess
from pathlib import Path

import inferlab_integration_specialized_engine as integration
import pytest
from inferlab_adapter_sdk import (
    AdapterOperationError,
    CaptureMechanism,
    GatewayTargetEngine,
    Parallelism,
    PlanServeInput,
    PromptCacheReadZeroRepresentation,
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
)
from inferlab_integration_specialized_engine import plan_serve, render_serve

ROOT = Path(__file__).parents[3]


def _plan_input(
    parallelism: Parallelism | None = None,
    **overrides: object,
) -> PlanServeInput:
    base: dict[str, object] = {
        "model": ServeModelInput(id="fixture-model", served_name="fixture-model"),
        "topology": ServeTopology.single,
        "gateway_backend": "smg",
        "pd_router_backend": None,
        "kv_transfer": None,
        "roles": [
            ServeRoleInput(
                id="serve",
                kind=ServeRoleKind.serve,
                replica_count=1,
                parallelism=parallelism or Parallelism(),
                settings={
                    "default_max_output_tokens": SettingValue(root=3),
                    "max_num_batched_tokens": SettingValue(root=12_000),
                },
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
    assert result.gateway.implementation == "tokenspeed-smg"
    assert result.gateway.render_source.value == "integration"
    assert result.gateway.endpoint.server_metrics is not None
    assert result.gateway.endpoint.server_metrics.model_dump(mode="json") == {
        "path": "/metrics",
        "port": "prometheus",
    }
    # A primed or prefix-sharing Bench needs cache-read usage, so the endpoint
    # states the reporting capability rather than leaving it unsaid.
    assert result.gateway.endpoint.prefix_cache_reset is not None
    # The SMG Gateway serves the primed conditioning fan-out, so a multi-target
    # primed Bench plans through the declared capability.
    conditioning = result.gateway.endpoint.prefix_cache_conditioning
    assert conditioning is not None
    assert conditioning.path == "/prime_prefix_cache"
    assert (
        result.gateway.endpoint.prompt_cache_read_zero_representation
        is PromptCacheReadZeroRepresentation.explicit
    )
    assert result.gateway.effective_settings["policy"].root == "least_load"
    target = result.gateway.targets[0].root
    assert isinstance(target, GatewayTargetEngine)
    assert target.role == "serve"
    assert result.roles[0].public_endpoint is None
    assert result.replicas[0].device_count == 1
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
                        "model_key": "dsv4",
                        "text": "dsv4:\n  - 3: 2.49\n",
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
            "start": {"method": "post", "path": "/start_profile", "body": None},
            "stop": {"method": "post", "path": "/stop_profile", "body": None},
        },
    }


@pytest.mark.parametrize("tensor_parallel_size", [2, 4])
def test_plan_resolves_one_process_pure_tp_as_one_device_set(
    tensor_parallel_size: int,
) -> None:
    result = plan_serve(
        _plan_input(
            Parallelism.model_validate({"outer": {"tensor_parallel_size": tensor_parallel_size}})
        )
    )

    effective = result.roles[0].effective_parallelism
    assert effective.outer is not None
    assert effective.outer.tensor_parallel_size == tensor_parallel_size
    assert effective.outer.pipeline_parallel_size == 1
    assert effective.attention is not None
    assert effective.attention.tensor_parallel_size == tensor_parallel_size
    assert effective.attention.data_parallel_size == 1
    assert effective.attention.context_parallel_size == 1
    assert effective.experts is not None
    assert effective.experts.tensor_parallel_size == tensor_parallel_size
    assert effective.experts.data_parallel_size == 1
    assert effective.experts.expert_parallel_size == 1
    assert effective.experts.dense_tensor_parallel_size == tensor_parallel_size
    assert result.replicas[0].device_count == tensor_parallel_size


def test_plan_rejects_non_tp_parallelism_and_conflicting_component_tp() -> None:
    with pytest.raises(AdapterOperationError, match="only tensor parallelism"):
        plan_serve(
            _plan_input(Parallelism.model_validate({"outer": {"pipeline_parallel_size": 2}}))
        )

    with pytest.raises(AdapterOperationError, match="match outer tensor parallelism"):
        plan_serve(
            _plan_input(
                Parallelism.model_validate(
                    {
                        "outer": {"tensor_parallel_size": 2},
                        "attention": {"tensor_parallel_size": 1},
                    }
                )
            )
        )


def test_plan_reads_the_tokenspeed_smg_distribution_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def package_version(distribution: str) -> str:
        requested.append(distribution)
        return "1.7.0"

    monkeypatch.setattr(integration, "version", package_version)

    result = plan_serve(_plan_input())

    assert requested == ["tokenspeed-smg"]
    assert result.gateway is not None
    assert result.gateway.implementation_version == "1.7.0"


def test_plan_rejects_non_contract_topologies_and_engine_specific_settings() -> None:
    with pytest.raises(AdapterOperationError, match="Gateway backend smg"):
        plan_serve(_plan_input(gateway_backend=None))
    with pytest.raises(AdapterOperationError, match="P/D Router"):
        plan_serve(_plan_input(pd_router_backend="smg"))

    request = _plan_input()
    request.roles[0].settings["engine_profile"] = SettingValue(root="qwen3-4b-sm120")
    with pytest.raises(AdapterOperationError, match="engine_profile"):
        plan_serve(request)


def _render_input(
    tensor_parallel_size: int = 1,
    engine_settings: dict[str, object] | None = None,
) -> RenderServeInput:
    plan_input = _plan_input(
        Parallelism.model_validate({"outer": {"tensor_parallel_size": tensor_parallel_size}})
    )
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
                "devices": list(range(tensor_parallel_size)),
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
                "endpoint": {"host": "gateway.example", "port": 30000},
                "ports": {"prometheus": {"host": "gateway.example", "port": 30001}},
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
        gateway_backend=plan_input.gateway_backend,
        pd_router_backend=plan_input.pd_router_backend,
        kv_transfer=plan_input.kv_transfer,
        profiling=None,
        allocations=allocations,
    )


def test_render_uses_only_the_canonical_tp2_worker_contract_and_smg() -> None:
    render_input = _render_input(tensor_parallel_size=2)
    result = render_serve(render_input)

    assert [process.root.process for process in result.processes] == ["server", "gateway"]
    engine = result.processes[0].root.command.argv
    assert engine == [
        "inferlab-token-engine",
        "smg-worker",
        "--listen",
        "engine.example:50051",
        "--model",
        "/models/fixture-model",
        "--served-model-name",
        "fixture-model",
        "--tensor-parallel-size",
        "2",
        "--default-max-output-tokens",
        "3",
        "--max-num-batched-tokens",
        "12000",
    ]
    gateway = result.processes[1].root.command.argv
    assert gateway[:4] == ["smg", "launch", "--host", "0.0.0.0"]
    assert gateway[gateway.index("--worker-urls") + 1] == "grpc://engine.example:50051"
    assert gateway[gateway.index("--worker-startup-timeout-secs") + 1] == "2147483647"
    assert gateway[gateway.index("--tokenizer-path") + 1] == "/models/fixture-model"
    assert gateway[gateway.index("--policy") + 1] == "least_load"
    assert "--pd-disaggregation" not in gateway
    assert all("grout" not in argument and "sm120" not in argument for argument in engine)


def test_plan_records_declared_engine_settings_without_inventing_omitted_ones() -> None:
    plan_input = _plan_input(Parallelism.model_validate({"outer": {"tensor_parallel_size": 2}}))
    plan_input.roles[0].settings["gpu_memory_utilization_percent"] = SettingValue.model_validate(90)
    plan_input.roles[0].settings["prefix_cache_ranks"] = SettingValue.model_validate(
        [{"cpu_bytes": 100, "numa_node": 3}, {"cpu_bytes": 200, "numa_node": 4}]
    )

    effective = plan_serve(plan_input).roles[0].effective_settings

    assert effective["gpu_memory_utilization_percent"].root == 90
    ranks = effective["prefix_cache_ranks"].root
    assert isinstance(ranks, list)
    declared: list[tuple[object, object]] = []
    for entry in ranks:
        fields = entry.root
        assert isinstance(fields, dict)
        declared.append((fields["cpu_bytes"].root, fields["numa_node"].root))
    assert declared == [(100, 3), (200, 4)]
    # An omitted setting renders no argument, so the record must not claim one.
    assert "workspace_reserve_mib" not in effective
    assert "prefix_cache_host_memory_percent" not in effective


def test_render_passes_every_declared_memory_and_prefix_cache_option() -> None:
    render_input = _render_input(
        tensor_parallel_size=2,
        engine_settings={
            "gpu_memory_utilization_percent": 90,
            "workspace_reserve_mib": 2048,
            "prefix_cache_gpu_entries": 16,
            "prefix_cache_ranks": [
                {"cpu_bytes": 100, "numa_node": 3},
                {"cpu_bytes": 200, "numa_node": 4},
            ],
        },
    )

    engine = render_serve(render_input).processes[0].root.command.argv

    # The worker pairs the per-rank lists by occurrence, so rank 0 is (100, 3).
    assert engine[engine.index("--max-num-batched-tokens") + 2 :] == [
        "--gpu-memory-utilization-percent",
        "90",
        "--workspace-reserve-mib",
        "2048",
        "--prefix-cache-gpu-entries",
        "16",
        "--prefix-cache-cpu-bytes-per-rank",
        "100",
        "--prefix-cache-cpu-bytes-per-rank",
        "200",
        "--prefix-cache-numa-node-per-rank",
        "3",
        "--prefix-cache-numa-node-per-rank",
        "4",
    ]


def test_render_rejects_per_rank_prefix_cache_that_does_not_cover_every_rank() -> None:
    render_input = _render_input(
        tensor_parallel_size=2,
        engine_settings={"prefix_cache_ranks": [{"cpu_bytes": 100, "numa_node": 3}]},
    )

    with pytest.raises(AdapterOperationError, match="declares 1 ranks"):
        render_serve(render_input)


def test_plan_rejects_two_host_prefix_cache_sizing_authorities() -> None:
    # Settings validation runs while planning, so the contradiction never
    # reaches a rendered command.
    with pytest.raises(AdapterOperationError, match="exactly one host sizing authority"):
        _render_input(
            tensor_parallel_size=1,
            engine_settings={
                "prefix_cache_host_memory_percent": 50,
                "prefix_cache_ranks": [{"cpu_bytes": 100, "numa_node": 3}],
            },
        )


def test_render_rejects_multi_process_rank_decomposition() -> None:
    render_input = _render_input(tensor_parallel_size=2)
    engine = render_input.allocations[0].root
    assert isinstance(engine, ServeProcessAllocationModelRank)
    engine.rank_count = 2

    with pytest.raises(AdapterOperationError, match="one rank process"):
        render_serve(render_input)


def test_render_rejects_an_incomplete_tp_device_set() -> None:
    render_input = _render_input(tensor_parallel_size=2)
    engine = render_input.allocations[0].root
    assert isinstance(engine, ServeProcessAllocationModelRank)
    engine.devices = engine.devices[:1]

    with pytest.raises(AdapterOperationError, match="one device per effective"):
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
