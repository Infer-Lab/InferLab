import json
from pathlib import Path
from typing import cast

import inferlab_measurement_sdk
import pytest
from inferlab_measurement_sdk import (
    BenchClientRequest,
    BenchRequestSourceInputRandomMixture,
    CaseBudgetExpired,
    CaseDeadline,
    EvalClientRequest,
    EvalClientResult,
    EvalDefinitionInputLmEval,
    EvalFailureKind,
    EvalMetricComparison,
    EvalMetricGateConclusion,
    EvalTaskSourceInputBundled,
    EvalTaskSourceInputWorkspaceYaml,
    MeasurementDataAssetPreparationRequest,
    MeasurementDataAssetPreparationResult,
    MeasurementDataAssetReadinessOpaque,
)
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "protocol" / "fixtures"
SCHEMA = ROOT / "protocol" / "schema" / "measurement-protocol-v1.schema.json"


def load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def test_public_sdk_excludes_adapter_models_and_runtime() -> None:
    assert not hasattr(inferlab_measurement_sdk, "AdapterRequest")
    assert not hasattr(inferlab_measurement_sdk, "PlanServeInput")
    assert not hasattr(inferlab_measurement_sdk, "run_adapter")


def test_case_deadline_consumes_one_clock_and_caps_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [10.0]
    monkeypatch.setattr("inferlab_measurement_sdk.runtime.time.monotonic", lambda: now[0])
    deadline = CaseDeadline(10.0)

    now[0] = 12.0
    assert deadline.remaining() == 8.0
    now[0] = 14.0
    assert deadline.remaining(5.0) == 5.0
    with pytest.raises(ValueError, match="attempt cap"):
        deadline.remaining(0.0)
    now[0] = 20.0
    with pytest.raises(CaseBudgetExpired, match="expired"):
        deadline.remaining()


def test_generated_models_preserve_workspace_yaml_eval_task_source() -> None:
    request = EvalClientRequest.model_validate(
        load_json(FIXTURES / "valid" / "eval-client-request-workspace-yaml.json")
    )

    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    source = definition.task.root
    assert isinstance(source, EvalTaskSourceInputWorkspaceYaml)
    assert source.path == "/workspace/evals/custom.yaml"
    assert definition.metric_filter == "strict-match"


def test_generated_models_preserve_bundled_eval_task_identity() -> None:
    request = EvalClientRequest.model_validate(
        load_json(FIXTURES / "valid" / "eval-client-request-bundled.json")
    )

    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    source = definition.task.root
    assert isinstance(source, EvalTaskSourceInputBundled)
    assert source.name == "estonia"
    assert source.task_identity == "inferlab_estonia"
    assert len(source.task_closure_sha256) == 64


def test_generated_models_preserve_typed_eval_probe_failure() -> None:
    result = EvalClientResult.model_validate(
        load_json(FIXTURES / "valid" / "eval-client-result-probe-failure.json")
    )

    assert result.failure_kind == EvalFailureKind.probe_generated_only_logprobs
    assert result.raw_artifacts[0].kind == "prompt-logprob-probe"


def test_generated_models_preserve_normalized_eval_metric_provenance() -> None:
    result = EvalClientResult.model_validate(
        load_json(FIXTURES / "valid" / "eval-client-result-normalized-metric.json")
    )

    metric = result.normalized_metrics["gsm8k:exact_match,strict-match"]
    assert metric.source_identity == "gsm8k"
    assert metric.native_metric_key == "exact_match,strict-match"
    assert result.gate is not None
    assert result.gate.comparison == EvalMetricComparison.at_least
    assert result.gate.conclusion == EvalMetricGateConclusion.passed
    assert result.native_exit_code == 0
    assert result.native_timed_out is False


def test_weighted_random_mixture_fixture_round_trips() -> None:
    request = BenchClientRequest.model_validate(
        load_json(FIXTURES / "valid" / "bench-client-request-random-mixture.json")
    )
    assert request.definition.request_source is not None
    source = request.definition.request_source.root

    assert isinstance(source, BenchRequestSourceInputRandomMixture)
    assert source.total_weight == 10
    assert len(source.shapes) == 2
    assert BenchClientRequest.model_validate(request.model_dump()) == request


def test_agentic_bench_fixtures_round_trip() -> None:
    request = BenchClientRequest.model_validate(
        load_json(FIXTURES / "valid" / "bench-client-request-agentic.json")
    )
    assert request.definition.agentic_source is not None
    assert request.definition.agentic_source.catalog.scenario == "inferencex-agentx-mvp"

    result = inferlab_measurement_sdk.BenchClientResult.model_validate(
        load_json(FIXTURES / "valid" / "bench-client-result-agentic.json")
    )
    assert result.agentic_evidence is not None
    assert result.agentic_evidence.run is not None
    assert result.agentic_evidence.run.submission_valid
    assert result.agentic_evidence.run.branch_stats.children_spawned == 1

    failed = inferlab_measurement_sdk.BenchClientResult.model_validate(
        load_json(FIXTURES / "valid" / "bench-client-result-agentic-source-failed.json")
    )
    assert failed.status is inferlab_measurement_sdk.ClientStatus.failed
    assert failed.agentic_evidence is not None
    assert (
        failed.agentic_evidence.source.observed_sha256
        != failed.agentic_evidence.source.expected_sha256
    )
    assert failed.agentic_evidence.run is None


def test_generated_schema_classifies_measurement_fixtures() -> None:
    bench_request = load_json(FIXTURES / "valid" / "bench-client-request-random-mixture.json")
    agentic_request = load_json(FIXTURES / "valid" / "bench-client-request-agentic.json")
    agentic_result = load_json(FIXTURES / "valid" / "bench-client-result-agentic.json")
    agentic_source_failure = load_json(
        FIXTURES / "valid" / "bench-client-result-agentic-source-failed.json"
    )
    eval_request = load_json(FIXTURES / "valid" / "eval-client-request-bundled.json")
    data_asset_request = load_json(FIXTURES / "valid" / "data-asset-preparation-request-eval.json")
    data_asset_result = load_json(FIXTURES / "valid" / "data-asset-preparation-result-opaque.json")
    validator = Draft202012Validator(load_json(SCHEMA))

    validator.validate({"bench_client_request": bench_request})
    validator.validate({"bench_client_request": agentic_request})
    validator.validate({"bench_client_result": agentic_result})
    validator.validate({"bench_client_result": agentic_source_failure})
    validator.validate({"eval_client_request": eval_request})
    validator.validate({"data_asset_preparation_request": data_asset_request})
    validator.validate({"data_asset_preparation_result": data_asset_result})


def test_generated_models_preserve_opaque_data_asset_readiness() -> None:
    request = MeasurementDataAssetPreparationRequest.model_validate(
        load_json(FIXTURES / "valid" / "data-asset-preparation-request-eval.json")
    )
    result = MeasurementDataAssetPreparationResult.model_validate(
        load_json(FIXTURES / "valid" / "data-asset-preparation-result-opaque.json")
    )

    assert request.phase.root.kind == "resolve"
    assert result.effective_selection is not None
    selection = result.effective_selection.root
    assert selection.kind == "eval"
    assert selection.data_files is not None
    assert selection.data_files.root == {
        "test": ["fixtures/test-000.jsonl", "fixtures/test-001.jsonl"]
    }
    assert result.readiness is not None
    assert isinstance(result.readiness.root, MeasurementDataAssetReadinessOpaque)
    assert result.readiness.root.deferred_source_access
