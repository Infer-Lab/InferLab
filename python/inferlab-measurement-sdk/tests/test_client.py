from pathlib import Path

from inferlab_measurement_sdk import (
    SCHEMA_VERSION,
    ClientStatus,
    MeasurementDataAssetPreparationResult,
    MeasurementDataAssetRemoteMetadataOutcome,
    MeasurementDataAssetSourceBytesOutcome,
    failed_data_asset_preparation_result,
    write_result,
)


def test_failed_data_asset_preparation_result_carries_the_error() -> None:
    result = failed_data_asset_preparation_result(ValueError("source unreadable"))

    assert result.schema_version == SCHEMA_VERSION
    assert result.status is ClientStatus.failed
    assert result.effective_selection is None
    assert result.readiness is None
    assert result.cache_stores == []
    assert result.remote_metadata is MeasurementDataAssetRemoteMetadataOutcome.unavailable
    assert result.source_bytes is MeasurementDataAssetSourceBytesOutcome.unavailable
    assert result.error == "source unreadable"


def test_write_result_replaces_the_target_atomically(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    temporary = tmp_path / ".result.json.tmp"

    first = failed_data_asset_preparation_result(RuntimeError("first"))
    write_result(target, first)
    assert not temporary.exists()
    assert (
        MeasurementDataAssetPreparationResult.model_validate_json(
            target.read_text(encoding="utf-8")
        )
        == first
    )

    second = failed_data_asset_preparation_result(RuntimeError("second"))
    write_result(target, second)
    assert not temporary.exists()
    assert (
        MeasurementDataAssetPreparationResult.model_validate_json(
            target.read_text(encoding="utf-8")
        )
        == second
    )


def test_an_ipv6_endpoint_url_is_bracketed() -> None:
    from inferlab_measurement_sdk._generated import ClientEndpointInput
    from inferlab_measurement_sdk.runtime import endpoint_url

    endpoint = ClientEndpointInput.model_validate(
        {
            "protocol": "http",
            "host": "2001:db8::1",
            "port": 8000,
            "completions_path": "/v1/completions",
            "chat_completions_path": "/v1/chat/completions",
        }
    )
    assert endpoint_url(endpoint, "/v1/models") == "http://[2001:db8::1]:8000/v1/models"
