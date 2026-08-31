import pytest
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    resolve_golden_acceptance_length,
)

FLAT_CURVE = """\
dsv4:
  - 1: 1.9
  - 2: 2.6
  - 4: 3.5
"""

MATRIX_CURVE = """\
dsv4:
  thinking_on:
    4: 3.5
  thinking_off:
    4: 3.1
minimaxm3:
  thinking_on:
    3: 2.78
"""


def test_flat_list_curve_resolves_by_draft_count() -> None:
    assert (
        resolve_golden_acceptance_length(
            curve_text=FLAT_CURVE, model_key="dsv4", thinking_mode=None, draft_count=2
        )
        == 2.6
    )


def test_thinking_mode_matrix_resolves_by_mode_and_draft_count() -> None:
    assert (
        resolve_golden_acceptance_length(
            curve_text=MATRIX_CURVE,
            model_key="dsv4",
            thinking_mode="thinking_off",
            draft_count=4,
        )
        == 3.1
    )
    assert (
        resolve_golden_acceptance_length(
            curve_text=MATRIX_CURVE,
            model_key="minimaxm3",
            thinking_mode="thinking_on",
            draft_count=3,
        )
        == 2.78
    )


def test_unknown_model_key_is_a_typed_failure_naming_the_key() -> None:
    with pytest.raises(AdapterOperationError, match="'unknown'") as captured:
        resolve_golden_acceptance_length(
            curve_text=FLAT_CURVE, model_key="unknown", thinking_mode=None, draft_count=2
        )
    assert captured.value.code == AdapterErrorCode.invalid_settings


def test_matrix_entry_without_a_shipped_thinking_mode_is_a_control_plane_bug() -> None:
    with pytest.raises(AdapterOperationError, match="thinking-mode shape") as captured:
        resolve_golden_acceptance_length(
            curve_text=MATRIX_CURVE, model_key="dsv4", thinking_mode=None, draft_count=4
        )
    assert captured.value.code == AdapterErrorCode.invalid_request


def test_thinking_mode_against_a_flat_list_entry_is_a_control_plane_bug() -> None:
    with pytest.raises(AdapterOperationError, match="flat list") as captured:
        resolve_golden_acceptance_length(
            curve_text=FLAT_CURVE, model_key="dsv4", thinking_mode="thinking_on", draft_count=2
        )
    assert captured.value.code == AdapterErrorCode.invalid_request


def test_missing_thinking_mode_is_a_typed_failure_naming_the_mode() -> None:
    with pytest.raises(AdapterOperationError, match="'thinking_off'") as captured:
        resolve_golden_acceptance_length(
            curve_text=MATRIX_CURVE,
            model_key="minimaxm3",
            thinking_mode="thinking_off",
            draft_count=3,
        )
    assert captured.value.code == AdapterErrorCode.invalid_settings


def test_missing_draft_entry_is_a_typed_failure_naming_the_draft_count() -> None:
    with pytest.raises(AdapterOperationError, match="draft count 3") as captured:
        resolve_golden_acceptance_length(
            curve_text=FLAT_CURVE, model_key="dsv4", thinking_mode=None, draft_count=3
        )
    assert captured.value.code == AdapterErrorCode.invalid_settings


def test_non_finite_curve_value_is_a_typed_failure() -> None:
    with pytest.raises(AdapterOperationError, match="finite") as captured:
        resolve_golden_acceptance_length(
            curve_text="dsv4:\n  - 2: .inf\n",
            model_key="dsv4",
            thinking_mode=None,
            draft_count=2,
        )
    assert captured.value.code == AdapterErrorCode.invalid_settings


def test_below_one_curve_value_is_a_typed_failure() -> None:
    with pytest.raises(AdapterOperationError, match="at least one") as captured:
        resolve_golden_acceptance_length(
            curve_text="dsv4:\n  - 2: 0.9\n",
            model_key="dsv4",
            thinking_mode=None,
            draft_count=2,
        )
    assert captured.value.code == AdapterErrorCode.invalid_settings


def test_a_non_mapping_curve_document_is_a_typed_failure() -> None:
    with pytest.raises(AdapterOperationError, match="model keys") as captured:
        resolve_golden_acceptance_length(
            curve_text="- dsv4\n", model_key="dsv4", thinking_mode=None, draft_count=2
        )
    assert captured.value.code == AdapterErrorCode.invalid_settings


def test_malformed_curve_yaml_is_a_typed_failure() -> None:
    with pytest.raises(AdapterOperationError, match="cannot parse") as captured:
        resolve_golden_acceptance_length(
            curve_text="dsv4: [unclosed\n", model_key="dsv4", thinking_mode=None, draft_count=2
        )
    assert captured.value.code == AdapterErrorCode.invalid_settings
