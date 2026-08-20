import pytest
from inferlab_adapter_sdk import AdapterErrorCode, AdapterOperationError
from inferlab_adapter_sdk.runtime import merge_serve_args, validate_extra_args

OPTION_ARITY: dict[str, int | None] = {
    "--block-size": 1,
    "--max-model-len": 1,
    "--trust-remote-code": 0,
    "--served-model-name": None,
}


def test_validate_extra_args_rejects_inferlab_owned_option() -> None:
    with pytest.raises(AdapterOperationError) as captured:
        validate_extra_args(["--block-size", "32"], OPTION_ARITY)

    assert captured.value.code == AdapterErrorCode.invalid_settings
    assert "--block-size" in captured.value.message


def test_validate_extra_args_rejects_equals_spelling_of_owned_option() -> None:
    with pytest.raises(AdapterOperationError, match="--max-model-len"):
        validate_extra_args(["--max-model-len=4096"], OPTION_ARITY)


def test_validate_extra_args_rejects_owned_flag_after_unknown_pass_through() -> None:
    with pytest.raises(AdapterOperationError, match="--trust-remote-code"):
        validate_extra_args(["--max-num-seqs", "16", "--trust-remote-code"], OPTION_ARITY)


def test_validate_extra_args_allows_unknown_options_and_passthrough() -> None:
    validate_extra_args(["--max-num-seqs", "16", "--", "--block-size", "32"], OPTION_ARITY)


def test_merge_serve_args_passes_unknown_options_through_and_appends_owned() -> None:
    merged = merge_serve_args(
        ["--max-num-seqs", "16"],
        ["--max-model-len", "4096"],
        OPTION_ARITY,
    )

    assert merged == ["--max-num-seqs", "16", "--max-model-len", "4096"]


def test_merge_serve_args_keeps_passthrough_sentinel_verbatim() -> None:
    merged = merge_serve_args(
        ["--max-num-seqs", "16", "--", "--block-size", "32"],
        ["--max-model-len", "4096"],
        OPTION_ARITY,
    )

    assert merged == [
        "--max-num-seqs",
        "16",
        "--max-model-len",
        "4096",
        "--",
        "--block-size",
        "32",
    ]


def test_merge_serve_args_rejects_owned_option_before_passthrough() -> None:
    with pytest.raises(AdapterOperationError, match="--served-model-name"):
        merge_serve_args(["--served-model-name", "shadow"], [], OPTION_ARITY)
