import pytest
from inferlab_adapter_sdk import AdapterErrorCode, AdapterOperationError
from inferlab_adapter_sdk.runtime import merge_serve_args, validate_extra_args

OWNED_OPTIONS: set[str] = {
    "--block-size",
    "--max-model-len",
    "--trust-remote-code",
    "--served-model-name",
}


def test_validate_extra_args_rejects_inferlab_owned_option() -> None:
    with pytest.raises(AdapterOperationError) as captured:
        validate_extra_args(["--block-size", "32"], OWNED_OPTIONS)

    assert captured.value.code == AdapterErrorCode.invalid_settings
    assert "--block-size" in captured.value.message


def test_validate_extra_args_rejects_equals_spelling_of_owned_option() -> None:
    with pytest.raises(AdapterOperationError, match="--max-model-len"):
        validate_extra_args(["--max-model-len=4096"], OWNED_OPTIONS)


def test_validate_extra_args_rejects_owned_flag_after_unknown_pass_through() -> None:
    with pytest.raises(AdapterOperationError, match="--trust-remote-code"):
        validate_extra_args(["--max-num-seqs", "16", "--trust-remote-code"], OWNED_OPTIONS)


def test_validate_extra_args_allows_unknown_options_and_passthrough() -> None:
    validate_extra_args(["--max-num-seqs", "16", "--", "--block-size", "32"], OWNED_OPTIONS)


def test_merge_serve_args_passes_unknown_options_through_and_appends_owned() -> None:
    merged = merge_serve_args(
        ["--max-num-seqs", "16"],
        ["--max-model-len", "4096"],
        OWNED_OPTIONS,
    )

    assert merged == ["--max-num-seqs", "16", "--max-model-len", "4096"]


def test_merge_serve_args_strips_passthrough_sentinel_from_engine_argv() -> None:
    merged = merge_serve_args(
        ["--max-num-seqs", "16", "--", "--block-size", "32"],
        ["--max-model-len", "4096"],
        OWNED_OPTIONS,
    )

    # The sentinel is an InferLab-side composition marker; argparse-based
    # engine launchers reject a literal "--", so only the tokens after it
    # land verbatim after the managed tail.
    assert merged == [
        "--max-num-seqs",
        "16",
        "--max-model-len",
        "4096",
        "--block-size",
        "32",
    ]


def test_merge_serve_args_rejects_owned_option_before_passthrough() -> None:
    with pytest.raises(AdapterOperationError, match="--served-model-name"):
        merge_serve_args(["--served-model-name", "shadow"], [], OWNED_OPTIONS)
