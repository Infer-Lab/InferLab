import os
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    SettingValue,
    SuppliedRenderInput,
    validate_extra_args,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field

# TensorRT-LLM declares its click options in underscore spellings plus short
# aliases and does no hyphen/underscore normalization, so the claim list must
# name every accepted spelling of every inferlab- or settings-owned option.
_INFERLAB_OWNED_OPTIONS: set[str] = {
    "--cluster_size",
    "--config",
    "--context_parallel_size",
    "--cp_size",
    "--custom_tokenizer",
    "--enable_attention_dp",
    "--enable_chunked_prefill",
    "--ep_size",
    "--extra_llm_api_options",
    "--free_gpu_memory_fraction",
    "--host",
    "--kv_cache_dtype",
    "--kv_cache_free_gpu_memory_fraction",
    "--max_batch_size",
    "--max_num_tokens",
    "--max_seq_len",
    "--moe_cluster_parallel_size",
    "--moe_expert_parallel_size",
    "--pipeline_parallel_size",
    "--port",
    "--pp_size",
    "--served_model_name",
    "--tensor_parallel_size",
    "--tp_size",
    "--trust_remote_code",
    "--tool_parser",
    "--reasoning_parser",
}

type YamlValue = bool | int | float | str | list[YamlValue] | dict[str, YamlValue]


class TrtllmServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_batch_size: int | None = Field(default=None, ge=1)
    max_num_tokens: int | None = Field(default=None, ge=1)
    max_seq_len: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    free_gpu_memory_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    enable_chunked_prefill: bool = False
    trust_remote_code: bool = False
    custom_tokenizer: str | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    # Source YAML; P/D composition overrides its transport and cache invariants.
    extra_llm_api_options: str | None = None
    extra_llm_api_options_patch: dict[str, YamlValue] | None = None
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


def _settings(values: dict[str, SettingValue]) -> TrtllmServeSettings:
    settings = validate_settings(TrtllmServeSettings, values)
    validate_extra_args(settings.extra_args or [], _INFERLAB_OWNED_OPTIONS)
    return settings


def _yaml_mapping(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM YAML {source} must be a mapping",
        )
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM YAML {source} must use string keys",
        )
    return cast(dict[str, object], mapping)


def _merge_yaml_patch(config: dict[str, object], patch: dict[str, YamlValue]) -> None:
    for key, value in patch.items():
        current = config.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_yaml_patch(_yaml_mapping(current, key), value)
        else:
            config[key] = value


def _render_source_path(state_dir: str, path: str) -> str:
    if Path(path).is_absolute():
        return path
    return os.path.normpath(Path(state_dir) / path)


def _read_operator_config(state_dir: str, path: str) -> str:
    """Plan-time read of the operator's source YAML through the workspace
    filesystem under the request-supplied state directory."""
    try:
        return Path(_render_source_path(state_dir, path)).read_text(encoding="utf-8")
    except OSError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"cannot read TensorRT-LLM extra_llm_api_options {path!r}: {error}",
        ) from error


def _parse_operator_config(text: str, path: str) -> dict[str, object]:
    try:
        value: object = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"cannot parse TensorRT-LLM extra_llm_api_options {path!r}: {error}",
        ) from error
    if value is None:
        return {}
    return dict(_yaml_mapping(value, repr(path)))


def _operator_config(
    settings: TrtllmServeSettings,
    state_dir: str,
    render_inputs: list[SuppliedRenderInput] | None = None,
) -> dict[str, object]:
    """The operator's effective extra_llm_api_options mapping: the source YAML
    plus the settings patch. At plan no supplied render inputs exist, so the
    source YAML is read through the workspace filesystem under the
    request-supplied state directory; at render the control-plane-supplied
    frozen text is matched by the source path the same spelling produces
    ([[RFC-0006:C-LAUNCH-FILES]])."""
    config: dict[str, object] = {}
    path = settings.extra_llm_api_options
    if path is not None:
        if render_inputs is None:
            text = _read_operator_config(state_dir, path)
        else:
            supplied = next(
                (
                    item
                    for item in render_inputs
                    if item.source_path == _render_source_path(state_dir, path)
                ),
                None,
            )
            if supplied is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"TensorRT-LLM render input {path!r} was not supplied",
                )
            text = supplied.text
        config = _parse_operator_config(text, path)
    _merge_yaml_patch(config, settings.extra_llm_api_options_patch or {})
    return config
