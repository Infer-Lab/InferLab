from inferlab_adapter_sdk import (
    SettingValue,
    validate_extra_args,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue]

_INFERLAB_OWNED_OPTIONS: set[str] = {
    "--block-size",
    "--compilation-config",
    "--data-parallel-size",
    "--decode-context-parallel-size",
    "--enable-auto-tool-choice",
    "--enable-expert-parallel",
    "--enable-flashinfer-autotune",
    "--enable-prompt-tokens-details",
    "--gpu-memory-utilization",
    "--headless",
    "--host",
    "--kv-cache-dtype",
    "--master-addr",
    "--master-port",
    "--max-model-len",
    "--nnodes",
    "--no-enable-flashinfer-autotune",
    "--node-rank",
    "--pipeline-parallel-size",
    "--port",
    "--prefill-context-parallel-size",
    "--profiler-config",
    "--reasoning-config",
    "--reasoning-parser",
    "--served-model-name",
    "--tensor-parallel-size",
    "--tokenizer-mode",
    "--tool-call-parser",
    "--trust-remote-code",
    "--kv-transfer-config",
}


class VllmServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_len: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    block_size: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = False
    compilation_config: dict[str, JsonValue] | None = None
    tokenizer_mode: str | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    enable_auto_tool_choice: bool | None = None
    reasoning_config: dict[str, JsonValue] | None = None
    enable_flashinfer_autotune: bool | None = None
    enable_prompt_tokens_details: bool = False
    kv_transfer_protocol: str | None = None
    mooncake_num_workers: int | None = Field(default=None, ge=1)
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


def _settings(values: dict[str, SettingValue]) -> VllmServeSettings:
    settings = validate_settings(VllmServeSettings, values)
    validate_extra_args(settings.extra_args or [], _INFERLAB_OWNED_OPTIONS)
    return settings
