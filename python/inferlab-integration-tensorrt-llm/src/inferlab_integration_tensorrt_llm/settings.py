from inferlab_adapter_sdk import (
    SettingValue,
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
