from inferlab_adapter_sdk import (
    SettingValue,
    validate_extra_args,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field

# TokenSpeed accepts several aliases for framework-owned values. Claim every
# accepted spelling so extra_args cannot shadow the resolved model, endpoint,
# process topology, parallelism, or typed settings.
_INFERLAB_OWNED_OPTIONS: set[str] = {
    "--attention-backend",
    "--attention-config.use-fp4-indexer-cache",
    "--attention-use-fp4-indexer-cache",
    "--attention_config.use_fp4_indexer_cache",
    "--attn-tp-size",
    "--block-size",
    "--chunked-prefill-size",
    "--control-port",
    "--data-parallel-size",
    "--dense-tp-size",
    "--disable-kvstore",
    "--disaggregation-bootstrap-port",
    "--disaggregation-mode",
    "--disaggregation-transfer-backend",
    "--dist-init-addr",
    "--enable-expert-parallel",
    "--enable-mixed-batch",
    "--enable-prefix-caching",
    "--ep-size",
    "--expert-parallel-size",
    "--gpu-memory-utilization",
    "--host",
    "--kv-cache-dtype",
    "--max-model-len",
    "--max-num-seqs",
    "--max-total-tokens",
    "--model",
    "--model-path",
    "--moe-backend",
    "--moe-tp-size",
    "--nnodes",
    "--node-rank",
    "--no-enable-prefix-caching",
    "--no-trust-remote-code",
    "--nprocs-per-node",
    "--port",
    "--pdlb-url",
    "--sampling-backend",
    "--served-model-name",
    "--tensor-parallel-size",
    "--tp",
    "--trust-remote-code",
    "--world-size",
}


class TokenspeedServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_len: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    max_num_seqs: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    chunked_prefill_size: int | None = Field(default=None, ge=1)
    block_size: int | None = Field(default=None, ge=1)
    moe_backend: str | None = None
    attention_backend: str | None = None
    sampling_backend: str | None = None
    attention_use_fp4_indexer_cache: bool = False
    enable_mixed_batch: bool = False
    enable_prefix_caching: bool = True
    disable_kvstore: bool = False
    trust_remote_code: bool = False
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


def _settings(values: dict[str, SettingValue]) -> TokenspeedServeSettings:
    settings = validate_settings(TokenspeedServeSettings, values)
    validate_extra_args(settings.extra_args or [], _INFERLAB_OWNED_OPTIONS)
    return settings
