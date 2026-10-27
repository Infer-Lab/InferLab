from inferlab_adapter_sdk import (
    SettingValue,
    validate_extra_args,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field

_INFERLAB_OWNED_OPTIONS: set[str] = {
    "--attention-context-parallel-size",
    "--attn-cp-size",
    "--context-length",
    "--cp-strategy",
    "--cuda-graph-max-bs-decode",
    "--data-parallel-size",
    "--dcp-size",
    "--decode-context-parallel-size",
    "--disaggregation-bootstrap-port",
    "--disaggregation-mode",
    "--disaggregation-transfer-backend",
    "--dist-init-addr",
    "--dp-size",
    "--enable-cache-report",
    "--enable-dp-attention",
    "--enable-metrics",
    "--enable-prefill-cp",
    "--ep-size",
    "--expert-parallel-size",
    "--host",
    "--kv-cache-dtype",
    "--mem-fraction-static",
    "--model-path",
    "--moe-data-parallel-size",
    "--moe-dense-tp-size",
    "--moe-runner-backend",
    "--nccl-init-addr",
    "--nnodes",
    "--node-rank",
    "--pipeline-parallel-size",
    "--port",
    "--pp-size",
    "--served-model-name",
    "--tensor-parallel-size",
    "--tp-size",
    "--trust-remote-code",
}


class SglangServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_length: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    mem_fraction_static: float | None = Field(default=None, gt=0.0, le=1.0)
    cuda_graph_max_bs_decode: int | None = Field(default=None, ge=1)
    moe_runner_backend: str | None = None
    trust_remote_code: bool = False
    enable_cache_report: bool = False
    enable_metrics: bool = False
    # The memory each SGLang instance contributes to a declared KV store,
    # split across its ranks by SGLang ([[RFC-0003:C-SERVE-KV-STORE]]).
    kv_store_global_segment_size: str | int | None = None
    kv_store_protocol: str | None = None
    kv_store_device_name: str | None = None
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


# Flags the managed argv owns once the server declares a KV store; without
# one, an operator may still run HiCache through `extra_args`.
_KV_STORE_OWNED_OPTIONS: set[str] = {
    "--enable-hierarchical-cache",
    "--enable-hicache",
    "--hicache-storage-backend",
    "--hicache-storage-backend-extra-config",
}


def _settings(values: dict[str, SettingValue]) -> SglangServeSettings:
    settings = validate_settings(SglangServeSettings, values)
    validate_extra_args(settings.extra_args or [], _INFERLAB_OWNED_OPTIONS)
    return settings
