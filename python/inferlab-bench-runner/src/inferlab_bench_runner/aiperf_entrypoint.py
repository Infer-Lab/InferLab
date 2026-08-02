"""Launch the pinned AIPerf CLI with InferLab's bounded 0.11 shim."""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module
from importlib.metadata import version
from typing import Protocol, cast

from .aiperf_phase_barrier import (
    PROFILE_BARRIER_ENV,
    Aiperf011ProfileBarrierStrategy,
)

SUPPORTED_AIPERF_VERSION = "0.11.0"


class PluginRegistry(Protocol):
    def register(self, category: object, name: str, cls: type[object]) -> None: ...

    def get_class(self, category: object, name: str) -> type[object]: ...


def _register_profile_barrier() -> None:
    observed_version = version("aiperf")
    if observed_version != SUPPORTED_AIPERF_VERSION:
        raise RuntimeError(
            "InferLab's profile barrier requires "
            f"AIPerf {SUPPORTED_AIPERF_VERSION}, found {observed_version}"
        )
    plugin_module = import_module("aiperf.plugin")
    enum_module = import_module("aiperf.plugin.enums")
    registry = cast(PluginRegistry, plugin_module.plugins)
    category = enum_module.PluginType.TIMING_STRATEGY
    registry.register(category, "request_rate", Aiperf011ProfileBarrierStrategy)
    if registry.get_class(category, "request_rate") is not Aiperf011ProfileBarrierStrategy:
        raise RuntimeError("AIPerf did not activate InferLab's profile barrier strategy")


def main() -> None:
    if os.environ.get(PROFILE_BARRIER_ENV) is not None:
        _register_profile_barrier()
    cli_module = import_module("aiperf.cli")
    app = cast(Callable[[], object], cli_module.app)
    app()


if __name__ == "__main__":
    main()
