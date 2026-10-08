"""Materialize a release-pinned AIPerf public dataset into canonical rows.

AIPerf owns every dataset-specific step: resolving content upstream cannot
redistribute, checking access to gated sources, and rejecting incomplete
results. This module asks the pinned AIPerf for the rows exactly as its own
public-dataset composer would and writes them as canonical JSONL; the control
plane then holds the bytes to the release catalog's pinned digest
(RFC-0004:C-BENCH-REQUEST-SOURCES).
"""

from __future__ import annotations

import asyncio
import os
from importlib import import_module
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    SCHEMA_VERSION,
    ClientStatus,
    JsonObject,
    MeasurementDataAssetPreparationPhaseMaterialize,
    MeasurementDataAssetPreparationRequest,
    MeasurementDataAssetPreparationResult,
    MeasurementDataAssetRemoteMetadataOutcome,
    MeasurementDataAssetSourceBytesOutcome,
    MeasurementDataAssetSourceInputAiperfPublicDataset,
)

from .population_types import json_line


def materialize_aiperf_public_dataset(
    request: MeasurementDataAssetPreparationRequest,
) -> MeasurementDataAssetPreparationResult:
    source = request.source.root
    if not isinstance(source, MeasurementDataAssetSourceInputAiperfPublicDataset):
        raise TypeError("AIPerf materialization requires an AIPerf public-dataset source")
    if not isinstance(request.phase.root, MeasurementDataAssetPreparationPhaseMaterialize):
        raise TypeError(
            f"unsupported AIPerf public-dataset phase {type(request.phase.root).__name__}"
        )
    # AIPerf's dataset cache is relative to the working directory; the control
    # plane names a machine-local root so it never lands in a workspace.
    cache_root = Path(source.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.chdir(cache_root)
    rows = load_public_dataset_rows(source.dataset)
    output = Path(source.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial")
    with partial.open("wb") as handle:
        for row in rows:
            handle.write(json_line(row))
    partial.replace(output)
    return MeasurementDataAssetPreparationResult(
        schema_version=SCHEMA_VERSION,
        status=ClientStatus.succeeded,
        effective_selection=None,
        readiness=None,
        cache_stores=[],
        # Whether AIPerf reached the network is its own cache's business.
        remote_metadata=MeasurementDataAssetRemoteMetadataOutcome.unavailable,
        source_bytes=MeasurementDataAssetSourceBytesOutcome.materialized,
        error=None,
    )


def load_public_dataset_rows(name: str) -> list[JsonObject]:
    """Return the rows the pinned AIPerf loads for one public dataset.

    The loader arguments come from AIPerf's own composer so InferLab keeps no
    second copy of how a public dataset is constructed; AIPerf's exceptions,
    including access and completeness failures, surface unchanged.
    """
    plugin_module = import_module("aiperf.plugin")
    enums_module = import_module("aiperf.plugin.enums")
    config_module = import_module("aiperf.config.dataset.config")
    composer_module = import_module("aiperf.dataset.composer.public")
    plugins = plugin_module.plugins
    category = enums_module.PluginType.PUBLIC_DATASET_LOADER
    loader_class = plugins.get_class(category, name)
    composer = object.__new__(composer_module.PublicDatasetComposer)
    composer._public_dataset = config_module.PublicDataset.model_validate(
        {"name": "default", "type": "public", "dataset": name}
    )
    kwargs = composer._build_loader_kwargs(name, loader_class)
    preflight = getattr(loader_class, "preflight_materialize", None)
    if preflight is not None:
        preflight(**({"hf_subset": kwargs["hf_subset"]} if "hf_subset" in kwargs else {}))
    loader = loader_class(run=None, **kwargs)
    data = asyncio.run(loader.load_dataset())
    rows = data.get("dataset") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"AIPerf public dataset {name!r} returned no row list")
    return cast(list[JsonObject], rows)
