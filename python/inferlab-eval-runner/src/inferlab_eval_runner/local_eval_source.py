"""Plan and bind immutable local-file closures for workspace lm-eval tasks."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    EvalPreparedSourceBinding,
    JsonObject,
    MeasurementDataAssetAcquiredSource,
    MeasurementDataAssetContentEntry,
    MeasurementDataAssetReadinessClosed,
)


@dataclass(frozen=True)
class LocalEvalSourcePlan:
    workspace_root: Path
    include_files: list[Path]
    data_files: dict[str, list[Path]]
    config: JsonObject


def _resolve_local_data_files(value: object, workspace_root: Path) -> dict[str, list[Path]]:
    data_files_module = importlib.import_module("datasets.data_files")
    sanitize_patterns = cast(Callable[[object], object], data_files_module.sanitize_patterns)
    try:
        patterns_value = sanitize_patterns(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the pinned datasets resolver rejected data_files: {error}") from error
    if not isinstance(patterns_value, dict) or not all(
        isinstance(split, str)
        and isinstance(patterns, list)
        and all(isinstance(pattern, str) for pattern in patterns)
        for split, patterns in patterns_value.items()
    ):
        raise ValueError("the pinned datasets resolver rejected the data_files shape")
    patterns = cast(dict[str, list[str]], patterns_value)
    if any("://" in pattern for split_patterns in patterns.values() for pattern in split_patterns):
        raise ValueError("remote data-file selectors are not local snapshot inputs")

    from_patterns = cast(Callable[..., object], data_files_module.DataFilesDict.from_patterns)
    try:
        resolved_value = from_patterns(patterns, base_path=str(workspace_root))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"the pinned datasets resolver rejected data_files: {error}") from error
    if not isinstance(resolved_value, dict) or not all(
        isinstance(split, str)
        and isinstance(paths, list)
        and all(isinstance(path, str) for path in paths)
        for split, paths in resolved_value.items()
    ):
        raise ValueError("the pinned datasets resolver returned an invalid local file mapping")

    resolved_files: dict[str, list[Path]] = {}
    for split, raw_paths in cast(dict[str, list[str]], resolved_value).items():
        if not raw_paths:
            raise ValueError(
                f"the pinned datasets resolver matched no local files for split {split!r}"
            )
        files: list[Path] = []
        for raw_path in raw_paths:
            if "://" in raw_path:
                raise ValueError("the pinned datasets resolver returned a non-local data file")
            try:
                path = Path(raw_path).resolve(strict=True)
                path.relative_to(workspace_root)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"resolved data file {raw_path!r} is not inside the workspace"
                ) from error
            if not path.is_file():
                raise ValueError(f"resolved data file {raw_path!r} is not a regular file")
            files.append(path)
        resolved_files[split] = files
    return resolved_files


def plan_local_eval_source(
    resolution: dict[str, object], workspace_root: Path
) -> tuple[LocalEvalSourcePlan | None, str | None]:
    config = resolution.get("effective_task_config")
    includes = resolution.get("include_closure")
    if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
        return None, "the owning resolver did not return one effective task configuration"
    if not isinstance(includes, list) or not all(isinstance(path, str) for path in includes):
        return None, "the owning resolver did not enumerate the task YAML include closure"
    dataset_path = config.get("dataset_path")
    dataset_kwargs = config.get("dataset_kwargs")
    if dataset_path not in {"json", "csv", "parquet", "text", "arrow"} or not isinstance(
        dataset_kwargs, dict
    ):
        return (
            None,
            "the workspace task is not one supported file-backed loader with "
            "dataset_kwargs.data_files",
        )
    data_files = dataset_kwargs.get("data_files")
    if data_files is None:
        return (
            None,
            "the workspace task is not one supported file-backed loader with "
            "dataset_kwargs.data_files",
        )
    if "data_dir" in dataset_kwargs:
        return None, "dataset_kwargs.data_dir changes the pinned loader base path"
    include_files = [Path(path).resolve(strict=True) for path in includes]
    if any("!function" in path.read_text(encoding="utf-8") for path in include_files):
        return None, "the task YAML closure contains a function reference outside the file set"
    try:
        resolved_data_files = _resolve_local_data_files(data_files, workspace_root)
    except (ImportError, ValueError) as error:
        return None, str(error)
    return (
        LocalEvalSourcePlan(
            workspace_root=workspace_root,
            include_files=include_files,
            data_files=resolved_data_files,
            config=cast(JsonObject, config),
        ),
        None,
    )


def snapshot_local_eval_source(
    plan: LocalEvalSourcePlan, artifact_dir: Path
) -> MeasurementDataAssetReadinessClosed:
    snapshot_root = artifact_dir / "prepared-source"
    source_files = sorted(
        {
            *plan.include_files,
            *(path for paths in plan.data_files.values() for path in paths),
        },
        key=lambda path: path.relative_to(plan.workspace_root).as_posix(),
    )
    closure: list[MeasurementDataAssetContentEntry] = []
    for source in source_files:
        relative = source.relative_to(plan.workspace_root)
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        target.write_bytes(content)
        target.chmod(0o444)
        digest = hashlib.sha256(content).hexdigest()
        relative_text = relative.as_posix()
        closure.append(MeasurementDataAssetContentEntry(relative_path=relative_text, sha256=digest))

    prepared_config = dict(plan.config)
    prepared_config.pop("include", None)
    prepared_data_files = {
        split: [str(snapshot_root / source.relative_to(plan.workspace_root)) for source in sources]
        for split, sources in plan.data_files.items()
    }
    dataset_kwargs = prepared_config.get("dataset_kwargs")
    if not isinstance(dataset_kwargs, dict):
        raise ValueError("resolved dataset_kwargs changed before local snapshot")
    prepared_config["dataset_kwargs"] = {
        **dataset_kwargs,
        "data_files": prepared_data_files,
    }
    yaml_module = importlib.import_module("yaml")
    safe_dump = cast(Callable[..., str], yaml_module.safe_dump)
    task_path = snapshot_root / "_inferlab_prepared_task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(safe_dump(prepared_config, sort_keys=False), encoding="utf-8")
    task_path.chmod(0o444)
    return MeasurementDataAssetReadinessClosed(
        kind="closed",
        acquired_source=MeasurementDataAssetAcquiredSource.model_validate(
            {
                "kind": "local_file_closure",
                "source_root": str(plan.workspace_root),
                "files": [entry.model_dump() for entry in closure],
            }
        ),
        verification=[],
        eval_binding=EvalPreparedSourceBinding(
            workspace_root=str(snapshot_root),
            task_path=str(task_path),
        ),
    )
