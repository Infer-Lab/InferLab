from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from inferlab_eval_runner.local_eval_source import (
    LocalEvalSourcePlan,
    _resolve_local_data_files,
    snapshot_local_eval_source,
)
from inferlab_measurement_sdk import JsonObject


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@runtime_checkable
class DatasetValues(Protocol):
    def to_list(self) -> list[dict[str, object]]: ...


def rows(dataset: object) -> dict[str, list[dict[str, object]]]:
    if not isinstance(dataset, Mapping):
        raise TypeError("datasets loader did not return a split mapping")
    result: dict[str, list[dict[str, object]]] = {}
    for split, values in dataset.items():
        if not isinstance(split, str) or not isinstance(values, DatasetValues):
            raise TypeError("datasets loader returned an invalid split")
        result[split] = values.to_list()
    return result


class EvalDataAssetSemantics(unittest.TestCase):
    def test_pinned_datasets_resolution_and_snapshot_binding_are_equivalent(self) -> None:
        datasets_module = importlib.import_module("datasets")
        load_dataset = cast(Callable[..., object], datasets_module.load_dataset)
        yaml_module = importlib.import_module("yaml")
        safe_load = cast(Callable[[str], object], yaml_module.safe_load)
        cases: dict[str, object] = {
            "exact": "data/a.jsonl",
            "list": ["data/a.jsonl", "data/b.jsonl"],
            "split_mapping": {
                "train": "data/a.jsonl",
                "validation": ["data/b.jsonl"],
            },
            "recursive_glob": "data/**",
        }
        for name, data_files in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                data = root / "data"
                (data / "nested").mkdir(parents=True)
                (data / "a.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
                (data / "b.jsonl").write_text('{"id":"b"}\n', encoding="utf-8")
                (data / "nested/c.jsonl").write_text('{"id":"c"}\n', encoding="utf-8")
                task = root / "task.yaml"
                task.write_text("task: fixture\ndataset_path: json\n", encoding="utf-8")

                resolved = _resolve_local_data_files(data_files, root)
                plan = LocalEvalSourcePlan(
                    workspace_root=root,
                    include_files=[task],
                    data_files=resolved,
                    config=cast(
                        JsonObject,
                        {
                            "task": "fixture",
                            "dataset_path": "json",
                            "dataset_kwargs": {"data_files": data_files},
                        },
                    ),
                )
                readiness = snapshot_local_eval_source(plan, root / "artifacts" / name)
                self.assertIsNotNone(readiness.eval_binding)
                binding = readiness.eval_binding
                if binding is None:
                    self.fail("snapshot did not return an Eval binding")
                prepared_value = safe_load(Path(binding.task_path).read_text(encoding="utf-8"))
                if not isinstance(prepared_value, dict):
                    self.fail("prepared Eval task is not a mapping")
                prepared = cast(dict[str, object], prepared_value)

                with working_directory(root):
                    original_dataset = load_dataset(
                        "json",
                        data_files=data_files,
                        cache_dir=root / "cache" / name / "original",
                    )
                (data / "a.jsonl").write_text('{"id":"mutated"}\n', encoding="utf-8")
                prepared_dataset = load_dataset(
                    "json",
                    data_files=cast(dict[str, object], prepared["dataset_kwargs"])["data_files"],
                    cache_dir=root / "cache" / name / "prepared",
                )
                self.assertEqual(rows(original_dataset), rows(prepared_dataset))


if __name__ == "__main__":
    unittest.main()
