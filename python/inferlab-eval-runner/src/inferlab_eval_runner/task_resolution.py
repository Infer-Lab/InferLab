from __future__ import annotations

import hashlib
import importlib
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_measurement_sdk import (
    EvalClientRequest,
    EvalDefinitionInputLmEval,
    EvalTaskSourceInputBuiltIn,
    EvalTaskSourceInputBundled,
    EvalTaskSourceInputWorkspaceYaml,
    JsonObject,
    endpoint_url,
)

PROMPT_LOGPROB_OUTPUT_TYPES: frozenset[str] = frozenset(
    {"loglikelihood", "loglikelihood_rolling", "multiple_choice"}
)


@dataclass(frozen=True)
class LmEvalRequestTarget:
    family: str
    model: str
    route_name: str
    url: str
    apply_chat_template: bool


@dataclass(frozen=True)
class PreparedLmEvalTask:
    resolution: JsonObject
    target: LmEvalRequestTarget
    requires_prompt_logprobs: bool


def render_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise ValueError(f"unsupported lm-eval argument value {value!r}")


def render_mapping(values: dict[str, object]) -> str:
    return ",".join(f"{key}={render_value(value)}" for key, value in values.items())


def lm_eval_task_argument(definition: EvalDefinitionInputLmEval) -> str:
    source = definition.task.root
    if isinstance(source, EvalTaskSourceInputBuiltIn):
        return source.name
    if isinstance(source, EvalTaskSourceInputBundled):
        return source.task_identity
    if isinstance(source, EvalTaskSourceInputWorkspaceYaml):
        return source.path
    raise TypeError(f"unsupported lm-eval task source {type(source).__name__}")


def repeated_base_seed(definition: EvalDefinitionInputLmEval) -> int:
    return definition.seed if definition.seed is not None else 1234


def load_yaml_include_mapping(path: Path) -> object:
    yaml_module = importlib.import_module("yaml")
    loader = cast(Callable[..., object], yaml_module.load)
    try:
        return loader(path.read_text(encoding="utf-8"), Loader=yaml_module.BaseLoader)
    except Exception as error:
        raise ValueError(f"task YAML {path} cannot be read: {error}") from error


def workspace_yaml_include_closure(
    task_yaml: Path,
    workspace_root: Path,
    source_exclusions: Sequence[Path] = (),
) -> list[Path]:
    """Resolve lm-eval YAML includes without importing task functions."""
    try:
        resolved_root = workspace_root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"workspace root {workspace_root} cannot be resolved: {error}") from error
    normalized_exclusions: list[Path] = []
    for exclusion in source_exclusions:
        if exclusion.is_absolute() or ".." in exclusion.parts:
            raise ValueError(f"workspace source exclusion {exclusion} is not workspace-relative")
        normalized_exclusions.append(exclusion)

    ordered: list[Path] = []
    visiting: set[Path] = set()
    visited: set[Path] = set()

    def visit(candidate: Path, field: str) -> None:
        candidate = candidate.resolve(strict=False)
        try:
            relative = candidate.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"{field} {candidate} escapes workspace root {resolved_root}"
            ) from error
        if any(
            relative == excluded or relative.is_relative_to(excluded)
            for excluded in normalized_exclusions
        ):
            raise ValueError(f"{field} {candidate} is excluded from workspace source identity")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{field} {candidate} cannot be resolved: {error}") from error
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"{field} {resolved} escapes workspace root {resolved_root}"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"{field} {resolved} is not a regular file")
        if resolved in visiting:
            raise ValueError(f"{field} {resolved} forms an include cycle")
        if resolved in visited:
            return

        visiting.add(resolved)
        ordered.append(resolved)
        raw = load_yaml_include_mapping(resolved)
        if not isinstance(raw, dict):
            raise ValueError(f"{field} {resolved} must contain a YAML mapping")
        includes = raw.get("include", [])
        if isinstance(includes, str):
            include_paths = [includes]
        elif isinstance(includes, list) and all(isinstance(item, str) for item in includes):
            include_paths = includes
        else:
            raise ValueError(f"task field include in {resolved} must be a path or path list")
        for include in include_paths:
            include_path = Path(include)
            if not include_path.is_absolute():
                include_path = resolved.parent / include_path
            visit(include_path, "task include")
        visiting.remove(resolved)
        visited.add(resolved)

    visit(task_yaml, "task YAML")
    for path in ordered:
        repo_root = path.parent
        while repo_root != resolved_root and not (repo_root / ".git").exists():
            repo_root = repo_root.parent
        relative = path.relative_to(repo_root)
        checked = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "--quiet", "--", str(relative)],
            check=False,
            text=True,
            capture_output=True,
        )
        if checked.returncode == 0:
            field = "task YAML" if path == ordered[0] else "task include"
            raise ValueError(f"{field} {path} is excluded from workspace source identity")
        if checked.returncode != 1:
            diagnostic = checked.stderr.strip() or f"exit status {checked.returncode}"
            raise ValueError(f"cannot verify source identity for task YAML {path}: {diagnostic}")
    return ordered


def load_lm_eval_yaml(path: Path) -> JsonObject:
    loader_module = importlib.import_module("lm_eval.tasks._yaml_loader")
    loader = cast(Callable[..., object], loader_module.load_yaml)
    loaded = loader(path, resolve_func=False, recursive=True)
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError(f"lm-eval task YAML {path} did not resolve to a string-keyed object")
    return cast(JsonObject, loaded)


def load_lm_eval_task_manager() -> object:
    tasks_module = importlib.import_module("lm_eval.tasks")
    manager_factory = cast(Callable[[], object], tasks_module.TaskManager)
    return manager_factory()


def resolved_output_type(identity: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"lm-eval task {identity!r} has invalid output_type")
    if value not in {
        "dynamic",
        "generate_until",
        "loglikelihood",
        "loglikelihood_rolling",
        "multiple_choice",
    }:
        raise ValueError(f"lm-eval task {identity!r} has unsupported output_type {value!r}")
    return value


def load_builtin_lm_eval_task(name: str) -> tuple[str, JsonObject, str]:
    manager = load_lm_eval_task_manager()
    catalog = getattr(manager, "all_tasks", None)
    individual_tasks = getattr(manager, "all_subtasks", None)
    if not isinstance(catalog, list) or not all(isinstance(item, str) for item in catalog):
        raise ValueError("lm-eval TaskManager returned no task catalog")
    if not isinstance(individual_tasks, list) or not all(
        isinstance(item, str) for item in individual_tasks
    ):
        raise ValueError("lm-eval TaskManager returned no individual task catalog")
    if name not in catalog:
        raise ValueError(f"lm-eval task field names unknown built-in task {name!r}")
    if name not in individual_tasks:
        raise ValueError(
            f"lm-eval selection {name!r} does not resolve to one individual task; "
            "select each task as a separate Eval definition in the recipe"
        )
    loader = getattr(manager, "load", None)
    if not callable(loader):
        raise ValueError("lm-eval TaskManager returned no task loader")
    loaded = loader(name)
    tasks = loaded.get("tasks") if isinstance(loaded, dict) else None
    if (
        not isinstance(tasks, dict)
        or len(tasks) != 1
        or not all(isinstance(identity, str) for identity in tasks)
    ):
        raise ValueError(
            f"lm-eval selection {name!r} does not resolve to one individual task; "
            "select each task as a separate Eval definition in the recipe"
        )
    identity, task = next(iter(tasks.items()))
    dump_config = getattr(task, "dump_config", None)
    if not callable(dump_config):
        raise ValueError(f"resolved lm-eval task {identity!r} cannot report its configuration")
    config_object = dump_config()
    if not isinstance(config_object, dict) or not all(
        isinstance(key, str) for key in config_object
    ):
        raise ValueError(f"resolved lm-eval task {identity!r} reported an invalid configuration")
    config = cast(JsonObject, config_object)
    task_index = getattr(manager, "task_index", None)
    indexed_entry = task_index.get(name) if isinstance(task_index, dict) else None
    indexed_kind = getattr(indexed_entry, "kind", None)
    kind_name = str(getattr(indexed_kind, "name", indexed_kind)).lower()
    output_type = (
        "dynamic"
        if kind_name == "py_task"
        else resolved_output_type(identity, getattr(task, "OUTPUT_TYPE", None))
    )
    config = {**config, "output_type": output_type}
    return identity, config, output_type


def effective_dataset_selection(config: JsonObject) -> JsonObject:
    evaluation_split = config.get("test_split")
    if evaluation_split is None:
        evaluation_split = config.get("validation_split")
    fewshot_split = config.get("fewshot_split")
    if fewshot_split is None:
        fewshot_split = config.get("training_split")
    return {
        "dataset_path": config.get("dataset_path"),
        "dataset_name": config.get("dataset_name"),
        "evaluation_split": evaluation_split,
        "fewshot_split": fewshot_split,
    }


def task_requires_prompt_logprobs(resolution: JsonObject) -> bool:
    identity = resolution.get("task_identity")
    output_type = resolution.get("output_type")
    if not isinstance(identity, str) or not isinstance(output_type, str):
        raise ValueError("resolved lm-eval task has no identity or output_type")
    if output_type in PROMPT_LOGPROB_OUTPUT_TYPES or output_type == "dynamic":
        return True
    if output_type != "generate_until":
        raise ValueError(f"lm-eval task {identity!r} has unsupported output_type {output_type!r}")
    return False


def resolve_lm_eval_target(
    request: EvalClientRequest, resolution: JsonObject
) -> LmEvalRequestTarget:
    identity = resolution.get("task_identity")
    output_type = resolution.get("output_type")
    if not isinstance(identity, str) or not isinstance(output_type, str):
        raise ValueError("resolved lm-eval task has no identity or output_type")
    if output_type == "generate_until":
        return LmEvalRequestTarget(
            family="chat_completions",
            model="local-chat-completions",
            route_name="chat_completions_path",
            url=endpoint_url(request.endpoint, request.endpoint.chat_completions_path),
            apply_chat_template=True,
        )
    if output_type in PROMPT_LOGPROB_OUTPUT_TYPES or output_type == "dynamic":
        return LmEvalRequestTarget(
            family="completions",
            model="local-completions",
            route_name="completions_path",
            url=endpoint_url(request.endpoint, request.endpoint.completions_path),
            apply_chat_template=False,
        )
    raise ValueError(
        f"lm-eval task {identity!r} has output_type {output_type!r}, "
        "so its request route cannot be selected"
    )


def resolve_lm_eval_task(
    request: EvalClientRequest, definition: EvalDefinitionInputLmEval
) -> JsonObject:
    source = definition.task.root
    if isinstance(source, EvalTaskSourceInputBuiltIn):
        task_identity, config, output_type = load_builtin_lm_eval_task(source.name)
        return {
            "schema_version": 1,
            "status": "resolved",
            "task_source": {"kind": "built_in", "name": source.name},
            "task_identity": task_identity,
            "output_type": output_type,
            "include_closure": [],
            "effective_task_config": config,
            "effective_dataset_selection": effective_dataset_selection(config),
            "tokenizer": {
                "locator": request.model.locator,
                "backend": "huggingface",
                "tokenized_requests": False,
            },
        }
    if isinstance(source, EvalTaskSourceInputBundled):
        task_yaml = Path(source.path).resolve(strict=True)
        root = task_yaml.parent
        assets = {
            "dataset": root / "dataset.json",
            "scorer": root / "estonia.py",
            "task_definition": task_yaml,
            "prompt": root / "prompt.txt",
        }
        for label, path in assets.items():
            if not path.is_file():
                raise ValueError(f"bundled task {source.name!r} has no {label} asset")
        digests = {
            label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in assets.items()
        }
        expected_digests = {
            "dataset": source.dataset_asset_sha256,
            "scorer": source.scorer_sha256,
            "task_definition": source.task_definition_sha256,
            "prompt": source.prompt_asset_sha256,
        }
        if digests != expected_digests:
            raise ValueError(
                f"bundled task {source.name!r} asset identity does not match "
                "the installed toolchain"
            )
        closure_digest = hashlib.sha256()
        for relative, path in [
            ("estonia/dataset.json", assets["dataset"]),
            ("estonia/estonia.py", assets["scorer"]),
            ("estonia/estonia.yaml", assets["task_definition"]),
            ("estonia/prompt.txt", assets["prompt"]),
        ]:
            contents = path.read_bytes()
            closure_digest.update(len(relative).to_bytes(8, "little"))
            closure_digest.update(relative.encode("utf-8"))
            closure_digest.update(len(contents).to_bytes(8, "little"))
            closure_digest.update(contents)
        if closure_digest.hexdigest() != source.task_closure_sha256:
            raise ValueError(
                f"bundled task {source.name!r} closure identity does not match "
                "the installed toolchain"
            )
        config = load_lm_eval_yaml(task_yaml)
        if config.get("task") != source.task_identity or config.get("group") is not None:
            raise ValueError(
                f"bundled task {source.name!r} does not resolve to its release task identity"
            )
        output_type = resolved_output_type(
            source.task_identity, config.get("output_type", "generate_until")
        )
        config = {**config, "output_type": output_type}
        return {
            "schema_version": 1,
            "status": "resolved",
            "task_source": {
                "kind": "bundled",
                "name": source.name,
                "task_closure_sha256": source.task_closure_sha256,
            },
            "task_identity": source.task_identity,
            "output_type": output_type,
            "include_closure": [str(path) for path in assets.values()],
            "bundled_assets": {
                "task_definition_sha256": source.task_definition_sha256,
                "prompt_asset_sha256": source.prompt_asset_sha256,
                "dataset_asset_sha256": source.dataset_asset_sha256,
                "scorer_sha256": source.scorer_sha256,
            },
            "effective_task_config": config,
            "effective_dataset_selection": effective_dataset_selection(config),
            "tokenizer": {
                "locator": request.model.locator,
                "backend": "huggingface",
                "tokenized_requests": False,
            },
        }
    if isinstance(source, EvalTaskSourceInputWorkspaceYaml):
        task_yaml = Path(source.path)
        workspace_root = Path(request.workspace_root)
        closure = workspace_yaml_include_closure(
            task_yaml,
            workspace_root,
            [Path(path) for path in request.workspace_source_exclusions],
        )
        resolved_task_yaml = closure[0]
        resolved_workspace_root = workspace_root.resolve(strict=True)
        config = load_lm_eval_yaml(resolved_task_yaml)
        workspace_task_identity = config.get("task")
        if (
            not isinstance(workspace_task_identity, str)
            or not workspace_task_identity
            or config.get("group") is not None
        ):
            raise ValueError(
                f"lm-eval task YAML {task_yaml} does not resolve to one individual task; "
                "select each task as a separate Eval definition in the recipe"
            )
        output_type = (
            "dynamic"
            if "class" in config
            else resolved_output_type(
                workspace_task_identity, config.get("output_type", "generate_until")
            )
        )
        config = {**config, "output_type": output_type}
        return {
            "schema_version": 1,
            "status": "resolved",
            "task_source": {
                "kind": "workspace_yaml",
                "workspace_relative_path": str(
                    resolved_task_yaml.relative_to(resolved_workspace_root)
                ),
                "resolved_path": str(resolved_task_yaml),
            },
            "task_identity": workspace_task_identity,
            "output_type": output_type,
            "include_closure": [str(path) for path in closure],
            "effective_task_config": config,
            "effective_dataset_selection": effective_dataset_selection(config),
            "tokenizer": {
                "locator": request.model.locator,
                "backend": "huggingface",
                "tokenized_requests": False,
            },
        }
    raise TypeError(f"unsupported lm-eval task source {type(source).__name__}")


def prepare_lm_eval_task(
    request: EvalClientRequest, definition: EvalDefinitionInputLmEval
) -> PreparedLmEvalTask:
    resolution = resolve_lm_eval_task(request, definition)
    if definition.trials > 1 and resolution.get("output_type") != "generate_until":
        identity = resolution.get("task_identity")
        output_type = resolution.get("output_type")
        raise ValueError(
            f"lm-eval task {identity!r} has resolved output_type {output_type!r}; "
            "trials greater than one require a resolved generate_until task"
        )
    target = resolve_lm_eval_target(request, resolution)
    requires_probe = task_requires_prompt_logprobs(resolution)
    resolution["request_target"] = {
        "family": target.family,
        "native_model": target.model,
        "selected_named_route": target.route_name,
        "effective_public_url": target.url,
        "apply_chat_template": target.apply_chat_template,
    }
    return PreparedLmEvalTask(
        resolution=resolution,
        target=target,
        requires_prompt_logprobs=requires_probe,
    )
