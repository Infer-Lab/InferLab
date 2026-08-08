#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--prepare-source", action="store_true")
args = parser.parse_args()
with open(args.input) as request_file:
    request = json.load(request_file)
if args.prepare_source:
    if os.environ.get("FIXTURE_SOURCE_PREPARATION_FAIL") == "1":
        with open(args.output, "w") as output_file:
            json.dump(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "effective_selection": None,
                    "readiness": None,
                    "cache_stores": [],
                    "remote_metadata": "unavailable",
                    "source_bytes": "unavailable",
                    "error": "fixture source preparation failed",
                },
                output_file,
            )
        raise SystemExit(0)
    definition = request["source"]["definition"]
    task = definition["task"]
    if os.environ.get("FIXTURE_LOCAL_SNAPSHOT") == "1":
        phase = request["phase"]["kind"]
        task_path = Path(task["path"])
        digest = hashlib.sha256(task_path.read_bytes()).hexdigest()
        readiness = None
        next_phase = "snapshot_local"
        source_bytes = "reused"
        if phase == "snapshot_local":
            snapshot_root = Path(request["artifact_dir"]) / "prepared-source"
            snapshot_root.mkdir(parents=True, exist_ok=True)
            prepared_task = snapshot_root / "_inferlab_prepared_task.yaml"
            prepared_task.write_bytes(task_path.read_bytes())
            readiness = {
                "kind": "closed",
                "acquired_source": {
                    "kind": "local_file_closure",
                    "source_root": str(task_path.parent),
                    "files": [
                        {"relative_path": task_path.name, "sha256": digest},
                    ],
                },
                "verification": [],
                "eval_binding": {
                    "workspace_root": str(snapshot_root),
                    "task_path": str(prepared_task),
                },
            }
            next_phase = None
        result = {
            "schema_version": 1,
            "status": "succeeded",
            "effective_selection": {
                "kind": "eval",
                "task_identity": "custom_eval",
                "dataset_path": "json",
                "dataset_name": None,
                "evaluation_split": "test",
                "fewshot_split": None,
                "data_files": {"train": ["evals/data.jsonl"]},
            },
            "readiness": readiness,
            "next_phase": next_phase,
            "cache_stores": [],
            "remote_metadata": "not_accessed",
            "source_bytes": source_bytes,
            "error": None,
        }
        with open(args.output, "w") as output_file:
            json.dump(result, output_file)
        raise SystemExit(0)
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "effective_selection": {
            "kind": "eval",
            "task_identity": task.get("name", task.get("task_identity", "custom")),
            "dataset_path": "fixture/dataset",
            "dataset_name": None,
            "evaluation_split": "test",
            "fewshot_split": None,
            "data_files": None,
        },
        "readiness": {
            "kind": "opaque",
            "reason": "fixture lm-eval source has no closed dataset interface",
            "unresolved_path": task.get("path"),
            "deferred_source_access": True,
        },
        "cache_stores": [],
        "remote_metadata": "not_accessed",
        "source_bytes": "not_accessed",
        "error": None,
    }
    with open(args.output, "w") as output_file:
        json.dump(result, output_file)
    raise SystemExit(0)
if os.environ.get("FIXTURE_EVAL_NO_RESULT") == "1":
    raise SystemExit(7)
if os.environ.get("FIXTURE_LOCAL_SNAPSHOT") == "1":
    prepared_source = request.get("prepared_source")
    if prepared_source is None or not Path(prepared_source["task_path"]).is_file():
        raise SystemExit("prepared Eval source binding was not received")
if os.environ.get("FIXTURE_EVAL_WAIT") == "1":
    if os.environ.get("FIXTURE_EVAL_NATIVE_CHECKPOINT") == "1":
        artifact_dir = os.path.join(os.path.dirname(args.output), "artifacts")
        raw_output_dir = os.path.join(artifact_dir, "lm-eval-output")
        process_path = os.path.join(artifact_dir, "lm-eval-process.json")
        os.makedirs(raw_output_dir, exist_ok=True)
        with open(process_path, "w") as process_file:
            json.dump({"native_command": ["fixture-eval"], "outcome": "running"}, process_file)
        checkpoint = {
            "schema_version": 1,
            "status": "failed",
            "metrics": {},
            "normalized_metrics": {},
            "gate": None,
            "native_command": ["fixture-eval"],
            "native_exit_code": None,
            "native_timed_out": False,
            "raw_artifacts": [
                {"name": "lm_eval_output", "kind": "directory", "path": raw_output_dir},
                {"name": "lm_eval_process", "kind": "lm-eval-process", "path": process_path},
            ],
            "error": "fixture native attempt did not finalize",
        }
        with open(args.output, "w") as output_file:
            json.dump(checkpoint, output_file)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "marker=sys.argv[1]; open(marker + '.tmp', 'w').write(str(os.getpid())); "
            "os.replace(marker + '.tmp', marker); time.sleep(60)",
            os.environ["FIXTURE_EVAL_MARKER"],
        ]
    )
    time.sleep(60)
else:
    with open(os.environ["FIXTURE_EVAL_MARKER"], "w") as marker:
        marker.write("ran")
kind = request["definition"]["kind"]
metrics = {"completed": 1.0}
normalized_metrics = {}
gate = None
if kind == "lm_eval":
    definition = request["definition"]
    score = float(os.environ.get("FIXTURE_GATE_SCORE", "0.95"))
    source = definition["task"].get("name", "custom")
    metric = definition["metric"]
    metric_filter = definition.get("metric_filter") or "none"
    native_key = f"{metric},{metric_filter}"
    normalized = {
        "source_identity": source,
        "metric": metric,
        "filter": metric_filter,
        "native_metric_key": native_key,
        "value": score,
        "higher_is_better": True,
        "prompt_authority": definition["prompt"],
    }
    metrics = {f"{source}:{native_key}": score}
    normalized_metrics = {f"{source}:{native_key}": normalized}
    gate = {
        "metric": normalized,
        "threshold": definition["threshold"],
        "comparison": "at_least",
        "conclusion": "passed" if score >= definition["threshold"] else "failed",
    }
schema_version = int(os.environ.get("FIXTURE_EVAL_SCHEMA_VERSION", "1"))
result = {
    "schema_version": schema_version,
    "status": "succeeded",
    "metrics": metrics,
    "normalized_metrics": normalized_metrics,
    "gate": gate,
    "native_command": ["fixture-eval"],
    "native_exit_code": 0 if kind == "lm_eval" else None,
    "native_timed_out": False,
    "raw_artifacts": [],
    "error": None,
}
if os.environ.get("FIXTURE_EVAL_ENVELOPE_EVOLVED"):
    # A future envelope: new version, unknown fields, none of the v1 fields.
    result = {"schema_version": 2, "frontier_field": {"nested": True}}
with open(args.output, "w") as output_file:
    json.dump(result, output_file)
if os.environ.get("FIXTURE_EVAL_EXIT_CODE"):
    sys.exit(int(os.environ["FIXTURE_EVAL_EXIT_CODE"]))
