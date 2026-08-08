#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def record_capture_event(event):
    path = os.environ.get("FIXTURE_CAPTURE_EVENTS")
    if path:
        with open(path, "a") as events:
            events.write(f"{event}\n")


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--prepare-source", action="store_true")
args = parser.parse_args()
with open(args.input) as handle:
    request = json.load(handle)
if args.prepare_source:
    if os.environ.get("FIXTURE_SOURCE_PREPARATION_FAIL") == "1":
        result = {
            "schema_version": 1,
            "status": "failed",
            "effective_selection": None,
            "readiness": None,
            "cache_stores": [],
            "remote_metadata": "unavailable",
            "source_bytes": "unavailable",
            "error": "fixture source preparation failed",
        }
        with open(args.output, "w") as handle:
            json.dump(result, handle)
        raise SystemExit(1)
    source = request["source"]["source"]
    catalog = source["catalog"]
    phase = request["phase"]
    selection = {
        "kind": "agentic",
        "repository": catalog["repository"],
        "requested_revision": catalog["revision"],
        "observed_revision": catalog["revision"],
        "filename": catalog["filename"],
    }
    readiness = None
    source_bytes = "not_accessed"
    cache_outcome = "partial_reuse"
    if phase["kind"] == "acquire":
        source_bytes = "reused"
        cache_outcome = "full_hit"
        readiness = {
            "kind": "closed",
            "acquired_source": {
                "kind": "release_qualified",
                "identity": f"sha256:{catalog['sha256']}",
                "closure": [{"relative_path": catalog["filename"], "sha256": catalog["sha256"]}],
            },
            "verification": [
                {
                    "subject": catalog["filename"],
                    "expected": catalog["sha256"],
                    "observed": catalog["sha256"],
                    "matched": True,
                }
            ],
        }
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "effective_selection": selection,
        "readiness": readiness,
        "cache_stores": [
            {
                "authority": "huggingface_hub",
                "purpose": "dataset_repository_files",
                "path": "/fixture/huggingface/hub",
                "outcome": cache_outcome,
            }
        ],
        "remote_metadata": "not_accessed",
        "source_bytes": source_bytes,
        "error": None,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle)
    raise SystemExit(0)
if args.prepare:
    artifact_dir = Path(request["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    required_entries = request["required_entries"]
    request_source = request.get("request_source")
    session_source = request.get("session_source")
    population_path = artifact_dir / "population.jsonl"
    population_digest = hashlib.sha256()
    session_templates = []
    with population_path.open("wb") as population:
        for index in range(required_entries):
            identity = f"fixture-{index:08}"
            if session_source is None:
                row = {
                    "session_id": identity,
                    "messages": [{"role": "user", "content": f"fixture prompt {index}"}],
                }
            else:
                row = {
                    "type": "multi_turn",
                    "session_id": identity,
                    "turns": [
                        {"type": "single_turn", "text": "first", "role": "user"},
                        {"type": "single_turn", "text": "second", "role": "user"},
                    ],
                }
                session_templates.append({"template_identity": identity, "turn_count": 2})
            encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            population.write(encoded)
            population_digest.update(encoded)
    evidence_path = artifact_dir / "population-evidence.jsonl"
    evidence_path.write_text("{}\n", encoding="utf-8")
    source = request_source or session_source
    catalog = source.get("catalog") if source is not None else None
    materialization_identity = (
        catalog["materialization_identity"]
        if catalog is not None
        else "inferlab-synthetic-prompt-authority-v4"
    )
    tpot_applicable = True
    if request_source is not None:
        if request_source["kind"] == "random":
            output_tokens = request_source["output_tokens"]
            tpot_applicable = not isinstance(output_tokens, int) or output_tokens >= 2
        elif request_source["kind"] == "random_mixture":
            tpot_applicable = all(shape["output_tokens"] >= 2 for shape in request_source["shapes"])
    elif session_source is not None and session_source.get("output_tokens") is not None:
        tpot_applicable = session_source["output_tokens"] >= 2
    prompt_token_targeting = None
    prefix_geometry = None
    prefix_conditioning = None
    if request_source is not None and request_source["kind"] in ("random", "random_mixture"):
        projection_template = (
            None
            if request["prompt"]["kind"] == "flat"
            else {
                "source": "tokenizer_default",
                "content": "{{ messages }}",
                "sha256": hashlib.sha256(b"{{ messages }}").hexdigest(),
            }
        )
        prompt_token_targeting = {
            "selected_prompt_tokens": {"minimum": 8, "maximum": 8, "mean": 8.0},
            "pre_template_content_tokens": {"minimum": 8, "maximum": 8, "mean": 8.0},
            "projection_template": projection_template,
            "exact_entries": required_entries,
            "fallback_entries": 0,
            "fallback_reasons": {},
        }
        if request["cache_start"] == "primed":
            canonical_prefix = "canonical prefix"
            canonical_prefix_path = artifact_dir / "canonical-prefix.txt"
            canonical_prefix_path.write_text(canonical_prefix, encoding="utf-8")
            canonical_prefix_sha256 = hashlib.sha256(canonical_prefix.encode()).hexdigest()
            prefix_geometry = {
                "shared_prefix_tokens": {"minimum": 8, "maximum": 8, "mean": 8.0},
                "unique_suffix_tokens": {"minimum": 0, "maximum": 0, "mean": 0.0},
                "maximum_shared_prefix_tokens": 8,
                "canonical_prefix_sha256": canonical_prefix_sha256,
                "full_prompt_entries": required_entries,
            }
            prefix_conditioning = {
                "path": str(canonical_prefix_path),
                "sha256": canonical_prefix_sha256,
                "prompt_tokens": 8,
            }
    if os.environ.get("FIXTURE_OMIT_PROMPT_TARGETING") == "1":
        prompt_token_targeting = None
    result = {
        "schema_version": 1,
        "status": "succeeded",
        "materialization_identity": materialization_identity,
        "requested_entries": required_entries,
        "candidate_entries": required_entries,
        "admitted_entries": required_entries,
        "ineligible_entries": 0,
        "ineligible_reasons": {},
        "population": {
            "path": str(population_path),
            "evidence_path": str(evidence_path),
            "sha256": population_digest.hexdigest(),
            "entries": required_entries,
            "tpot_applicable": tpot_applicable,
            "session_templates": session_templates,
        },
        "input_tokens": {"minimum": 8, "maximum": 8, "mean": 8.0},
        "output_tokens": {"minimum": 2, "maximum": 2, "mean": 2.0},
        "prompt_token_targeting": prompt_token_targeting,
        "prefix_geometry": prefix_geometry,
        "prefix_conditioning": prefix_conditioning,
        "evidence_path": str(evidence_path),
        "error": None,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle)
    raise SystemExit(0)
if os.environ.get("FIXTURE_RECORD_CLIENT_START") == "1":
    record_capture_event("client_started")
barrier = os.environ.get("INFERLAB_AIPERF_PROFILE_BARRIER")
if barrier:
    record_capture_event("warmup_complete")
    if os.environ.get("FIXTURE_BENCH_FAIL_BEFORE_PROFILE") == "1":
        raise SystemExit(7)
    host, port = barrier.rsplit(":", 1)
    with socket.create_connection((host, int(port))) as connection:
        if os.environ.get("FIXTURE_BENCH_INVALID_BARRIER") == "1":
            connection.sendall(b"invalid-readiness\n")
            raise SystemExit(8)
        connection.sendall(b"profiling-ready\n")
        if connection.makefile("rb").readline() != b"capture-open\n":
            raise RuntimeError("fixture profiling release was not acknowledged")
    record_capture_event("profiling_started")
failed = os.environ.get("FIXTURE_BENCH_FAIL") == "1"
rate = float(request["case"]["load_shape"].get("request_rate", 1.0))
request_count = request["case"]["request_count"]
request_slo = request["definition"].get("request_slo")
request_slo_result = None
if request_slo is not None:
    duration = request_count / rate
    request_slo_result = {
        "good_requests": request_count,
        "good_request_ratio": 1.0,
        "goodput": rate,
        "profiling_duration_seconds": duration,
        "profiling_duration_source": "native-profiling-request-window",
        "request_count_reconciled": True,
        "native_aggregate_good_request_count": request_count,
        "native_aggregate_good_request_count_consistent": True,
    }
artifacts = []
if os.environ.get("FIXTURE_BENCH_INTERRUPT_WAIT") == "1":
    artifact = os.path.join(os.path.dirname(args.output), "artifacts", "partial.txt")
    os.makedirs(os.path.dirname(artifact), exist_ok=True)
    Path(artifact).write_text("partial\n", encoding="utf-8")
    artifacts = [{"name": "partial", "kind": "fixture", "path": artifact}]
    failed = True
result = {
    "schema_version": int(os.environ.get("FIXTURE_BENCH_SCHEMA_VERSION", "1")),
    "status": "failed" if failed else "succeeded",
    "completed_requests": request_count,
    "failed_requests": 1 if failed else 0,
    "normalization_schema": "aiperf-summary-v1",
    "metrics": {
        "request_throughput": rate,
        "output_throughput": rate * 1000.0,
        "total_token_throughput": rate * 9000.0,
        "mean_prompt_tokens": 8000.0,
        "min_prompt_tokens": 8000.0,
        "max_prompt_tokens": 8000.0,
        "stddev_prompt_tokens": 0.0,
        "p50_prompt_tokens": 8000.0,
        "p90_prompt_tokens": 8000.0,
        "p95_prompt_tokens": 8000.0,
        "p99_prompt_tokens": 8000.0,
        "mean_request_latency_ms": rate * 90.0,
        "min_request_latency_ms": rate * 70.0,
        "max_request_latency_ms": rate * 120.0,
        "stddev_request_latency_ms": rate * 10.0,
        "p50_request_latency_ms": rate * 90.0,
        "p90_request_latency_ms": rate * 100.0,
        "p95_request_latency_ms": rate * 105.0,
        "p99_request_latency_ms": rate * 110.0,
        "mean_ttft_ms": rate * 80.0,
        "min_ttft_ms": rate * 60.0,
        "max_ttft_ms": rate * 110.0,
        "stddev_ttft_ms": rate * 10.0,
        "p50_ttft_ms": rate * 80.0,
        "p90_ttft_ms": rate * 90.0,
        "p95_ttft_ms": rate * 95.0,
        "p99_ttft_ms": rate * 100.0,
        "mean_tpot_ms": rate * 10.0,
        "min_tpot_ms": rate * 8.0,
        "max_tpot_ms": rate * 13.0,
        "stddev_tpot_ms": rate,
        "p50_tpot_ms": rate * 10.0,
        "p90_tpot_ms": rate * 11.0,
        "p95_tpot_ms": rate * 11.5,
        "p99_tpot_ms": rate * 12.0,
        **({"good_request_ratio": 1.0, "goodput": rate} if request_slo else {}),
    },
    "request_slo": request_slo_result,
    "native_command": ["fixture-bench"],
    "native_exit_code": 143 if os.environ.get("FIXTURE_BENCH_INTERRUPT_WAIT") == "1" else 0,
    "raw_artifacts": artifacts,
    "error": (
        "fixture bench interruption"
        if os.environ.get("FIXTURE_BENCH_INTERRUPT_WAIT") == "1"
        else ("fixture bench failure" if failed else None)
    ),
}
if request["definition"]["cache_start"] == "primed":
    result["prompt_cache_observations"] = [
        {
            "request_id": index,
            "prompt_tokens": 8000,
            "cache_read_tokens": 8000,
            "uncached_prompt_tokens": 0,
            "cache_read_ratio": 1.0,
        }
        for index in range(request_count)
    ]
    result["metrics"].update(
        {
            "mean_prompt_cache_read_tokens": 8000.0,
            "min_prompt_cache_read_tokens": 8000.0,
            "max_prompt_cache_read_tokens": 8000.0,
            "stddev_prompt_cache_read_tokens": 0.0,
            "p50_prompt_cache_read_tokens": 8000.0,
            "p90_prompt_cache_read_tokens": 8000.0,
            "p95_prompt_cache_read_tokens": 8000.0,
            "p99_prompt_cache_read_tokens": 8000.0,
            "mean_uncached_prompt_tokens": 0.0,
            "min_uncached_prompt_tokens": 0.0,
            "max_uncached_prompt_tokens": 0.0,
            "stddev_uncached_prompt_tokens": 0.0,
            "p50_uncached_prompt_tokens": 0.0,
            "p90_uncached_prompt_tokens": 0.0,
            "p95_uncached_prompt_tokens": 0.0,
            "p99_uncached_prompt_tokens": 0.0,
            "prompt_cache_read_ratio": 1.0,
        }
    )
if os.environ.get("FIXTURE_BENCH_INTERRUPT_WAIT") == "1":
    with open(args.output, "w") as handle:
        json.dump(result, handle)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ]
    )
    Path(os.environ["FIXTURE_BENCH_MARKER"]).write_text(str(child.pid), encoding="utf-8")
    time.sleep(60)
with open(os.environ["FIXTURE_BENCH_MARKER"], "w") as marker:
    marker.write("ran")
if os.environ.get("FIXTURE_BENCH_WAIT") == "1":
    time.sleep(1)
if os.environ.get("FIXTURE_BENCH_ENVELOPE_EVOLVED"):
    result = {"schema_version": 2, "frontier_field": {"nested": True}}
with open(args.output, "w") as handle:
    json.dump(result, handle)
