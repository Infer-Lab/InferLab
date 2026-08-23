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
    if request_source is not None and request_source["kind"] == "replay":
        # Mirror the real runner's replay contract: the file is copied byte
        # for byte, entry output classes must not mix TPOT applicability, and
        # an insufficient population fails without repeating entries.
        payload = Path(request["source_path"]).read_bytes()
        replay_entries = [json.loads(line) for line in payload.split(b"\n") if line.strip()]
        replay_failure = None
        replay_flat = request["prompt"]["kind"] in ("flat", "rendered_chat")
        for line_number, entry in enumerate(replay_entries, start=1):
            output_length = entry.get("output_length")
            if not isinstance(output_length, int) or output_length < 1:
                replay_failure = (
                    f"replay population line {line_number}: entry output_length "
                    "must be a positive integer"
                )
                break
            if replay_flat and not isinstance(entry.get("text_input"), str):
                replay_failure = (
                    f"replay population line {line_number}: flat prompt entries "
                    "require a non-empty text_input string"
                )
                break
            if not replay_flat and not isinstance(entry.get("messages"), list):
                replay_failure = (
                    f"replay population line {line_number}: server_chat prompt entries "
                    "require a non-empty messages list"
                )
                break
        replay_outputs = [entry.get("output_length", 1) for entry in replay_entries]
        if replay_failure is None and len(replay_entries) < required_entries:
            replay_failure = (
                f"replay population has {len(replay_entries)} entries, "
                f"requires {required_entries}; entries are never repeated"
            )
        elif (
            replay_failure is None
            and any(output == 1 for output in replay_outputs)
            and any(output >= 2 for output in replay_outputs)
        ):
            replay_failure = "replay population must not mix TPOT applicability classes"
        if replay_failure is not None:
            result = {
                "schema_version": 1,
                "status": "failed",
                "materialization_identity": "inferlab-replay-population-v1",
                "requested_entries": required_entries,
                "candidate_entries": len(replay_entries),
                "admitted_entries": len(replay_entries),
                "ineligible_entries": 0,
                "ineligible_reasons": {},
                "population": None,
                "input_tokens": None,
                "output_tokens": None,
                "evidence_path": None,
                "error": replay_failure,
            }
            with open(args.output, "w") as handle:
                json.dump(result, handle)
            raise SystemExit(0)
        population_path = artifact_dir / "population.jsonl"
        population_path.write_bytes(payload)
        evidence_path = artifact_dir / "population-evidence.jsonl"
        evidence_path.write_text("{}\n", encoding="utf-8")
        prefix_geometry = None
        prefix_conditioning = None
        sharing = request_source.get("prefix_sharing")
        if sharing is not None:
            first_words = replay_entries[0]["text_input"].split()
            if "shared_prefix_tokens" in sharing:
                shared = sharing["shared_prefix_tokens"]
            else:
                shared = int(len(first_words) * sharing["shared_prefix_ratio"])
            canonical = " ".join(first_words[:shared])
            canonical_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
            prefix_geometry = {
                "shared_prefix_tokens": {
                    "minimum": shared,
                    "maximum": shared,
                    "mean": float(shared),
                },
                "unique_suffix_tokens": {
                    "minimum": 0,
                    "maximum": len(first_words) - shared,
                    "mean": float(len(first_words) - shared),
                },
                "maximum_shared_prefix_tokens": shared,
                "canonical_prefix_sha256": canonical_sha256,
                "full_prompt_entries": 0,
            }
            if request["cache_start"] == "primed":
                canonical_prefix_path = artifact_dir / "canonical-prefix.txt"
                canonical_prefix_path.write_text(canonical, encoding="utf-8")
                prefix_conditioning = {
                    "path": str(canonical_prefix_path),
                    "sha256": canonical_sha256,
                    "prompt_tokens": shared,
                }
        result = {
            "schema_version": 1,
            "status": "succeeded",
            "materialization_identity": "inferlab-replay-population-v1",
            "requested_entries": required_entries,
            "candidate_entries": len(replay_entries),
            "admitted_entries": len(replay_entries),
            "ineligible_entries": 0,
            "ineligible_reasons": {},
            "population": {
                "path": str(population_path),
                "evidence_path": str(evidence_path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "entries": len(replay_entries),
                "tpot_applicable": all(output >= 2 for output in replay_outputs),
                "session_templates": [],
            },
            "input_tokens": {"minimum": 4, "maximum": 4, "mean": 4.0},
            "output_tokens": {
                "minimum": min(replay_outputs),
                "maximum": max(replay_outputs),
                "mean": sum(replay_outputs) / len(replay_outputs),
            },
            "prompt_token_targeting": None,
            "prefix_geometry": prefix_geometry,
            "prefix_conditioning": prefix_conditioning,
            "evidence_path": str(evidence_path),
            "error": None,
        }
        with open(args.output, "w") as handle:
            json.dump(result, handle)
        raise SystemExit(0)
    corpus = request_source.get("corpus") if request_source is not None else None
    if request_source is not None and request_source["kind"] == "random" and corpus is not None:
        # Mirror the real runner's corpus contract: exact-length slices of the
        # corpus token stream (whitespace words stand in for tokens here), one
        # fixed corpus slice as the shared prefix, and a typed failure when
        # the corpus cannot serve the largest selected input target.
        corpus_words = Path(request["source_path"]).read_text(encoding="utf-8").split()
        input_target = request_source["input_tokens"]
        if not isinstance(input_target, int):
            input_target = input_target["max"]
        output_target = request_source["output_tokens"]
        if not isinstance(output_target, int):
            output_target = output_target["max"]
        if len(corpus_words) < input_target:
            result = {
                "schema_version": 1,
                "status": "failed",
                "materialization_identity": "inferlab-corpus-slice-v1",
                "requested_entries": required_entries,
                "candidate_entries": required_entries,
                "admitted_entries": 0,
                "ineligible_entries": 0,
                "ineligible_reasons": {},
                "population": None,
                "input_tokens": None,
                "output_tokens": None,
                "evidence_path": None,
                "error": (
                    f"corpus token stream has {len(corpus_words)} tokens, shorter than "
                    f"the largest selected input-token target {input_target}"
                ),
            }
            with open(args.output, "w") as handle:
                json.dump(result, handle)
            raise SystemExit(0)
        sharing = request_source.get("prefix_sharing")
        shared = 0
        if sharing is not None:
            if "shared_prefix_tokens" in sharing:
                shared = sharing["shared_prefix_tokens"]
            else:
                shared = int(input_target * sharing["shared_prefix_ratio"])
        seed = request["seed"]
        population_path = artifact_dir / "population.jsonl"
        evidence_path = artifact_dir / "population-evidence.jsonl"
        population_digest = hashlib.sha256()
        with population_path.open("wb") as population_file:
            evidence_lines = []
            for index in range(required_entries):
                if shared > 0:
                    suffix_length = input_target - shared
                    suffix_offset = (seed + index) % (len(corpus_words) - suffix_length + 1)
                    text = " ".join(
                        corpus_words[:shared]
                        + corpus_words[suffix_offset : suffix_offset + suffix_length]
                    )
                    slice_offset, slice_length = suffix_offset, suffix_length
                else:
                    slice_offset = (seed + index) % (len(corpus_words) - input_target + 1)
                    slice_length = input_target
                    text = " ".join(corpus_words[slice_offset : slice_offset + input_target])
                row = {
                    "session_id": f"inferlab-{index:08}",
                    "text_input": text,
                    "output_length": output_target,
                    "extra": {"ignore_eos": True, "min_tokens": output_target},
                }
                encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                population_file.write(encoded)
                population_digest.update(encoded)
                evidence_lines.append(
                    json.dumps(
                        {
                            "population_index": index,
                            "selected_prompt_tokens": input_target,
                            "selected_output_tokens": output_target,
                            "corpus_slice_offset": slice_offset,
                            "corpus_slice_length": slice_length,
                            "corpus_shared_slice_offset": 0 if shared > 0 else None,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        evidence_path.write_text("\n".join(evidence_lines) + "\n", encoding="utf-8")
        prefix_geometry = None
        prefix_conditioning = None
        if shared > 0:
            canonical = " ".join(corpus_words[:shared])
            canonical_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
            prefix_geometry = {
                "shared_prefix_tokens": {
                    "minimum": shared,
                    "maximum": shared,
                    "mean": float(shared),
                },
                "unique_suffix_tokens": {
                    "minimum": input_target - shared,
                    "maximum": input_target - shared,
                    "mean": float(input_target - shared),
                },
                "maximum_shared_prefix_tokens": shared,
                "canonical_prefix_sha256": canonical_sha256,
                "full_prompt_entries": required_entries if shared == input_target else 0,
            }
            if request["cache_start"] == "primed":
                canonical_prefix_path = artifact_dir / "canonical-prefix.txt"
                canonical_prefix_path.write_text(canonical, encoding="utf-8")
                prefix_conditioning = {
                    "path": str(canonical_prefix_path),
                    "sha256": canonical_sha256,
                    "prompt_tokens": shared,
                }
        result = {
            "schema_version": 1,
            "status": "succeeded",
            "materialization_identity": "inferlab-corpus-slice-v1",
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
                "tpot_applicable": output_target >= 2,
                "session_templates": [],
            },
            "input_tokens": {
                "minimum": input_target,
                "maximum": input_target,
                "mean": float(input_target),
            },
            "output_tokens": {
                "minimum": output_target,
                "maximum": output_target,
                "mean": float(output_target),
            },
            "prompt_token_targeting": {
                "selected_prompt_tokens": {
                    "minimum": input_target,
                    "maximum": input_target,
                    "mean": float(input_target),
                },
                "pre_template_content_tokens": {
                    "minimum": input_target,
                    "maximum": input_target,
                    "mean": float(input_target),
                },
                "projection_template": None,
                "exact_entries": required_entries,
                "fallback_entries": 0,
                "fallback_reasons": {},
            },
            "prefix_geometry": prefix_geometry,
            "prefix_conditioning": prefix_conditioning,
            "evidence_path": str(evidence_path),
            "error": None,
        }
        with open(args.output, "w") as handle:
            json.dump(result, handle)
        raise SystemExit(0)
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
