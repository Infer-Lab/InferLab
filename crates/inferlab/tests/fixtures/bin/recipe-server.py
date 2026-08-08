#!/usr/bin/env python3
import http.server
import json
import os
import sys
import threading
import time


def record_capture_event(event):
    path = os.environ.get("FIXTURE_CAPTURE_EVENTS")
    if path:
        with open(path, "a") as events:
            events.write(f"{event}\n")


def register_with_reaper():
    # Cross-process registry entry for the test-side reaper; the file layout
    # is the protocol (see tests/support/mod.rs). Only a detached group
    # leader registers: anything else dies with its parent.
    registry = os.environ.get("FIXTURE_REAPER_REGISTRY")
    if not registry or os.getpgid(0) != os.getpid():
        return
    pgid = os.getpid()
    with open(f"/proc/{pgid}/stat") as stat:
        starttime = stat.read().rsplit(")", 1)[1].split()[19]
    entry = "\n".join(
        [
            os.environ["FIXTURE_REAPER_OWNER"],
            starttime,
            os.environ["FIXTURE_REAPER_WORKSPACE"],
        ]
    )
    path = os.path.join(registry, f"{pgid}.grp")
    temp = f"{path}.tmp.{pgid}"
    with open(temp, "w") as handle:
        handle.write(entry)
    os.rename(temp, path)


register_with_reaper()
time.sleep(float(os.environ.get("FIXTURE_READY_DELAY_SECONDS", "0")))
if os.environ.get("FIXTURE_EXIT_BEFORE_READY"):
    sys.exit(7)
host, port, *extra = sys.argv[1:]
port = int(port)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirected":
            body = json.dumps({"choices": [{"text": "redirected"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/query":
            body = json.dumps({"0": {"engine_id": f"fixture-{port}"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200 if self.path in ["/health", "/v1/models"] else 404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/v1/completions":
            if os.environ.get("FIXTURE_SMOKE_REDIRECT") == "1":
                self.send_response(302)
                self.send_header("Location", "/redirected")
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if request.get("prompt") == "canonical prefix":
                conditioning_request = os.environ.get("FIXTURE_CONDITIONING_REQUEST")
                if conditioning_request:
                    with open(conditioning_request, "w") as handle:
                        json.dump(request, handle)
                if os.environ.get("FIXTURE_RECORD_CACHE_PREPARATION") == "1":
                    record_capture_event("cache_conditioning")
            marker = os.environ.get("FIXTURE_SMOKE_MARKER")
            if marker:
                with open(f"{marker}.tmp", "w") as handle:
                    handle.write("started")
                os.replace(f"{marker}.tmp", marker)
            time.sleep(float(os.environ.get("FIXTURE_SMOKE_DELAY_SECONDS", "0")))
            response = {
                "id": "fixture-completion",
                "object": "text_completion",
                "model": request["model"],
                "choices": [{"index": 0, "text": " San Francisco", "finish_reason": "stop"}],
            }
            if request.get("prompt") == "canonical prefix":
                response["usage"] = {
                    "prompt_tokens": 8,
                    "completion_tokens": 1,
                    "total_tokens": 9,
                    "prompt_tokens_details": {"cached_tokens": 0},
                }
            if "kv_transfer_params" in request:
                response["kv_transfer_params"] = request["kv_transfer_params"]
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/start_profile":
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length)) if length else None
            if self.headers.get("Content-Type") != "application/json" or request != {
                "activities": ["CUDA_PROFILER"]
            }:
                self.send_response(400)
                self.end_headers()
                return
            time.sleep(float(os.environ.get("FIXTURE_START_PROFILE_DELAY_SECONDS", "0")))
            record_capture_event("capture_open")
        if self.path == "/stop_profile":
            if (
                int(self.headers.get("Content-Length", "0")) != 0
                or self.headers.get("Content-Type") is not None
            ):
                self.send_response(400)
                self.end_headers()
                return
            record_capture_event("capture_close")
            if not os.environ.get("FIXTURE_STOP_PROFILE_SKIP_REPORT"):
                state_path = os.environ["FIXTURE_NSYS_STATE"]
                with open(state_path) as state_file:
                    output, count, index, session = state_file.read().split("\t")
                index = int(index) + 1
                with open(f"{output}.{index}.nsys-rep", "w") as report:
                    report.write("fixture\n")
                with open(state_path, "w") as state_file:
                    state_file.write(f"{output}\t{count}\t{index}\t{session}")
            if os.environ.get("FIXTURE_STOP_PROFILE_FAIL"):
                self.send_response(500)
                self.end_headers()
                return
        status = (
            200 if self.path in ["/reset_prefix_cache", "/start_profile", "/stop_profile"] else 404
        )
        if self.path == "/reset_prefix_cache":
            status = int(os.environ.get("FIXTURE_RESET_STATUS", "200"))
            if os.environ.get("FIXTURE_RECORD_CACHE_PREPARATION") == "1":
                record_capture_event("cache_reset")
        self.send_response(status)
        self.end_headers()

    def log_message(self, format, *args):
        pass


if extra:
    threading.Thread(
        target=http.server.HTTPServer((host, int(extra[0])), Handler).serve_forever,
        daemon=True,
    ).start()
http.server.HTTPServer((host, port), Handler).serve_forever()
