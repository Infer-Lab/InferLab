#!/usr/bin/env python3
"""Fake `ssh` for every test harness.

Mirrors OpenSSH as InferLab invokes it: `ssh <options> -- <target> <remote
command words>`. Like the real client, it joins the remote words with spaces
and hands the line to the remote account's shell; this fake runs it with bash
on the local host. A remote command other than a log read first prints a
login banner, as interactive login initialization on real hosts can, so
callers must find their results by marker rather than by position.

Switches (all optional):
  FAKE_SSH_EVENTS          append "<target> <operation>" per call
  FAKE_SSH_ENV_LOG         append whether the dynamic-linker overrides reached ssh
  FAKE_SSH_HANG_PREFLIGHT  on a preflight call, write this process id there and hang
  FIXTURE_SCENARIO         JSON faults: ssh_hang_rm, ssh_fail_cleanup, ssh_swallow_handle
The remote command sees FAKE_SSH_TARGET, which other fakes use to act per machine.
"""

import json
import os
import subprocess
import sys
import time


def scenario():
    path = os.environ.get("FIXTURE_SCENARIO")
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle)


def operation_of(command):
    for marker, operation in (
        ("INFERLAB_LAUNCH_FILE", "materialize"),
        ("INFERLAB_PREFLIGHT", "preflight"),
        ("INFERLAB_HANDLE", "launch"),
        ("INFERLAB_CLEANUP", "cleanup"),
        ("INFERLAB_HARDWARE", "hardware"),
        ("INFERLAB_DEVICE_RESIDUAL", "residual"),
    ):
        if marker in command:
            return operation
    return "status"


def append(path, line):
    with open(path, "a") as handle:
        handle.write(line + "\n")


fault = scenario()
args = sys.argv[1:]
if "--" not in args:
    sys.stderr.write(f"unexpected ssh fixture arguments: {args!r}\n")
    sys.exit(255)
remote = args[args.index("--") + 1 :]
target, words = remote[0], remote[1:]
env_log = os.environ.get("FAKE_SSH_ENV_LOG")
if env_log:
    append(
        env_log,
        "{} ld_library_path={} ld_preload={}".format(
            target,
            "set" if os.environ.get("LD_LIBRARY_PATH") else "",
            "set" if os.environ.get("LD_PRELOAD") else "",
        ),
    )
command = " ".join(words)
env = dict(os.environ, FAKE_SSH_TARGET=target)
events = os.environ.get("FAKE_SSH_EVENTS")
if words[:1] == ["cat"]:
    if events:
        append(events, f"{target} logs")
    sys.exit(subprocess.run(["bash", "-c", command], env=env).returncode)
operation = operation_of(command)
if events:
    append(events, f"{target} {operation}")
hang = os.environ.get("FAKE_SSH_HANG_PREFLIGHT")
if operation == "preflight" and hang:
    with open(hang, "w") as handle:
        handle.write(str(os.getpid()))
    time.sleep(3600)
if fault.get("ssh_hang_rm") and (
    "docker rm -f" in command or "docker container inspect" in command
):
    # A live connection to a wedged remote daemon: removal requests and
    # container observations never return.
    time.sleep(3600)
if fault.get("ssh_fail_cleanup") and "read pid expected" in command:
    # The incomplete-launch cleanup script (its handle read is distinctive)
    # fails while the connection is fine.
    sys.stderr.write("fixture: forced launcher cleanup failure\n")
    sys.exit(6)
sys.stdout.write("fixture login banner\n")
sys.stdout.flush()
# The login shell runs the joined line: `bash -lic '<script>'`.
line = command.replace("bash -lic", "bash -c", 1)
if fault.get("ssh_swallow_handle"):
    # Deliver everything except the launch handle, as a broken transport
    # or interposed shell might.
    result = subprocess.run(["bash", "-c", line], env=env, capture_output=True)
    kept = [row for row in result.stdout.splitlines() if b"INFERLAB_HANDLE" not in row]
    sys.stdout.buffer.write(b"\n".join(kept) + b"\n")
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
sys.exit(subprocess.run(["bash", "-c", line], env=env).returncode)
