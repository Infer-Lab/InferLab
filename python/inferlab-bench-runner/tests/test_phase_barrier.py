import asyncio
import socket
import sys
import threading
from pathlib import Path
from typing import cast

import pytest
from inferlab_bench_runner import aiperf_phase_barrier
from inferlab_bench_runner.aiperf import (
    aiperf_config,
    prepare_aiperf_execution,
)
from inferlab_bench_runner.aiperf_phase_barrier import (
    AiperfAgenticProfileBarrierStrategy,
    AiperfProfileBarrierStrategy,
    WarmupExpectation,
    await_capture_open,
    warmup_completion_error,
)
from inferlab_measurement_sdk import (
    CaseDeadline,
)

from .support import (
    request,
)


def test_config_maps_native_warmup_before_the_concurrency_profile(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 2},
            warmup_request_count=2,
        )
    )
    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])

    assert dataset["entries"] == 6
    assert dataset["sampling"] == "sequential"
    assert benchmark["warmup"] == {
        "type": "concurrency",
        "concurrency": 2,
        "requests": 2,
    }
    assert benchmark["profiling"] == {
        "type": "concurrency",
        "concurrency": 2,
        "requests": 4,
    }


def test_profile_command_uses_the_release_owned_aiperf_entrypoint(tmp_path: Path) -> None:
    prepared = prepare_aiperf_execution(
        request(tmp_path, {"kind": "concurrency_limited", "concurrency": 2}),
        CaseDeadline(5.0),
    )

    assert prepared.command[:3] == [
        sys.executable,
        "-m",
        "inferlab_bench_runner.aiperf_entrypoint",
    ]
    assert prepared.command[3:] == ["profile", "--config", str(prepared.config_path)]


def test_warmup_gate_requires_the_native_request_phase_to_drain_without_errors() -> None:
    complete = {
        "final_requests_sent": 2,
        "final_requests_completed": 2,
        "final_requests_cancelled": 0,
        "final_request_errors": 0,
        "final_sent_sessions": 2,
        "final_completed_sessions": 2,
        "final_cancelled_sessions": 0,
    }

    assert warmup_completion_error(WarmupExpectation(requests=2, sessions=None), complete) is None
    assert "errors" in (
        warmup_completion_error(
            WarmupExpectation(requests=2, sessions=None),
            {**complete, "final_request_errors": 1},
        )
        or ""
    )


def test_warmup_gate_requires_complete_native_sessions() -> None:
    incomplete = {
        "final_requests_sent": 4,
        "final_requests_completed": 4,
        "final_requests_cancelled": 0,
        "final_request_errors": 0,
        "final_sent_sessions": 2,
        "final_completed_sessions": 1,
        "final_cancelled_sessions": 0,
    }

    error = warmup_completion_error(WarmupExpectation(requests=None, sessions=2), incomplete)

    assert error is not None
    assert "sessions" in error


def test_profile_barrier_waits_for_capture_open_acknowledgement() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    observed: list[bytes] = []

    def acknowledge() -> None:
        connection, _ = listener.accept()
        with connection:
            observed.append(connection.makefile("rb").readline())
            connection.sendall(b"capture-open\n")
        listener.close()

    server = threading.Thread(target=acknowledge)
    server.start()
    await_capture_open(f"{host}:{port}")
    server.join()

    assert observed == [b"profiling-ready\n"]


def test_profile_barrier_forwards_the_pinned_timing_strategy_error_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[object, str | None]] = []

    class Delegate:
        async def handle_credit_return(self, credit: object, *, error: str | None = None) -> None:
            observed.append((credit, error))

    delegate = Delegate()
    monkeypatch.delenv("INFERLAB_AIPERF_PROFILE_BARRIER", raising=False)
    monkeypatch.setattr(
        aiperf_phase_barrier,
        "_native_strategy_factory",
        lambda: lambda **_kwargs: delegate,
    )
    config = type(
        "Config",
        (),
        {
            "phase": "profiling",
            "total_expected_requests": 1,
            "expected_num_sessions": None,
        },
    )()
    strategy = AiperfProfileBarrierStrategy(config=config)
    credit = object()

    asyncio.run(strategy.handle_credit_return(credit, error="context overflow"))

    assert observed == [(credit, "context overflow")]


def test_agentic_profile_barrier_preserves_native_warmup_and_branch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[object, str | None]] = []

    class Delegate:
        wants_returns_after_sending_complete = True
        allows_pending_branch_handoff_after_sending_complete = True

        async def setup_phase(self) -> None:
            return None

        async def execute_phase(self) -> None:
            return None

        async def handle_credit_return(self, credit: object, *, error: str | None = None) -> None:
            observed.append((credit, error))

        def record_warmup_failure(self, trace_id: str) -> None:
            observed.append((trace_id, "warmup"))

        def report_warmup_failures(self) -> None:
            observed.append(("report", None))

    delegate = Delegate()
    monkeypatch.delenv("INFERLAB_AIPERF_PROFILE_BARRIER", raising=False)
    monkeypatch.setattr(
        aiperf_phase_barrier,
        "_native_agentic_strategy_factory",
        lambda: lambda **_kwargs: delegate,
    )
    config = type(
        "Config",
        (),
        {
            "phase": "profiling",
            "total_expected_requests": None,
            "expected_num_sessions": None,
        },
    )()
    strategy = AiperfAgenticProfileBarrierStrategy(config=config)
    credit = object()

    asyncio.run(strategy.handle_credit_return(credit, error="context overflow"))
    strategy.record_warmup_failure("trace-1")
    strategy.report_warmup_failures()

    assert strategy.wants_returns_after_sending_complete
    assert strategy.allows_pending_branch_handoff_after_sending_complete
    assert observed == [
        (credit, "context overflow"),
        ("trace-1", "warmup"),
        ("report", None),
    ]
