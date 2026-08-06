"""Coordinate pinned AIPerf warmup completion with Rust-owned capture."""

from __future__ import annotations

import asyncio
import importlib
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

PROFILE_BARRIER_ENV = "INFERLAB_AIPERF_PROFILE_BARRIER"
PROFILE_READY = b"profiling-ready\n"
CAPTURE_OPEN = b"capture-open\n"


@dataclass(frozen=True)
class WarmupExpectation:
    requests: int | None
    sessions: int | None


class PhaseConfig(Protocol):
    phase: object
    total_expected_requests: int | None
    expected_num_sessions: int | None


class CreditCounter(Protocol):
    final_requests_sent: int | None
    final_requests_completed: int | None
    final_requests_cancelled: int | None
    final_request_errors: int | None
    final_sent_sessions: int | None
    final_completed_sessions: int | None
    final_cancelled_sessions: int | None


class PhaseProgress(Protocol):
    counter: CreditCounter


class NativeTimingStrategy(Protocol):
    async def setup_phase(self) -> None: ...

    async def execute_phase(self) -> None: ...

    async def handle_credit_return(self, credit: object, *, error: str | None = None) -> None: ...

    def set_request_rate(self, new_rate: float) -> None: ...

    @property
    def wants_returns_after_sending_complete(self) -> bool: ...

    @property
    def allows_pending_branch_handoff_after_sending_complete(self) -> bool: ...

    def record_warmup_failure(self, trace_id: str) -> None: ...

    def report_warmup_failures(self) -> None: ...


class NativeTimingStrategyFactory(Protocol):
    def __call__(self, **kwargs: object) -> NativeTimingStrategy: ...


@dataclass(frozen=True)
class WarmupCheckpoint:
    expectation: WarmupExpectation
    progress: PhaseProgress


_warmup_checkpoint: WarmupCheckpoint | None = None


def warmup_completion_error(
    expectation: WarmupExpectation, counts: Mapping[str, int | None]
) -> str | None:
    request_values = (
        counts.get("final_requests_sent"),
        counts.get("final_requests_completed"),
        counts.get("final_requests_cancelled"),
        counts.get("final_request_errors"),
    )
    if any(value is None for value in request_values):
        return "AIPerf warmup did not publish terminal request counts"
    sent, completed, cancelled, errors = request_values
    if cancelled != 0:
        return f"AIPerf warmup cancelled {cancelled} requests"
    if errors != 0:
        return f"AIPerf warmup reported {errors} request errors"
    if sent != completed:
        return f"AIPerf warmup sent {sent} requests but completed {completed}"
    if expectation.requests is not None and sent != expectation.requests:
        return f"AIPerf warmup completed {sent} requests, expected {expectation.requests}"
    if expectation.sessions is None:
        return None

    session_values = (
        counts.get("final_sent_sessions"),
        counts.get("final_completed_sessions"),
        counts.get("final_cancelled_sessions"),
    )
    if any(value is None for value in session_values):
        return "AIPerf warmup did not publish terminal session counts"
    sent_sessions, completed_sessions, cancelled_sessions = session_values
    if cancelled_sessions != 0:
        return f"AIPerf warmup cancelled {cancelled_sessions} sessions"
    if sent_sessions != expectation.sessions or completed_sessions != expectation.sessions:
        return (
            "AIPerf warmup sessions did not drain: "
            f"sent={sent_sessions}, completed={completed_sessions}, "
            f"expected={expectation.sessions}"
        )
    return None


def await_capture_open(address: str) -> None:
    host, separator, raw_port = address.rpartition(":")
    if not separator or not host:
        raise RuntimeError(f"invalid InferLab profile barrier address {address!r}")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise RuntimeError(f"invalid InferLab profile barrier port {raw_port!r}") from error
    with socket.create_connection((host, port)) as connection:
        connection.sendall(PROFILE_READY)
        acknowledgement = connection.makefile("rb").readline()
    if acknowledgement != CAPTURE_OPEN:
        raise RuntimeError(
            "InferLab profile barrier closed without acknowledging the capture window"
        )


def _native_strategy_factory() -> NativeTimingStrategyFactory:
    module = importlib.import_module("aiperf.timing.strategies.request_rate")
    return cast(NativeTimingStrategyFactory, module.RequestRateStrategy)


def _native_agentic_strategy_factory() -> NativeTimingStrategyFactory:
    module = importlib.import_module("aiperf.timing.strategies.agentic_replay")
    return cast(NativeTimingStrategyFactory, module.AgenticReplayStrategy)


def _counter_values(counter: CreditCounter) -> dict[str, int | None]:
    return {
        "final_requests_sent": counter.final_requests_sent,
        "final_requests_completed": counter.final_requests_completed,
        "final_requests_cancelled": counter.final_requests_cancelled,
        "final_request_errors": counter.final_request_errors,
        "final_sent_sessions": counter.final_sent_sessions,
        "final_completed_sessions": counter.final_completed_sessions,
        "final_cancelled_sessions": counter.final_cancelled_sessions,
    }


class AiperfProfileBarrierStrategy:
    """Delegate native request-rate timing and gate its profiling setup."""

    def __init__(self, **kwargs: object) -> None:
        global _warmup_checkpoint

        self._delegate = _native_strategy_factory()(**kwargs)
        config = cast(PhaseConfig, kwargs["config"])
        phase = str(config.phase)
        self._release_address: str | None = None
        if phase == "warmup":
            _warmup_checkpoint = WarmupCheckpoint(
                expectation=WarmupExpectation(
                    requests=config.total_expected_requests,
                    sessions=config.expected_num_sessions,
                ),
                progress=cast(PhaseProgress, kwargs["progress"]),
            )
        elif phase == "profiling":
            self._release_address = os.environ.get(PROFILE_BARRIER_ENV)

    async def setup_phase(self) -> None:
        global _warmup_checkpoint

        await self._delegate.setup_phase()
        if self._release_address is None:
            return
        checkpoint = _warmup_checkpoint
        _warmup_checkpoint = None
        if checkpoint is None:
            raise RuntimeError("AIPerf profiling reached the barrier without a warmup checkpoint")
        error = warmup_completion_error(
            checkpoint.expectation, _counter_values(checkpoint.progress.counter)
        )
        if error is not None:
            raise RuntimeError(error)
        await asyncio.to_thread(await_capture_open, self._release_address)

    async def execute_phase(self) -> None:
        await self._delegate.execute_phase()

    async def handle_credit_return(self, credit: object, *, error: str | None = None) -> None:
        await self._delegate.handle_credit_return(credit, error=error)

    def set_request_rate(self, new_rate: float) -> None:
        self._delegate.set_request_rate(new_rate)


class AiperfAgenticProfileBarrierStrategy:
    """Delegate native AgentX scheduling and gate its profiling setup."""

    def __init__(self, **kwargs: object) -> None:
        global _warmup_checkpoint

        self._delegate = _native_agentic_strategy_factory()(**kwargs)
        config = cast(PhaseConfig, kwargs["config"])
        phase = str(config.phase)
        self._release_address: str | None = None
        if phase == "warmup":
            _warmup_checkpoint = WarmupCheckpoint(
                expectation=WarmupExpectation(
                    requests=config.total_expected_requests,
                    sessions=config.expected_num_sessions,
                ),
                progress=cast(PhaseProgress, kwargs["progress"]),
            )
        elif phase == "profiling":
            self._release_address = os.environ.get(PROFILE_BARRIER_ENV)

    @property
    def wants_returns_after_sending_complete(self) -> bool:
        return self._delegate.wants_returns_after_sending_complete

    @property
    def allows_pending_branch_handoff_after_sending_complete(self) -> bool:
        return self._delegate.allows_pending_branch_handoff_after_sending_complete

    async def setup_phase(self) -> None:
        global _warmup_checkpoint

        await self._delegate.setup_phase()
        if self._release_address is None:
            return
        checkpoint = _warmup_checkpoint
        _warmup_checkpoint = None
        if checkpoint is None:
            raise RuntimeError("AIPerf AgentX profiling reached the barrier without warmup")
        error = warmup_completion_error(
            checkpoint.expectation, _counter_values(checkpoint.progress.counter)
        )
        if error is not None:
            raise RuntimeError(error)
        await asyncio.to_thread(await_capture_open, self._release_address)

    async def execute_phase(self) -> None:
        await self._delegate.execute_phase()

    async def handle_credit_return(self, credit: object, *, error: str | None = None) -> None:
        await self._delegate.handle_credit_return(credit, error=error)

    def record_warmup_failure(self, trace_id: str) -> None:
        self._delegate.record_warmup_failure(trace_id)

    def report_warmup_failures(self) -> None:
        self._delegate.report_warmup_failures()
