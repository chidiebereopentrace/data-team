"""Per-stage latency budgets. An overrun keeps the evidence already in hand."""
from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from math import isfinite
from threading import BoundedSemaphore
from typing import Callable, Iterator, TypeVar


T = TypeVar("T")
_deadline: ContextVar[float | None] = ContextVar("rag_deadline", default=None)


@contextmanager
def request_budget(seconds: float | None = None, *, reserve_s: float = 0.0) -> Iterator[None]:
    now = time.perf_counter()
    parent = _deadline.get()
    cutoff = now + seconds if seconds is not None else None
    if parent is not None:
        cutoff = min(parent, cutoff) if cutoff is not None else parent
    token = _deadline.set(cutoff - reserve_s if cutoff is not None else None)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_seconds(limit: float | None = None) -> float | None:
    cutoff = _deadline.get()
    if cutoff is None:
        return limit
    remaining = max(0.0, cutoff - time.perf_counter())
    return min(limit, remaining) if limit is not None else remaining


class RetrievalWorkers:
    """Bound in-flight service calls, including calls that outlive a request."""

    def __init__(self, name: str, workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self._slots = BoundedSemaphore(workers)

    def try_submit(self, call: Callable[[], T]) -> Future[T] | None:
        if not self._slots.acquire(blocking=False):
            return None
        try:
            future = self._executor.submit(call)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future


def stage_budget_seconds(env_var: str, *, default_s: float) -> float:
    raw = os.environ.get(env_var, "").strip()
    try:
        budget = float(raw) if raw else float(default_s)
    except ValueError:
        budget = float(default_s)
    budget = max(0.0, budget) if isfinite(budget) else default_s
    capped = remaining_seconds(budget)
    return budget if capped is None else capped
