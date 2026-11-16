"""Per-stage latency budgets. An overrun keeps the evidence already in hand."""
from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from math import isfinite
from threading import BoundedSemaphore
from typing import Callable, TypeVar


T = TypeVar("T")


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
    return max(0.0, budget) if isfinite(budget) else default_s
