"""Periodic copy status, including while the worker waits on network or database I/O."""
from __future__ import annotations

import threading
import time


class CopyProgress:
    def __init__(self, table, *, interval=15, emit=None):
        self.table = table
        self.interval = interval
        self.emit = emit or (lambda message: print(message, flush=True))
        self.started = time.monotonic()
        self.phase = 'waiting for BigQuery query/results'
        self.copied = 0
        self.total = None
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def update(self, phase, *, copied=None, total=None):
        with self.lock:
            self.phase = phase
            if copied is not None:
                self.copied = copied
            if total is not None:
                self.total = total

    def report(self):
        with self.lock:
            phase, copied, total = self.phase, self.copied, self.total
        elapsed = max(0, time.monotonic() - self.started)
        count = f'{copied:,}'
        if total is not None:
            count += f'/{total:,}'
            if total:
                count += f' ({100 * copied / total:.1f}%)'
        rate = copied / elapsed if elapsed else 0
        self.emit(f'Progress {self.table}: {count} rows staged; {phase}; '
                  f'elapsed {elapsed:.0f}s; average {rate:,.0f} rows/s')

    def _watch(self):
        while not self.stop.wait(self.interval):
            self.report()

    def __enter__(self):
        self.report()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.thread.join()
        if exc_type:
            self.update('interrupted; current table copy not committed' if exc_type is KeyboardInterrupt
                        else 'failed; current table copy not committed')
        self.report()
