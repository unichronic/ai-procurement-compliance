"""Request limiting and input bounds.

Not a substitute for a real gateway, but the endpoints do embedding work per
request, so an unbounded client can trivially saturate the process. These are
the floor: a per-IP token bucket and hard caps on input size.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

MAX_QUERY_CHARS = 8_000
MAX_DOCUMENT_CHARS = 400_000


class RateLimiter:
    """Fixed-window-per-caller limiter with a bounded memory footprint."""

    def __init__(self, limit: int = RATE_LIMIT_REQUESTS,
                 window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.limit = limit
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> bool:
        """True if allowed. Prunes expired entries so idle callers cost nothing."""
        now = time.time() if now is None else now
        cutoff = now - self.window

        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < cutoff:
                hits.popleft()

            if not hits and key in self._hits and len(self._hits) > 10_000:
                # Don't let one-shot callers grow the table without bound.
                self._hits.pop(key, None)
                hits = self._hits[key]

            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return 0
            return max(1, int(self.window - (now - hits[0])))
