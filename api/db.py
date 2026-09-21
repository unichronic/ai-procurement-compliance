"""Database connection pool for the API layer."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row

# No credential default. A baked-in username and password is the one that ends
# up in production because nobody noticed the env var was missing; failing to
# start is the safer outcome, and the message says exactly what to set.
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Export it before starting, e.g.\n"
        "  export DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME\n"
        "See .env.example."
    )


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Yield a short-lived connection with dict-row factory."""
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn
