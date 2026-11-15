"""Bounded, metadata-only audit storage for knowledge base requests."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path, max_runs: int = 500):
        if not 1 <= max_runs <= 10_000:
            raise ValueError("Audit retention must be between 1 and 10000 runs")
        self.path = path
        self.max_runs = max_runs

    def write(self, report: dict) -> None:
        # KnowledgeService builds report from enumerated statuses, internal IDs
        # and metrics. Keep raw question, source paths and model data out.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY, recorded_at TEXT NOT NULL, report TEXT NOT NULL
                )
            """)
            db.execute(
                "INSERT INTO runs (recorded_at, report) VALUES (?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    json.dumps(report, ensure_ascii=False),
                ),
            )
            db.execute(
                """
                DELETE FROM runs WHERE id < (
                    SELECT MIN(id) FROM (SELECT id FROM runs ORDER BY id DESC LIMIT ?)
                )
            """,
                (self.max_runs,),
            )

    def recent(self, limit: int = 20) -> list[dict]:
        if not 1 <= limit <= 10_000:
            raise ValueError("Audit limit must be between 1 and 10000")
        if not self.path.is_file():
            return []
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            return [
                {"recorded_at": time, **json.loads(report)}
                for time, report in db.execute(
                    "SELECT recorded_at, report FROM runs ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            ]
