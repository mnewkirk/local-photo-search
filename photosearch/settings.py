"""Server-side UI preferences.

A small namespaced key/value store for things a person sets once and expects to
find next time — currently the Split planner's defaults.

**Why not the replica DB.** ``sync-replica.sh`` swaps ``PHOTOSEARCH_DB``
wholesale on every sync, so anything written there is destroyed the next time
the replica pulls. These live in the same **sidecar** file as the photobook
curation state (``PHOTOSEARCH_BOOKS_DB``), which exists for exactly that reason.

Values are JSON, so a namespace can hold whatever shape its page needs without
a migration each time a control is added.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SettingsStore:
    """Namespaced key/value settings in the sidecar sqlite file."""

    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                namespace  TEXT NOT NULL,
                key        TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (namespace, key)
            );
            """
        )
        self.conn.commit()

    # -- reads -------------------------------------------------------------

    def get_all(self, namespace: str) -> dict:
        """Every setting in a namespace. Unreadable rows are skipped rather
        than failing the whole page — a bad value should not lock you out of
        your own defaults."""
        rows = self.conn.execute(
            "SELECT key, value_json FROM app_settings WHERE namespace = ?",
            (namespace,)).fetchall()
        out = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value_json"])
            except (ValueError, TypeError):
                logger.warning("settings: bad JSON at %s.%s", namespace, r["key"])
        return out

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value_json FROM app_settings WHERE namespace = ? AND key = ?",
            (namespace, key)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except (ValueError, TypeError):
            return default

    # -- writes ------------------------------------------------------------

    def set_many(self, namespace: str, values: dict) -> dict:
        """Upsert a whole namespace's worth at once, and return what was
        stored — the caller echoes it back so the UI shows the canonical values
        rather than what it hoped it sent."""
        with self.conn:
            for k, v in values.items():
                self.conn.execute(
                    "INSERT INTO app_settings (namespace, key, value_json, updated_at) "
                    "VALUES (?, ?, ?, datetime('now')) "
                    "ON CONFLICT(namespace, key) DO UPDATE SET "
                    "  value_json = excluded.value_json, "
                    "  updated_at = excluded.updated_at",
                    (namespace, k, json.dumps(v)))
        return self.get_all(namespace)

    def clear(self, namespace: str) -> None:
        """Drop a namespace so the page falls back to its built-in defaults."""
        with self.conn:
            self.conn.execute("DELETE FROM app_settings WHERE namespace = ?",
                              (namespace,))

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
