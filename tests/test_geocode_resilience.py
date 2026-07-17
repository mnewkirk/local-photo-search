"""The ingest/index reverse-geocode must tolerate a busy DB.

A heavy maintenance-sweep stage plus the worker fleet can hold SQLite's single
write lock past the 60s busy_timeout; the geocode write then raises 'database is
locked'. It must DEFER (leave place_name NULL for the next pass), not abort the
whole index. Observed on the NAS 2026-07-17.
"""
import sqlite3

import pytest

from photosearch.db import PhotoDB
from photosearch.index import _write_geocode_results


def _seed_gps_photo(db, pid_path="2026/2026-07-17/a.jpg"):
    db.set_photo_root("/photos")
    pid = db.add_photo(filepath=pid_path, filename="a.jpg",
                       date_taken="2026-07-17 10:00:00",
                       gps_lat=47.6, gps_lon=-122.3)
    db.conn.commit()
    return pid


def test_geocode_write_defers_on_locked_db_instead_of_raising(tmp_db_path, monkeypatch):
    from photosearch import geocode
    monkeypatch.setattr(geocode, "reverse_geocode_batch",
                        lambda coords: ["Seattle, WA"] * len(coords))

    with PhotoDB(tmp_db_path) as db:
        pid = _seed_gps_photo(db)
        ungeo = db.conn.execute(
            "SELECT id, gps_lat, gps_lon FROM photos WHERE place_name IS NULL"
        ).fetchall()

        # Simulate the busy NAS: every write raises 'database is locked'.
        def _locked(*a, **k):
            raise sqlite3.OperationalError("database is locked")
        monkeypatch.setattr(db, "update_photo", _locked)

        # Must NOT raise — deferral, not abort.
        filled = _write_geocode_results(db, ungeo)
        assert filled == 0

        # Batch mode was reset, so the connection is still usable.
        assert db._batch_mode is False
        # place_name is still NULL (will fill next pass), GPS intact.
        row = db.conn.execute(
            "SELECT gps_lat, place_name FROM photos WHERE id=?", (pid,)).fetchone()
        assert row["gps_lat"] == 47.6
        assert row["place_name"] is None


def test_geocode_write_fills_place_name_when_db_is_free(tmp_db_path, monkeypatch):
    from photosearch import geocode
    monkeypatch.setattr(geocode, "reverse_geocode_batch",
                        lambda coords: ["Seattle, WA"] * len(coords))

    with PhotoDB(tmp_db_path) as db:
        pid = _seed_gps_photo(db)
        ungeo = db.conn.execute(
            "SELECT id, gps_lat, gps_lon FROM photos WHERE place_name IS NULL"
        ).fetchall()
        filled = _write_geocode_results(db, ungeo)
        assert filled == 1
        row = db.conn.execute(
            "SELECT place_name FROM photos WHERE id=?", (pid,)).fetchone()
        assert row["place_name"] == "Seattle, WA"


def test_abort_batch_rolls_back_and_exits_batch_mode(tmp_db_path):
    with PhotoDB(tmp_db_path) as db:
        _seed_gps_photo(db)
        db.begin_batch(batch_size=200)
        assert db._batch_mode is True
        db.abort_batch()
        assert db._batch_mode is False
        assert db._batch_count == 0
        # Connection still works after an abort.
        n = db.conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"]
        assert n == 1
