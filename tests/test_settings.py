"""Tests for server-side UI settings.

The behaviour that matters: these live in the SIDECAR file, not the replica DB,
because sync-replica.sh swaps the replica wholesale and would wipe them. And
every written value is validated, since they drive the Split planner's geometry
and a bad one would be hard to back out of from the UI.
"""

import json

import pytest
from fastapi.testclient import TestClient

from photosearch.settings import SettingsStore


def _store(tmp_path):
    return SettingsStore(str(tmp_path / "sidecar.db"))


class TestStore:
    def test_roundtrips_values_of_every_shape(self, tmp_path):
        with _store(tmp_path) as st:
            st.set_many("split", {
                "wall": 65, "gutter": 0.5, "mode": "window", "wallOnly": True,
                "minDpi": 240, "maxCrop": 0.3,
                "sizes": ["13×19", "11×14"], "count": "any",
            })
            got = st.get_all("split")
        assert got["wall"] == 65
        assert got["gutter"] == 0.5
        assert got["wallOnly"] is True
        assert got["sizes"] == ["13×19", "11×14"]
        assert got["count"] == "any"

    def test_none_survives_as_none_not_missing(self, tmp_path):
        """sizes=None means 'all sheet sizes' — distinct from unset."""
        with _store(tmp_path) as st:
            st.set_many("split", {"sizes": None})
            assert st.get_all("split") == {"sizes": None}
            assert st.get("split", "sizes", "fallback") is None

    def test_set_many_upserts(self, tmp_path):
        with _store(tmp_path) as st:
            st.set_many("split", {"wall": 65, "minDpi": 240})
            out = st.set_many("split", {"wall": 40})
        assert out == {"wall": 40, "minDpi": 240}   # merges, does not replace

    def test_namespaces_are_isolated(self, tmp_path):
        with _store(tmp_path) as st:
            st.set_many("split", {"wall": 65})
            st.set_many("other", {"wall": 12})
            assert st.get_all("split") == {"wall": 65}
            assert st.get_all("other") == {"wall": 12}

    def test_clear_drops_only_its_namespace(self, tmp_path):
        with _store(tmp_path) as st:
            st.set_many("split", {"wall": 65})
            st.set_many("other", {"wall": 12})
            st.clear("split")
            assert st.get_all("split") == {}
            assert st.get_all("other") == {"wall": 12}

    def test_unknown_namespace_reads_empty(self, tmp_path):
        with _store(tmp_path) as st:
            assert st.get_all("nope") == {}
            assert st.get("nope", "k", "dflt") == "dflt"

    def test_a_corrupt_row_does_not_break_the_rest(self, tmp_path):
        """One bad value must not lock you out of all your defaults."""
        with _store(tmp_path) as st:
            st.set_many("split", {"wall": 65, "mode": "window"})
            st.conn.execute(
                "UPDATE app_settings SET value_json='{not json' WHERE key='mode'")
            st.conn.commit()
            got = st.get_all("split")
        assert got == {"wall": 65}

    def test_survives_reopening(self, tmp_path):
        with _store(tmp_path) as st:
            st.set_many("split", {"wall": 42})
        with _store(tmp_path) as st2:
            assert st2.get_all("split")["wall"] == 42


class TestRoutes:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from photosearch import web
        from photosearch.db import PhotoDB
        dbpath = tmp_path / "r.db"
        PhotoDB(str(dbpath)).close()
        monkeypatch.setattr(web, "_db_path", str(dbpath))
        monkeypatch.setattr(web, "_books_db_path", str(tmp_path / "sidecar.db"))
        return TestClient(web.app)

    def test_first_visit_returns_empty_not_404(self, client):
        r = client.get("/api/settings/split")
        assert r.status_code == 200
        assert r.json()["values"] == {}

    def test_put_then_get_roundtrips(self, client):
        vals = {"wall": 40, "gutter": 0.25, "mode": "continue",
                "wallOnly": False, "minDpi": 150, "maxCrop": 0.5,
                "sizes": ["8×10"], "count": 4}
        r = client.put("/api/settings/split", json={"values": vals})
        assert r.status_code == 200, r.text
        assert r.json()["values"] == vals
        assert client.get("/api/settings/split").json()["values"] == vals

    def test_delete_falls_back_to_builtins(self, client):
        client.put("/api/settings/split", json={"values": {"wall": 40}})
        assert client.delete("/api/settings/split").status_code == 200
        assert client.get("/api/settings/split").json()["values"] == {}

    def test_unknown_namespace_is_rejected(self, client):
        assert client.get("/api/settings/bogus").status_code == 404
        assert client.put("/api/settings/bogus", json={"values": {}}).status_code == 404

    def test_unknown_key_is_named_not_dropped(self, client):
        r = client.put("/api/settings/split", json={"values": {"nope": 1}})
        assert r.status_code == 400
        assert "nope" in r.json()["detail"]

    @pytest.mark.parametrize("vals", [
        {"wall": 0},                 # below the floor
        {"wall": 500},               # past the ceiling
        {"gutter": -1},
        {"mode": "sideways"},
        {"wallOnly": "yes"},         # must be a real bool
        {"minDpi": 5},
        {"maxCrop": 1.5},
        {"sizes": [13, 19]},         # labels, not numbers
        {"count": 0},
        {"count": "lots"},
    ])
    def test_out_of_range_values_are_rejected(self, client, vals):
        r = client.put("/api/settings/split", json={"values": vals})
        assert r.status_code == 400
        assert list(vals)[0] in r.json()["detail"]

    def test_a_rejected_write_leaves_the_stored_values_alone(self, client):
        client.put("/api/settings/split", json={"values": {"wall": 40}})
        client.put("/api/settings/split", json={"values": {"wall": 9999}})
        assert client.get("/api/settings/split").json()["values"] == {"wall": 40}

    def test_settings_live_outside_the_replica_db(self, client, tmp_path):
        """The point of the sidecar: a replica sync swaps the replica DB, so
        anything stored there would be destroyed."""
        client.put("/api/settings/split", json={"values": {"wall": 40}})
        assert (tmp_path / "sidecar.db").exists()
        import sqlite3
        names = {r[0] for r in sqlite3.connect(tmp_path / "r.db").execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "app_settings" not in names
