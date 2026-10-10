"""A new description re-queues everything derived from the old one.

category-content and keywords are extracted FROM the description, and verify
checks it. On 2026-10-04 the fleet ran verify LAST, so its rewrites (16% of a
batch) left categories/keywords extracted from text verify had just judged
hallucinated — 13,637 such photos library-wide. Now the server NULLs the
dependents (and their ledger rows) whenever a description is written over.
"""

import json

import pytest

from tests.test_worker_submit_resilience import _open, client  # noqa: F401  (fixture)


def _seed_downstream(photo_id=1):
    with _open() as db:
        db.conn.execute(
            "UPDATE photos SET categories=?, keywords=?, verified_at='2090-01-01T00:00:00', "
            "verification_status='pass', tags=NULL WHERE id=?",
            (json.dumps(["beach"]), json.dumps(["dog"]), photo_id))
        for p in ("category-content", "keywords", "verify"):
            db.conn.execute(
                "INSERT INTO worker_processed (photo_id, pass_type, attempts) VALUES (?, ?, 1)",
                (photo_id, p))
        db.conn.commit()


def _row(photo_id=1):
    with _open() as db:
        r = db.conn.execute("SELECT * FROM photos WHERE id=?", (photo_id,)).fetchone()
        ledger = {x[0] for x in db.conn.execute(
            "SELECT pass_type FROM worker_processed WHERE photo_id=?", (photo_id,))}
        return dict(r), ledger


def _submit(client, pass_type, key, results):
    r = client.post("/api/worker/submit-results", json={
        "batch_id": "no-claim", "pass_type": pass_type, key: results,
        "model": "m", "model_version": "v"})
    assert r.status_code == 200, r.text


def test_verify_rewrite_requeues_text_passes_but_keeps_its_verdict(client):
    _seed_downstream()
    _submit(client, "verify", "verify_results", [{
        "photo_id": 1, "status": "regenerated", "verified_at": "2090-02-02T00:00:00",
        "hallucination_flags": "[]", "description": "a dog on sand"}])
    row, ledger = _row()
    assert row["description"] == "a dog on sand"
    assert row["categories"] is None and row["keywords"] is None
    assert not ledger & {"category-content", "keywords"}
    assert row["verification_status"] == "regenerated"   # this result IS the verify


def test_verify_pass_without_rewrite_touches_nothing(client):
    _seed_downstream()
    _submit(client, "verify", "verify_results", [{
        "photo_id": 1, "status": "pass", "verified_at": "2090-02-02T00:00:00"}])
    row, ledger = _row()
    assert json.loads(row["categories"]) == ["beach"]
    assert {"category-content", "keywords"} <= ledger


def test_old_worker_retag_is_not_written_to_the_legacy_column(client):
    _submit(client, "verify", "verify_results", [{
        "photo_id": 1, "status": "regenerated", "verified_at": "2090-02-02T00:00:00",
        "description": "x", "tags": ["sunny"]}])
    assert _row()[0]["tags"] is None


def test_redescribe_requeues_text_passes_and_verify(client):
    _seed_downstream()
    _submit(client, "describe", "describe_results",
            [{"photo_id": 1, "description": "a new description"}])
    row, ledger = _row()
    assert row["categories"] is None and row["keywords"] is None
    assert row["verified_at"] is None and row["verification_status"] is None
    assert not ledger & {"category-content", "keywords", "verify"}


def test_empty_describe_result_leaves_downstream_alone(client):
    _seed_downstream()
    _submit(client, "describe", "describe_results", [{"photo_id": 1, "description": None}])
    assert json.loads(_row()[0]["categories"]) == ["beach"]


def test_fleet_order_runs_verify_before_the_text_passes():
    from photosearch import batch_state, rerun
    for order in (batch_state.WORKER_PASSES, rerun.ALL_PASSES):
        i = order.index
        assert i("describe") < i("verify") < i("category-content")
        assert i("verify") < i("keywords")
    assert tuple(rerun.ALL_PASSES) == tuple(batch_state.WORKER_PASSES)


def test_workers_start_sorts_passes_into_dependency_order(monkeypatch, tmp_path):
    from photosearch import admin_api
    script = tmp_path / "run-workers.sh"
    script.write_text("")
    seen = {}
    monkeypatch.setattr(admin_api, "_run_workers_script", lambda: str(script))
    monkeypatch.setattr(admin_api, "_fleet_env", lambda: {})

    class _P:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _P()
    monkeypatch.setattr(admin_api.subprocess, "run", fake_run)
    try:
        admin_api.admin_workers_start(admin_api.WorkersStartRequest(
            passes=["category-visual", "keywords", "verify", "category-content"]))
    except Exception:
        pass
    cmd = seen["cmd"]
    assert cmd[cmd.index("-p") + 1] == "verify,category-content,keywords,category-visual"
