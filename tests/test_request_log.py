"""The persistent API request log (photosearch/request_log.py).

The NAS container's stdout is discarded on every redeploy, so this file is the
only record of how real requests performed.
"""

import json

import pytest
from click.testing import CliRunner

from photosearch import request_log


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "request_log.jsonl"
    monkeypatch.setenv("PHOTOSEARCH_REQUEST_LOG", str(path))
    request_log.reset()
    yield path
    request_log.reset()


def _records(path):
    request_log.reset()  # drain the background queue to the file
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_api_requests_are_logged_with_timing(client, log_file):
    client.get("/api/search?person=Alice&sort=date_desc")
    client.get("/api/persons")
    client.get("/")  # a page, not the API: not logged
    recs = _records(log_file)
    assert [r["path"] for r in recs] == ["/api/search", "/api/persons"]
    assert recs[0]["query"] == "person=Alice&sort=date_desc"
    assert recs[0]["method"] == "GET" and recs[0]["status"] == 200
    assert recs[0]["ms"] >= 0 and recs[0]["ts"].endswith("+00:00")


def test_ui_requests_get_an_inferred_intent_from_page_and_parameters(client, log_file):
    client.get("/api/search?person=Alice&camera=X100&date_from=2026-09-26"
               "&date_to=2026-09-27&sort=date_desc",
               headers={"Referer": "http://nas:8000/?person=Alice"})
    client.get("/api/faces/suggest-person?person=Koa&date_from=2026-09-19"
               "&date_to=2026-09-19",
               headers={"Referer": "http://nas:8000/faces?date_from=2026-09-19"})
    search, suggest = _records(log_file)
    assert search["source"] == "ui" and search["intent_inferred"] is True
    assert search["page"] == "/?person=Alice"
    assert search["intent"] == ("Search photos: person Alice, camera X100, "
                                "2026-09-26 to 2026-09-27 (sorted date_desc) "
                                "(from the search page)")
    assert suggest["intent"].startswith("Find more faces of Koa")
    assert "2026-09-19" in suggest["intent"] and "faces page" in suggest["intent"]


def test_claude_states_its_own_intent_in_headers(client, log_file):
    client.get("/api/persons", headers={
        "X-Photosearch-Source": "claude",
        "X-Photosearch-Intent": "Check who is registered before tagging Koa"})
    (rec,) = _records(log_file)
    assert rec["source"] == "claude"
    assert rec["intent"] == "Check who is registered before tagging Koa"
    assert "intent_inferred" not in rec


def test_unlabelled_python_clients_read_as_script(client, log_file):
    client.get("/api/persons", headers={"User-Agent": "Python-urllib/3.12"})
    client.get("/api/persons", headers={"User-Agent": "curl/8.5"})
    script, other = _records(log_file)
    assert script["source"] == "script" and other["source"] == "other"


def test_agent_tool_calls_carry_the_question(db, log_file):
    from photosearch import agent
    agent._logged_tool_call(db, "list_people", {}, "who is in the library?")
    (rec,) = _records(log_file)
    assert rec["source"] == "agent" and rec["method"] == "TOOL"
    assert rec["path"] == "list_people"
    assert rec["intent"] == "Ask: who is in the library?"


def test_mcp_tool_calls_log_claudes_intent_to_their_own_file(db, log_file):
    from photosearch import mcp_server
    mcp_server.call_tool_logged(db, "list_people",
                                {"intent": "Find Koa's registered name"})
    mcp_server.call_tool_logged(db, "list_people", {})
    request_log.reset()
    mcp_file = log_file.with_name("request_log.mcp.jsonl")
    stated, missing = [json.loads(l) for l in mcp_file.read_text().splitlines()]
    assert stated["source"] == "claude-mcp" and stated["path"] == "list_people"
    assert stated["intent"] == "Find Koa's registered name"
    assert "intent" not in json.loads(stated["query"])  # stripped before the tool
    assert missing["intent_inferred"] is True
    assert not log_file.exists() or "list_people" not in log_file.read_text()


def test_every_mcp_tool_advertises_an_optional_intent():
    from photosearch import mcp_server
    schema = mcp_server._with_intent({"type": "object", "properties": {"q": {}},
                                      "required": ["q"]})
    assert "intent" in schema["properties"] and schema["required"] == ["q"]


def test_failed_requests_are_logged_with_their_status(client, log_file):
    client.get("/api/photos/999999")
    (rec,) = _records(log_file)
    assert rec["status"] == 404


def test_disabled_by_zero(client, tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_REQUEST_LOG", "0")
    request_log.reset()
    client.get("/api/persons")
    request_log.reset()
    assert not list(tmp_path.glob("*.jsonl"))


def test_an_unwritable_log_never_fails_the_request(client, tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_REQUEST_LOG",
                       str(tmp_path / "missing-dir" / "log.jsonl"))
    request_log.reset()
    assert client.get("/api/persons").status_code == 200
    request_log.reset()


def test_request_stats_summarises_by_endpoint(client, log_file):
    from cli import cli
    for _ in range(3):
        client.get("/api/persons")
    client.get("/api/photos/1")
    client.get("/api/batches")  # polling: hidden by default
    request_log.reset()
    out = CliRunner().invoke(cli, ["request-stats", "--file", str(log_file)])
    assert out.exit_code == 0, out.output
    assert "GET /api/persons" in out.output and "GET /api/photos/{id}" in out.output
    assert "/api/batches" not in out.output
    out = CliRunner().invoke(cli, ["request-stats", "--file", str(log_file),
                                   "--include-polling"])
    assert "GET /api/batches" in out.output


def test_rotated_backups_are_read_oldest_first(tmp_path):
    path = tmp_path / "r.jsonl"
    (tmp_path / "r.jsonl.1").write_text('{"n": 1}\n')
    path.write_text('{"n": 2}\nnot json\n')
    assert [r["n"] for r in request_log.read_records(str(path))] == [1, 2]


def test_logs_are_merged_by_time(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"ts": "2026-10-05T01", "n": 1}\n{"ts": "2026-10-05T03", "n": 3}\n')
    b.write_text('{"ts": "2026-10-05T02", "n": 2}\n')
    assert [r["n"] for r in request_log.read_records(str(a), str(b))] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Replica -> NAS calls say they come from the replica, and why
# ---------------------------------------------------------------------------

from photosearch import request_intent


def test_outbound_headers_carry_the_causing_request_intent():
    token = request_intent.set_current_intent("Review team faces (from the faces page)")
    try:
        h = request_intent.outbound_headers("Fetch preview of photo 7")
    finally:
        request_intent.reset_current_intent(token)
    assert h["X-Photosearch-Source"] == "replica"
    assert h["User-Agent"] == "photosearch-replica"
    assert h["X-Photosearch-Intent"] == (
        "Fetch preview of photo 7 - for: Review team faces (from the faces page)")
    # Outside any request: just the purpose.
    assert request_intent.outbound_headers("x")["X-Photosearch-Intent"] == "x"


def test_non_ascii_intent_survives_the_header_round_trip():
    token = request_intent.set_current_intent("Ask: photos of Zoë ≥ 2024\nnext line")
    try:
        sent = request_intent.outbound_headers("Fetch")["X-Photosearch-Intent"]
    finally:
        request_intent.reset_current_intent(token)
    sent.encode("latin-1")  # must be transmittable as a header
    assert request_intent.explicit_intent({"x-photosearch-intent": sent}) == (
        "Fetch - for: Ask: photos of Zoë ≥ 2024 next line")


def test_any_photosearch_user_agent_reads_as_replica():
    assert request_intent.classify_source(
        "/api/photos/1/preview", {"user-agent": "photosearch-rerank"}) == "replica"


def test_carry_context_reaches_threads_and_pools():
    import concurrent.futures as cf
    import threading
    token = request_intent.set_current_intent("Verify face labels")
    try:
        got = []
        t = threading.Thread(target=request_intent.carry_context(
            lambda: got.append(request_intent._current_intent.get())))
        t.start(); t.join()
        with cf.ThreadPoolExecutor(4) as ex:
            got += list(ex.map(request_intent.carry_context(
                lambda _: request_intent._current_intent.get()), range(8)))
    finally:
        request_intent.reset_current_intent(token)
    assert got == ["Verify face labels"] * 9


def test_replica_preview_fetch_tells_the_nas_why(client, db, log_file, monkeypatch):
    """A preview the replica must pull from the NAS goes with the replica
    source and the intent of the page request that needed it."""
    import urllib.request
    from photosearch import web
    seen = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"\xff\xd8\xff\xd9"

    def fake_urlopen(req, timeout=None):
        seen.update({k.lower(): v for k, v in req.header_items()})
        return _Resp()

    monkeypatch.setattr(web, "_nas_url", "http://nas.invalid:8000")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    pid = db.conn.execute("SELECT id FROM photos LIMIT 1").fetchone()[0]
    client.get(f"/api/photos/{pid}/preview",
               headers={"Referer": "http://replica:8001/faces?date_from=2026-10-03"})
    assert seen.get("x-photosearch-source") == "replica"
    intent = request_intent.explicit_intent(
        {"x-photosearch-intent": seen["x-photosearch-intent"]})
    assert intent.startswith(f"Fetch preview of photo {pid}")
    assert intent.endswith(f"- for: Show preview of photo {pid} (from the faces page)")


def test_worker_session_labels_itself():
    from photosearch.worker import WorkerClient
    wc = WorkerClient("http://nas.invalid:8000", probe=False)
    assert wc.session.headers["X-Photosearch-Source"] == "worker"
