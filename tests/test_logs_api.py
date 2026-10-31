"""The /logs page's read-only endpoints (photosearch/logs_api.py)."""

import json
import os

import pytest

from photosearch import logs_api, request_log


def _write(path, recs):
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture
def req_log(tmp_path, monkeypatch):
    path = tmp_path / "request_log.jsonl"
    monkeypatch.setenv("PHOTOSEARCH_REQUEST_LOG", str(path))
    request_log.reset()
    yield path
    request_log.reset()


def test_tail_records_is_newest_first_and_continues_into_backups(tmp_path):
    p = str(tmp_path / "r.jsonl")
    _write(p + ".1", [{"ts": "a", "n": i} for i in range(3)])
    _write(p, [{"ts": "b", "n": i} for i in range(3, 6)])
    recs, _ = request_log.tail_records(p, last=5)
    assert [r["n"] for r in recs] == [5, 4, 3, 2, 1]


def test_tail_records_reads_across_block_boundaries(tmp_path):
    p = str(tmp_path / "r.jsonl")
    _write(p, [{"n": i, "pad": "x" * 50} for i in range(400)])
    lines = list(request_log._reverse_lines(p, block=64))
    assert [json.loads(r)["n"] for r in lines] == list(range(399, -1, -1))


def test_tail_records_stops_scanning_when_nothing_matches(tmp_path):
    p = str(tmp_path / "r.jsonl")
    _write(p, [{"n": i} for i in range(50)])
    recs, scanned = request_log.tail_records(p, last=10, match=lambda r: False, max_scan=20)
    assert recs == [] and scanned == 20


def test_requests_endpoint_filters(client, req_log):
    request_log.reset()
    _write(req_log, [
        {"ts": "1", "method": "GET", "path": "/api/search", "status": 200, "ms": 50, "source": "ui",
         "intent": "Search for dogs"},
        {"ts": "2", "method": "GET", "path": "/api/faces/groups", "status": 500, "ms": 2500,
         "source": "claude", "intent": "debug"},
        {"ts": "3", "method": "POST", "path": "/api/faces/bulk-assign", "status": 200, "ms": 10,
         "source": "ui"},
    ])
    # The test client's own requests are logged to the same file; keep to the seeded ones.
    get = lambda q: [r for r in client.get("/api/logs/requests?" + q).json()["records"]
                     if r["ts"] in ("1", "2", "3")]
    assert [r["ts"] for r in get("q=faces")][:2] == ["3", "2"]
    assert [r["ts"] for r in get("errors=true")] == ["2"]
    assert [r["ts"] for r in get("min_ms=1000")] == ["2"]
    assert [r["ts"] for r in get("source=ui&method=POST")] == ["3"]
    assert [r["ts"] for r in get("q=DOGS")] == ["1"]


def test_mcp_stream_reads_its_own_file(client, req_log):
    request_log.reset()
    _write(str(req_log).replace(".jsonl", ".mcp.jsonl"),
           [{"ts": "1", "method": "TOOL", "path": "search_photos", "status": 200}])
    d = client.get("/api/logs/requests?stream=mcp").json()
    assert [r["path"] for r in d["records"]] == ["search_photos"]
    assert d["path"].endswith("request_log.mcp.jsonl")


def test_host_nas_needs_a_replica(client, req_log):
    assert client.get("/api/logs/requests?host=nas").status_code == 400


def test_ask_logs_by_day_and_available_days(client, tmp_path):
    d = os.environ["PHOTOSEARCH_ASK_LOG_DIR"]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "ask-2026-01-02.md"), "w") as fh:
        fh.write("## 2026-01-02 10:00:00 · q\nbody\n")
    r = client.get("/api/logs?date=2026-01-02").json()
    assert r["days"][0]["date"] == "2026-01-02" and "body" in r["days"][0]["content"]
    assert "2026-01-02" in r["available"]
    assert client.get("/api/logs?date=../../etc").status_code == 422


def test_fleet_logs_are_tailed_with_liveness(client, tmp_path, monkeypatch):
    fleet = tmp_path / "photosearch-worker-fleet-ui"
    fleet.mkdir()
    (fleet / "worker-1.log").write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    (fleet / "worker-1.pid").write_text(str(os.getpid()))
    monkeypatch.setattr(logs_api, "FLEET_LOG_GLOB",
                        str(tmp_path / "photosearch-worker-fleet*" / "worker-*.log"))
    w = client.get("/api/logs/fleet?lines=10").json()["workers"][0]
    assert w["fleet"] == "ui" and w["worker"] == "worker-1" and w["alive"] is True
    assert w["lines"] == [f"line {i}" for i in range(90, 100)]
