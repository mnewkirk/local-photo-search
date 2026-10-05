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
