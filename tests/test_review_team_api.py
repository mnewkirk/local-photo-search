"""Tests for POST /api/faces/review-team — the review-faces CLI as an endpoint.

The pipeline itself is covered by the face_review module's own tests; these
cover the endpoint contract: validation, the SSE terminal events, and that the
preview fetch is the in-process one (not ~800 HTTP requests to ourselves).
"""

import io
import json

import pytest


def _sse_events(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[len("data:"):].strip()))
            except ValueError:
                pass
    return out


def _terminal(text):
    evs = [e for e in _sse_events(text)
           if e.get("type") in ("done", "cancelled", "fatal")]
    assert evs, f"no terminal event in stream: {text[:400]}"
    return evs[-1]


@pytest.fixture
def blue_preview(monkeypatch):
    """Stub the preview fetch with an image whose torso band is solid blue.

    Patching `_preview_bytes` (rather than the HTTP route) is deliberate: it
    also pins that the endpoint goes through that helper, which is what keeps
    the NAS from issuing a thousand requests to itself.
    """
    from PIL import Image
    from photosearch import web

    calls = []

    def fake(photo_id):
        calls.append(photo_id)
        # Solid blue, so the torso sample lands on a readable hue wherever
        # the band happens to fall.
        im = Image.new("RGB", (1920, 1280), (20, 60, 200))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        return buf.getvalue()

    monkeypatch.setattr(web, "_preview_bytes", fake)
    return calls


def test_date_is_required(client):
    r = client.post("/api/faces/review-team", json={})
    assert r.status_code == 400


def test_no_faces_on_the_date_is_a_fatal_event_not_a_500(client):
    """A day with nothing above the quality floor is a normal outcome. It has
    to arrive as an SSE 'fatal' — the work runs on a background thread, so an
    exception there would never reach the client as an HTTP error."""
    r = client.post("/api/faces/review-team", json={"date": "1999-01-01"})
    assert r.status_code == 200
    ev = _terminal(r.text)
    assert ev["type"] == "fatal"
    assert "No faces" in ev["message"]


def test_happy_path_groups_the_days_faces(client, blue_preview):
    r = client.post("/api/faces/review-team",
                    json={"date": "2026-03-13", "team_hue": 220, "tolerance": 40})
    assert r.status_code == 200
    ev = _terminal(r.text)
    assert ev["type"] == "done", ev
    assert ev["date"] == "2026-03-13"
    assert ev["team_hue"] == 220
    assert "stats" in ev and "groups" in ev
    assert blue_preview, "the endpoint must fetch previews via _preview_bytes"


def test_start_event_reports_the_scale_before_the_slow_part(client, blue_preview):
    """Sampling is minutes long, so the client needs the denominator up front
    rather than after the work finishes."""
    r = client.post("/api/faces/review-team",
                    json={"date": "2026-03-13", "team_hue": 220})
    start = [e for e in _sse_events(r.text) if e.get("type") == "start"]
    assert start, "no start event"
    assert start[0]["faces"] > 0
    assert start[0]["photos"] > 0
    assert "named" in start[0]


def test_explicit_hue_skips_learning_and_is_reported_as_given(client, blue_preview):
    """Hue is normally learned from faces already named that day. When those
    are themselves suspect (a bad temporal match pass), the override is the
    escape hatch — so it must actually take effect and say so."""
    r = client.post("/api/faces/review-team",
                    json={"date": "2026-03-13", "team_hue": 111, "tolerance": 40})
    evs = _sse_events(r.text)
    hue_ev = [e for e in evs if e.get("phase") == "hue"]
    assert hue_ev and hue_ev[0]["team_hue"] == 111
    assert hue_ev[0]["source"] == "given"
    assert _terminal(r.text)["team_hue"] == 111


def test_unlearnable_hue_is_a_clear_fatal_not_a_crash(client, monkeypatch):
    """No readable torso on any named face → the run can't proceed. The user
    needs to be told to name someone or set the hue, not shown a traceback."""
    from photosearch import web

    def blank(photo_id):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (0, 0, 0)).save(buf, format="JPEG")
        return buf.getvalue()

    monkeypatch.setattr(web, "_preview_bytes", blank)
    r = client.post("/api/faces/review-team", json={"date": "2026-03-13"})
    ev = _terminal(r.text)
    assert ev["type"] == "fatal"
    assert "team colour" in ev["message"]


def test_ungrouped_bucket_is_labelled_separately(client, blue_preview):
    """DBSCAN noise is a bag of different people. It must never look like just
    another nameable group — assigning it wholesale would be catastrophic."""
    r = client.post("/api/faces/review-team",
                    json={"date": "2026-03-13", "team_hue": 220, "tolerance": 40})
    ev = _terminal(r.text)
    for g in ev["groups"]:
        if g["group_id"] < 0:
            assert g["label"] == "Ungrouped"
            break
