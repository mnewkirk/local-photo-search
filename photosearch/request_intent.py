"""Who made an API request, and why — for the request log.

Source, in order of precedence:
  1. `X-Photosearch-Source` header — callers that know who they are (Claude
     sessions send `claude`).
  2. `/api/worker/*` -> `worker`.
  3. A `Referer` header -> `ui` (the browser names the page it came from).
  4. A Python HTTP client (urllib / httpx / requests) -> `script`: the
     replica forwarding to the NAS, the fleet's helpers, ad-hoc scripts.
  5. Otherwise `other` (curl without headers, a bookmark, ...).

A call the replica forwards to the NAS sends `X-Photosearch-Source: replica`
via `outbound_headers`, with an intent naming its own purpose plus the intent
of the request that caused it ("fetch preview 123 - for: Review team faces
(from the faces page)"). Anything with a `photosearch-*` User-Agent that
still forgets the header is classified `replica` too.

Intent: an explicit `X-Photosearch-Intent` header wins; a handler can set
`request.state.log_intent` (the Ask agent uses the question); otherwise it is
inferred from the endpoint and its parameters by `infer_intent`, and the log
line says `"intent_inferred": true`.
"""

from __future__ import annotations

import contextvars
import re
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urlsplit

MAX_INTENT = 300

_PAGES = {
    "": "search page", "faces": "faces page", "merges": "merge review",
    "geotag": "geotag page", "map": "map", "review": "review page",
    "status": "status page", "batches": "batches page",
    "admin": "maintenance page", "book": "photobook builder",
    "split": "split planner", "eval": "eval page", "collections": "collections",
}


def classify_source(path: str, headers) -> str:
    explicit = (headers.get("x-photosearch-source") or "").strip().lower()
    if explicit:
        return re.sub(r"[^a-z0-9_.-]", "", explicit)[:32] or "other"
    if path.startswith("/api/worker/"):
        return "worker"
    if headers.get("referer"):
        return "ui"
    ua = (headers.get("user-agent") or "").lower()
    if ua.startswith("photosearch-"):
        return "replica"
    if ua.startswith(("python-urllib", "python-httpx", "python-requests")):
        return "script"
    return "other"


def page_of(referer: Optional[str]) -> Optional[str]:
    """`/faces?date_from=...` from a full Referer URL."""
    if not referer:
        return None
    parts = urlsplit(referer)
    return parts.path + (f"?{parts.query}" if parts.query else "")


def _page_label(page: Optional[str]) -> Optional[str]:
    if not page:
        return None
    first = page.split("?")[0].strip("/").split("/")[0]
    return _PAGES.get(first, f"/{first} page")


def _params(query: str) -> dict:
    return {k: v[-1] for k, v in parse_qs(query, keep_blank_values=False).items()}


def _dates(p: dict) -> str:
    a, b = p.get("date_from"), p.get("date_to")
    if a and b:
        return a if a == b else f"{a} to {b}"
    if a:
        return f"from {a}"
    if b:
        return f"until {b}"
    return ""


def _filters(p: dict) -> str:
    bits = []
    for key, label in (("q", "query"), ("person", "person"), ("camera", "camera"),
                       ("location", "place"), ("place", "place"),
                       ("category", "category"), ("visual_tag", "look"),
                       ("keyword", "keyword"), ("style_tag", "style"),
                       ("color", "colour"), ("match_source", "label source")):
        if p.get(key):
            bits.append(f"{label} {p[key]!r}" if key == "q" else f"{label} {p[key]}")
    for key, label in (("min_quality", "quality ≥"), ("min_aesthetic", "aesthetic pct ≥"),
                       ("min_day_aesthetic", "day pct ≥")):
        if p.get(key):
            bits.append(f"{label} {p[key]}")
    d = _dates(p)
    if d:
        bits.append(d)
    return ", ".join(bits)


def _search(m, p):
    f = _filters(p) or "everything"
    extra = []
    if p.get("sort"):
        extra.append(f"sorted {p['sort']}")
    if p.get("offset") not in (None, "0"):
        extra.append(f"from result {p['offset']}")
    return f"Search photos: {f}" + (f" ({', '.join(extra)})" if extra else "")


def _scoped(text: str) -> Callable:
    def build(m, p):
        f = _filters(p)
        return text.format(**m.groupdict(), **{k: v for k, v in p.items()}) + (
            f" — {f}" if f else "")
    return build


def _fixed(text: str) -> Callable:
    return lambda m, p: text.format(**m.groupdict())


# (method or None for any, path regex, builder(match, params) -> str)
_RULES: list[tuple[Optional[str], str, Callable]] = [
    ("GET", r"/api/search", _search),
    (None, r"/api/ask", _fixed("Ask the agent a question")),
    ("GET", r"/api/persons", _fixed("Load the people list")),
    ("GET", r"/api/cameras", _fixed("Load the camera list")),
    ("GET", r"/api/collections", _fixed("List collections")),
    ("GET", r"/api/stats(/.*)?", _fixed("Load library statistics")),
    ("GET", r"/api/health", _fixed("Check the server is reachable")),
    ("GET", r"/api/photos/geojson", _fixed("Load map points")),
    ("GET", r"/api/photos/(?P<id>\d+)/thumbnail", _fixed("Show thumbnail of photo {id}")),
    ("GET", r"/api/photos/(?P<id>\d+)/preview", _fixed("Show preview of photo {id}")),
    ("GET", r"/api/photos/(?P<id>\d+)/full", _fixed("Fetch the original of photo {id}")),
    ("GET", r"/api/photos/(?P<id>\d+)/mirror-fields",
     _fixed("Mirror photo {id} to the replica")),
    ("GET", r"/api/photos/(?P<id>\d+)", _fixed("Open photo {id}")),
    ("GET", r"/api/faces/crop/(?P<id>\d+)", _fixed("Show face crop {id}")),
    ("GET", r"/api/faces/groups", _scoped("List face groups")),
    ("GET", r"/api/faces/group/(?P<kind>\w+)/(?P<id>\d+)/photos",
     _scoped("Open {kind} face group {id}")),
    ("GET", r"/api/faces/suggest-person",
     _scoped("Find more faces of {person}")),
    ("GET", r"/api/faces/verify-labels", _scoped("Verify face labels")),
    ("GET", r"/api/faces/label-health", _fixed("Check face-label health")),
    ("GET", r"/api/faces/unmatch-preview", _scoped("Preview bulk unmatch")),
    ("GET", r"/api/faces/person/(?P<id>\d+)/inspect",
     _scoped("Inspect the faces of person {id}")),
    ("GET", r"/api/faces/label-conflicts", _scoped("Find clusters mixing two names")),
    ("POST", r"/api/faces/bulk-assign", _fixed("Assign or clear faces in bulk")),
    ("POST", r"/api/faces/(?P<id>\d+)/assign", _fixed("Name face {id}")),
    ("POST", r"/api/faces/(?P<id>\d+)/clear", _fixed("Clear the name on face {id}")),
    ("POST", r"/api/faces/review-team", _scoped("Review team faces")),
    ("GET", r"/api/geotag/folders", _fixed("List folders for geotagging")),
    ("GET", r"/api/geotag/folder-photos", _scoped("Open folder {folder} for geotagging")),
    ("GET", r"/api/geotag/known-places", _fixed("Load known places for geotagging")),
    ("GET", r"/api/geocode/search", _scoped("Look up place {q}")),
    ("POST", r"/api/photos/bulk-set-location", _fixed("Set GPS on photos")),
    ("POST", r"/api/photos/bulk-set-tags", _fixed("Set tags on photos")),
    ("GET", r"/api/review/.*", _scoped("Review a shoot")),
    ("GET", r"/api/batches", _fixed("Poll ingest batches")),
    ("GET", r"/api/batches/(?P<id>\d+)", _fixed("Poll batch {id}")),
    (None, r"/api/admin/maintenance-.*", _fixed("Maintenance status / sweep")),
    ("GET", r"/api/admin/workers/queue-status", _fixed("Poll the worker queue")),
    ("GET", r"/api/admin/incoming-status", _fixed("Poll incoming files")),
    ("GET", r"/api/admin/version", _fixed("Check the deployed version")),
    ("POST", r"/api/admin/rerun-passes", _fixed("Re-run index passes on photos")),
    ("POST", r"/api/admin/mirror-photos", _fixed("Mirror photos from the NAS")),
    (None, r"/api/worker/(?P<op>[\w-]+)", _fixed("Worker fleet: {op}")),
]
_COMPILED = [(m, re.compile(rx + r"/?$"), b) for m, rx, b in _RULES]


def infer_intent(method: str, path: str, query: str,
                 page: Optional[str] = None) -> str:
    p = {k: unquote(v) for k, v in _params(query).items()}
    text = None
    for m, rx, build in _COMPILED:
        if m and m != method:
            continue
        hit = rx.match(path)
        if hit:
            try:
                text = build(hit, p)
            except (KeyError, IndexError):
                text = None
            break
    if text is None:
        text = f"{method} {re.sub(r'/[0-9]+', '/{id}', path)}"
    label = _page_label(page)
    if label:
        text = f"{text} (from the {label})"
    return text[:MAX_INTENT]


def explicit_intent(headers) -> Optional[str]:
    v = (headers.get("x-photosearch-intent") or "").strip()
    try:
        # HTTP headers are decoded as latin-1; curl sends UTF-8 bytes.
        v = v.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return v[:MAX_INTENT] or None


# The intent of the request currently being served, so a call it causes to
# the NAS can say what it is for. Set by web._log_request_timing. Thread
# pools do NOT inherit it; submit with contextvars.copy_context().run (as
# face_review does) to carry it across.
_current_intent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "photosearch_request_intent", default=None)


def set_current_intent(intent: Optional[str]):
    return _current_intent.set(intent)


def reset_current_intent(token) -> None:
    _current_intent.reset(token)


def _header_safe(text: str) -> str:
    """Header values travel as latin-1: send the UTF-8 bytes (explicit_intent
    decodes them back), with no line breaks."""
    text = " ".join(text.split())[:MAX_INTENT]
    return text.encode("utf-8").decode("latin-1")


def outbound_headers(purpose: str, source: str = "replica") -> dict:
    """Headers for a call this server makes to the NAS on someone's behalf."""
    cause = _current_intent.get()
    intent = f"{purpose} - for: {cause}" if cause else purpose
    return {"User-Agent": f"photosearch-{source}",
            "X-Photosearch-Source": source,
            "X-Photosearch-Intent": _header_safe(intent)}


def carry_context(fn: Callable) -> Callable:
    """Wrap `fn` so it runs with the CURRENT context (and so the current
    request's intent) in whichever thread calls it. Each call gets its own
    copy, so the wrapper is safe for a thread pool's concurrent tasks."""
    parent = contextvars.copy_context()

    def run(*args, **kwargs):
        return parent.copy().run(fn, *args, **kwargs)
    return run
