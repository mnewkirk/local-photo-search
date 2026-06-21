"""Shared LLM tool layer for local-photo-search (M24).

A single registry of tools that let a language model *plan* a search instead
of the caller hand-assembling structured filters. Defined once here and
consumed by two adapters:

  - ``photosearch/mcp_server.py``  — an MCP server over streamable HTTP
  - ``web.py``  ``POST /api/ask``   — an in-app agent loop on the local LLM

Each tool is a :class:`ToolSpec` with a name, an LLM-facing description, a
JSON Schema for its arguments, and a handler ``fn(db, args) -> result``.
``call_tool(db, name, args)`` dispatches; ``openai_tools()`` and
``mcp_tools()`` project the same schemas into the two wire formats so there
is exactly one definition of what "search" means to a model.

Design notes:
  - ``search`` (and therefore ``clip_embed`` / torch) is imported lazily
    inside handlers, so importing this module is cheap and unit-testable
    without the CLIP stack loaded.
  - Tools never return raw image bytes. ``get_photo_image`` returns a cached
    thumbnail *path* + mime type; each adapter decides how to encode it (MCP
    → ImageContent bytes; the web agent → a thumbnail URL). Whether images
    are exposed at all is an adapter-level policy (see
    ``PHOTOSEARCH_MCP_ALLOW_IMAGES``), not a property of the tool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .db import PhotoDB

# How long a description we hand back per search hit. The model pays for every
# token; full LLaVA descriptions run several hundred chars. 240 keeps the gist.
_DESC_TRUNCATE = 240

# Default / max result counts for search_photos. The model can ask for more,
# but a hard cap keeps a runaway request from dumping the whole library into
# the context window.
_SEARCH_DEFAULT_LIMIT = 30
_SEARCH_MAX_LIMIT = 100

# Default / max rows for the grounding list_* tools.
_LIST_DEFAULT_LIMIT = 50
_LIST_MAX_LIMIT = 500

_THUMB_SIZE = 600  # px long edge — matches web.py's thumbnail cache.


# ---------------------------------------------------------------------------
# Tool registry plumbing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """One LLM-callable tool: its schema and its handler."""
    name: str
    description: str
    parameters: dict          # JSON Schema (object) for the arguments
    handler: Callable[[PhotoDB, dict], Any]


_REGISTRY: dict[str, ToolSpec] = {}


def _register(spec: ToolSpec) -> ToolSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get_tool(name: str) -> Optional[ToolSpec]:
    return _REGISTRY.get(name)


def all_tools() -> list[ToolSpec]:
    """Every registered tool, in a stable registration order."""
    return list(_REGISTRY.values())


def call_tool(db: PhotoDB, name: str, args: Optional[dict] = None) -> Any:
    """Dispatch a tool call. Raises KeyError for an unknown tool name."""
    spec = _REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"unknown tool: {name!r}")
    return spec.handler(db, args or {})


def openai_tools(include_images: bool = True) -> list[dict]:
    """Project the registry into OpenAI/LM-Studio ``tools=[...]`` format.

    ``include_images=False`` drops ``get_photo_image`` — used when image
    returns are disabled so the model is never told the tool exists.
    """
    out = []
    for spec in _REGISTRY.values():
        if not include_images and spec.name == "get_photo_image":
            continue
        out.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        })
    return out


def mcp_tools(include_images: bool = True) -> list[dict]:
    """Project the registry into the shape an MCP ``list_tools`` needs:
    ``[{name, description, inputSchema}]``. The low-level MCP server wraps
    these in ``types.Tool(...)``.
    """
    out = []
    for spec in _REGISTRY.values():
        if not include_images and spec.name == "get_photo_image":
            continue
        out.append({
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.parameters,
        })
    return out


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _clamp(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _truncate(text: Optional[str], n: int = _DESC_TRUNCATE) -> Optional[str]:
    if not text:
        return text
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _coerce_str_list(value) -> list[str]:
    """Coerce a tool argument into a list of strings.

    Tolerates models that JSON-encode the array as a *string* (small models
    often emit ``people='["Calvin"]'`` instead of ``["Calvin"]``) or pass a
    bare scalar. Correct reasoning shouldn't be penalized by a serialization
    quirk — both the agent and the MCP server route through here.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if str(v).strip()]
            except (ValueError, TypeError):
                pass
        return [s] if s else []
    return [str(value)]


def _json_array(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (ValueError, TypeError):
        return []


def _compact_hit(photo: dict) -> dict:
    """The per-result shape handed to the model — small on purpose."""
    pid = photo.get("id") or photo.get("photo_id")
    return {
        "id": pid,
        "filename": photo.get("filename"),
        "date_taken": photo.get("date_taken"),
        "place_name": photo.get("place_name"),
        "description": _truncate(photo.get("description")),
        "categories": _json_array(photo.get("categories")),
        "aesthetic_score": photo.get("aesthetic_score"),
        "score": photo.get("rrf_score") if photo.get("rrf_score") is not None
                 else photo.get("score"),
        "thumbnail_url": f"/api/photos/{pid}/thumbnail" if pid else None,
    }


def _resolve_person_names(db: PhotoDB, names: list[str]) -> tuple[list[int], list[str]]:
    """Map a list of person names to ids (case-insensitive, exact).

    Returns ``(ids, unresolved_names)`` so the caller can tell the model
    which names didn't match anything registered.
    """
    ids: list[int] = []
    unresolved: list[str] = []
    for raw in names:
        name = (raw or "").strip()
        if not name:
            continue
        row = db.conn.execute(
            "SELECT id FROM persons WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        if row:
            ids.append(row["id"])
        else:
            unresolved.append(name)
    return ids, unresolved


# ---------------------------------------------------------------------------
# Tool: get_library_overview
# ---------------------------------------------------------------------------

def _h_get_library_overview(db: PhotoDB, args: dict) -> dict:
    c = db.conn
    total = c.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"]
    # GLOB-filter to well-formed YYYY-MM-DD values: a handful of production
    # rows have corrupt date_taken (stray control bytes) that MIN/MAX would
    # otherwise surface as a garbage date range to the model.
    drange = c.execute(
        "SELECT MIN(date_taken) AS lo, MAX(date_taken) AS hi FROM photos "
        "WHERE date_taken GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*'"
    ).fetchone()

    def _count(where: str) -> int:
        return c.execute(f"SELECT COUNT(*) AS n FROM photos WHERE {where}").fetchone()["n"]

    return {
        "total_photos": total,
        "date_taken_min": drange["lo"],
        "date_taken_max": drange["hi"],
        "described": _count("description IS NOT NULL"),
        "categorized": _count("categories IS NOT NULL AND categories != '[]'"),
        "with_gps": _count("gps_lat IS NOT NULL"),
        "quality_scored": _count("aesthetic_score IS NOT NULL"),
        "registered_people": c.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"],
        "photos_with_faces": c.execute(
            "SELECT COUNT(DISTINCT photo_id) AS n FROM faces").fetchone()["n"],
    }


_register(ToolSpec(
    name="get_library_overview",
    description=(
        "Get a high-level summary of the photo library: total photo count, the "
        "earliest and latest photo dates, how many photos have descriptions / "
        "categories / GPS / quality scores, and how many people are registered. "
        "ALWAYS call this first when starting a new search session — it tells you "
        "what data is available to filter on and what date range exists."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_h_get_library_overview,
))


# ---------------------------------------------------------------------------
# Tool: list_people
# ---------------------------------------------------------------------------

def _h_list_people(db: PhotoDB, args: dict) -> dict:
    limit = _clamp(args.get("limit"), _LIST_DEFAULT_LIMIT, 1, _LIST_MAX_LIMIT)
    q = (args.get("query") or "").strip()
    sql = (
        "SELECT p.name, COUNT(DISTINCT f.photo_id) AS photo_count "
        "FROM persons p LEFT JOIN faces f ON f.person_id = p.id "
    )
    params: list = []
    if q:
        sql += "WHERE LOWER(p.name) LIKE LOWER(?) "
        params.append(f"%{q}%")
    sql += "GROUP BY p.id ORDER BY photo_count DESC LIMIT ?"
    params.append(limit)
    rows = db.conn.execute(sql, params).fetchall()
    return {"people": [{"name": r["name"], "photo_count": r["photo_count"]} for r in rows]}


_register(ToolSpec(
    name="list_people",
    description=(
        "List the people registered for face search, with how many photos each "
        "appears in. You MUST call this before filtering a search by a person's "
        "name — only names returned here can be used in search_photos' `people` "
        "argument. Optionally pass `query` to substring-filter the names."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Case-insensitive substring to filter names by."},
            "limit": {"type": "integer", "description": "Max people to return (default 50)."},
        },
        "additionalProperties": False,
    },
    handler=_h_list_people,
))


# ---------------------------------------------------------------------------
# Tool: list_places
# ---------------------------------------------------------------------------

def _h_list_places(db: PhotoDB, args: dict) -> dict:
    limit = _clamp(args.get("limit"), _LIST_DEFAULT_LIMIT, 1, _LIST_MAX_LIMIT)
    q = (args.get("query") or "").strip()
    like = f"%{q}%" if q else None

    out: list[dict] = []

    # Flat reverse-geocoded place_name (always present).
    if like:
        rows = db.conn.execute(
            "SELECT place_name AS v, COUNT(*) AS c FROM photos "
            "WHERE place_name IS NOT NULL AND place_name LIKE ? "
            "GROUP BY place_name ORDER BY c DESC LIMIT ?", (like, limit)).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT place_name AS v, COUNT(*) AS c FROM photos "
            "WHERE place_name IS NOT NULL "
            "GROUP BY place_name ORDER BY c DESC LIMIT ?", (limit,)).fetchall()
    out.extend({"value": r["v"], "kind": "place_name", "count": r["c"]} for r in rows)

    # Structured columns (schema v19) — graceful if not yet migrated.
    for col in ("country", "admin1", "admin2", "locality"):
        try:
            if like:
                rows = db.conn.execute(
                    f"SELECT {col} AS v, COUNT(*) AS c FROM photos "
                    f"WHERE {col} IS NOT NULL AND {col} LIKE ? "
                    f"GROUP BY {col} ORDER BY c DESC LIMIT ?", (like, limit)).fetchall()
            else:
                rows = db.conn.execute(
                    f"SELECT {col} AS v, COUNT(*) AS c FROM photos "
                    f"WHERE {col} IS NOT NULL "
                    f"GROUP BY {col} ORDER BY c DESC LIMIT ?", (limit,)).fetchall()
            out.extend({"value": r["v"], "kind": col, "count": r["c"]} for r in rows)
        except Exception:
            pass  # column absent on an old DB — skip silently.

    return {"places": out}


_register(ToolSpec(
    name="list_places",
    description=(
        "List place values present in the library — reverse-geocoded place "
        "names plus structured country / admin1 (state) / admin2 (county) / "
        "locality values — with photo counts. Use this to ground a location "
        "filter: resolve a vague phrase like 'our Italy trip' to the actual "
        "strings the database holds before passing one to search_photos' "
        "`location`. Optionally pass `query` to substring-filter."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Case-insensitive substring to filter places by."},
            "limit": {"type": "integer",
                      "description": "Max values per kind to return (default 50)."},
        },
        "additionalProperties": False,
    },
    handler=_h_list_places,
))


# ---------------------------------------------------------------------------
# Tool: list_vocab
# ---------------------------------------------------------------------------

_VOCAB_COLUMNS = {"categories": "categories",
                  "visual_tags": "visual_tags",
                  "keywords": "keywords"}


def _aggregate_json_vocab(db: PhotoDB, column: str, q: str, limit: int) -> list[dict]:
    """Count distinct values across a JSON-array column. This is a full scan
    of the non-null rows (the values live inside JSON, so SQL can't GROUP
    them) — acceptable for a grounding tool the model calls a few times, not
    on the hot search path.
    """
    counts: dict[str, int] = {}
    q_lower = q.lower() if q else None
    for row in db.conn.execute(
        f"SELECT {column} AS raw FROM photos "
        f"WHERE {column} IS NOT NULL AND {column} != '[]'"
    ):
        for item in _json_array(row["raw"]):
            if not isinstance(item, str):
                continue
            if q_lower and q_lower not in item.lower():
                continue
            counts[item] = counts.get(item, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"value": v, "count": c} for v, c in ranked[:limit]]


def _h_list_vocab(db: PhotoDB, args: dict) -> dict:
    limit = _clamp(args.get("limit"), _LIST_DEFAULT_LIMIT, 1, _LIST_MAX_LIMIT)
    q = (args.get("query") or "").strip()
    kind = (args.get("kind") or "").strip().lower()
    kinds = [kind] if kind in _VOCAB_COLUMNS else list(_VOCAB_COLUMNS)
    return {k: _aggregate_json_vocab(db, _VOCAB_COLUMNS[k], q, limit) for k in kinds}


_register(ToolSpec(
    name="list_vocab",
    description=(
        "List the controlled vocabulary the library actually uses: distinct "
        "`categories` (content categories), `visual_tags` (visual-quality tags), "
        "and `keywords`, each with photo counts. Call this before using "
        "search_photos' `category` / `visual_tag` / `keyword` arguments so you "
        "filter on terms that exist instead of guessing. Pass `kind` to restrict "
        "to one of categories|visual_tags|keywords, and `query` to substring-filter."
    ),
    parameters={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["categories", "visual_tags", "keywords"],
                     "description": "Restrict to one vocabulary; omit for all three."},
            "query": {"type": "string",
                      "description": "Case-insensitive substring to filter values by."},
            "limit": {"type": "integer",
                      "description": "Max values per kind to return (default 50)."},
        },
        "additionalProperties": False,
    },
    handler=_h_list_vocab,
))


# ---------------------------------------------------------------------------
# Tool: search_photos
# ---------------------------------------------------------------------------

_VALID_SORTS = ("date_desc", "date_asc", "quality_desc", "relevance", "subject")


def _subject_ranked_search(db: PhotoDB, args: dict, person_ids: list, limit: int) -> dict:
    """Flat top-N search ranked by how FOREGROUND the person is (prominence
    sweet-spot band, then quality) — for 'best photos of X where X is the
    foreground subject' in a single scope (no per-bucket spread). Same logic as
    representatives(rank_by='subject') but a flat list, not one-per-bucket."""
    where, params = _build_filter_sql(db, args)
    ph = ",".join("?" * len(person_ids))
    prom_join = (
        f"JOIN (SELECT f.photo_id, MAX("
        f"((f.bbox_right - f.bbox_left) * (f.bbox_bottom - f.bbox_top) * 1.0) "
        f"/ (NULLIF(po.image_width,0) * NULLIF(po.image_height,0))) AS _prom "
        f"FROM faces f JOIN photos po ON po.id = f.photo_id "
        f"WHERE f.person_id IN ({ph}) GROUP BY f.photo_id) pm ON pm.photo_id = photos.id")
    order = (f"(pm._prom BETWEEN {_SUBJECT_PROM_MIN} AND {_SUBJECT_PROM_MAX}) DESC, "
             f"aesthetic_score IS NULL, aesthetic_score DESC, pm._prom DESC")
    dedupe = args.get("dedupe", True)
    fetch = min(limit * _DEDUP_OVERFETCH, 600) if dedupe else limit
    try:
        total = db.conn.execute(
            f"SELECT COUNT(*) FROM photos {prom_join} WHERE {where}",
            (*person_ids, *params)).fetchone()[0]
        rows = [dict(r) for r in db.conn.execute(
            f"SELECT photos.*, pm._prom AS _prom FROM photos {prom_join} "
            f"WHERE {where} ORDER BY {order} LIMIT ?",
            (*person_ids, *params, fetch)).fetchall()]
    except Exception as exc:
        return {"error": f"subject search failed: {exc}"}
    if dedupe:
        rows = _dedupe_ranked(rows, limit)
    results = []
    for d in rows:
        hit = _compact_hit(d)
        if d.get("_prom") is not None:
            hit["subject_prominence"] = round(d["_prom"], 4)
        results.append(hit)
    return {"total": total, "returned": len(results), "sort": "subject",
            "deduped": dedupe, "results": results}


def _h_search_photos(db: PhotoDB, args: dict) -> dict:
    from .search import search_combined  # lazy — pulls in CLIP.

    limit = _clamp(args.get("limit"), _SEARCH_DEFAULT_LIMIT, 1, _SEARCH_MAX_LIMIT)
    raw_sort = args.get("sort")

    people = _coerce_str_list(args.get("people"))
    person_ids, unresolved = _resolve_person_names(db, people) if people else ([], [])

    # Subject sort: flat list ranked by the person's foreground prominence.
    if raw_sort == "subject" and person_ids:
        return _subject_ranked_search(db, args, person_ids, limit)
    sort = raw_sort if raw_sort in _VALID_SORTS and raw_sort != "subject" else None

    # If the model asked for people that don't exist, don't silently run an
    # unfiltered search — return the miss so it can correct via list_people.
    if people and not person_ids:
        return {
            "total": 0,
            "results": [],
            "unresolved_people": unresolved,
            "note": "None of the requested people are registered. Call list_people "
                    "to see valid names.",
        }

    min_quality = args.get("min_quality")
    try:
        min_quality = float(min_quality) if min_quality is not None else None
    except (TypeError, ValueError):
        min_quality = None

    # Default sort: relevance when there's a semantic query, else newest-first.
    query = (args.get("query") or "").strip() or None
    if sort is None:
        sort = "relevance" if query else "date_desc"

    results, total = search_combined(
        db=db,
        query=query,
        person_ids=person_ids or None,
        location=(args.get("location") or "").strip() or None,
        date_from=(args.get("date_from") or "").strip() or None,
        date_to=(args.get("date_to") or "").strip() or None,
        color=(args.get("color") or "").strip() or None,
        category=(args.get("category") or "").strip() or None,
        visual_tag=(args.get("visual_tag") or "").strip() or None,
        keyword=(args.get("keyword") or "").strip() or None,
        min_quality=min_quality,
        sort=sort,
        limit=limit,
        with_total=True,
    )

    out: dict = {
        "total": total,
        "returned": len(results),
        "sort": sort,
        "results": [_compact_hit(r) for r in results],
    }
    if unresolved:
        out["unresolved_people"] = unresolved
    return out


_register(ToolSpec(
    name="search_photos",
    description=(
        "Search the photo library. Fill in only the filters you've inferred from "
        "the user's request; omit the rest. Filters combine as an AND-intersection. "
        "`query` is free-text semantic search (CLIP) — use it for visual content "
        "('sunset', 'birthday cake'); omit it for pure metadata searches. `people` "
        "is a list of registered names (validate with list_people first) and "
        "matches photos containing ALL of them. `location` matches a place/region "
        "(validate with list_places). `category`/`visual_tag`/`keyword` filter on "
        "the controlled vocabulary (validate with list_vocab). Dates are "
        "YYYY-MM-DD. Returns the matching photos plus `total` — if `total` is far "
        "larger or smaller than expected, adjust the filters and search again."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Free-text semantic (CLIP) query for visual "
                      "content (e.g. 'sunset', 'birthday cake'). Supports "
                      "exclusion: prefix a term with '-' or say 'no <thing>' to "
                      "filter it out, e.g. 'landscape -people' or 'beach no "
                      "crowds'. Do NOT put ranking words like 'best'/'top' here "
                      "— use the sort/min_quality fields for those."},
            "people": {"type": "array", "items": {"type": "string"},
                       "description": "Registered person names; matches photos with ALL of them."},
            "location": {"type": "string",
                         "description": "Place/region name (see list_places)."},
            "date_from": {"type": "string", "description": "Earliest date, YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Latest date, YYYY-MM-DD."},
            "color": {"type": "string", "description": "Dominant color name or #hex."},
            "category": {"type": "string", "description": "Exact content category (see list_vocab)."},
            "visual_tag": {"type": "string", "description": "Visual-quality tag (see list_vocab)."},
            "keyword": {"type": "string", "description": "Keyword substring (see list_vocab)."},
            "min_quality": {"type": "number",
                            "description": "Minimum aesthetic score, 1-10."},
            "sort": {"type": "string", "enum": list(_VALID_SORTS),
                     "description": "Result order; defaults to relevance for a query, "
                                    "else date_desc. 'subject' (with a `people` filter) "
                                    "ranks by how foreground/prominent that person is — "
                                    "for 'best photos of X where X is the main subject'."},
            "limit": {"type": "integer",
                      "description": f"Max results (default {_SEARCH_DEFAULT_LIMIT}, "
                                     f"max {_SEARCH_MAX_LIMIT})."},
        },
        "additionalProperties": False,
    },
    handler=_h_search_photos,
))


# ---------------------------------------------------------------------------
# Tool: summarize  (faceting / aggregation)
# ---------------------------------------------------------------------------

_STRUCT_LOC_CACHE: dict = {}


def _has_structured_location(db) -> bool:
    key = getattr(db, "db_path", "?")
    if key not in _STRUCT_LOC_CACHE:
        try:
            db.conn.execute("SELECT country FROM photos LIMIT 1")
            _STRUCT_LOC_CACHE[key] = True
        except Exception:
            _STRUCT_LOC_CACHE[key] = False
    return _STRUCT_LOC_CACHE[key]


def _build_filter_sql(db, args: dict) -> tuple[str, list]:
    """Build a WHERE clause + params from the structured filters (the same set
    search_photos accepts, minus the free-text CLIP `query`). Mirrors
    search_combined's structured semantics closely enough for faceting."""
    clauses: list[str] = []
    params: list = []

    people = _coerce_str_list(args.get("people"))
    if people:
        ids, _ = _resolve_person_names(db, people)
        if ids:
            ph = ",".join("?" * len(ids))
            clauses.append(
                f"photos.id IN (SELECT photo_id FROM faces WHERE person_id IN ({ph}) "
                f"GROUP BY photo_id HAVING COUNT(DISTINCT person_id) = ?)")
            params.extend(ids)
            params.append(len(ids))
        else:
            clauses.append("0")  # unknown person → match nothing

    loc = (args.get("location") or "").strip()
    if loc:
        from .geocode import country_name_to_code
        sub = ["place_name LIKE ?"]
        params.append(f"%{loc}%")
        code = country_name_to_code(loc)
        if code:
            sub.append("place_name LIKE ?")
            params.append(f"%, {code}")
        if _has_structured_location(db):
            for col in ("country", "admin1", "admin2", "locality"):
                sub.append(f"LOWER({col}) = LOWER(?)")
                params.append(loc)
        clauses.append("(" + " OR ".join(sub) + ")")

    df = (args.get("date_from") or "").strip()
    dt = (args.get("date_to") or "").strip()
    if df:
        clauses.append("date_taken >= ?")
        params.append(df)
    if dt:
        clauses.append("date_taken <= ?")
        params.append(dt + " 23:59:59")

    # JSON-array token filters.
    for key, col in (("category", "categories"), ("visual_tag", "visual_tags")):
        v = (args.get(key) or "").strip()
        if v:
            clauses.append(f"{col} LIKE ?")
            params.append(f'%"{v}"%')
    kw = (args.get("keyword") or "").strip()
    if kw:
        clauses.append("keywords LIKE ?")
        params.append(f"%{kw}%")

    mq = args.get("min_quality")
    if mq is not None:
        try:
            clauses.append("aesthetic_score >= ?")
            params.append(float(mq))
        except (TypeError, ValueError):
            pass

    return (" AND ".join(clauses) if clauses else "1"), params


# group_by → (SELECT expression, extra WHERE to keep buckets clean)
_GROUP_EXPR = {
    "year": ("substr(date_taken,1,4)", "date_taken GLOB '[0-9][0-9][0-9][0-9]-*'"),
    "month": ("substr(date_taken,1,7)", "date_taken GLOB '[0-9][0-9][0-9][0-9]-*'"),
    "location": ("place_name", "place_name IS NOT NULL"),
    "camera_model": ("camera_model", "camera_model IS NOT NULL"),
}


def _h_summarize(db: PhotoDB, args: dict) -> dict:
    group_by = (args.get("group_by") or "year").strip().lower()
    limit = _clamp(args.get("limit"), 50, 1, 500)
    where, params = _build_filter_sql(db, args)

    if group_by == "person":
        sql = ("SELECT pe.name AS bucket, COUNT(DISTINCT photos.id) AS n "
               "FROM photos JOIN faces f ON f.photo_id = photos.id "
               "JOIN persons pe ON pe.id = f.person_id "
               f"WHERE {where} GROUP BY pe.id ORDER BY n DESC LIMIT ?")
    elif group_by in _GROUP_EXPR:
        expr, extra = _GROUP_EXPR[group_by]
        order = "bucket DESC" if group_by in ("year", "month") else "n DESC"
        sql = (f"SELECT {expr} AS bucket, COUNT(*) AS n FROM photos "
               f"WHERE ({where}) AND {extra} GROUP BY bucket ORDER BY {order} LIMIT ?")
    else:
        return {"error": f"unsupported group_by: {group_by!r}. "
                         "Use year | month | location | person | camera_model."}

    try:
        rows = db.conn.execute(sql, (*params, limit)).fetchall()
    except Exception as exc:
        return {"error": f"summarize failed: {exc}"}

    buckets = [{"value": r["bucket"], "count": r["n"]}
               for r in rows if r["bucket"] is not None]
    return {
        "group_by": group_by,
        # NB: person buckets can overlap (a photo with 2 people counts twice),
        # so this is the sum of buckets, not necessarily distinct photos.
        "total_in_buckets": sum(b["count"] for b in buckets),
        "buckets": buckets,
    }


_register(ToolSpec(
    name="summarize",
    description=(
        "Aggregate/COUNT photos by a dimension instead of returning them — for "
        "'which years / how many / when / who / how often' questions. Accepts "
        "the same STRUCTURED filters as search_photos (people, location, "
        "date_from/to, category, visual_tag, keyword, min_quality) but NOT the "
        "free-text `query`. `group_by` is one of: year, month, location, person, "
        "camera_model. Returns counts per bucket. Use it to answer multi-step "
        "questions: e.g. for 'which year were we in both New York and France', "
        "call summarize(location='New York', group_by='year') and "
        "summarize(location='France', group_by='year'), intersect the years, "
        "then search_photos for that year."
    ),
    parameters={
        "type": "object",
        "properties": {
            "group_by": {"type": "string",
                         "enum": ["year", "month", "location", "person", "camera_model"],
                         "description": "Dimension to count by (default year)."},
            "people": {"type": "array", "items": {"type": "string"},
                       "description": "Registered names; matches photos with ALL of them."},
            "location": {"type": "string", "description": "Place/region name."},
            "date_from": {"type": "string", "description": "Earliest date, YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Latest date, YYYY-MM-DD."},
            "category": {"type": "string", "description": "Exact content category."},
            "visual_tag": {"type": "string", "description": "Visual-quality tag."},
            "keyword": {"type": "string", "description": "Keyword substring."},
            "min_quality": {"type": "number", "description": "Minimum aesthetic score, 1-10."},
            "limit": {"type": "integer", "description": "Max buckets (default 50)."},
        },
        "additionalProperties": False,
    },
    handler=_h_summarize,
))


# ---------------------------------------------------------------------------
# Tool: representatives  (top-N best per bucket — "one per year")
# ---------------------------------------------------------------------------

def _h_representatives(db: PhotoDB, args: dict) -> dict:
    bucket = (args.get("bucket") or "year").strip().lower()
    n = _clamp(args.get("n"), 1, 1, 10)
    rank_by = (args.get("rank_by") or "quality").strip().lower()
    where, params = _build_filter_sql(db, args)
    order_quality = "aesthetic_score IS NULL, aesthetic_score DESC"

    # Subject-prominence ranking: rank each bucket by how FOREGROUND the named
    # person is (their largest face's area as a fraction of the image), then by
    # fewer total faces, then quality. Needs a person filter. Excludes the
    # background/incidental shots that pure-aesthetic ranking surfaces.
    person_ids: list = []
    if rank_by == "subject":
        person_ids, _ = _resolve_person_names(db, _coerce_str_list(args.get("people")))
    use_subject = rank_by == "subject" and person_ids and bucket in _GROUP_EXPR

    prom_join, prom_select, prom_params, order_within = "", "", [], order_quality
    if use_subject:
        ph = ",".join("?" * len(person_ids))
        prom_join = (
            f"JOIN (SELECT f.photo_id, MAX("
            f"((f.bbox_right - f.bbox_left) * (f.bbox_bottom - f.bbox_top) * 1.0) "
            f"/ (NULLIF(po.image_width,0) * NULLIF(po.image_height,0))) AS _prom "
            f"FROM faces f JOIN photos po ON po.id = f.photo_id "
            f"WHERE f.person_id IN ({ph}) GROUP BY f.photo_id) pm ON pm.photo_id = photos.id")
        prom_select = ", pm._prom AS _prom"
        # Prefer photos where the person is a featured subject in a REAL photo —
        # a prominence "sweet spot" band — then rank those by aesthetic quality.
        # Pure face-size ranking over-picks extreme close-ups (haircut selfies);
        # plain quality picks background shots. Band-then-quality avoids both.
        order_within = (
            f"(pm._prom BETWEEN {_SUBJECT_PROM_MIN} AND {_SUBJECT_PROM_MAX}) DESC, "
            f"{order_quality}, pm._prom DESC")
        prom_params = list(person_ids)

    if bucket == "person":
        inner = (f"SELECT photos.*, pe.name AS _bucket, "
                 f"ROW_NUMBER() OVER (PARTITION BY pe.id ORDER BY {order_quality}) AS _rn "
                 f"FROM photos JOIN faces f ON f.photo_id = photos.id "
                 f"JOIN persons pe ON pe.id = f.person_id WHERE {where}")
        outer_order = "_bucket"
        full_params = (*params, n)
    elif bucket in _GROUP_EXPR:
        expr, extra = _GROUP_EXPR[bucket]
        inner = (f"SELECT photos.*, {expr} AS _bucket{prom_select}, "
                 f"ROW_NUMBER() OVER (PARTITION BY {expr} ORDER BY {order_within}) AS _rn "
                 f"FROM photos {prom_join} WHERE ({where}) AND {extra}")
        outer_order = "_bucket DESC" if bucket in ("year", "month") else "_bucket"
        full_params = (*prom_params, *params, n)
    else:
        return {"error": f"unsupported bucket: {bucket!r}. "
                         "Use year | month | location | person | camera_model."}

    dedupe = args.get("dedupe", True)
    # Over-fetch per bucket so dedup can still return n distinct photos.
    fetch_rn = min(n * _DEDUP_OVERFETCH, 60) if dedupe and n > 1 else n
    sql = f"SELECT * FROM ({inner}) WHERE _rn <= ? ORDER BY {outer_order}, _rn LIMIT 2000"
    try:
        params_with_fetch = full_params[:-1] + (fetch_rn,)
        rows = [dict(r) for r in db.conn.execute(sql, params_with_fetch).fetchall()]
    except Exception as exc:
        return {"error": f"representatives failed: {exc}"}

    if dedupe and n > 1:
        rows = _dedupe_ranked(rows, n)

    results = []
    for d in rows:
        hit = _compact_hit(d)
        hit["bucket"] = d.get("_bucket")
        if d.get("_prom") is not None:
            hit["subject_prominence"] = round(d["_prom"], 4)
        results.append(hit)
    return {
        "bucket": bucket,
        "n_per_bucket": n,
        "ranked_by": "subject" if use_subject else "quality",
        "deduped": bool(dedupe and n > 1),
        "buckets": len({h["bucket"] for h in results}),
        "returned": len(results),
        "results": results,
    }


_register(ToolSpec(
    name="representatives",
    description=(
        "Return the top-N best (by aesthetic quality) photos PER bucket — e.g. "
        "the single best photo of EACH year ('one per year'), a few from each "
        "trip, or one of each person. Use this when the user wants a "
        "representative SPREAD across years/months/locations/people, NOT a flat "
        "ranked list (search_photos can't do per-bucket selection). Same "
        "structured filters as search_photos (no free-text `query`). `bucket` is "
        "year|month|location|person|camera_model; `n` is photos per bucket "
        "(default 1). `rank_by`: 'quality' (default, aesthetic) or 'subject' — "
        "with 'subject' (+ a `people` filter) each bucket is ranked by how "
        "FOREGROUND/prominent that person is (their face size in frame), which "
        "excludes shots where they're in the background. Examples: 'best photo "
        "of Matt, one per year' → people=['Matt'], bucket='year', n=1. 'one per "
        "year where Matt is the foreground subject' → add rank_by='subject'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "bucket": {"type": "string",
                       "enum": ["year", "month", "location", "person", "camera_model"],
                       "description": "Group photos by this; return the best N of each."},
            "n": {"type": "integer", "description": "Photos per bucket (default 1, max 10)."},
            "rank_by": {"type": "string", "enum": ["quality", "subject"],
                        "description": "Rank each bucket by aesthetic quality (default) "
                        "or by how foreground/prominent the filtered person is."},
            "dedupe": {"type": "boolean",
                       "description": "Skip near-duplicate/burst shots (default true); "
                       "ensures the N per bucket are distinct moments."},
            "people": {"type": "array", "items": {"type": "string"},
                       "description": "Registered names; matches photos with ALL of them."},
            "location": {"type": "string", "description": "Place/region name."},
            "date_from": {"type": "string", "description": "Earliest date, YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Latest date, YYYY-MM-DD."},
            "category": {"type": "string", "description": "Exact content category."},
            "visual_tag": {"type": "string", "description": "Visual-quality tag."},
            "keyword": {"type": "string", "description": "Keyword substring."},
            "min_quality": {"type": "number", "description": "Minimum aesthetic score, 1-10."},
        },
        "additionalProperties": False,
    },
    handler=_h_representatives,
))


# ---------------------------------------------------------------------------
# Tool: get_photo
# ---------------------------------------------------------------------------

def _h_get_photo(db: PhotoDB, args: dict) -> dict:
    pid = args.get("photo_id")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return {"error": "photo_id must be an integer"}

    photo = db.get_photo(pid)
    if not photo:
        return {"error": f"no photo with id {pid}"}

    people = [r["name"] for r in db.conn.execute(
        "SELECT DISTINCT p.name FROM faces f JOIN persons p ON p.id = f.person_id "
        "WHERE f.photo_id = ? ORDER BY p.name", (pid,)).fetchall()]
    face_count = db.conn.execute(
        "SELECT COUNT(*) AS n FROM faces WHERE photo_id = ?", (pid,)).fetchone()["n"]

    return {
        "id": pid,
        "filename": photo.get("filename"),
        "date_taken": photo.get("date_taken"),
        "description": photo.get("description"),
        "categories": _json_array(photo.get("categories")),
        "visual_tags": _json_array(photo.get("visual_tags")),
        "keywords": _json_array(photo.get("keywords")),
        "place_name": photo.get("place_name"),
        "gps_lat": photo.get("gps_lat"),
        "gps_lon": photo.get("gps_lon"),
        "aesthetic_score": photo.get("aesthetic_score"),
        "camera_model": photo.get("camera_model"),
        "people": people,
        "face_count": face_count,
        "stack": db.get_photo_stack(pid),
        "thumbnail_url": f"/api/photos/{pid}/thumbnail",
    }


_register(ToolSpec(
    name="get_photo",
    description=(
        "Get full detail on one photo by id: its description, categories / "
        "visual tags / keywords, EXIF, GPS and place, the registered people in "
        "it, and any burst-stack info. Use this to drill into a specific result "
        "before answering."
    ),
    parameters={
        "type": "object",
        "properties": {
            "photo_id": {"type": "integer", "description": "The photo id."},
        },
        "required": ["photo_id"],
        "additionalProperties": False,
    },
    handler=_h_get_photo,
))


# ---------------------------------------------------------------------------
# Tool: rerank_photos  (VLM re-ranking — a vision model looks at each candidate)
# ---------------------------------------------------------------------------

_RERANK_MAX_K = 24
_RERANK_WORKERS = 4

# Subject-prominence "sweet spot": a person's largest face filling this fraction
# of the frame reads as a featured subject in a real photo. Below ≈ background;
# above ≈ a face-filling close-up (e.g. a haircut/mirror selfie). Calibrated
# from the face-prominence distribution (~p55–p95) on the 163k library.
_SUBJECT_PROM_MIN = 0.04
_SUBJECT_PROM_MAX = 0.22

# Two photos taken within this many seconds are treated as the same moment
# (burst / near-duplicate) for dedup. Bursts are seconds apart; this also
# catches "took 3 of the same pose" without dropping genuinely different shots.
_DEDUP_WINDOW_S = 45
_DEDUP_OVERFETCH = 6  # fetch this × n per bucket so dedup still yields n


def _parse_dt(s):
    import datetime
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(str(s).replace(" ", "T")[:19])
    except (ValueError, TypeError):
        return None


def _dedupe_ranked(rows: list, n: int) -> list:
    """Take up to `n` per bucket from ranked (best-first) photo rows, skipping
    near-duplicates: identical file_hash, or taken within _DEDUP_WINDOW_S of an
    already-kept photo in the same bucket. Rows must carry date_taken,
    file_hash, and _bucket (None for a flat, single-bucket list)."""
    out: list = []
    per: dict = {}
    for d in rows:
        sel = per.setdefault(d.get("_bucket"), [])
        if len(sel) >= n:
            continue
        h = d.get("file_hash")
        if h and any(s.get("file_hash") == h for s in sel):
            continue
        t = _parse_dt(d.get("date_taken"))
        if t and any((st := _parse_dt(s.get("date_taken"))) is not None
                     and abs((t - st).total_seconds()) < _DEDUP_WINDOW_S for s in sel):
            continue
        sel.append(d)
        out.append(d)
    return out


def _thumb_b64(db: PhotoDB, photo_id: int) -> Optional[str]:
    """Base64 JPEG thumbnail for a photo. On a replica (no local originals)
    fetch from the NAS web API; otherwise generate locally."""
    import base64
    nas = (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/")
    if nas:
        import urllib.request
        try:
            req = urllib.request.Request(f"{nas}/api/photos/{photo_id}/thumbnail",
                                         headers={"User-Agent": "photosearch-rerank"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return base64.b64encode(r.read()).decode("ascii")
        except Exception:
            return None
    photo = db.get_photo(photo_id)
    if not photo:
        return None
    try:
        with open(_thumb_path(db, photo), "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except Exception:
        return None


def _vision_score(base_url: str, model: str, image_b64: str, criteria: str) -> Optional[dict]:
    """Ask the vision model how well one image matches `criteria`. Returns
    {"score": 0-1, "reason": str} or None on failure."""
    import urllib.request
    prompt = (
        "You are scoring how well a photo matches a request.\n"
        f'Request: "{criteria}".\n'
        "Look at the image and rate from 0.0 to 1.0 how well it matches "
        "(1.0 = excellent match, 0.0 = does not match at all). Judge the actual "
        "visible content. Reply with ONLY a JSON object: "
        '{"score": <number 0-1>, "reason": "<one short phrase>"}.'
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}],
        "max_tokens": 150,
        "temperature": 0,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"] or ""
    except Exception:
        return None
    # Extract the JSON object from the (possibly fenced) reply.
    s = content.strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(s[a:b + 1])
        score = float(obj.get("score"))
    except (ValueError, TypeError):
        return None
    return {"score": max(0.0, min(1.0, score)), "reason": str(obj.get("reason", ""))[:120]}


def _h_rerank_photos(db: PhotoDB, args: dict) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    ids = args.get("photo_ids") or []
    if isinstance(ids, str):
        ids = _coerce_str_list(ids)
    try:
        ids = [int(x) for x in ids][:_RERANK_MAX_K]
    except (TypeError, ValueError):
        return {"error": "photo_ids must be a list of integers"}
    criteria = (args.get("criteria") or "").strip()
    if not ids or not criteria:
        return {"error": "photo_ids and criteria are required"}

    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    model = os.environ.get("PHOTOSEARCH_LLM_VISUAL_MODEL")
    photos = {pid: db.get_photo(pid) for pid in ids}

    def _compact(pid, score=None, reason=None):
        p = photos.get(pid)
        hit = _compact_hit(p) if p else {"id": pid}
        if score is not None:
            hit["rerank_score"] = round(score, 3)
            hit["rerank_reason"] = reason
        return hit

    # No vision backend configured → return the candidates in their input order,
    # unscored (graceful fallback; the agent still gets results).
    if not (base and model):
        return {"reranked": False,
                "note": "no vision model configured (PHOTOSEARCH_LLM_VISUAL_MODEL); "
                        "returning candidates unranked",
                "criteria": criteria,
                "results": [_compact(pid) for pid in ids]}

    def _score(pid):
        b64 = _thumb_b64(db, pid)
        if not b64:
            return pid, None
        return pid, _vision_score(base, model, b64, criteria)

    scores: dict = {}
    with ThreadPoolExecutor(max_workers=_RERANK_WORKERS) as pool:
        for pid, sc in pool.map(_score, ids):
            scores[pid] = sc

    # Scored photos first (by score desc), then any that failed (input order).
    scored_ids = [pid for pid in ids if scores.get(pid)]
    scored_ids.sort(key=lambda pid: scores[pid]["score"], reverse=True)
    failed_ids = [pid for pid in ids if not scores.get(pid)]

    results = ([_compact(pid, scores[pid]["score"], scores[pid]["reason"]) for pid in scored_ids]
               + [_compact(pid) for pid in failed_ids])
    return {
        "reranked": True,
        "criteria": criteria,
        "model": model,
        "scored": len(scored_ids),
        "results": results,
    }


_register(ToolSpec(
    name="rerank_photos",
    description=(
        "Re-rank a set of candidate photos by having a VISION model actually "
        "LOOK at each one and score how well it matches a visual criterion. Use "
        "this for precision — when the user wants THE specific shot or a visual "
        "judgment that metadata can't capture: 'the one where Ellie is blowing "
        "out candles', 'make sure Matt is the primary subject in the foreground, "
        "not the background', 'the sharpest / best-composed one'. Workflow: run "
        "search_photos (or representatives) first to get candidates, then call "
        "rerank_photos(photo_ids=<the ids you got>, criteria=<the visual thing to "
        "judge>). Returns the photos re-sorted best-first with a per-photo "
        "rerank_score + reason. Slower than a normal search (it inspects each "
        "image), so pass a focused candidate set (<=24)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "photo_ids": {"type": "array", "items": {"type": "integer"},
                          "description": "Candidate photo ids to re-rank (from a prior search)."},
            "criteria": {"type": "string",
                         "description": "The visual thing to judge each photo against."},
        },
        "required": ["photo_ids", "criteria"],
        "additionalProperties": False,
    },
    handler=_h_rerank_photos,
))


# ---------------------------------------------------------------------------
# Tool: get_photo_image  (gated at the adapter layer)
# ---------------------------------------------------------------------------

def _thumb_path(db: PhotoDB, photo: dict) -> str:
    """Return a cached thumbnail path for a photo, generating it on demand.

    Mirrors web.py's cache: <db_parent>/thumbnails/<id>_thumb.jpg keyed by
    photo id. Kept here (rather than imported from web.py) so the tool layer
    doesn't pull in the FastAPI app.
    """
    from PIL import Image, ImageOps

    db_parent = Path(db.db_path).resolve().parent
    thumb_dir = db_parent / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"{photo['id']}_thumb.jpg"
    if thumb_path.exists():
        return str(thumb_path)

    filepath = db.resolve_filepath(photo.get("filepath", ""))
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"original not found: {filepath}")

    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((_THUMB_SIZE, _THUMB_SIZE), Image.LANCZOS)
    img.save(thumb_path, "JPEG", quality=85)
    return str(thumb_path)


def _h_get_photo_image(db: PhotoDB, args: dict) -> dict:
    """Return a cached thumbnail *path* + mime type. The calling adapter
    decides how to encode it (MCP reads the bytes into an ImageContent; the
    web agent hands back the thumbnail URL). Whether this tool is exposed at
    all is an adapter policy — see PHOTOSEARCH_MCP_ALLOW_IMAGES.
    """
    pid = args.get("photo_id")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return {"error": "photo_id must be an integer"}

    photo = db.get_photo(pid)
    if not photo:
        return {"error": f"no photo with id {pid}"}
    try:
        path = _thumb_path(db, photo)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    return {"photo_id": pid, "thumbnail_path": path, "mime_type": "image/jpeg",
            "thumbnail_url": f"/api/photos/{pid}/thumbnail"}


_register(ToolSpec(
    name="get_photo_image",
    description=(
        "Return a thumbnail of one photo so you can visually inspect it — useful "
        "to verify or re-rank ambiguous search results. Only call this when "
        "looking at the actual pixels would change your answer; prefer the text "
        "description from get_photo when that suffices."
    ),
    parameters={
        "type": "object",
        "properties": {
            "photo_id": {"type": "integer", "description": "The photo id."},
        },
        "required": ["photo_id"],
        "additionalProperties": False,
    },
    handler=_h_get_photo_image,
))
