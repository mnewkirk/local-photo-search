"""Tests for the shared LLM tool layer (photosearch/tools.py — M24).

Exercises each tool against the conftest `db` fixture (5 photos; persons
Alex/Jamie/Sam; vocab on DSC04922) plus the schema-projection helpers and
the call_tool dispatch. Search tests stay on metadata filters (no `query`)
so they're deterministic without real CLIP inference.
"""

import pytest

from photosearch import tools


# ---------------------------------------------------------------------------
# Registry + schema projections
# ---------------------------------------------------------------------------

# Read-only tools advertised by default (include_images=True, include_writes=False).
READ_TOOLS = {
    "get_library_overview", "list_people", "list_places", "list_vocab",
    "search_photos", "summarize", "representatives", "daily_highlights",
    "group_into_chapters", "daily_scene_breakdown", "suggest_layout",
    "rerank_photos", "get_photo", "get_photo_image",
}
# M26b mutation tools — gated out of the default projections.
WRITE_TOOLS = {"set_photo_location", "set_photo_tags", "add_to_collection"}
EXPECTED_TOOLS = READ_TOOLS | WRITE_TOOLS


def test_registry_has_all_tools():
    assert {t.name for t in tools.all_tools()} == EXPECTED_TOOLS


def test_every_schema_is_a_valid_object_schema():
    for spec in tools.all_tools():
        assert spec.parameters["type"] == "object"
        assert "properties" in spec.parameters
        assert spec.description and len(spec.description) > 20


def test_openai_projection_shape():
    # Default projection hides the write tools (include_writes=False).
    fns = tools.openai_tools()
    assert {t["function"]["name"] for t in fns} == READ_TOOLS
    for t in fns:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_mcp_projection_shape():
    specs = tools.mcp_tools()
    assert {s["name"] for s in specs} == READ_TOOLS
    for s in specs:
        assert "inputSchema" in s and s["inputSchema"]["type"] == "object"


def test_include_images_flag_drops_image_tool():
    assert "get_photo_image" not in {
        t["function"]["name"] for t in tools.openai_tools(include_images=False)}
    assert "get_photo_image" not in {
        s["name"] for s in tools.mcp_tools(include_images=False)}
    # …and is present when enabled.
    assert "get_photo_image" in {
        s["name"] for s in tools.mcp_tools(include_images=True)}


def test_include_writes_flag_gates_mutation_tools():
    # Off by default in both projections…
    assert not (WRITE_TOOLS & {
        t["function"]["name"] for t in tools.openai_tools()})
    assert not (WRITE_TOOLS & {s["name"] for s in tools.mcp_tools()})
    # …and present when opted in.
    assert WRITE_TOOLS <= {
        t["function"]["name"] for t in tools.openai_tools(include_writes=True)}
    assert WRITE_TOOLS <= {
        s["name"] for s in tools.mcp_tools(include_writes=True)}


def test_call_tool_unknown_raises():
    with pytest.raises(KeyError):
        tools.call_tool(None, "no_such_tool", {})


# ---------------------------------------------------------------------------
# get_library_overview
# ---------------------------------------------------------------------------

def test_library_overview(db):
    ov = tools.call_tool(db, "get_library_overview", {})
    assert ov["total_photos"] == 5
    assert ov["registered_people"] == 3
    # Faces exist on 894, 907, 922, 878 → 4 distinct photos.
    assert ov["photos_with_faces"] == 4
    assert ov["date_taken_min"] <= ov["date_taken_max"]
    # DSC04878 has GPS in the sample data.
    assert ov["with_gps"] >= 1


# ---------------------------------------------------------------------------
# list_people
# ---------------------------------------------------------------------------

def test_list_people_counts(db):
    people = {p["name"]: p["photo_count"]
              for p in tools.call_tool(db, "list_people", {})["people"]}
    assert people == {"Alex": 3, "Jamie": 2, "Sam": 1}


def test_list_people_query_filter(db):
    res = tools.call_tool(db, "list_people", {"query": "ale"})["people"]
    assert [p["name"] for p in res] == ["Alex"]


# ---------------------------------------------------------------------------
# list_places
# ---------------------------------------------------------------------------

def test_list_places_place_name_counts(db):
    places = tools.call_tool(db, "list_places", {})["places"]
    pn = {p["value"]: p["count"] for p in places if p["kind"] == "place_name"}
    assert pn["Big Sur, CA"] == 3
    assert pn["Morro Bay, CA"] == 2


def test_list_places_query_filter(db):
    places = tools.call_tool(db, "list_places", {"query": "morro"})["places"]
    assert all("morro" in p["value"].lower() for p in places)
    assert any(p["value"] == "Morro Bay, CA" for p in places)


# ---------------------------------------------------------------------------
# list_vocab
# ---------------------------------------------------------------------------

def test_list_vocab_all_kinds(db):
    vocab = tools.call_tool(db, "list_vocab", {})
    assert set(vocab) == {"categories", "visual_tags", "keywords", "style_tags"}
    cats = {c["value"] for c in vocab["categories"]}
    assert {"landscape", "people"} <= cats
    vis = {v["value"] for v in vocab["visual_tags"]}
    assert {"dramatic", "golden hour"} <= vis


def test_list_vocab_covers_every_vocab_filter():
    """search_photos exposes category / visual_tag / keyword / style_tag; each
    must be groundable via list_vocab, or the model can only guess the value
    (qwen3.5-9b picked style_tag='moody' — a filter that matched nothing —
    because style_tags was unlistable)."""
    kinds = set(tools.get_tool("list_vocab").parameters["properties"]["kind"]["enum"])
    props = tools.get_tool("search_photos").parameters["properties"]
    for arg, kind in (("category", "categories"), ("visual_tag", "visual_tags"),
                      ("keyword", "keywords"), ("style_tag", "style_tags")):
        assert arg in props
        assert kind in kinds, f"{arg} filter has no list_vocab kind"


def test_list_vocab_style_tags(db):
    vocab = tools.call_tool(db, "list_vocab", {"kind": "style_tags"})
    assert set(vocab) == {"style_tags"}
    assert isinstance(vocab["style_tags"], list)


def test_list_vocab_kind_restriction(db):
    vocab = tools.call_tool(db, "list_vocab", {"kind": "keywords"})
    assert set(vocab) == {"keywords"}
    kws = {k["value"] for k in vocab["keywords"]}
    assert "sunset" in kws


def test_list_vocab_query_filter(db):
    vocab = tools.call_tool(db, "list_vocab", {"kind": "categories", "query": "land"})
    assert [c["value"] for c in vocab["categories"]] == ["landscape"]


# ---------------------------------------------------------------------------
# search_photos
# ---------------------------------------------------------------------------

def test_search_by_single_person(db):
    res = tools.call_tool(db, "search_photos", {"people": ["Alex"]})
    assert res["total"] == 3
    assert res["returned"] == 3
    # Compact-hit contract.
    hit = res["results"][0]
    assert set(hit) >= {"id", "filename", "date_taken", "place_name",
                        "description", "categories", "thumbnail_url"}
    assert hit["thumbnail_url"] == f"/api/photos/{hit['id']}/thumbnail"


def test_search_people_and_intersection(db):
    # Alex ∩ Jamie → DSC04907 + DSC04922.
    res = tools.call_tool(db, "search_photos", {"people": ["Alex", "Jamie"]})
    assert res["total"] == 2
    names = {h["filename"] for h in res["results"]}
    assert names == {"DSC04907.JPG", "DSC04922.JPG"}


def test_search_people_case_insensitive(db):
    res = tools.call_tool(db, "search_photos", {"people": ["alex"]})
    assert res["total"] == 3


def test_search_people_stringified_array(db):
    # Small models often JSON-encode the array as a string; coerce it.
    res = tools.call_tool(db, "search_photos", {"people": '["Alex"]'})
    assert res["total"] == 3
    res2 = tools.call_tool(db, "search_photos", {"people": '["Alex", "Jamie"]'})
    assert res2["total"] == 2


def test_search_people_bare_string(db):
    res = tools.call_tool(db, "search_photos", {"people": "Alex"})
    assert res["total"] == 3


def test_coerce_str_list():
    assert tools._coerce_str_list('["a","b"]') == ["a", "b"]
    assert tools._coerce_str_list("Alex") == ["Alex"]
    assert tools._coerce_str_list(["a", "b"]) == ["a", "b"]
    assert tools._coerce_str_list(None) == []
    assert tools._coerce_str_list("") == []


def test_search_unresolved_person_reported(db):
    res = tools.call_tool(db, "search_photos", {"people": ["Nobody"]})
    assert res["total"] == 0
    assert res["unresolved_people"] == ["Nobody"]
    assert "note" in res


def test_search_sort_subject(db):
    # Flat subject-ranked search: returns Alex's photos with prominence scores.
    res = tools.call_tool(db, "search_photos", {"people": ["Alex"], "sort": "subject"})
    assert res["sort"] == "subject"
    assert res["total"] == 3
    assert all("subject_prominence" in h for h in res["results"])


def test_search_sort_subject_without_people_falls_back(db):
    # No people → can't subject-rank; falls back to a normal search (no crash).
    res = tools.call_tool(db, "search_photos", {"location": "Big Sur", "sort": "subject"})
    assert res.get("sort") != "subject"
    assert res["total"] >= 3  # the 3 Big Sur photos (bbox fallback may add nearby)


def test_search_by_category(db):
    res = tools.call_tool(db, "search_photos", {"category": "landscape"})
    assert res["total"] == 1
    assert res["results"][0]["filename"] == "DSC04922.JPG"


def test_search_by_location(db):
    # _search_by_location may also union a Nominatim bbox fallback (which can
    # pull in the nearby GPS-tagged Morro Bay shot), so assert the three
    # place_name-tagged Big Sur photos are present rather than an exact count.
    res = tools.call_tool(db, "search_photos", {"location": "Big Sur"})
    names = {h["filename"] for h in res["results"]}
    assert {"DSC04894.JPG", "DSC04907.JPG", "DSC04922.JPG"} <= names


def test_search_limit_clamped(db):
    res = tools.call_tool(db, "search_photos", {"people": ["Alex"], "limit": 1})
    assert res["returned"] == 1
    assert res["total"] == 3  # total reflects all matches, not the page


def test_search_description_truncated(db):
    long = "x " * 400
    pid = db.add_photo(filepath="z/long.JPG", filename="long.JPG",
                       date_taken="2026-03-13T10:00:00", description=long,
                       place_name="Big Sur, CA")
    res = tools.call_tool(db, "search_photos", {"location": "Big Sur"})
    hit = next(h for h in res["results"] if h["id"] == pid)
    assert len(hit["description"]) <= tools._DESC_TRUNCATE
    assert hit["description"].endswith("…")


# ---------------------------------------------------------------------------
# summarize (faceting)
# ---------------------------------------------------------------------------

def test_summarize_by_year(db):
    res = tools.call_tool(db, "summarize", {"group_by": "year"})
    buckets = {b["value"]: b["count"] for b in res["buckets"]}
    # All 5 fixture photos are 2026-03-13.
    assert buckets.get("2026") == 5


def test_summarize_by_location(db):
    res = tools.call_tool(db, "summarize", {"group_by": "location"})
    buckets = {b["value"]: b["count"] for b in res["buckets"]}
    assert buckets["Big Sur, CA"] == 3
    assert buckets["Morro Bay, CA"] == 2


def test_summarize_by_person(db):
    res = tools.call_tool(db, "summarize", {"group_by": "person"})
    buckets = {b["value"]: b["count"] for b in res["buckets"]}
    assert buckets == {"Alex": 3, "Jamie": 2, "Sam": 1}


def test_summarize_with_filter(db):
    # Alex's photos are all in Big Sur → group by location yields just that.
    res = tools.call_tool(db, "summarize", {"people": ["Alex"], "group_by": "location"})
    buckets = {b["value"]: b["count"] for b in res["buckets"]}
    assert buckets == {"Big Sur, CA": 3}


def test_summarize_location_filter_by_year(db):
    res = tools.call_tool(db, "summarize", {"location": "Big Sur", "group_by": "year"})
    buckets = {b["value"]: b["count"] for b in res["buckets"]}
    assert buckets.get("2026") == 3


def test_summarize_bad_group_by(db):
    res = tools.call_tool(db, "summarize", {"group_by": "nonsense"})
    assert "error" in res


# ---------------------------------------------------------------------------
# representatives (top-N per bucket)
# ---------------------------------------------------------------------------

def test_representatives_by_location_one_each(db):
    # Best (highest aesthetic_score) per place: Big Sur → DSC04922 (9.1),
    # Morro Bay → DSC04878 (6.2).
    res = tools.call_tool(db, "representatives", {"bucket": "location", "n": 1})
    by_bucket = {h["bucket"]: h["filename"] for h in res["results"]}
    assert by_bucket["Big Sur, CA"] == "DSC04922.JPG"
    assert by_bucket["Morro Bay, CA"] == "DSC04878.JPG"
    assert res["buckets"] == 2


def test_representatives_n_per_bucket(db):
    res = tools.call_tool(db, "representatives", {"bucket": "location", "n": 2})
    # 2 places × up to 2 each = 4 (Big Sur has 3, Morro Bay has 2).
    assert res["returned"] == 4


def test_representatives_with_person_filter_by_year(db):
    # Alex's photos are all 2026; best is DSC04922.
    res = tools.call_tool(db, "representatives",
                          {"people": ["Alex"], "bucket": "year", "n": 1})
    assert res["returned"] == 1
    assert res["results"][0]["filename"] == "DSC04922.JPG"
    assert res["results"][0]["bucket"] == "2026"


def test_representatives_bad_bucket(db):
    res = tools.call_tool(db, "representatives", {"bucket": "nonsense"})
    assert "error" in res


def test_representatives_rank_by_subject(db):
    res = tools.call_tool(db, "representatives",
                          {"people": ["Alex"], "bucket": "location", "rank_by": "subject"})
    assert res["ranked_by"] == "subject"
    # Subject ranking attaches a prominence score to each hit.
    assert all("subject_prominence" in h for h in res["results"])


def test_representatives_subject_falls_back_without_people(db):
    # rank_by=subject needs a people filter; without one it ranks by quality.
    res = tools.call_tool(db, "representatives", {"bucket": "year", "rank_by": "subject"})
    assert res["ranked_by"] == "quality"


def test_dedupe_ranked_skips_burst_and_hash():
    rows = [
        {"id": 1, "_bucket": "2026", "date_taken": "2026-01-01 10:00:00", "file_hash": "a"},
        {"id": 2, "_bucket": "2026", "date_taken": "2026-01-01 10:00:03", "file_hash": "b"},  # burst (3s)
        {"id": 3, "_bucket": "2026", "date_taken": "2026-01-01 14:00:00", "file_hash": "c"},  # distinct
        {"id": 4, "_bucket": "2026", "date_taken": "2026-06-01 09:00:00", "file_hash": "a"},  # dup hash of 1
    ]
    kept = [d["id"] for d in tools._dedupe_ranked(rows, n=3)]
    assert kept == [1, 3]  # 2 (burst of 1) and 4 (hash dup of 1) skipped


# ---------------------------------------------------------------------------
# rerank_photos (VLM re-ranking)
# ---------------------------------------------------------------------------

def test_rerank_falls_back_without_vision_model(db, monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_LLM_VISUAL_MODEL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    ids = list(db._test_photo_ids.values())
    res = tools.call_tool(db, "rerank_photos", {"photo_ids": ids, "criteria": "a dog"})
    assert res["reranked"] is False
    assert [h["id"] for h in res["results"]] == ids  # input order preserved


def test_rerank_sorts_by_vision_score(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://fake:1234/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "fake-vl")
    monkeypatch.setattr(tools, "_thumb_b64", lambda db, pid: "ZmFrZQ==")
    # Score = id mod 10 / 10 → deterministic, higher id → higher score.
    monkeypatch.setattr(tools, "_vision_score",
                        lambda base, model, b64, crit: {"score": (b64 and 0.0) or 0.0})

    ids = list(db._test_photo_ids.values())
    # give each id a distinct score via a dict
    score_map = {pid: i / 10 for i, pid in enumerate(ids)}
    monkeypatch.setattr(tools, "_thumb_b64", lambda db, pid: f"id:{pid}")
    monkeypatch.setattr(tools, "_vision_score",
                        lambda base, model, b64, crit:
                        {"score": score_map[int(b64.split(':')[1])], "reason": "ok"})

    res = tools.call_tool(db, "rerank_photos", {"photo_ids": ids, "criteria": "x"})
    assert res["reranked"] is True
    got = [h["id"] for h in res["results"]]
    assert got == sorted(ids, key=lambda p: score_map[p], reverse=True)
    assert all("rerank_score" in h for h in res["results"])


def test_rerank_top_n_caps_and_keeps_best(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://fake:1234/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "fake-vl")
    ids = list(db._test_photo_ids.values())
    score_map = {pid: i / 10 for i, pid in enumerate(ids)}
    monkeypatch.setattr(tools, "_thumb_b64", lambda db, pid: f"id:{pid}")
    monkeypatch.setattr(tools, "_vision_score",
                        lambda base, model, b64, crit:
                        {"score": score_map[int(b64.split(':')[1])], "reason": "x"})
    res = tools.call_tool(db, "rerank_photos",
                          {"photo_ids": ids, "criteria": "x", "top_n": 2})
    assert res["returned"] == 2
    # The two highest-scoring ids only.
    top2 = sorted(ids, key=lambda p: score_map[p], reverse=True)[:2]
    assert [h["id"] for h in res["results"]] == top2


def test_rerank_requires_args(db):
    assert "error" in tools.call_tool(db, "rerank_photos", {"photo_ids": [], "criteria": "x"})
    assert "error" in tools.call_tool(db, "rerank_photos", {"photo_ids": [1], "criteria": ""})


# ---------------------------------------------------------------------------
# get_photo
# ---------------------------------------------------------------------------

def test_get_photo_detail_with_people(db):
    pid = db._test_photo_ids["DSC04894.JPG"]
    photo = tools.call_tool(db, "get_photo", {"photo_id": pid})
    assert photo["id"] == pid
    assert set(photo["people"]) == {"Alex", "Sam"}
    assert photo["face_count"] == 2
    assert photo["thumbnail_url"] == f"/api/photos/{pid}/thumbnail"


def test_get_photo_missing(db):
    res = tools.call_tool(db, "get_photo", {"photo_id": 999999})
    assert "error" in res


def test_get_photo_bad_id(db):
    res = tools.call_tool(db, "get_photo", {"photo_id": "abc"})
    assert "error" in res


# ---------------------------------------------------------------------------
# get_photo_image — returns an error (not a raise) when the original is
# absent. The fixture photos have no real bytes on disk, which exercises the
# graceful-failure path without needing image files.
# ---------------------------------------------------------------------------

def test_get_photo_image_missing_original(db):
    pid = db._test_photo_ids["DSC04894.JPG"]
    res = tools.call_tool(db, "get_photo_image", {"photo_id": pid})
    assert "error" in res


# ---------------------------------------------------------------------------
# Face-framing filters: only_these_people / faces_in_frame (from the Ask-log
# review — "only the four of us / nobody cropped" is metadata, not vision).
# bbox tuple order is (top, right, bottom, left); fixture images are
# 7008x4672, so the 1% edge margin is ~70px wide / ~47px tall.
# ---------------------------------------------------------------------------

IN_FRAME_BBOX = (1000, 4000, 2000, 1000)   # (top, right, bottom, left) — fully inside
EDGE_BBOX = (1000, 4000, 2000, 30)         # left=30px → cut off at the left edge


def _add_photo_with_faces(db, filename, faces, score=8.0,
                          date="2026-03-14T10:00:00"):
    """faces: list of (person_id|None, bbox). Returns the new photo id."""
    pid = db.add_photo(filepath=f"2026/march/{filename}", filename=filename,
                       date_taken=date, aesthetic_score=score,
                       image_width=7008, image_height=4672)
    for person_id, bbox in faces:
        fid = db.add_face(pid, bbox, [0.1] * 512, person_id=person_id)
        if person_id is not None:
            db.assign_face_to_person(fid, person_id, "strict")
    return pid


def test_truthy_coercion():
    # Small models sometimes send the JSON bool as a string.
    assert tools._truthy(True) and tools._truthy("true") and tools._truthy("1")
    assert tools._truthy("YES") and tools._truthy(1)
    assert not tools._truthy(False) and not tools._truthy("false")
    assert not tools._truthy("0") and not tools._truthy(None) and not tools._truthy(0)


def test_search_only_these_people_drops_photos_with_extras(db):
    alex = db._test_person_ids["Alex"]
    sam = db._test_person_ids["Sam"]
    # EXTRA has Alex + Sam + an unknown third face → 3 faces total.
    extra = _add_photo_with_faces(db, "EXTRA_TRIO.JPG", [
        (alex, IN_FRAME_BBOX), (sam, IN_FRAME_BBOX), (None, IN_FRAME_BBOX)])
    # DSC04894 has exactly Alex + Sam (2 faces).
    just_two = db._test_photo_ids["DSC04894.JPG"]

    base = tools.call_tool(db, "search_photos", {"people": ["Alex", "Sam"]})
    assert {just_two, extra} <= {r["id"] for r in base["results"]}

    filt = tools.call_tool(db, "search_photos",
                           {"people": ["Alex", "Sam"], "only_these_people": True})
    ids = {r["id"] for r in filt["results"]}
    assert just_two in ids       # exactly the two named → kept
    assert extra not in ids      # third person present → dropped
    assert filt["face_filtered"] is True
    # `total` is the pre-filter match count; `returned` reflects survivors.
    assert filt["returned"] <= filt["total"]


def test_search_faces_in_frame_drops_edge_cropped(db):
    alex = db._test_person_ids["Alex"]
    sam = db._test_person_ids["Sam"]
    inside = _add_photo_with_faces(db, "INSIDE.JPG", [
        (alex, IN_FRAME_BBOX), (sam, IN_FRAME_BBOX)])
    cropped = _add_photo_with_faces(db, "CROPPED.JPG", [
        (alex, IN_FRAME_BBOX), (sam, EDGE_BBOX)])

    res = tools.call_tool(db, "search_photos",
                          {"people": ["Alex", "Sam"], "faces_in_frame": True})
    ids = {r["id"] for r in res["results"]}
    assert inside in ids
    assert cropped not in ids                                # Sam at left edge
    assert db._test_photo_ids["DSC04894.JPG"] not in ids     # fixture left-edge faces


def test_faces_in_frame_scoped_to_named_people(db):
    # An UNNAMED bystander cropped at the edge must not disqualify the photo —
    # faces_in_frame with a people filter checks only the named people's faces.
    alex = db._test_person_ids["Alex"]
    sam = db._test_person_ids["Sam"]
    pid = _add_photo_with_faces(db, "BYSTANDER.JPG", [
        (alex, IN_FRAME_BBOX), (sam, IN_FRAME_BBOX), (None, EDGE_BBOX)])
    res = tools.call_tool(db, "search_photos",
                          {"people": ["Alex", "Sam"], "faces_in_frame": True})
    assert pid in {r["id"] for r in res["results"]}


def test_representatives_only_these_people_sql_path(db):
    # representatives flows through _build_filter_sql (SQL), not the search
    # post-filter — exercise that branch separately.
    alex = db._test_person_ids["Alex"]
    sam = db._test_person_ids["Sam"]
    # Higher quality than DSC04894 (7.1) so it wins n=1 when NOT filtered.
    trio = _add_photo_with_faces(db, "TRIO_HIQ.JPG", [
        (alex, IN_FRAME_BBOX), (sam, IN_FRAME_BBOX), (None, IN_FRAME_BBOX)],
        score=9.9)
    unfiltered = tools.call_tool(db, "representatives",
        {"people": ["Alex", "Sam"], "bucket": "year", "n": 1})
    assert unfiltered["results"][0]["id"] == trio            # wins on quality

    filtered = tools.call_tool(db, "representatives",
        {"people": ["Alex", "Sam"], "bucket": "year", "n": 1,
         "only_these_people": True})
    ids = {r["id"] for r in filtered["results"]}
    assert trio not in ids                                   # extra face → out
    assert db._test_photo_ids["DSC04894.JPG"] in ids         # exactly two → in


def test_summarize_faces_in_frame_sql_path(db):
    alex = db._test_person_ids["Alex"]
    sam = db._test_person_ids["Sam"]
    _add_photo_with_faces(db, "INSIDE2.JPG", [
        (alex, IN_FRAME_BBOX), (sam, IN_FRAME_BBOX)])
    # people=[Alex,Sam] matches DSC04894 (edge faces) + INSIDE2 (in-frame).
    base = tools.call_tool(db, "summarize",
        {"people": ["Alex", "Sam"], "group_by": "year"})
    framed = tools.call_tool(db, "summarize",
        {"people": ["Alex", "Sam"], "group_by": "year", "faces_in_frame": True})
    base_2026 = {b["value"]: b["count"] for b in base["buckets"]}.get("2026", 0)
    framed_2026 = {b["value"]: b["count"] for b in framed["buckets"]}.get("2026", 0)
    assert framed_2026 == base_2026 - 1     # DSC04894 (edge) dropped, INSIDE2 kept


def test_face_filter_specs_exposed():
    by_name = {s.name: s for s in tools.all_tools()}
    for name in ("search_photos", "summarize", "representatives"):
        props = by_name[name].parameters["properties"]
        assert "only_these_people" in props
        assert "faces_in_frame" in props


# ---------------------------------------------------------------------------
# representatives max_buckets — "top N photos, no more than 1 from each X"
# (from the Ask-log review: the agent stuffed the "10" into n and got ~10 PER
# location; max_buckets caps the bucket count instead, best-first).
# ---------------------------------------------------------------------------

def test_representatives_max_buckets_caps_and_ranks(db):
    # Fixture has 2 locations (Big Sur ×3, Morro Bay ×2). n=1 → 2 buckets;
    # max_buckets=1 keeps only the higher-quality location's representative.
    full = tools.call_tool(db, "representatives", {"bucket": "location", "n": 1})
    assert full["buckets"] == 2

    capped = tools.call_tool(db, "representatives",
                             {"bucket": "location", "n": 1, "max_buckets": 1})
    assert capped["returned"] == 1
    assert capped["buckets"] == 1
    # Big Sur's best (DSC04922, 9.1) beats Morro Bay's best (DSC04878, 6.2).
    assert capped["results"][0]["filename"] == "DSC04922.JPG"


def test_representatives_max_buckets_is_not_per_bucket_count(db):
    # The bug being fixed: a large value must NOT multiply results per bucket.
    res = tools.call_tool(db, "representatives",
                          {"bucket": "location", "n": 1, "max_buckets": 50})
    # Only 2 locations exist → at most 2 results regardless of the cap.
    assert res["returned"] == 2
    assert res["buckets"] == 2


def test_representatives_max_buckets_orders_best_first(db):
    # With max_buckets set, output is a ranked "top" list (best-first), unlike
    # the default chronological/alphabetical spread.
    res = tools.call_tool(db, "representatives",
                          {"bucket": "location", "n": 1, "max_buckets": 10})
    scores = [r.get("aesthetic_score") for r in res["results"]]
    non_null = [s for s in scores if s is not None]
    assert non_null == sorted(non_null, reverse=True)


def test_representatives_max_buckets_spec_exposed():
    spec = {s.name: s for s in tools.all_tools()}["representatives"]
    assert "max_buckets" in spec.parameters["properties"]


# ---------------------------------------------------------------------------
# daily_highlights
# ---------------------------------------------------------------------------

def test_daily_highlights_registered_and_schema():
    spec = {s.name: s for s in tools.all_tools()}.get("daily_highlights")
    assert spec is not None
    props = spec.parameters["properties"]
    assert "per_day" in props and "window_minutes" in props


def test_daily_highlights_collapses_within_window(db):
    # Fixture: 5 photos on 2026-03-13 at 10:00, 10:05, 11:30, 14:00, 16:00.
    # The 10:00 (6.2) and 10:05 (5.4) are 5 min apart → collapse to the best.
    res = tools.call_tool(db, "daily_highlights",
                          {"per_day": 20, "window_minutes": 10})
    assert res["returned"] == 4
    names = [r["filename"] for r in res["results"]]
    assert "DSC04878.JPG" in names       # 10:00, higher score → kept
    assert "DSC04880.JPG" not in names   # 10:05, near-dup → dropped
    # Chronological output order.
    dts = [r["date_taken"] for r in res["results"]]
    assert dts == sorted(dts)


def test_daily_highlights_window_zero_keeps_all(db):
    res = tools.call_tool(db, "daily_highlights", {"window_minutes": 0})
    assert res["returned"] == 5


def test_daily_highlights_per_day_cap(db):
    res = tools.call_tool(db, "daily_highlights",
                          {"per_day": 2, "window_minutes": 10})
    assert res["returned"] == 2
    # Keeps the two best distinct moments (9.1 and 8.3).
    scores = sorted(r["aesthetic_score"] for r in res["results"])
    assert scores == [8.3, 9.1]


def test_daily_highlights_day_summary_places(db):
    res = tools.call_tool(db, "daily_highlights", {"window_minutes": 10})
    assert res["days"] == 1
    summary = res["day_summary"][0]
    assert summary["day"] == "2026-03-13"
    assert "Big Sur, CA" in summary["places"]
    # Every result carries its geotagged place for location highlighting.
    assert all("place_name" in r for r in res["results"])


# ---------------------------------------------------------------------------
# camera filter + camera_model on compact hits (Ask card badge)
# ---------------------------------------------------------------------------

def test_search_photos_camera_filter(db):
    res = tools.call_tool(db, "search_photos", {"camera": "ILCE-7M4"})
    assert res["total"] == 5
    # Every hit carries camera_model so the grid's shared 📷 badge renders.
    assert all(h["camera_model"] == "ILCE-7M4" for h in res["results"])


def test_search_photos_camera_filter_no_match(db):
    res = tools.call_tool(db, "search_photos", {"camera": "KODAK PIXPRO WPZ2"})
    assert res["total"] == 0


def test_representatives_camera_filter(db):
    # Camera flows through _build_filter_sql into the faceting tools too.
    res = tools.call_tool(db, "representatives",
                          {"bucket": "location", "n": 1, "camera": "ILCE-7M4"})
    assert res["returned"] >= 1
    res0 = tools.call_tool(db, "representatives",
                           {"bucket": "location", "n": 1, "camera": "nope"})
    assert res0["returned"] == 0


# ---------------------------------------------------------------------------
# group_into_chapters
# ---------------------------------------------------------------------------

def test_group_into_chapters_by_place(db):
    res = tools.call_tool(db, "group_into_chapters", {"min_photos": 1})
    titles = [c["title"] for c in res["chapters"]]
    # Chronological: Morro Bay (10:00, 10:05) then Big Sur (11:30, 14:00, 16:00).
    assert titles == ["Morro Bay, CA", "Big Sur, CA"]
    assert res["chapters"][0]["photo_count"] == 2
    assert res["chapters"][1]["photo_count"] == 3
    assert res["chapters"][0]["date_from"] == "2026-03-13"
    # Representative photos come back so the grid can render them.
    assert res["returned"] >= 1
    assert all(h.get("chapter") for h in res["results"])


def test_group_into_chapters_min_photos_drops_small(db):
    res = tools.call_tool(db, "group_into_chapters", {"min_photos": 3})
    # Only Big Sur (3) survives; Morro Bay (2) is dropped as transient.
    assert [c["title"] for c in res["chapters"]] == ["Big Sur, CA"]
    assert res["dropped_small_chapters"] == 1


# ---------------------------------------------------------------------------
# daily_scene_breakdown
# ---------------------------------------------------------------------------

def test_daily_scene_breakdown_splits_on_gap_and_place(db):
    res = tools.call_tool(db, "daily_scene_breakdown",
                          {"date": "2026-03-13", "gap_minutes": 40})
    # Morro (10:00-10:05), then Big Sur splits at each >40min gap (11:30/14:00/16:00).
    assert res["scenes_found"] == 4
    assert res["scenes"][0]["place"] == "Morro Bay, CA"
    assert res["scenes"][0]["photo_count"] == 2
    assert all(s["place"] == "Big Sur, CA" for s in res["scenes"][1:])


def test_daily_scene_breakdown_requires_date(db):
    assert "error" in tools.call_tool(db, "daily_scene_breakdown", {})


# ---------------------------------------------------------------------------
# suggest_layout
# ---------------------------------------------------------------------------

def test_suggest_layout_partitions_and_picks_hero(db):
    ids = [h["id"] for h in
           tools.call_tool(db, "search_photos", {"camera": "ILCE-7M4"})["results"]]
    res = tools.call_tool(db, "suggest_layout",
                          {"photo_ids": ids, "spread_count": 2})
    assert res["spread_count"] == 2
    assert res["photo_count"] == 5
    # 5 photos → spreads of 3 and 2.
    assert sorted(s["photo_count"] for s in res["spreads"]) == [2, 3]
    # Every spread names a hero drawn from its own photos.
    for s in res["spreads"]:
        assert s["hero_id"] in s["photo_ids"]
    assert {"matched 2-up", "asymmetric collage"} == {s["archetype"] for s in res["spreads"]}


def test_suggest_layout_requires_ids(db):
    assert "error" in tools.call_tool(db, "suggest_layout", {"photo_ids": []})


# ---------------------------------------------------------------------------
# Routing guidance shared by both adapters (MCP instructions / agent prompt)
# ---------------------------------------------------------------------------

def test_tool_descriptions_do_not_mandate_grounding():
    """The grounding directive is adapter-specific: the agent pre-injects the
    library facts and should skip the list_* tools, an MCP client must call
    them. So the descriptions must not hardcode either stance — they used to say
    'ALWAYS call this first' / 'You MUST call this before', contradicting the
    agent's system prompt in the same context window."""
    for name in ("get_library_overview", "list_people"):
        desc = tools.get_tool(name).description
        assert "ALWAYS call this first" not in desc
        assert "You MUST call this before" not in desc


def test_grounding_directives_are_opposites():
    assert "go STRAIGHT to search_photos" in tools.GROUNDING_WITH_FACTS
    assert "Never stop after a list_* call" in tools.GROUNDING_WITHOUT_FACTS


def test_daily_highlights_does_not_claim_flat_best_of_range():
    """'best photos from <date range>' is search_photos(sort='quality_desc').
    daily_highlights used to claim that exact phrasing and won the tool choice
    on qwen3.5-9b for p03/p10."""
    desc = tools.get_tool("daily_highlights").description
    assert "ONLY use this when" in desc
    assert "is NOT this tool" in desc
    assert "sort='quality_desc'" in desc


def test_server_instructions_carry_routing_and_gate_writes():
    plain = tools.server_instructions(include_writes=False)
    assert tools.ROUTING_GUIDANCE in plain
    assert tools.GROUNDING_WITHOUT_FACTS in plain
    assert "set_photo_location" not in plain

    writeful = tools.server_instructions(include_writes=True)
    assert tools.WRITE_GUIDANCE in writeful

    withfacts = tools.server_instructions(library_facts="People: Calvin, Ellie.")
    assert tools.GROUNDING_WITH_FACTS in withfacts
    assert "LIBRARY FACTS:" in withfacts
    assert tools.GROUNDING_WITHOUT_FACTS not in withfacts
