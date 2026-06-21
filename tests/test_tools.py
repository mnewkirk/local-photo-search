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

EXPECTED_TOOLS = {
    "get_library_overview", "list_people", "list_places", "list_vocab",
    "search_photos", "summarize", "representatives", "rerank_photos",
    "get_photo", "get_photo_image",
}


def test_registry_has_all_tools():
    assert {t.name for t in tools.all_tools()} == EXPECTED_TOOLS


def test_every_schema_is_a_valid_object_schema():
    for spec in tools.all_tools():
        assert spec.parameters["type"] == "object"
        assert "properties" in spec.parameters
        assert spec.description and len(spec.description) > 20


def test_openai_projection_shape():
    fns = tools.openai_tools()
    assert {t["function"]["name"] for t in fns} == EXPECTED_TOOLS
    for t in fns:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_mcp_projection_shape():
    specs = tools.mcp_tools()
    assert {s["name"] for s in specs} == EXPECTED_TOOLS
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
    assert set(vocab) == {"categories", "visual_tags", "keywords"}
    cats = {c["value"] for c in vocab["categories"]}
    assert {"landscape", "people"} <= cats
    vis = {v["value"] for v in vocab["visual_tags"]}
    assert {"dramatic", "golden hour"} <= vis


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
