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
    "search_photos", "summarize", "get_photo", "get_photo_image",
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
