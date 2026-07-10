"""Camera badge/filter: the /api/cameras dropdown source, the search camera
filter, and the geotag camera/date filters."""


def _seed(db):
    db.set_photo_root("/photos")
    a = db.add_photo(filepath="2026/2026-06-01_trip/a.jpg", filename="a.jpg",
                     date_taken="2026-06-01 10:00:00", camera_make="Sony",
                     camera_model="TESTCAM-A")
    b = db.add_photo(filepath="2026/2026-06-01_trip/b.jpg", filename="b.jpg",
                     date_taken="2026-06-02 10:00:00", camera_make="Sony",
                     camera_model="TESTCAM-A")
    c = db.add_photo(filepath="2026/2026-06-01_trip/c.jpg", filename="c.jpg",
                     date_taken="2026-06-03 10:00:00", camera_make="Apple",
                     camera_model="TESTCAM-B")
    db.conn.commit()
    return a, b, c


def test_search_combined_has_camera_param():
    import inspect
    from photosearch.search import search_combined
    assert "camera" in inspect.signature(search_combined).parameters


def test_api_cameras_lists_models_with_counts(client, db):
    _seed(db)
    cams = client.get("/api/cameras").json()["cameras"]
    by_model = {c["model"]: c for c in cams}
    assert by_model["TESTCAM-A"]["count"] == 2
    assert by_model["TESTCAM-A"]["make"] == "Sony"
    assert by_model["TESTCAM-B"]["count"] == 1
    # ordered by most-recently-used: TESTCAM-B (2026-06-03) is newer than
    # TESTCAM-A (2026-06-02), so it sorts ahead despite fewer photos.
    order = [c["model"] for c in cams]
    assert order.index("TESTCAM-B") < order.index("TESTCAM-A")


def test_search_camera_filter(client, db):
    a, b, c = _seed(db)
    res = client.get("/api/search", params={"camera": "TESTCAM-A"}).json()
    ids = {r["id"] for r in res["results"]}
    assert ids == {a, b}
    # payload carries the camera fields for the badge
    assert all("camera_model" in r for r in res["results"])


def test_search_camera_intersects_with_date(client, db):
    a, b, c = _seed(db)
    res = client.get("/api/search", params={
        "camera": "TESTCAM-A", "date_from": "2026-06-02", "date_to": "2026-06-02"}).json()
    assert {r["id"] for r in res["results"]} == {b}


def test_geotag_folder_photos_camera_filter(client, db):
    a, b, c = _seed(db)
    # No GPS on any → all show; camera filter narrows to the Sony pair.
    r = client.get("/api/geotag/folder-photos", params={
        "folder": "2026/2026-06-01_trip", "camera": "TESTCAM-A"}).json()
    assert {p["id"] for p in r["photos"]} == {a, b}
