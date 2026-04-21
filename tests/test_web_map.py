"""Integration tests for /api/photos/geojson and the /map page."""

from photosearch.db import PhotoDB


def test_geojson_returns_only_gps_rows(client, db):
    with PhotoDB(db.db_path) as pdb:
        pdb.add_photo(filepath="/gps1.jpg", filename="gps1.jpg",
                      date_taken="2024-08-10T14:30:00",
                      gps_lat=47.6, gps_lon=-122.3,
                      place_name="Seattle, Washington, US")
        pdb.add_photo(filepath="/gps2.jpg", filename="gps2.jpg",
                      date_taken="2023-01-01T00:00:00",
                      gps_lat=35.0, gps_lon=139.0,
                      location_source="inferred",
                      location_confidence=0.82,
                      place_name="Tokyo, Tokyo, JP")
        pdb.add_photo(filepath="/nogps.jpg", filename="nogps.jpg",
                      date_taken="2024-08-10T14:30:00")
        pdb.conn.commit()

    r = client.get("/api/photos/geojson")
    assert r.status_code == 200
    body = r.json()

    assert body["count"] == len(body["points"])
    returned = {p[0]: p for p in body["points"]}

    gps1 = next(p for p in body["points"] if p[1] == 47.6)
    assert gps1[2] == -122.3
    assert gps1[3] == "exif"          # add_photo auto-stamp
    assert gps1[4] == 2024
    assert gps1[5] == "Seattle, Washington, US"

    gps2 = next(p for p in body["points"] if p[1] == 35.0)
    assert gps2[3] == "inferred"
    assert gps2[4] == 2023
    assert gps2[5] == "Tokyo, Tokyo, JP"

    # The no-GPS row must not appear.
    nogps_row = db.conn.execute(
        "SELECT id FROM photos WHERE filename='nogps.jpg'"
    ).fetchone()
    assert nogps_row["id"] not in returned


def test_geojson_handles_missing_date_and_place(client, db):
    with PhotoDB(db.db_path) as pdb:
        pdb.add_photo(filepath="/nodate.jpg", filename="nodate.jpg",
                      gps_lat=10.0, gps_lon=20.0)
        pdb.conn.commit()

    r = client.get("/api/photos/geojson")
    assert r.status_code == 200
    row = next(p for p in r.json()["points"] if p[1] == 10.0)
    assert row[4] is None
    assert row[5] is None


def test_map_page_served(client):
    r = client.get("/map")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text or "<!doctype html>" in r.text
