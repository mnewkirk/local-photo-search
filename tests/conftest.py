"""Shared fixtures for the local-photo-search test suite.

Provides an in-memory PhotoDB and a FastAPI TestClient, both populated
with a realistic set of sample data — no real photos or ML models needed.
"""

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the project root is on sys.path so `import photosearch` works
# regardless of how pytest is invoked.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest

# ---------------------------------------------------------------------------
# Mock heavy ML dependencies that aren't available in the test environment.
# Must happen before any photosearch module is imported.
# Only mock modules that genuinely can't be imported — if the real package
# is installed (e.g. in the project venv), leave it alone so tests like
# test_face_matching.py that need the real CLIP model keep working.
# ---------------------------------------------------------------------------
_OPTIONAL_DEPS = (
    "torch", "open_clip", "insightface", "onnxruntime",
    "ollama", "cv2", "scipy", "scipy.cluster",
    "scipy.cluster.hierarchy", "scipy.spatial",
    "scipy.spatial.distance",
)
for _mod in _OPTIONAL_DEPS:
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

# Provide a mock embed_text only when CLIP isn't really available, so
# search_combined can be called without actual CLIP inference in unit tests.
import photosearch.clip_embed as _clip_mod
try:
    import open_clip as _oc_check
    if isinstance(_oc_check, MagicMock):
        raise ImportError("mocked")
except ImportError:
    _clip_mod.embed_text = lambda query: [0.0] * 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(dim: int = 512, seed: int = 0) -> list[float]:
    """Generate a deterministic pseudo-random unit vector."""
    import hashlib
    h = hashlib.sha512(seed.to_bytes(4, "big")).digest()
    raw = [((b - 128) / 128.0) for b in h]
    # Pad / truncate to dim
    vec = (raw * (dim // len(raw) + 1))[:dim]
    norm = sum(v ** 2 for v in vec) ** 0.5
    return [v / norm for v in vec]


def _serialize(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Sample data constants
# ---------------------------------------------------------------------------

SAMPLE_PHOTOS = [
    {
        "filepath": "2026/march/DSC04878.JPG",
        "filename": "DSC04878.JPG",
        "date_taken": "2026-03-13T10:00:00",
        "description": "Rocky coastline with crashing waves and overcast sky. No people visible.",
        "aesthetic_score": 6.2,
        "dominant_colors": json.dumps(["#3b5998", "#8b9dc3", "#dfe3ee"]),
        "tags": json.dumps(["ocean", "rocks", "waves", "coastline"]),
        "place_name": "Morro Bay, CA",
        "camera_model": "ILCE-7M4",
        "focal_length": "70/1",
        "f_number": "28/10",
        "iso": 200,
        "image_width": 7008,
        "image_height": 4672,
        "gps_lat": 35.3733,
        "gps_lon": -120.8658,
    },
    {
        "filepath": "2026/march/DSC04880.JPG",
        "filename": "DSC04880.JPG",
        "date_taken": "2026-03-13T10:05:00",
        "description": "Seagulls flying over a sandy beach at low tide.",
        "aesthetic_score": 5.4,
        "dominant_colors": json.dumps(["#c2b280", "#87ceeb", "#f5f5dc"]),
        "tags": json.dumps(["beach", "seagulls", "sand", "sky"]),
        "place_name": "Morro Bay, CA",
        "camera_model": "ILCE-7M4",
        "focal_length": "35/1",
        "f_number": "56/10",
        "iso": 100,
        "image_width": 7008,
        "image_height": 4672,
    },
    {
        "filepath": "2026/march/DSC04894.JPG",
        "filename": "DSC04894.JPG",
        "date_taken": "2026-03-13T11:30:00",
        "description": "Two people standing on a rocky overlook, ocean in the background.",
        "aesthetic_score": 7.1,
        "dominant_colors": json.dumps(["#708090", "#2f4f4f", "#87ceeb"]),
        "tags": json.dumps(["people", "overlook", "ocean", "rocks"]),
        "place_name": "Big Sur, CA",
        "camera_model": "ILCE-7M4",
        "focal_length": "50/1",
        "f_number": "28/10",
        "iso": 100,
        "image_width": 7008,
        "image_height": 4672,
    },
    {
        "filepath": "2026/march/DSC04907.JPG",
        "filename": "DSC04907.JPG",
        "date_taken": "2026-03-13T14:00:00",
        "description": "Family walking along a coastal trail, wildflowers on both sides.",
        "aesthetic_score": 8.3,
        "dominant_colors": json.dumps(["#228b22", "#87ceeb", "#daa520"]),
        "tags": json.dumps(["family", "trail", "wildflowers", "coastal"]),
        "place_name": "Big Sur, CA",
        "camera_model": "ILCE-7M4",
        "focal_length": "24/1",
        "f_number": "40/10",
        "iso": 100,
        "image_width": 7008,
        "image_height": 4672,
    },
    {
        "filepath": "2026/march/DSC04922.JPG",
        "filename": "DSC04922.JPG",
        "date_taken": "2026-03-13T16:00:00",
        "description": "Sunset over the Pacific with silhouettes of two people on the cliff.",
        "aesthetic_score": 9.1,
        "dominant_colors": json.dumps(["#ff6347", "#ff8c00", "#4b0082"]),
        "tags": json.dumps(["sunset", "silhouette", "people", "cliff", "pacific"]),
        "place_name": "Big Sur, CA",
        "camera_model": "ILCE-7M4",
        "focal_length": "85/1",
        "f_number": "20/10",
        "iso": 400,
        "image_width": 7008,
        "image_height": 4672,
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a path for a temporary database file."""
    return str(tmp_path / "test.db")


@pytest.fixture
def db(tmp_db_path):
    """An in-memory-like PhotoDB with sample photos, persons, faces, and collections."""
    from photosearch.db import PhotoDB

    database = PhotoDB(tmp_db_path)
    database.set_photo_root("/photos")

    # Insert sample photos
    photo_ids = {}
    for p in SAMPLE_PHOTOS:
        pid = database.add_photo(**p)
        photo_ids[p["filename"]] = pid

    # Add CLIP embeddings (deterministic fake vectors)
    for i, (fname, pid) in enumerate(photo_ids.items()):
        emb = _make_embedding(512, seed=i)
        database.add_clip_embedding(pid, emb)

    # Create persons
    alex_id = database.add_person("Alex")
    jamie_id = database.add_person("Jamie")
    sam_id = database.add_person("Sam")

    # Add faces with bounding boxes
    face1 = database.add_face(
        photo_ids["DSC04894.JPG"], (100, 200, 250, 50),
        _make_embedding(512, seed=100), person_id=alex_id,
    )
    database.assign_face_to_person(face1, alex_id, "strict")

    face2 = database.add_face(
        photo_ids["DSC04894.JPG"], (100, 400, 250, 300),
        _make_embedding(512, seed=101), person_id=sam_id,
    )
    database.assign_face_to_person(face2, sam_id, "strict")

    face3 = database.add_face(
        photo_ids["DSC04907.JPG"], (80, 180, 200, 60),
        _make_embedding(512, seed=102), person_id=jamie_id,
    )
    database.assign_face_to_person(face3, jamie_id, "strict")

    face4 = database.add_face(
        photo_ids["DSC04907.JPG"], (90, 350, 170, 260),
        _make_embedding(512, seed=103), person_id=alex_id,
    )
    database.assign_face_to_person(face4, alex_id, "temporal")

    face5 = database.add_face(
        photo_ids["DSC04922.JPG"], (50, 150, 180, 30),
        _make_embedding(512, seed=104), person_id=jamie_id,
    )
    database.assign_face_to_person(face5, jamie_id, "strict")

    face6 = database.add_face(
        photo_ids["DSC04922.JPG"], (60, 320, 190, 200),
        _make_embedding(512, seed=105), person_id=alex_id,
    )
    database.assign_face_to_person(face6, alex_id, "manual")

    # An unknown face in a cluster (no person assigned)
    face7 = database.add_face(
        photo_ids["DSC04878.JPG"], (300, 400, 380, 320),
        _make_embedding(512, seed=106), cluster_id=99,
    )

    # Create a collection
    coll_id = database.create_collection("Best of March", "Top picks from March trip")
    database.add_photos_to_collection(coll_id, [
        photo_ids["DSC04907.JPG"],
        photo_ids["DSC04922.JPG"],
    ])

    # Store IDs for test access
    database._test_photo_ids = photo_ids
    database._test_person_ids = {"Alex": alex_id, "Jamie": jamie_id, "Sam": sam_id}
    database._test_face_ids = {
        "alex_894": face1, "sam_894": face2,
        "jamie_907": face3, "alex_907": face4,
        "jamie_922": face5, "alex_922": face6,
        "unknown_878": face7,
    }
    database._test_collection_id = coll_id

    yield database
    database.close()


@pytest.fixture
def client(db, tmp_path):
    """FastAPI TestClient wired to the test database."""
    from fastapi.testclient import TestClient
    from photosearch import web

    # Point the web module at our test DB
    original_db_path = web._db_path
    original_photo_root = web._photo_root
    original_thumb_dir = web._thumb_dir

    web._db_path = db.db_path
    web._photo_root = "/photos"
    web._thumb_dir = str(tmp_path / "thumbs")
    os.makedirs(web._thumb_dir, exist_ok=True)

    with TestClient(web.app) as c:
        yield c

    # Restore
    web._db_path = original_db_path
    web._photo_root = original_photo_root
    web._thumb_dir = original_thumb_dir
