"""
Integration tests against real sample photos in ../Photos/sample.

These tests exercise the actual ML pipelines (EXIF, CLIP, faces, descriptions,
colors, quality scoring, and combined search) against the 7-photo sample set.
They require:
  - The full project venv with all dependencies installed
  - Sample photos present at ../Photos/sample/ (relative to project root)
  - Ollama running locally with the llava model pulled (for description tests)

Run:
  pytest tests/test_integration.py -v

Skip slow pipelines (descriptions, quality):
  pytest tests/test_integration.py -v -m "not slow"
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

SAMPLE_DIR = str((_root.parent / "Photos" / "sample").resolve())
SAMPLE_PHOTOS = [
    "DSC04878.JPG",
    "DSC04880.JPG",
    "DSC04894.JPG",
    "DSC04895.JPG",
    "DSC04899.JPG",
    "DSC04907.JPG",
    "DSC04922.JPG",
]

# ---------------------------------------------------------------------------
# Ground truth — known facts about the sample photos
# ---------------------------------------------------------------------------

# EXIF: all 7 photos are from the same Sony camera, taken 2026-03-13
EXPECTED_CAMERA = "SONY"
EXPECTED_DATE_PREFIX = "2026-03-13"

# Approximate date_taken for ordering (HH:MM from EXIF)
EXPECTED_TIMES = {
    "DSC04878.JPG": "10:00",
    "DSC04880.JPG": "10:01",
    "DSC04894.JPG": "10:09",
    "DSC04895.JPG": "10:09",
    "DSC04899.JPG": "10:18",
    "DSC04907.JPG": "10:22",
    "DSC04922.JPG": "10:30",
}

# Face counts per photo (InsightFace detection)
EXPECTED_FACE_COUNTS = {
    "DSC04878.JPG": 0,   # landscape
    "DSC04880.JPG": 0,   # landscape
    "DSC04894.JPG": 2,   # Calvin + Nicole
    "DSC04895.JPG": 0,   # Eleanor facing away — undetectable
    "DSC04899.JPG": 0,   # landscape
    "DSC04907.JPG": 2,   # Eleanor + Calvin (small faces)
    "DSC04922.JPG": 2,   # Eleanor + Calvin
}

# Photos that should have people-related words in their description
PHOTOS_WITH_PEOPLE = {"DSC04894.JPG", "DSC04895.JPG", "DSC04907.JPG", "DSC04922.JPG"}
PHOTOS_WITHOUT_PEOPLE = {"DSC04878.JPG", "DSC04880.JPG", "DSC04899.JPG"}


# ---------------------------------------------------------------------------
# Requirement checks
# ---------------------------------------------------------------------------

def _check_sample_photos():
    """Return list of issues with sample photo availability."""
    issues = []
    if not Path(SAMPLE_DIR).is_dir():
        issues.append(
            f"Sample photo directory not found: {SAMPLE_DIR}\n"
            f"Expected at: ../Photos/sample/ relative to project root"
        )
        return issues
    for photo in SAMPLE_PHOTOS:
        if not (Path(SAMPLE_DIR) / photo).exists():
            issues.append(f"Missing sample photo: {photo}")
    return issues


def _check_dependency(module_name, install_hint=None):
    """Try to import a module and return an error string if unavailable."""
    try:
        __import__(module_name)
        return None
    except ImportError:
        hint = install_hint or f"pip install {module_name}"
        return f"Required module '{module_name}' not installed. Run: {hint}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_dir():
    """Verify sample photos exist and return the directory path."""
    issues = _check_sample_photos()
    if issues:
        pytest.fail(
            "Sample photos not available:\n  "
            + "\n  ".join(issues)
            + "\n\nPlace 7 sample JPEGs in ../Photos/sample/ relative to the project root."
        )
    return SAMPLE_DIR


@pytest.fixture(scope="module")
def sample_photo(sample_dir):
    """Return path to the first sample photo (for single-photo tests)."""
    return str(Path(sample_dir) / SAMPLE_PHOTOS[0])


@pytest.fixture(scope="module")
def all_sample_paths(sample_dir):
    """Return list of paths to all 7 sample photos."""
    return [str(Path(sample_dir) / p) for p in SAMPLE_PHOTOS]


@pytest.fixture(scope="module")
def integration_db(sample_dir):
    """Create a temporary database and index all sample photos with EXIF only.

    This is a lightweight fixture — no CLIP, faces, or descriptions.
    Other fixtures build on it.
    """
    from photosearch.db import PhotoDB
    from photosearch.exif import extract_exif, find_raw_pair
    from photosearch.index import file_hash

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        with PhotoDB(db_path) as db:
            db.set_photo_root(sample_dir)
            db.begin_batch(batch_size=100)
            for photo_name in SAMPLE_PHOTOS:
                photo_path = str(Path(sample_dir) / photo_name)
                meta = extract_exif(photo_path)
                meta["filepath"] = db.relative_filepath(str(Path(photo_path).resolve()))
                raw = find_raw_pair(photo_path)
                if raw:
                    meta["raw_filepath"] = db.relative_filepath(str(Path(raw).resolve()))
                meta["file_hash"] = file_hash(photo_path)
                db.add_photo(**meta)
            db.end_batch()
            yield db
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# EXIF extraction tests
# ---------------------------------------------------------------------------

class TestExif:
    """Test EXIF metadata extraction from real photos."""

    def test_exif_dependency(self):
        err = _check_dependency("exifread")
        if err:
            pytest.fail(err)

    def test_extract_all_photos(self, sample_dir):
        from photosearch.exif import extract_exif

        for photo_name in SAMPLE_PHOTOS:
            path = str(Path(sample_dir) / photo_name)
            meta = extract_exif(path)

            assert meta["filename"] == photo_name
            assert meta["filepath"] is not None

    def test_camera_make(self, sample_dir):
        from photosearch.exif import extract_exif

        for photo_name in SAMPLE_PHOTOS:
            meta = extract_exif(str(Path(sample_dir) / photo_name))
            assert meta["camera_make"] is not None, f"{photo_name}: missing camera_make"
            assert EXPECTED_CAMERA in meta["camera_make"].upper(), (
                f"{photo_name}: expected '{EXPECTED_CAMERA}' in camera_make, "
                f"got '{meta['camera_make']}'"
            )

    def test_date_taken(self, sample_dir):
        from photosearch.exif import extract_exif

        for photo_name in SAMPLE_PHOTOS:
            meta = extract_exif(str(Path(sample_dir) / photo_name))
            assert meta["date_taken"] is not None, f"{photo_name}: missing date_taken"
            assert meta["date_taken"].startswith(EXPECTED_DATE_PREFIX), (
                f"{photo_name}: expected date starting with '{EXPECTED_DATE_PREFIX}', "
                f"got '{meta['date_taken']}'"
            )

    def test_image_dimensions(self, sample_dir):
        from photosearch.exif import extract_exif

        for photo_name in SAMPLE_PHOTOS:
            meta = extract_exif(str(Path(sample_dir) / photo_name))
            # Sony A7 IV shoots at 7008x4672 or similar high-res
            assert meta["image_width"] is not None, f"{photo_name}: missing image_width"
            assert meta["image_height"] is not None, f"{photo_name}: missing image_height"
            assert meta["image_width"] > 1000, f"{photo_name}: width {meta['image_width']} too small"
            assert meta["image_height"] > 1000, f"{photo_name}: height {meta['image_height']} too small"

    def test_exposure_fields_present(self, sample_dir):
        from photosearch.exif import extract_exif

        for photo_name in SAMPLE_PHOTOS:
            meta = extract_exif(str(Path(sample_dir) / photo_name))
            # All 7 photos are outdoor Sony shots — should have full exposure data
            assert meta["focal_length"] is not None, f"{photo_name}: missing focal_length"
            assert meta["iso"] is not None, f"{photo_name}: missing iso"

    def test_chronological_order(self, sample_dir):
        """Photos should be in chronological order by filename."""
        from photosearch.exif import extract_exif

        dates = []
        for photo_name in SAMPLE_PHOTOS:
            meta = extract_exif(str(Path(sample_dir) / photo_name))
            dates.append((photo_name, meta["date_taken"]))

        for i in range(len(dates) - 1):
            assert dates[i][1] <= dates[i + 1][1], (
                f"Photos not in chronological order: "
                f"{dates[i][0]} ({dates[i][1]}) > {dates[i+1][0]} ({dates[i+1][1]})"
            )


# ---------------------------------------------------------------------------
# Database indexing tests
# ---------------------------------------------------------------------------

class TestDatabaseIndexing:
    """Test that indexing sample photos produces correct DB records."""

    def test_all_photos_indexed(self, integration_db):
        count = integration_db.conn.execute(
            "SELECT COUNT(*) FROM photos"
        ).fetchone()[0]
        assert count == len(SAMPLE_PHOTOS), (
            f"Expected {len(SAMPLE_PHOTOS)} photos in DB, got {count}"
        )

    def test_photo_metadata_stored(self, integration_db):
        for photo_name in SAMPLE_PHOTOS:
            row = integration_db.conn.execute(
                "SELECT * FROM photos WHERE filename = ?", (photo_name,)
            ).fetchone()
            assert row is not None, f"{photo_name} not found in DB"
            assert row["camera_make"] is not None
            assert row["date_taken"] is not None

    def test_file_hashes_unique(self, integration_db):
        """Each photo should have a unique hash."""
        rows = integration_db.conn.execute(
            "SELECT file_hash FROM photos WHERE file_hash IS NOT NULL"
        ).fetchall()
        hashes = [r["file_hash"] for r in rows]
        assert len(hashes) == len(SAMPLE_PHOTOS)
        assert len(set(hashes)) == len(hashes), "Duplicate file hashes found"

    def test_file_hashes_are_sha256(self, integration_db):
        rows = integration_db.conn.execute(
            "SELECT file_hash FROM photos WHERE file_hash IS NOT NULL"
        ).fetchall()
        for row in rows:
            h = row["file_hash"]
            assert len(h) == 64, f"Hash should be 64 hex chars, got {len(h)}"
            assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Color extraction tests
# ---------------------------------------------------------------------------

class TestColors:
    """Test dominant color extraction from real photos."""

    def test_color_dependency(self):
        err = _check_dependency("colorthief", "pip install colorthief")
        if err:
            pytest.fail(err)

    def test_extract_colors(self, sample_dir):
        from photosearch.colors import extract_dominant_colors

        for photo_name in SAMPLE_PHOTOS:
            path = str(Path(sample_dir) / photo_name)
            colors = extract_dominant_colors(path)
            assert colors is not None, f"{photo_name}: color extraction returned None"
            assert len(colors) >= 3, (
                f"{photo_name}: expected at least 3 colors, got {len(colors)}"
            )

    def test_colors_are_hex(self, sample_dir):
        from photosearch.colors import extract_dominant_colors

        colors = extract_dominant_colors(str(Path(sample_dir) / SAMPLE_PHOTOS[0]))
        for color in colors:
            assert color.startswith("#"), f"Color should start with #, got {color}"
            assert len(color) == 7, f"Hex color should be 7 chars (#rrggbb), got {color}"

    def test_colors_to_json_roundtrip(self, sample_dir):
        import json
        from photosearch.colors import extract_dominant_colors, colors_to_json

        colors = extract_dominant_colors(str(Path(sample_dir) / SAMPLE_PHOTOS[0]))
        json_str = colors_to_json(colors)
        roundtripped = json.loads(json_str)
        assert roundtripped == colors


# ---------------------------------------------------------------------------
# CLIP embedding tests
# ---------------------------------------------------------------------------

class TestClipEmbedding:
    """Test CLIP image and text embeddings on real photos."""

    def test_clip_dependency(self):
        err = _check_dependency("open_clip", "pip install open-clip-torch")
        if err:
            pytest.fail(err)

    def test_embed_single_image(self, sample_photo):
        from photosearch.clip_embed import embed_image

        embedding = embed_image(sample_photo)
        assert embedding is not None, "embed_image returned None"
        assert len(embedding) == 512, f"Expected 512-dim, got {len(embedding)}"

    def test_embed_text(self):
        from photosearch.clip_embed import embed_text

        embedding = embed_text("people outdoors")
        assert embedding is not None, "embed_text returned None"
        assert len(embedding) == 512, f"Expected 512-dim, got {len(embedding)}"

    def test_embeddings_are_normalized(self, sample_photo):
        """CLIP embeddings should be L2-normalized (unit vectors)."""
        import math
        from photosearch.clip_embed import embed_image, embed_text

        img_emb = embed_image(sample_photo)
        txt_emb = embed_text("a photo")
        for label, emb in [("image", img_emb), ("text", txt_emb)]:
            magnitude = math.sqrt(sum(x * x for x in emb))
            assert abs(magnitude - 1.0) < 0.01, (
                f"{label} embedding not unit-normalized: magnitude={magnitude:.4f}"
            )

    def test_embed_batch(self, all_sample_paths):
        from photosearch.clip_embed import embed_images_batch

        embeddings = embed_images_batch(all_sample_paths, batch_size=4)
        assert len(embeddings) == len(SAMPLE_PHOTOS)
        for i, emb in enumerate(embeddings):
            assert emb is not None, f"{SAMPLE_PHOTOS[i]}: batch embedding returned None"
            assert len(emb) == 512

    def test_different_images_have_different_embeddings(self, sample_dir):
        """Distinct photos should produce distinct (non-identical) embeddings."""
        from photosearch.clip_embed import embed_image

        emb_a = embed_image(str(Path(sample_dir) / "DSC04894.JPG"))
        emb_b = embed_image(str(Path(sample_dir) / "DSC04878.JPG"))

        # They should not be identical
        assert emb_a != emb_b, "Two different photos produced identical embeddings"

        # But they're from the same shoot, so they should still be somewhat similar
        sim = sum(a * b for a, b in zip(emb_a, emb_b))
        assert sim > 0.5, (
            f"Photos from the same shoot should have reasonable similarity, "
            f"got {sim:.3f}"
        )

    def test_text_image_relevance(self, sample_dir):
        """'people' query should be more similar to people photos than landscapes."""
        from photosearch.clip_embed import embed_image, embed_text

        text_emb = embed_text("people standing outdoors")
        people_emb = embed_image(str(Path(sample_dir) / "DSC04894.JPG"))
        landscape_emb = embed_image(str(Path(sample_dir) / "DSC04878.JPG"))

        sim_people = sum(a * b for a, b in zip(text_emb, people_emb))
        sim_landscape = sum(a * b for a, b in zip(text_emb, landscape_emb))

        assert sim_people > sim_landscape, (
            f"'people standing outdoors' should match people photo "
            f"(sim={sim_people:.3f}) more than landscape (sim={sim_landscape:.3f})"
        )


# ---------------------------------------------------------------------------
# Face detection tests
# ---------------------------------------------------------------------------

class TestFaceDetection:
    """Test face detection and encoding on real photos."""

    def test_face_dependency(self):
        err = _check_dependency("insightface", "pip install insightface onnxruntime")
        if err:
            pytest.fail(err)

    def test_detect_faces_per_photo(self, sample_dir):
        from photosearch.faces import detect_faces

        for photo_name, expected_count in EXPECTED_FACE_COUNTS.items():
            path = str(Path(sample_dir) / photo_name)
            faces = detect_faces(path)
            assert len(faces) == expected_count, (
                f"{photo_name}: expected {expected_count} faces, "
                f"detected {len(faces)}"
            )

    def test_face_encoding_dimension(self, sample_dir):
        from photosearch.faces import detect_faces

        # Use a photo known to have faces
        faces = detect_faces(str(Path(sample_dir) / "DSC04894.JPG"))
        assert len(faces) >= 2
        for i, face in enumerate(faces):
            assert len(face["encoding"]) == 512, (
                f"Face {i}: expected 512-dim encoding, got {len(face['encoding'])}"
            )

    def test_face_bounding_boxes(self, sample_dir):
        from photosearch.faces import detect_faces

        faces = detect_faces(str(Path(sample_dir) / "DSC04894.JPG"))
        for i, face in enumerate(faces):
            top, right, bottom, left = face["bbox"]
            assert top < bottom, f"Face {i}: top ({top}) >= bottom ({bottom})"
            assert left < right, f"Face {i}: left ({left}) >= right ({right})"
            # Bounding box should be in original image coordinates (large values)
            assert bottom - top > 20, f"Face {i}: bbox too small vertically"
            assert right - left > 20, f"Face {i}: bbox too small horizontally"

    def test_detection_scores(self, sample_dir):
        from photosearch.faces import detect_faces

        faces = detect_faces(str(Path(sample_dir) / "DSC04894.JPG"))
        for i, face in enumerate(faces):
            assert 0.0 < face["det_score"] <= 1.0, (
                f"Face {i}: det_score {face['det_score']} out of range (0, 1]"
            )

    def test_same_person_encodings_closer(self, sample_dir):
        """Faces of the same person across photos should be closer than different people."""
        import math
        from photosearch.faces import detect_faces

        def l2_dist(a, b):
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

        # DSC04894 has Calvin + Nicole, DSC04922 has Eleanor + Calvin
        faces_894 = detect_faces(str(Path(sample_dir) / "DSC04894.JPG"))
        faces_922 = detect_faces(str(Path(sample_dir) / "DSC04922.JPG"))

        assert len(faces_894) == 2 and len(faces_922) == 2

        # Find the closest pair across the two photos (should be Calvin-Calvin)
        all_dists = []
        for i, f1 in enumerate(faces_894):
            for j, f2 in enumerate(faces_922):
                d = l2_dist(f1["encoding"], f2["encoding"])
                all_dists.append((d, i, j))
        all_dists.sort()

        # The closest cross-photo pair should be well within match tolerance
        from photosearch.faces import MATCH_TOLERANCE
        closest_dist = all_dists[0][0]
        assert closest_dist < MATCH_TOLERANCE, (
            f"Closest cross-photo face pair has L2={closest_dist:.3f}, "
            f"which exceeds MATCH_TOLERANCE={MATCH_TOLERANCE}"
        )

    def test_match_face(self, sample_dir):
        from photosearch.faces import detect_faces, match_face

        faces_894 = detect_faces(str(Path(sample_dir) / "DSC04894.JPG"))
        faces_922 = detect_faces(str(Path(sample_dir) / "DSC04922.JPG"))

        # Use first face from 894 as query, match against 922's faces
        query = faces_894[0]["encoding"]
        known = [f["encoding"] for f in faces_922]
        matches = match_face(query, known)

        # Should match at most one face (the same person)
        assert len(matches) <= 1, (
            f"Query face matched {len(matches)} faces — expected 0 or 1"
        )

    def test_encode_reference_photo(self, sample_dir):
        """encode_reference_photo should extract a single encoding from a face photo."""
        from photosearch.faces import encode_reference_photo

        # DSC04894 has 2 faces — should pick the largest
        encoding = encode_reference_photo(str(Path(sample_dir) / "DSC04894.JPG"))
        assert encoding is not None, "encode_reference_photo returned None"
        assert len(encoding) == 512

    def test_no_faces_returns_empty(self, sample_dir):
        from photosearch.faces import detect_faces

        faces = detect_faces(str(Path(sample_dir) / "DSC04878.JPG"))
        assert faces == [], f"Landscape photo should have 0 faces, got {len(faces)}"


# ---------------------------------------------------------------------------
# Description generation tests (requires Ollama)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDescriptions:
    """Test LLaVA description generation via Ollama."""

    def test_describe_photo(self, sample_photo):
        from photosearch.describe import describe_photo

        description = describe_photo(sample_photo)
        assert description is not None, "describe_photo returned None"
        assert len(description) > 20, (
            f"Description too short ({len(description)} chars): {description!r}"
        )

    def test_descriptions_mention_people(self, sample_dir):
        """Photos with people should get descriptions mentioning people."""
        from photosearch.describe import describe_photo

        people_words = {"person", "people", "man", "woman", "boy", "girl",
                        "child", "children", "kid", "figure", "someone",
                        "individual", "couple", "family", "standing",
                        "walking", "sitting"}

        for photo_name in ["DSC04894.JPG", "DSC04922.JPG"]:
            path = str(Path(sample_dir) / photo_name)
            desc = describe_photo(path)
            assert desc is not None, f"{photo_name}: no description generated"
            desc_lower = desc.lower()
            has_people_word = any(w in desc_lower for w in people_words)
            assert has_people_word, (
                f"{photo_name}: description should mention people but doesn't.\n"
                f"Description: {desc}"
            )

    def test_landscape_descriptions_no_people(self, sample_dir):
        """Landscape photos should not mention people in their descriptions."""
        from photosearch.describe import describe_photo

        for photo_name in ["DSC04878.JPG", "DSC04899.JPG"]:
            path = str(Path(sample_dir) / photo_name)
            desc = describe_photo(path)
            assert desc is not None, f"{photo_name}: no description generated"
            # These are pure landscape — description should indicate no people
            desc_lower = desc.lower()
            # We check that descriptions don't affirmatively place people in the scene
            affirming_people = {"group of people", "two people", "a man", "a woman",
                                "a boy", "a girl", "several people", "a couple"}
            for phrase in affirming_people:
                assert phrase not in desc_lower, (
                    f"{photo_name}: landscape description incorrectly mentions "
                    f"'{phrase}'. Description: {desc}"
                )

    def test_tag_photo(self, sample_photo):
        from photosearch.describe import tag_photo

        tags = tag_photo(sample_photo)
        assert tags is not None, "tag_photo returned None"
        assert isinstance(tags, list)
        assert len(tags) >= 1, "Expected at least 1 tag"
        # Tags should be short strings
        for tag in tags:
            assert isinstance(tag, str)
            assert len(tag) < 50, f"Tag too long: {tag!r}"


# ---------------------------------------------------------------------------
# Quality scoring tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestQualityScoring:
    """Test aesthetic quality scoring on real photos."""

    def test_quality_dependency(self):
        # Quality scoring uses CLIP ViT-L/14
        err = _check_dependency("open_clip", "pip install open-clip-torch")
        if err:
            pytest.fail(err)

    def test_score_photo(self, sample_photo):
        from photosearch.quality import score_photo

        score = score_photo(sample_photo)
        assert score is not None, "score_photo returned None"
        assert 1.0 <= score <= 10.0, f"Score {score} out of expected range [1, 10]"

    def test_score_photos_batch(self, all_sample_paths):
        from photosearch.quality import score_photos_batch

        scores = score_photos_batch(all_sample_paths, batch_size=4)
        assert len(scores) == len(SAMPLE_PHOTOS)
        for i, score in enumerate(scores):
            assert score is not None, f"{SAMPLE_PHOTOS[i]}: score is None"
            assert 1.0 <= score <= 10.0, (
                f"{SAMPLE_PHOTOS[i]}: score {score} out of range"
            )

    def test_analyze_photo_concepts(self, sample_photo):
        from photosearch.quality import analyze_photo_concepts

        analysis = analyze_photo_concepts(sample_photo)
        assert analysis is not None, "analyze_photo_concepts returned None"
        assert "strengths" in analysis
        assert "weaknesses" in analysis
        assert "scores" in analysis
        assert isinstance(analysis["strengths"], list)
        assert len(analysis["strengths"]) > 0


# ---------------------------------------------------------------------------
# Semantic search integration tests
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    """Test the full semantic search pipeline against an indexed sample set.

    Uses hardcoded descriptions so these tests work without Ollama.
    The descriptions mirror what LLaVA actually generates for these photos.
    """

    # Hardcoded descriptions matching LLaVA output for search relevance testing.
    # This avoids requiring Ollama for the semantic search integration tests.
    SAMPLE_DESCRIPTIONS = {
        "DSC04878.JPG": "Rocky coastline with crashing waves and no people visible.",
        "DSC04880.JPG": "Seagulls flying over a sandy beach. No one is present.",
        "DSC04894.JPG": "Two people standing on a rocky overlook above the ocean.",
        "DSC04895.JPG": "A young girl walking along a coastal trail, seen from behind.",
        "DSC04899.JPG": "Dramatic sunset over the Pacific Ocean with no people in sight.",
        "DSC04907.JPG": "Family walking along a coastal trail with ocean in the background.",
        "DSC04922.JPG": "Sunset over the Pacific with silhouettes of two people on the cliff.",
    }

    @pytest.fixture(scope="class")
    def search_db(self, sample_dir):
        """Index all sample photos with CLIP + faces + descriptions into a temp DB."""
        from photosearch.db import PhotoDB
        from photosearch.exif import extract_exif, find_raw_pair
        from photosearch.index import file_hash
        from photosearch.clip_embed import embed_image
        from photosearch.faces import detect_faces
        from photosearch.colors import extract_dominant_colors, colors_to_json

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            with PhotoDB(db_path) as db:
                db.set_photo_root(sample_dir)
                db.begin_batch(batch_size=100)

                for photo_name in SAMPLE_PHOTOS:
                    photo_path = str(Path(sample_dir) / photo_name)

                    # EXIF + metadata
                    meta = extract_exif(photo_path)
                    meta["filepath"] = db.relative_filepath(
                        str(Path(photo_path).resolve())
                    )
                    raw = find_raw_pair(photo_path)
                    if raw:
                        meta["raw_filepath"] = db.relative_filepath(
                            str(Path(raw).resolve())
                        )
                    meta["file_hash"] = file_hash(photo_path)
                    photo_id = db.add_photo(**meta)

                    # CLIP embedding
                    embedding = embed_image(photo_path)
                    if embedding:
                        db.add_clip_embedding(photo_id, embedding)

                    # Colors
                    colors = extract_dominant_colors(photo_path)
                    if colors:
                        db.conn.execute(
                            "UPDATE photos SET dominant_colors = ? WHERE id = ?",
                            (colors_to_json(colors), photo_id),
                        )

                    # Hardcoded description (avoids Ollama dependency)
                    desc = self.SAMPLE_DESCRIPTIONS.get(photo_name)
                    if desc:
                        db.conn.execute(
                            "UPDATE photos SET description = ? WHERE id = ?",
                            (desc, photo_id),
                        )

                    # Face detection
                    faces = detect_faces(photo_path)
                    for face in faces:
                        db.add_face(
                            photo_id,
                            face["bbox"],
                            face["encoding"],
                        )

                db.end_batch()
                db.conn.commit()

                # Sanity check: verify embeddings were stored
                clip_count = db.conn.execute(
                    "SELECT COUNT(*) FROM clip_embeddings"
                ).fetchone()[0]
                assert clip_count == len(SAMPLE_PHOTOS), (
                    f"search_db fixture: expected {len(SAMPLE_PHOTOS)} CLIP "
                    f"embeddings, got {clip_count}"
                )

                yield db
        finally:
            os.unlink(db_path)

    def test_search_returns_results(self, search_db):
        from photosearch.search import search_semantic

        # Use a very permissive min_score and debug to diagnose any issues
        results = search_semantic(search_db, "outdoor scene", limit=10, min_score=-1.0, debug=True)

        # Diagnostic info for failures
        clip_count = search_db.conn.execute(
            "SELECT COUNT(*) FROM clip_embeddings"
        ).fetchone()[0]
        photo_count = search_db.conn.execute(
            "SELECT COUNT(*) FROM photos"
        ).fetchone()[0]
        desc_count = search_db.conn.execute(
            "SELECT COUNT(*) FROM photos WHERE description IS NOT NULL"
        ).fetchone()[0]

        # Also verify search_clip works directly
        from photosearch.clip_embed import embed_text
        query_emb = embed_text("outdoor scene")
        direct_results = search_db.search_clip(query_emb, limit=10) if query_emb else []

        assert len(results) > 0, (
            f"Semantic search returned no results.\n"
            f"  DB state: {photo_count} photos, {clip_count} embeddings, {desc_count} descriptions\n"
            f"  embed_text returned: {'None' if query_emb is None else f'{len(query_emb)}-dim vector'}\n"
            f"  search_clip direct: {len(direct_results)} results\n"
            f"  Check stderr for debug output from search_semantic."
        )

    def test_people_query_ranks_people_photos_higher(self, search_db):
        """'people' query should rank people photos above landscapes."""
        from photosearch.search import search_semantic

        results = search_semantic(search_db, "people outdoors", limit=10, min_score=-1.0)
        scores = {r["filename"]: r.get("score", 0) for r in results}

        # At least one people photo should score higher than all landscape photos
        people_scores = [scores[f] for f in PHOTOS_WITH_PEOPLE if f in scores]
        landscape_scores = [scores[f] for f in PHOTOS_WITHOUT_PEOPLE if f in scores]

        if people_scores and landscape_scores:
            assert max(people_scores) > max(landscape_scores), (
                f"Best people photo score ({max(people_scores):.3f}) should beat "
                f"best landscape score ({max(landscape_scores):.3f})\n"
                f"People scores: {people_scores}\n"
                f"Landscape scores: {landscape_scores}"
            )

    def test_landscape_query(self, search_db):
        """'rocky coastline' should surface landscape photos."""
        from photosearch.search import search_semantic

        results = search_semantic(search_db, "rocky coastline waves", limit=7, min_score=-1.0)
        assert len(results) > 0, (
            f"Search for 'rocky coastline waves' returned no results. "
            f"DB has {search_db.conn.execute('SELECT COUNT(*) FROM clip_embeddings').fetchone()[0]} embeddings."
        )
        # DSC04878 description is "Rocky coastline with crashing waves..."
        filenames = [r["filename"] for r in results]
        assert "DSC04878.JPG" in filenames, (
            f"DSC04878.JPG (rocky coastline) should appear in results for "
            f"'rocky coastline waves', got: {filenames}"
        )

    def test_exclusion_syntax(self, search_db):
        """'outdoor -people' should exclude photos whose descriptions mention people."""
        from photosearch.search import search_semantic

        results = search_semantic(search_db, "outdoor -people", limit=10, min_score=-1.0)

        # Should return some results (the landscape photos)
        assert len(results) > 0, "Exclusion query returned no results"

        filenames = {r["filename"] for r in results}

        # Photos with "people" in their description should be excluded
        for f in ["DSC04894.JPG", "DSC04922.JPG"]:
            assert f not in filenames, (
                f"{f} should be excluded by '-people' but appeared in results"
            )


# ---------------------------------------------------------------------------
# Combined search integration tests
# ---------------------------------------------------------------------------

class TestCombinedSearch:
    """Test the combined search pipeline with real data."""

    @pytest.fixture(scope="class")
    def combined_db(self, sample_dir):
        """Index sample photos with EXIF + CLIP + faces for combined search."""
        import struct
        from photosearch.db import PhotoDB
        from photosearch.exif import extract_exif, find_raw_pair
        from photosearch.index import file_hash
        from photosearch.clip_embed import embed_image
        from photosearch.faces import detect_faces, match_faces_to_persons, FACE_ENCODING_DIM

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            with PhotoDB(db_path) as db:
                db.set_photo_root(sample_dir)
                db.begin_batch(batch_size=100)

                for photo_name in SAMPLE_PHOTOS:
                    photo_path = str(Path(sample_dir) / photo_name)
                    meta = extract_exif(photo_path)
                    meta["filepath"] = db.relative_filepath(
                        str(Path(photo_path).resolve())
                    )
                    meta["file_hash"] = file_hash(photo_path)
                    photo_id = db.add_photo(**meta)

                    embedding = embed_image(photo_path)
                    if embedding:
                        db.add_clip_embedding(photo_id, embedding)

                    faces = detect_faces(photo_path)
                    for face in faces:
                        db.add_face(photo_id, face["bbox"], face["encoding"])

                db.end_batch()
                db.conn.commit()

                # Add hardcoded descriptions for search scoring
                for photo_name in SAMPLE_PHOTOS:
                    desc = TestSemanticSearch.SAMPLE_DESCRIPTIONS.get(photo_name)
                    if desc:
                        db.conn.execute(
                            "UPDATE photos SET description = ? WHERE filename = ?",
                            (desc, photo_name),
                        )
                db.conn.commit()

                # Register a person using a face from DSC04894.
                # There's no add_face_reference() method — use raw SQL like cli.py.
                faces_894 = detect_faces(str(Path(sample_dir) / "DSC04894.JPG"))
                if len(faces_894) >= 1:
                    person_id = db.add_person("TestPerson")
                    cur = db.conn.execute(
                        "INSERT INTO face_references (person_id, source_path) VALUES (?, ?)",
                        (person_id, "test_integration"),
                    )
                    ref_id = cur.lastrowid
                    db.conn.execute(
                        "INSERT INTO face_ref_encodings (ref_id, encoding) VALUES (?, ?)",
                        (ref_id, struct.pack(f"{FACE_ENCODING_DIM}f", *faces_894[0]["encoding"])),
                    )
                    db.conn.commit()
                    match_faces_to_persons(db)

                yield db
        finally:
            os.unlink(db_path)

    def test_search_by_person(self, combined_db):
        from photosearch.search import search_combined

        results = search_combined(combined_db, person="TestPerson", limit=10)
        # Should find at least the photo where we registered the person
        assert len(results) >= 1, "Person search returned no results"

    def test_search_by_query(self, combined_db):
        from photosearch.search import search_combined

        results = search_combined(combined_db, query="ocean", limit=10, min_score=-1.0)
        assert len(results) > 0, "Query search returned no results"

    def test_search_with_date_filter(self, combined_db):
        from photosearch.search import search_combined

        # All photos are from 2026-03-13
        results = search_combined(
            combined_db, query="outdoor",
            date_from="2026-03-13", date_to="2026-03-13",
            limit=10, min_score=-1.0,
        )
        assert len(results) > 0, "Date-filtered search returned no results"

    def test_search_excludes_wrong_date(self, combined_db):
        from photosearch.search import search_combined

        # No photos from 2020
        results = search_combined(
            combined_db, query="outdoor",
            date_from="2020-01-01", date_to="2020-12-31",
            limit=10, min_score=-1.0,
        )
        assert len(results) == 0, (
            f"Expected no results for 2020 date range, got {len(results)}"
        )


# ---------------------------------------------------------------------------
# Full indexing pipeline test
# ---------------------------------------------------------------------------

class TestIndexDirectory:
    """Test the index_directory orchestration function."""

    def test_index_sample_directory(self, sample_dir):
        from photosearch.db import PhotoDB
        from photosearch.index import index_directory

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            index_directory(
                sample_dir,
                db_path=db_path,
                enable_clip=True,
                enable_colors=True,
                enable_faces=True,
                enable_describe=False,  # skip Ollama to keep test faster
                enable_quality=False,
            )

            with PhotoDB(db_path) as db:
                photo_count = db.conn.execute(
                    "SELECT COUNT(*) FROM photos"
                ).fetchone()[0]
                assert photo_count == len(SAMPLE_PHOTOS), (
                    f"Expected {len(SAMPLE_PHOTOS)} photos, got {photo_count}"
                )

                # Verify CLIP embeddings were generated
                clip_count = db.conn.execute(
                    "SELECT COUNT(*) FROM clip_embeddings"
                ).fetchone()[0]
                assert clip_count == len(SAMPLE_PHOTOS), (
                    f"Expected {len(SAMPLE_PHOTOS)} CLIP embeddings, got {clip_count}"
                )

                # Verify faces were detected
                face_count = db.conn.execute(
                    "SELECT COUNT(*) FROM faces"
                ).fetchone()[0]
                expected_faces = sum(EXPECTED_FACE_COUNTS.values())
                assert face_count == expected_faces, (
                    f"Expected {expected_faces} faces, got {face_count}"
                )

                # Verify colors were extracted
                color_count = db.conn.execute(
                    "SELECT COUNT(*) FROM photos WHERE dominant_colors IS NOT NULL"
                ).fetchone()[0]
                assert color_count == len(SAMPLE_PHOTOS), (
                    f"Expected {len(SAMPLE_PHOTOS)} photos with colors, got {color_count}"
                )
        finally:
            os.unlink(db_path)

    def test_skip_existing(self, sample_dir):
        """Running index_directory twice should skip already-indexed photos."""
        from photosearch.db import PhotoDB
        from photosearch.index import index_directory

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # First run
            index_directory(
                sample_dir, db_path=db_path,
                enable_clip=False, enable_colors=False,
                enable_faces=False, enable_describe=False,
            )

            with PhotoDB(db_path) as db:
                count_1 = db.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]

            # Second run — should skip all
            index_directory(
                sample_dir, db_path=db_path,
                enable_clip=False, enable_colors=False,
                enable_faces=False, enable_describe=False,
            )

            with PhotoDB(db_path) as db:
                count_2 = db.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]

            assert count_1 == count_2 == len(SAMPLE_PHOTOS), (
                f"Photo count changed between runs: {count_1} → {count_2}"
            )
        finally:
            os.unlink(db_path)
