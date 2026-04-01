"""
Tests for photosearch/verify.py — hallucination detection and verification.

Covers:
  - _extract_nouns: pure text processing (stop words, dedup, short words)
  - _flag_by_clip: threshold logic with synthetic scores
  - clip_score_description / clip_score_tags: CLIP-based scoring (mocked embed_text)
  - llm_verify_description: response parsing (mocked Ollama)
  - verify_photo: full two-pass pipeline
  - _save_verification: DB persistence

Uses mocked CLIP and Ollama where needed; no real ML models required
for unit tests.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from photosearch.verify import (
    _extract_nouns,
    _flag_by_clip,
    _save_verification,
    clip_score_description,
    clip_score_tags,
    llm_verify_description,
    verify_photo,
    verify_photos,
    _STOP_WORDS,
    _ABSTRACT_PHRASES,
)
from photosearch.db import PhotoDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_vec(dim: int = 512, seed: int = 0) -> list[float]:
    """Create a deterministic unit vector as a plain list."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _mock_embed_text_factory(base_seed: int = 0):
    """Return a mock embed_text that produces deterministic vectors per query string."""
    cache = {}
    def _embed(query: str):
        if query not in cache:
            seed = base_seed + hash(query) % 10000
            cache[query] = _make_unit_vec(seed=seed)
        return cache[query]
    return _embed


# ---------------------------------------------------------------------------
# _extract_nouns tests
# ---------------------------------------------------------------------------

class TestExtractNouns:
    """Tests for the noun extraction heuristic."""

    def test_basic_extraction(self):
        """Should extract content words, skipping stop words."""
        nouns = _extract_nouns("A dog sitting on a bench in the park")
        assert "dog" in nouns
        assert "bench" in nouns
        assert "park" in nouns

    def test_stop_words_removed(self):
        """Common stop words should not appear in results."""
        nouns = _extract_nouns("The cat is sitting on the mat")
        for word in ["the", "is", "on"]:
            assert word not in nouns

    def test_short_words_removed(self):
        """Words with 2 or fewer characters should be excluded."""
        nouns = _extract_nouns("Go to NY on a bus")
        assert "go" not in nouns
        assert "to" not in nouns
        assert "ny" not in nouns
        assert "bus" in nouns

    def test_deduplication(self):
        """Repeated words should appear only once."""
        nouns = _extract_nouns("tree tree tree tree forest tree")
        assert nouns.count("tree") == 1

    def test_abstract_phrases_excluded(self):
        """Words in _ABSTRACT_PHRASES should be excluded."""
        nouns = _extract_nouns("The composition has great lighting and contrast with a mountain")
        for word in ["composition", "lighting", "contrast"]:
            assert word not in nouns
        assert "mountain" in nouns

    def test_empty_string(self):
        """Empty input should return empty list."""
        assert _extract_nouns("") == []

    def test_all_stop_words(self):
        """Input of only stop words should return empty list."""
        assert _extract_nouns("the and or but in on at to for") == []

    def test_hyphenated_words(self):
        """Hyphenated words should be kept as one token."""
        nouns = _extract_nouns("A well-lit mountain with a snow-capped peak")
        # The regex captures hyphenated forms
        hyphenated = [n for n in nouns if "-" in n]
        assert len(hyphenated) >= 1

    def test_case_insensitive(self):
        """Extraction should be case-insensitive."""
        nouns = _extract_nouns("A Mountain and a RIVER near the Valley")
        assert "mountain" in nouns
        assert "river" in nouns
        assert "valley" in nouns


# ---------------------------------------------------------------------------
# _flag_by_clip tests
# ---------------------------------------------------------------------------

class TestFlagByClip:
    """Tests for the CLIP threshold flagging logic."""

    def test_flags_below_absolute_threshold(self):
        """Items below clip_threshold should be flagged."""
        desc_scores = [
            {"noun": "dog", "similarity": 0.30},
            {"noun": "unicorn", "similarity": 0.10},
        ]
        tag_scores = [
            {"tag": "beach", "similarity": 0.25},
        ]
        desc_flagged, tag_flagged, all_items = _flag_by_clip(
            desc_scores, tag_scores, clip_threshold=0.20
        )
        flagged_nouns = {f["noun"] for f in desc_flagged}
        assert "unicorn" in flagged_nouns
        assert "dog" not in flagged_nouns

    def test_flags_below_relative_threshold(self):
        """Items far below the median should be flagged even if above absolute threshold."""
        # All items have high similarity except one outlier
        desc_scores = [
            {"noun": "ocean", "similarity": 0.50},
            {"noun": "waves", "similarity": 0.48},
            {"noun": "rocks", "similarity": 0.47},
            {"noun": "sand", "similarity": 0.45},
            {"noun": "alien", "similarity": 0.15},  # far below median
        ]
        tag_scores = []
        desc_flagged, tag_flagged, all_items = _flag_by_clip(
            desc_scores, tag_scores, clip_threshold=0.10  # low absolute
        )
        flagged_nouns = {f["noun"] for f in desc_flagged}
        assert "alien" in flagged_nouns

    def test_no_flags_when_all_similar(self):
        """If all scores are high and close together, nothing should be flagged."""
        desc_scores = [
            {"noun": "ocean", "similarity": 0.40},
            {"noun": "waves", "similarity": 0.38},
            {"noun": "rocks", "similarity": 0.39},
        ]
        tag_scores = []
        desc_flagged, tag_flagged, all_items = _flag_by_clip(
            desc_scores, tag_scores, clip_threshold=0.18
        )
        assert len(desc_flagged) == 0

    def test_empty_scores(self):
        """Empty input should return empty flags."""
        desc_flagged, tag_flagged, all_items = _flag_by_clip([], [], 0.18)
        assert desc_flagged == []
        assert tag_flagged == []
        assert all_items == []

    def test_tag_flagging(self):
        """Tags should be flagged independently from description nouns."""
        desc_scores = []
        tag_scores = [
            {"tag": "sunset", "similarity": 0.35},
            {"tag": "spaceship", "similarity": 0.05},
        ]
        desc_flagged, tag_flagged, all_items = _flag_by_clip(
            desc_scores, tag_scores, clip_threshold=0.20
        )
        flagged_tags = {f["tag"] for f in tag_flagged}
        assert "spaceship" in flagged_tags
        assert "sunset" not in flagged_tags

    def test_all_items_returned(self):
        """all_items should contain every scored item from both sources."""
        desc_scores = [{"noun": "dog", "similarity": 0.30}]
        tag_scores = [{"tag": "beach", "similarity": 0.25}]
        _, _, all_items = _flag_by_clip(desc_scores, tag_scores, 0.18)
        assert len(all_items) == 2
        types = {item["type"] for item in all_items}
        assert types == {"description", "tag"}


# ---------------------------------------------------------------------------
# clip_score_description / clip_score_tags tests (mocked CLIP)
# ---------------------------------------------------------------------------

class TestClipScoring:
    """Tests for CLIP-based description and tag scoring (with mocked embed_text)."""

    @patch("photosearch.clip_embed.embed_text")
    def test_score_description_returns_sorted(self, mock_embed):
        """Results should be sorted by similarity ascending."""
        # Return different vectors for different nouns
        mock_embed.side_effect = _mock_embed_text_factory()
        photo_emb = _make_unit_vec(seed=42)

        scored = clip_score_description(photo_emb, "A dog running on the beach near a lighthouse")
        assert len(scored) > 0
        # Verify ascending sort
        sims = [s["similarity"] for s in scored]
        assert sims == sorted(sims)

    @patch("photosearch.clip_embed.embed_text")
    def test_score_description_has_nouns(self, mock_embed):
        """Each result should have 'noun' and 'similarity' keys."""
        mock_embed.side_effect = _mock_embed_text_factory()
        photo_emb = _make_unit_vec(seed=42)

        scored = clip_score_description(photo_emb, "A mountain with trees and a river")
        for item in scored:
            assert "noun" in item
            assert "similarity" in item
            assert isinstance(item["similarity"], float)

    @patch("photosearch.clip_embed.embed_text")
    def test_score_description_empty(self, mock_embed):
        """Empty description should return empty list."""
        scored = clip_score_description(_make_unit_vec(), "")
        assert scored == []
        mock_embed.assert_not_called()

    @patch("photosearch.clip_embed.embed_text")
    def test_score_tags_returns_sorted(self, mock_embed):
        """Tag scores should be sorted ascending by similarity."""
        mock_embed.side_effect = _mock_embed_text_factory()
        photo_emb = _make_unit_vec(seed=42)

        scored = clip_score_tags(photo_emb, ["sunset", "ocean", "mountain", "car"])
        sims = [s["similarity"] for s in scored]
        assert sims == sorted(sims)

    @patch("photosearch.clip_embed.embed_text")
    def test_score_tags_has_tag_key(self, mock_embed):
        """Each result should have 'tag' and 'similarity' keys."""
        mock_embed.side_effect = _mock_embed_text_factory()
        photo_emb = _make_unit_vec(seed=42)

        scored = clip_score_tags(photo_emb, ["beach", "waves"])
        for item in scored:
            assert "tag" in item
            assert "similarity" in item

    @patch("photosearch.clip_embed.embed_text")
    def test_score_tags_empty(self, mock_embed):
        """Empty tags list should return empty list."""
        scored = clip_score_tags(_make_unit_vec(), [])
        assert scored == []

    @patch("photosearch.clip_embed.embed_text")
    def test_score_description_skips_none_embeddings(self, mock_embed):
        """If embed_text returns None for a noun, it should be skipped."""
        def _sometimes_none(query):
            if "unicorn" in query:
                return None
            return _make_unit_vec(seed=hash(query) % 10000)

        mock_embed.side_effect = _sometimes_none
        photo_emb = _make_unit_vec(seed=42)

        scored = clip_score_description(photo_emb, "A unicorn standing near a castle")
        nouns_in_result = {s["noun"] for s in scored}
        assert "unicorn" not in nouns_in_result
        assert "castle" in nouns_in_result


# ---------------------------------------------------------------------------
# llm_verify_description tests (mocked Ollama)
# ---------------------------------------------------------------------------

class TestLlmVerifyDescription:
    """Tests for parsing LLM verification responses."""

    @patch("photosearch.describe._ollama_chat_with_retry")
    def test_all_correct_response(self, mock_chat):
        """'ALL CORRECT' response should return empty list."""
        mock_chat.return_value = "ALL CORRECT"
        result = llm_verify_description("/fake/photo.jpg", "A dog on a beach", ["dog", "beach"])
        assert result == []

    @patch("photosearch.describe._ollama_chat_with_retry")
    def test_wrong_items_parsed(self, mock_chat):
        """'WRONG:' lines should be parsed into confirmed hallucinations."""
        mock_chat.return_value = "WRONG: unicorn\nWRONG: spaceship"
        result = llm_verify_description("/fake/photo.jpg", "A unicorn near a spaceship", [])
        assert len(result) == 2
        nouns = {r["noun"] for r in result}
        assert "unicorn" in nouns
        assert "spaceship" in nouns
        for r in result:
            assert r["llm_says"] == "NO"

    @patch("photosearch.describe._ollama_chat_with_retry")
    def test_mixed_response(self, mock_chat):
        """Only 'WRONG:' lines should be extracted, ignoring other text."""
        mock_chat.return_value = (
            "The description is mostly accurate.\n"
            "WRONG: dragon\n"
            "The rest looks fine."
        )
        result = llm_verify_description("/fake/photo.jpg", "A dragon on a beach", [])
        assert len(result) == 1
        assert result[0]["noun"] == "dragon"

    @patch("photosearch.describe._ollama_chat_with_retry")
    def test_empty_description_and_tags(self, mock_chat):
        """Empty description and tags should return empty without calling LLM."""
        result = llm_verify_description("/fake/photo.jpg", "", [])
        assert result == []
        mock_chat.assert_not_called()

    @patch("photosearch.describe._ollama_chat_with_retry")
    def test_ollama_failure_returns_empty(self, mock_chat):
        """If Ollama raises an exception, should return empty list gracefully."""
        mock_chat.side_effect = Exception("Connection refused")
        result = llm_verify_description("/fake/photo.jpg", "A dog", ["dog"])
        assert result == []

    @patch("photosearch.describe._ollama_chat_with_retry")
    def test_none_response_returns_empty(self, mock_chat):
        """If Ollama returns None, should return empty list."""
        mock_chat.return_value = None
        result = llm_verify_description("/fake/photo.jpg", "A dog", [])
        assert result == []

    @patch("photosearch.describe._ollama_chat_with_retry")
    def test_wrong_strips_trailing_period(self, mock_chat):
        """Trailing periods on WRONG items should be stripped."""
        mock_chat.return_value = "WRONG: a flying saucer."
        result = llm_verify_description("/fake/photo.jpg", "desc", [])
        assert result[0]["noun"] == "a flying saucer"

    @patch("photosearch.describe._ollama_chat_with_retry")
    def test_case_insensitive_wrong(self, mock_chat):
        """'wrong:' should be matched case-insensitively."""
        mock_chat.return_value = "wrong: helicopter"
        result = llm_verify_description("/fake/photo.jpg", "desc", [])
        assert len(result) == 1
        assert result[0]["noun"] == "helicopter"


# ---------------------------------------------------------------------------
# _save_verification tests
# ---------------------------------------------------------------------------

class TestSaveVerification:
    """Tests for writing verification status to the DB."""

    @pytest.fixture
    def verify_db(self, tmp_path):
        db_path = str(tmp_path / "verify_test.db")
        db = PhotoDB(db_path)
        pid = db.add_photo(filepath="/test/photo.jpg", filename="photo.jpg")
        db.conn.commit()
        db._test_pid = pid
        yield db
        db.close()

    def test_saves_pass_status(self, verify_db):
        _save_verification(verify_db, verify_db._test_pid, "pass", [])
        row = verify_db.conn.execute(
            "SELECT verification_status, verified_at FROM photos WHERE id = ?",
            (verify_db._test_pid,),
        ).fetchone()
        assert row["verification_status"] == "pass"
        assert row["verified_at"] is not None

    def test_saves_fail_status_with_flags(self, verify_db):
        flags = [{"type": "description", "noun": "unicorn", "similarity": 0.12}]
        _save_verification(verify_db, verify_db._test_pid, "fail", flags)
        row = verify_db.conn.execute(
            "SELECT verification_status, hallucination_flags FROM photos WHERE id = ?",
            (verify_db._test_pid,),
        ).fetchone()
        assert row["verification_status"] == "fail"
        loaded_flags = json.loads(row["hallucination_flags"])
        assert loaded_flags[0]["noun"] == "unicorn"

    def test_overwrites_previous_status(self, verify_db):
        _save_verification(verify_db, verify_db._test_pid, "fail", [])
        _save_verification(verify_db, verify_db._test_pid, "pass", [])
        row = verify_db.conn.execute(
            "SELECT verification_status FROM photos WHERE id = ?",
            (verify_db._test_pid,),
        ).fetchone()
        assert row["verification_status"] == "pass"


# ---------------------------------------------------------------------------
# verify_photo integration test (mocked CLIP + Ollama)
# ---------------------------------------------------------------------------

class TestVerifyPhoto:
    """Integration tests for the full two-pass verification pipeline."""

    @pytest.fixture
    def photo_db(self, tmp_path):
        """DB with a photo that has a description, tags, and CLIP embedding."""
        db_path = str(tmp_path / "verify_int_test.db")
        db = PhotoDB(db_path)
        db.set_photo_root(str(tmp_path))

        # Create a dummy image file
        img_dir = tmp_path / "photos"
        img_dir.mkdir()
        (img_dir / "test.jpg").write_bytes(b"fake")

        pid = db.add_photo(
            filepath="photos/test.jpg",
            filename="test.jpg",
            description="A dog playing fetch on a sandy beach with waves in the background",
            tags=json.dumps(["dog", "beach", "waves", "fetch"]),
        )
        emb = _make_unit_vec(seed=42)
        db.add_clip_embedding(pid, emb)
        db.conn.commit()

        db._test_pid = pid
        db._test_emb = emb
        yield db
        db.close()

    @patch("photosearch.clip_embed.embed_text")
    def test_pass_when_no_flags(self, mock_embed, photo_db):
        """Photo should pass when CLIP doesn't flag anything."""
        # Make all nouns score high similarity
        mock_embed.side_effect = lambda q: _make_unit_vec(seed=42)

        photo = dict(photo_db.conn.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_db._test_pid,)
        ).fetchone())

        result = verify_photo(photo_db, photo, photo_embedding=photo_db._test_emb)
        assert result["status"] == "pass"
        assert result["llm_confirmed"] == []
        assert result["regenerated"] is False

    @patch("photosearch.verify.llm_verify_description")
    @patch("photosearch.verify.clip_score_tags")
    @patch("photosearch.verify.clip_score_description")
    def test_fail_when_hallucination_confirmed(self, mock_desc_score, mock_tag_score,
                                                mock_llm_verify, photo_db):
        """Photo should fail when both CLIP and LLM confirm a hallucination."""
        # CLIP flags "unicorn" with very low similarity
        mock_desc_score.return_value = [
            {"noun": "dog", "similarity": 0.35},
            {"noun": "beach", "similarity": 0.30},
            {"noun": "unicorn", "similarity": 0.05},
        ]
        mock_tag_score.return_value = [
            {"tag": "dog", "similarity": 0.35},
            {"tag": "beach", "similarity": 0.30},
        ]
        # LLM confirms unicorn is wrong
        mock_llm_verify.return_value = [{"noun": "unicorn", "llm_says": "NO"}]

        photo = dict(photo_db.conn.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_db._test_pid,)
        ).fetchone())

        # Mock embed_text for the CLIP cross-check (Pass 3)
        with patch("photosearch.clip_embed.embed_text") as mock_embed:
            # Return a vector with low similarity to photo for "unicorn"
            mock_embed.return_value = _make_unit_vec(seed=9999)
            result = verify_photo(
                photo_db, photo, photo_embedding=photo_db._test_emb,
                auto_regenerate=False,
            )

        assert result["status"] == "fail"
        assert len(result["llm_confirmed"]) >= 1

    def test_pass_with_no_description(self, photo_db):
        """Photo with no description should pass immediately."""
        photo_db.conn.execute(
            "UPDATE photos SET description = NULL, tags = NULL WHERE id = ?",
            (photo_db._test_pid,),
        )
        photo_db.conn.commit()

        photo = dict(photo_db.conn.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_db._test_pid,)
        ).fetchone())

        result = verify_photo(photo_db, photo)
        assert result["status"] == "pass"

    def test_pass_with_no_embedding(self, photo_db):
        """Photo with no CLIP embedding should pass (can't verify)."""
        # Remove the embedding
        photo_db.conn.execute(
            "DELETE FROM clip_embeddings WHERE photo_id = ?", (photo_db._test_pid,)
        )
        photo_db.conn.commit()

        photo = dict(photo_db.conn.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_db._test_pid,)
        ).fetchone())

        result = verify_photo(photo_db, photo)
        assert result["status"] == "pass"

    @patch("photosearch.clip_embed.embed_text")
    def test_llm_all_forces_llm_pass(self, mock_embed, photo_db):
        """llm_all=True should run LLM even when CLIP doesn't flag anything."""
        # All nouns score high — CLIP won't flag
        mock_embed.side_effect = lambda q: _make_unit_vec(seed=42)

        photo = dict(photo_db.conn.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_db._test_pid,)
        ).fetchone())

        with patch("photosearch.verify.llm_verify_description") as mock_llm:
            mock_llm.return_value = []  # LLM says all correct
            result = verify_photo(
                photo_db, photo, photo_embedding=photo_db._test_emb,
                llm_all=True,
            )
            mock_llm.assert_called_once()

        assert result["status"] == "pass"


# ---------------------------------------------------------------------------
# verify_photos batch test
# ---------------------------------------------------------------------------

class TestVerifyPhotos:
    """Tests for the batch verification wrapper."""

    @pytest.fixture
    def batch_db(self, tmp_path):
        db_path = str(tmp_path / "batch_test.db")
        db = PhotoDB(db_path)
        db.set_photo_root(str(tmp_path))

        img_dir = tmp_path / "photos"
        img_dir.mkdir()

        pids = []
        for i in range(3):
            fname = f"photo_{i}.jpg"
            (img_dir / fname).write_bytes(b"fake")
            pid = db.add_photo(
                filepath=f"photos/{fname}",
                filename=fname,
                description=f"Description for photo {i}",
                tags=json.dumps([f"tag_{i}"]),
            )
            db.add_clip_embedding(pid, _make_unit_vec(seed=i))
            pids.append(pid)

        db.conn.commit()
        db._test_pids = pids
        yield db
        db.close()

    @patch("photosearch.clip_embed.embed_text")
    def test_batch_processes_all_photos(self, mock_embed, batch_db):
        """Batch verification should process every photo."""
        mock_embed.side_effect = lambda q: _make_unit_vec(seed=42)

        photos = [dict(r) for r in batch_db.conn.execute("SELECT * FROM photos").fetchall()]
        stats = verify_photos(batch_db, photos=photos)
        assert stats["checked"] == 3
        assert stats["total"] == 3

    @patch("photosearch.clip_embed.embed_text")
    def test_batch_returns_zero_when_empty(self, mock_embed, batch_db):
        """Empty photo list should return zero counts."""
        stats = verify_photos(batch_db, photos=[])
        assert stats["total"] == 0
        assert stats["checked"] == 0
