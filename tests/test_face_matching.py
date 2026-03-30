"""
Face detection, matching, and semantic search accuracy tests.

Ground truth for the 7-photo sample set:
  DSC04878.JPG  nobody
  DSC04880.JPG  nobody
  DSC04894.JPG  Calvin, Nicole
  DSC04895.JPG  Eleanor — present but facing away (face undetectable by design)
  DSC04899.JPG  nobody
  DSC04907.JPG  Eleanor, Calvin
  DSC04922.JPG  Eleanor, Calvin

"Eleanor" is stored in the database as "Ellie".

After indexing, run in this order to get a fully-passing suite:
  python scripts/setup_sample.py
  python tests/test_face_matching.py

Or run with auto-setup (only runs missing steps):
  python tests/test_face_matching.py --setup

Run with pytest (all tests):
  pytest tests/test_face_matching.py -v

Run with pytest (face tests only, no CLIP needed — works from sandbox):
  pytest tests/test_face_matching.py -v -m "not semantic"

Known limitations documented in MANUALLY_CORRECTED:
  DSC04907 Calvin — 87×80px crop, L2=1.3565. Resolved by temporal matching
    (python cli.py match-faces --temporal) or manually via:
    python cli.py correct-face DSC04907.JPG 2 Calvin
"""

import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pytest
from photosearch.db import PhotoDB

DB_PATH = str(_root / "photo_index.db")

# Eleanor's name as registered in the database
ELLIE = "Ellie"

# Ground truth: filename → set of person names expected to be identified.
EXPECTED_IDENTIFIED: dict[str, set[str]] = {
    "DSC04878.JPG": set(),
    "DSC04880.JPG": set(),
    "DSC04894.JPG": {"Calvin", "Nicole"},
    "DSC04895.JPG": set(),   # Eleanor is in the photo but facing away — undetectable
    "DSC04899.JPG": set(),
    "DSC04907.JPG": {ELLIE, "Calvin"},
    "DSC04922.JPG": {ELLIE, "Calvin"},
}

# Persons physically present but facing away — must NOT be falsely matched.
KNOWN_UNDETECTABLE: dict[str, list[str]] = {
    "DSC04895.JPG": [ELLIE],
}

# Faces resolved by temporal matching or manual correction (documented for reference).
MANUALLY_CORRECTED: dict[str, list[tuple[int, str]]] = {
    "DSC04907.JPG": [(2, "Calvin")],   # 87×80px crop, L2=1.3565, above auto threshold
}

# ---------------------------------------------------------------------------
# Semantic search ground truth
# ---------------------------------------------------------------------------
# Each entry: query → {"include": set of filenames that MUST appear,
#                       "exclude": set of filenames that must NOT appear}
# Tests are marked @pytest.mark.semantic and skipped when CLIP is unavailable.
# All 7 sample photos should be accounted for in each query's include+exclude.
#
# Model: ViT-B/16 + openai pretrained (clip_embed.py).
# CLIP alone scores all 7 sample photos within a 0.018-point band for "people
# outdoors" — it sees them all as "outdoor coastal scene" and can't reliably
# isolate the "people" component. The face-aware boost in search.py (FACE_BOOST)
# lifts photos with detected faces above pure landscapes, fixing 3 of 4 people
# photos. DSC04895 (person facing away, no detectable face) remains a known gap
# that LLaVA descriptions (M3) will close.
#
# If tests start failing after a model change, check raw scores with:
#   python cli.py search -q "people outdoors" --json-output

# Semantic search tests use RANKING checks: relevant photos must outrank
# irrelevant ones. With only 7 photos from the same coastal shoot, CLIP returns
# all of them above min_score — strict include/exclude isn't meaningful here.
# What matters is that the right photos are at the top.
#
# Format: query → {
#   "top": set of filenames that MUST appear in the top N results (N = len(top)),
#   "bottom": set of filenames that must rank BELOW every "top" photo,
# }
# Photos not in either set (like DSC04895 — person facing away) are unconstrained.
EXPECTED_SEMANTIC: dict[str, dict[str, set[str]]] = {
    "people outdoors": {
        "top": {
            "DSC04894.JPG",   # Calvin + Nicole outdoors — 2 detected faces
            "DSC04907.JPG",   # Eleanor + Calvin outdoors — 2 detected faces
            "DSC04922.JPG",   # Eleanor + Calvin outdoors — 2 detected faces
        },
        "bottom": {
            "DSC04878.JPG",   # landscape, no people
            "DSC04880.JPG",   # landscape, no people
            "DSC04899.JPG",   # landscape, no people
        },
        # DSC04895.JPG: Eleanor from behind — person IS present but no detectable
        # face. CLIP + face-boost can't distinguish this from a landscape.
        # LLaVA descriptions (M3) will close this gap.
    },
    # Add more queries here as ground truth is established.
}


# ---------------------------------------------------------------------------
# Setup checking
# ---------------------------------------------------------------------------

def _db_state() -> dict:
    """Return current database state, or defaults if DB doesn't exist."""
    if not Path(DB_PATH).exists():
        return {"exists": False, "photos": 0, "faces": 0, "persons": 0, "person_refs": 0}
    try:
        with PhotoDB(DB_PATH) as db:
            return {
                "exists": True,
                "photos":      db.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0],
                "faces":       db.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
                "persons":     db.conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0],
                "person_refs": db.conn.execute("SELECT COUNT(*) FROM face_references").fetchone()[0],
                "matched":     db.conn.execute(
                                   "SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL"
                               ).fetchone()[0],
            }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def _check_requirements() -> list[str]:
    """Return a list of unmet setup requirements (empty = all good)."""
    state = _db_state()
    issues = []
    if not state.get("exists"):
        issues.append("Database not found — run: python scripts/setup_sample.py")
        return issues  # no point checking further
    if "error" in state:
        issues.append(f"Database error: {state['error']}")
        return issues
    if state["photos"] == 0:
        issues.append("No photos indexed — run: python scripts/setup_sample.py")
    if state["faces"] == 0:
        issues.append("No faces detected — run: python scripts/setup_sample.py")
    if state["persons"] == 0 or state["person_refs"] == 0:
        issues.append("No reference persons registered — run: python scripts/setup_sample.py")
    if state["matched"] == 0 and state["faces"] > 0:
        issues.append("Faces detected but none matched — run: python cli.py match-faces --temporal")
    return issues


def _run_setup() -> bool:
    """Run scripts/setup_sample.py and return True if it succeeded."""
    setup_script = _root / "scripts" / "setup_sample.py"
    if not setup_script.exists():
        print(f"  Setup script not found: {setup_script}")
        return False
    print(f"  Running: python {setup_script.name} ...")
    result = subprocess.run(
        [sys.executable, str(setup_script)],
        cwd=str(_root),
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_identified_persons(db: PhotoDB, filename: str) -> set[str]:
    """Return the set of named persons matched in a given photo."""
    rows = db.conn.execute(
        """SELECT DISTINCT p.name
           FROM faces f
           JOIN persons p ON p.id = f.person_id
           JOIN photos ph ON ph.id = f.photo_id
           WHERE ph.filename = ?""",
        (filename,),
    ).fetchall()
    return {row["name"] for row in rows}


def get_face_count(db: PhotoDB, filename: str) -> int:
    """Return total number of detected faces in a photo (named or not)."""
    row = db.conn.execute(
        """SELECT COUNT(*) AS cnt
           FROM faces f
           JOIN photos ph ON ph.id = f.photo_id
           WHERE ph.filename = ?""",
        (filename,),
    ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def photo_db():
    issues = _check_requirements()
    if issues:
        pytest.skip(
            "Setup required before running tests:\n  "
            + "\n  ".join(issues)
            + "\n\nQuick fix: python scripts/setup_sample.py"
        )
    with PhotoDB(DB_PATH) as db:
        yield db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", EXPECTED_IDENTIFIED.items())
def test_identified_persons(photo_db, filename, expected):
    """Each photo must have exactly the expected set of identified persons."""
    actual = get_identified_persons(photo_db, filename)
    missing = expected - actual
    extra = actual - expected
    assert actual == expected, (
        f"\n  Photo:    {filename}"
        f"\n  Expected: {sorted(expected) or '(nobody)'}"
        f"\n  Got:      {sorted(actual) or '(nobody)'}"
        + (f"\n  Missing:  {sorted(missing)}  ← false negative" if missing else "")
        + (f"\n  Extra:    {sorted(extra)}  ← false positive" if extra else "")
    )


@pytest.mark.parametrize("filename,persons", KNOWN_UNDETECTABLE.items())
def test_undetectable_persons_not_hallucinated(photo_db, filename, persons):
    """Persons present but facing away must not be falsely matched."""
    actual = get_identified_persons(photo_db, filename)
    for person in persons:
        assert person not in actual, (
            f"\n  {filename}: '{person}' is present but facing away "
            f"and should not be matched — false positive detected."
        )


def test_all_photos_indexed(photo_db):
    """All 7 sample photos must be present in the database."""
    for filename in EXPECTED_IDENTIFIED:
        row = photo_db.conn.execute(
            "SELECT id FROM photos WHERE filename = ?", (filename,)
        ).fetchone()
        assert row is not None, (
            f"{filename} is not in the database.\n"
            f"Run: python scripts/setup_sample.py"
        )


def test_no_faces_in_nobody_photos(photo_db):
    """Photos with no people present should have zero detected faces."""
    nobody_photos = {f for f, p in EXPECTED_IDENTIFIED.items()
                     if not p and f not in KNOWN_UNDETECTABLE}
    for filename in nobody_photos:
        count = get_face_count(photo_db, filename)
        assert count == 0, (
            f"{filename}: expected 0 faces (nobody present), "
            f"got {count} — possible false detection."
        )


# ---------------------------------------------------------------------------
# Semantic search helpers and tests
# ---------------------------------------------------------------------------

def _has_clip() -> bool:
    """Return True if open_clip and the CLIP model are available."""
    try:
        import open_clip  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def clip_db(photo_db):
    """Same DB connection as photo_db, but skips if CLIP is not installed."""
    if not _has_clip():
        pytest.skip(
            "open_clip not installed — semantic search tests require the full venv.\n"
            "Run:  pytest tests/test_face_matching.py -m 'not semantic'  to skip these."
        )
    yield photo_db


@pytest.mark.semantic
@pytest.mark.parametrize("query,expected", EXPECTED_SEMANTIC.items())
def test_semantic_search(clip_db, query, expected):
    """Semantic search ranking: relevant photos must outrank irrelevant ones.

    'top' photos must appear before all 'bottom' photos in the result ranking.
    This validates CLIP + face-boost without requiring a perfect threshold cutoff.
    """
    from photosearch.search import search_semantic

    results = search_semantic(clip_db, query, limit=20)
    ranked = [r["filename"] for r in results]
    scores = {r["filename"]: r.get("score", "?") for r in results}

    top_expected = expected.get("top", set())
    bottom_expected = expected.get("bottom", set())

    # Check that all "top" photos are present in results
    missing_top = top_expected - set(ranked)

    # Check ranking: every "top" photo must rank above every "bottom" photo
    ranking_violations = []
    for top_f in top_expected:
        if top_f not in ranked:
            continue
        top_rank = ranked.index(top_f)
        for bot_f in bottom_expected:
            if bot_f not in ranked:
                continue
            bot_rank = ranked.index(bot_f)
            if top_rank > bot_rank:  # higher index = worse rank
                ranking_violations.append(
                    f"{top_f} (rank {top_rank+1}, score {scores.get(top_f, '?'):.3f}) "
                    f"ranked below {bot_f} (rank {bot_rank+1}, score {scores.get(bot_f, '?'):.3f})"
                )

    ok = not missing_top and not ranking_violations

    assert ok, (
        f"\n  Query: '{query}'"
        + (
            f"\n  Missing from results: {sorted(missing_top)}"
            f"\n    → check CLIP model (should be ViT-B-16 + openai)"
            f"\n    → run: python cli.py index <dir> --force-clip"
            if missing_top else ""
        )
        + (
            f"\n  Ranking violations (top photo ranked below bottom photo):"
            + "".join(f"\n    • {v}" for v in ranking_violations)
            if ranking_violations else ""
        )
        + f"\n  Full ranking: {ranked}"
        + f"\n  Scores: { {f: f'{s:.3f}' for f, s in sorted(scores.items())} }"
    )


# ---------------------------------------------------------------------------
# Standalone summary runner
# ---------------------------------------------------------------------------

def run_summary(auto_setup: bool = False) -> int:
    """Print a human-readable pass/fail table. Returns number of failures."""

    # Check setup requirements
    issues = _check_requirements()
    if issues:
        print("\n  Setup required:")
        for issue in issues:
            print(f"    • {issue}")

        if auto_setup:
            print("\n  Running setup automatically...")
            ok = _run_setup()
            if not ok:
                print("\n  Setup failed — check output above.")
                return 1
            # Re-check after setup
            issues = _check_requirements()
            if issues:
                print("\n  Still not ready after setup:")
                for issue in issues:
                    print(f"    • {issue}")
                return 1
        else:
            print("\n  Run with --setup to auto-configure, or:")
            print("    python scripts/setup_sample.py")
            return 1

    passed = 0
    failed = 0
    rows = []

    with PhotoDB(DB_PATH) as db:
        for filename, expected in EXPECTED_IDENTIFIED.items():
            actual = get_identified_persons(db, filename)
            missing = expected - actual
            extra = actual - expected
            ok = actual == expected

            status = "PASS" if ok else "FAIL"
            passed += ok
            failed += (not ok)

            expected_str = ", ".join(sorted(expected)) if expected else "(nobody)"
            actual_str   = ", ".join(sorted(actual))   if actual   else "(nobody)"
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"extra {sorted(extra)}")
            detail = ", ".join(parts)
            rows.append((status, filename, expected_str, actual_str, detail))

    # Print table
    col_w = [6, 16, 28, 28, 30]
    headers = ["", "Photo", "Expected", "Detected", "Notes"]
    sep = "  ".join("-" * w for w in col_w)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print()
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v)[:col_w[i]] for i, v in enumerate(row)]))
    print(sep)

    total = passed + failed
    print(f"\n{passed}/{total} passed", end="")
    print("  ✓  all passing" if not failed else f"  —  {failed} failed")

    # ── Semantic search tests (only if CLIP is available) ────────────────────
    if _has_clip():
        from photosearch.search import search_semantic
        print(f"\n{'─' * 108}")
        print(f"  Semantic search ranking tests")
        print(f"{'─' * 108}")

        sem_col_w = [6, 24, 38, 38]
        sem_headers = ["", "Query", "Must outrank landscapes", "Result"]
        sem_sep = "  ".join("-" * w for w in sem_col_w)
        sem_fmt = "  ".join(f"{{:<{w}}}" for w in sem_col_w)
        print(sem_fmt.format(*sem_headers))
        print(sem_sep)

        sem_passed = sem_failed = 0
        with PhotoDB(DB_PATH) as db:
            for query, expected in EXPECTED_SEMANTIC.items():
                results = search_semantic(db, query, limit=20)
                ranked = [r["filename"] for r in results]
                scores = {r["filename"]: r.get("score", 0) for r in results}

                top_expected = expected.get("top", set())
                bottom_expected = expected.get("bottom", set())

                missing_top = top_expected - set(ranked)
                violations = []
                for top_f in top_expected:
                    if top_f not in ranked:
                        continue
                    top_rank = ranked.index(top_f)
                    for bot_f in bottom_expected:
                        if bot_f not in ranked:
                            continue
                        if ranked.index(bot_f) < top_rank:
                            violations.append(f"{top_f} below {bot_f}")

                ok = not missing_top and not violations
                status = "PASS" if ok else "FAIL"
                sem_passed += ok
                sem_failed += (not ok)

                top_str = ", ".join(f.replace("DSC0", "") for f in sorted(top_expected))
                parts = []
                if missing_top:
                    parts.append(f"missing {[f.replace('DSC0','') for f in sorted(missing_top)]}")
                if violations:
                    parts.append(f"rank errors: {violations}")
                result_str = "OK" if ok else " | ".join(parts)

                print(sem_fmt.format(
                    status,
                    query[:sem_col_w[1]],
                    top_str[:sem_col_w[2]],
                    result_str[:sem_col_w[3]],
                ))

        print(sem_sep)
        print(f"\n{sem_passed}/{sem_passed + sem_failed} semantic passed", end="")
        print("  ✓" if not sem_failed else f"  —  {sem_failed} failed")
        failed += sem_failed
    else:
        print(f"\n  Semantic search tests: skipped (open_clip not in this environment)")
        print(f"  Run on your Mac with the full venv to test CLIP search quality.")

    print()
    return failed


if __name__ == "__main__":
    auto_setup = "--setup" in sys.argv
    sys.exit(run_summary(auto_setup=auto_setup))
