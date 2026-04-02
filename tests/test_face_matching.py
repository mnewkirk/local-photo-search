"""
Face detection and matching accuracy tests.

Ground truth for the 7-photo sample set:
  DSC04878.JPG  nobody
  DSC04880.JPG  nobody
  DSC04894.JPG  Calvin, Nicole
  DSC04895.JPG  Ellie — present but facing away (face undetectable by design)
  DSC04899.JPG  nobody
  DSC04907.JPG  Ellie, Calvin
  DSC04922.JPG  Ellie, Calvin

"Ellie" is stored in the database as "Ellie".

After indexing, run in this order to get a fully-passing suite:
  python scripts/setup_sample.py
  python tests/test_face_matching.py

Or run with auto-setup (only runs missing steps):
  python tests/test_face_matching.py --setup

Run with pytest:
  pytest tests/test_face_matching.py -v

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

# Ellie's name as registered in the database
ELLIE = "Ellie"

# Ground truth: filename → set of person names expected to be identified.
EXPECTED_IDENTIFIED: dict[str, set[str]] = {
    "DSC04878.JPG": set(),
    "DSC04880.JPG": set(),
    "DSC04894.JPG": {"Calvin", "Nicole"},
    "DSC04895.JPG": set(),   # Ellie is in the photo but facing away — undetectable
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

    print()
    return failed


if __name__ == "__main__":
    auto_setup = "--setup" in sys.argv
    sys.exit(run_summary(auto_setup=auto_setup))
