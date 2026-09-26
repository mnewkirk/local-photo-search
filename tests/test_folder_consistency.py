"""`photos.folder` must follow `photos.filepath` through every writer.

`folder` (schema v25) is dirname(filepath) and is what ingest batches
(`WHERE folder = ?`), /review, /geotag and the worker fleet's directory scope
read. `relocate-into-year-dirs` and `db.remap_paths` used to rewrite filepath
alone, leaving folder stale and silently dropping the photos from all of
those. Every writer now routes through `db.set_photo_filepath`.
"""

import os
import re
from pathlib import Path

from click.testing import CliRunner

from photosearch.db import PhotoDB, folder_for_path, set_photo_filepath

REPO = Path(__file__).resolve().parent.parent


def _folders(db):
    return {r["id"]: (r["filepath"], r["folder"]) for r in db.conn.execute(
        "SELECT id, filepath, folder FROM photos")}


def _assert_consistent(db):
    for pid, (fp, folder) in _folders(db).items():
        assert folder == os.path.dirname(fp), (pid, fp, folder)


def test_folder_for_path_matches_add_photo(tmp_db_path):
    with PhotoDB(tmp_db_path) as db:
        for fp in ("2026/2026-09-19_ILCE-7RM6/DSC01.JPG", "/abs/dir/x.jpg", "top.jpg"):
            pid = db.add_photo(filepath=fp, filename=os.path.basename(fp))
            row = db.conn.execute("SELECT folder FROM photos WHERE id=?", (pid,)).fetchone()
            assert row["folder"] == folder_for_path(fp)


def test_remap_paths_rederives_folder(tmp_db_path):
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="/old/root/2024/a.jpg", filename="a.jpg",
                     raw_filepath="/old/root/2024/a.ARW")
        db.add_photo(filepath="/other/b.jpg", filename="b.jpg")
        assert db.remap_paths("/old/root", "/new/root") == 1
        _assert_consistent(db)
        row = db.get_photo_by_path("/new/root/2024/a.jpg")
        assert row["folder"] == "/new/root/2024"
        assert row["raw_filepath"] == "/new/root/2024/a.ARW"
        # the untouched row keeps its folder
        assert db.get_photo_by_path("/other/b.jpg")["folder"] == "/other"
        # and a folder-keyed lookup (batches, /review) now finds the moved photo
        assert [r[0] for r in db.conn.execute(
            "SELECT id FROM photos WHERE folder = ?", ("/new/root/2024",))] == [row["id"]]


def test_remap_paths_cli_rederives_folder(tmp_db_path):
    from cli import cli
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="/old/2024/a.jpg", filename="a.jpg")
    res = CliRunner().invoke(cli, ["remap-paths", "/old", "/photos", "--db", tmp_db_path])
    assert res.exit_code == 0, res.output
    with PhotoDB(tmp_db_path) as db:
        _assert_consistent(db)
        assert db.get_photo_by_path("/photos/2024/a.jpg")["folder"] == "/photos/2024"


def test_relocate_into_year_dirs_rederives_folder(tmp_db_path, tmp_path):
    from cli import cli
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2013-05-04 Party/a.jpg", filename="a.jpg",
                     raw_filepath="2013-05-04 Party/a.NEF")
        db.add_photo(filepath="2013-05-04 Party/b.jpg", filename="b.jpg")
        db.add_photo(filepath="2020/2020-01-01_x/c.jpg", filename="c.jpg")
    res = CliRunner().invoke(cli, [
        "relocate-into-year-dirs", "--db", tmp_db_path,
        "--conflicts-file", str(tmp_path / "conflicts.txt")])
    assert res.exit_code == 0, res.output
    with PhotoDB(tmp_db_path) as db:
        _assert_consistent(db)
        a = db.get_photo_by_path("2013/2013-05-04 Party/a.jpg")
        assert a["folder"] == "2013/2013-05-04 Party"
        assert a["raw_filepath"] == "2013/2013-05-04 Party/a.NEF"
        assert db.get_photo_by_path("2013/2013-05-04 Party/b.jpg")["raw_filepath"] is None
        ids = {r[0] for r in db.conn.execute(
            "SELECT id FROM photos WHERE folder = ?", ("2013/2013-05-04 Party",))}
        assert len(ids) == 2  # batch-membership query sees both moved photos


def test_update_photo_filepath_rederives_folder(tmp_db_path):
    with PhotoDB(tmp_db_path) as db:
        pid = db.add_photo(filepath="a/b/c.jpg", filename="c.jpg")
        db.update_photo(pid, filepath="x/y/c.jpg")
        _assert_consistent(db)


def test_set_photo_filepath_keeps_or_sets_raw(tmp_db_path):
    with PhotoDB(tmp_db_path) as db:
        pid = db.add_photo(filepath="a/c.jpg", filename="c.jpg", raw_filepath="a/c.ARW")
        set_photo_filepath(db.conn, pid, "b/c.jpg")
        db.conn.commit()
        row = db.get_photo(pid)
        assert (row["folder"], row["raw_filepath"]) == ("b", "a/c.ARW")
        set_photo_filepath(db.conn, pid, "b/c.jpg", raw_filepath=None)
        db.conn.commit()
        assert db.get_photo(pid)["raw_filepath"] is None


def test_no_bare_filepath_update_outside_the_primitive():
    """A new `UPDATE photos SET filepath` that skips folder is the bug again."""
    # Only SQL inside a string literal (a quote right before UPDATE), so prose
    # in docstrings/comments describing the old bug doesn't trip it.
    pat = re.compile(r"[\"']\s*UPDATE\s+photos\s+SET\s+filepath\b", re.IGNORECASE)
    offenders = []
    files = [REPO / "cli.py", *sorted((REPO / "photosearch").glob("*.py"))]
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line) and "folder" not in line:
                offenders.append(f"{f.name}:{n}: {line.strip()}")
    assert not offenders, "route these through db.set_photo_filepath:\n" + "\n".join(offenders)
