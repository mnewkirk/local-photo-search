"""CLI tests for the set-filepath command."""

from click.testing import CliRunner

from cli import cli


def test_cli_set_filepath_updates_photo(tmp_db_path):
    """Test that set-filepath updates a photo's filepath and folder."""
    from photosearch.db import PhotoDB

    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2026/2026-09-12/IMG_001.JPG", filename="IMG_001.JPG")
        db.conn.commit()
        photo_id = db.conn.execute("SELECT id FROM photos LIMIT 1").fetchone()["id"]

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["set-filepath", str(photo_id), "2026/2026-10-15/IMG_001_renamed.JPG", "--db", tmp_db_path]
    )

    assert result.exit_code == 0
    assert "Updated photo" in result.output
    assert "2026/2026-09-12/IMG_001.JPG" in result.output
    assert "2026/2026-10-15/IMG_001_renamed.JPG" in result.output

    # Verify the DB was actually updated
    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute(
            "SELECT filepath, folder FROM photos WHERE id = ?",
            (photo_id,)
        ).fetchone()
        assert row["filepath"] == "2026/2026-10-15/IMG_001_renamed.JPG"
        # folder should be re-derived automatically
        assert row["folder"] == "2026/2026-10-15"


def test_cli_set_filepath_photo_not_found(tmp_db_path):
    """Test that set-filepath fails gracefully when photo doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["set-filepath", "99999", "2026/2026-10-15/IMG_001.JPG", "--db", tmp_db_path]
    )

    assert result.exit_code != 0
    assert "Photo 99999 not found" in result.output


def test_cli_set_filepath_derives_folder(tmp_db_path):
    """Test that folder column is automatically re-derived when filepath changes."""
    from photosearch.db import PhotoDB

    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2026/2026-09-12/IMG_001.JPG", filename="IMG_001.JPG")
        db.conn.commit()
        photo_id = db.conn.execute("SELECT id FROM photos LIMIT 1").fetchone()["id"]

    # Move to a different date folder
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["set-filepath", str(photo_id), "2025/2025-12-25/holiday.JPG", "--db", tmp_db_path]
    )

    assert result.exit_code == 0

    # Verify folder was re-derived
    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute(
            "SELECT filepath, folder FROM photos WHERE id = ?",
            (photo_id,)
        ).fetchone()
        assert row["filepath"] == "2025/2025-12-25/holiday.JPG"
        assert row["folder"] == "2025/2025-12-25"
