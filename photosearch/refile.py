"""Re-file historical `YYYY-MM-DD_unknown-camera/` folders onto the right body.

`unknown-camera` is the SD-card importer's fallback label, not a fact: the
Windows importer reads the camera model through the shell property store, which
had no codec for a new body's RAWs, so every one of those `.ARW` files arrived
under `_incoming/unknown-camera/` and ingest filed it to
`/photos/YYYY/YYYY-MM-DD_unknown-camera/` — away from its own JPEGs in
`/photos/YYYY/YYYY-MM-DD_<Model>/`.

`ingest._file_suffix` self-corrects going forward. This module repairs the
files already on disk, and is built to be boring about it:

* **The model comes only from the file's OWN EXIF.** Never from the date, never
  from a sibling folder, never from a same-stem JPEG (unless the operator opts
  into the narrowly-gated `infer_from_sibling`). Two bodies were in use on
  several of these days, so a date-based guess silently mixes two cameras'
  files — an error nobody would ever notice.
* **The date comes only from the source folder's name.** Ingest already dated
  these files; re-deriving from EXIF could split a shoot across a midnight or
  timezone edge.
* **The move cannot overwrite, at the syscall level.** Not "we checked first" —
  `os.rename` and `shutil.copy2` both clobber, and the check-then-move window is
  real: `YYYY-MM-DD_<today>_unknown-camera` is one of the targets, the nightly
  ingest cron and the SD-card importer write into the very folder we move into,
  and Sony DSC names repeat across cards. See `_move_file`.
* **Nothing is ever deleted.** A name collision at the destination is resolved by
  hashing: identical content leaves the source alone (`duplicate_left`),
  different content leaves it alone too (`conflict`).
* **The audit is written before the move and confirmed after it**, so a kill at
  any point leaves `undo_refile` enough to put the tree back.

IO shape matters: this runs on a 4-core NAS with spinning disks. `extract_exif`
reads the file header only (``exifread.process_file(..., details=False)``);
hashing is whole-file and therefore happens ONLY on a name collision, never as
part of the normal path and never during a dry run.
"""

from __future__ import annotations

import csv
import errno
import os
import re
import shutil
import sqlite3
import stat as stat_mod
from pathlib import Path
from typing import Callable, Iterable, Optional

from .db import PhotoDB, _folder_of, set_photo_filepath
from .exif import extract_exif
from .index import file_hash
from .ingest import (
    ALL_MEDIA_EXTENSIONS,
    UNDATED_DIRNAME,
    _file_suffix,
    _no_lock,
    _sweep_lock,
)

# The fallback source label this tool repairs. Deliberately the literal ingest
# uses (`ingest._DEFAULT_BARE_SOURCES`), because it is also the folder-name
# suffix on disk.
UNKNOWN_SUFFIX = "unknown-camera"

# `2026-09-19_unknown-camera` under a 4-digit year dir. Anything else — a named
# model folder, a bare `unknown-camera`, `notadate_unknown-camera` — is ignored.
_FOLDER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_" + re.escape(UNKNOWN_SUFFIX) + r"$")
_YEAR_RE = re.compile(r"^\d{4}$")

# Audit CSV header. Ordering is part of the contract `undo_refile` reads back.
AUDIT_FIELDS = ["action", "source", "destination", "model", "size", "hash", "reason"]

# errno values that mean "hardlink is not available here", as opposed to "that
# destination is taken". Only these fall through to the copy path.
_LINK_UNSUPPORTED = {errno.EXDEV, errno.EPERM, errno.EMLINK}
for _name in ("ENOTSUP", "EOPNOTSUPP"):
    if hasattr(errno, _name):
        _LINK_UNSUPPORTED.add(getattr(errno, _name))

ProgressFn = Callable[[dict], None]


class DestinationExists(OSError):
    """The destination path was already taken at the moment of the move.

    Raised instead of overwriting. The caller handles it exactly like a
    collision detected up front — hash compare, then `duplicate_left` or
    `conflict` — and never retries under another name.
    """


def _zero_counts() -> dict:
    return {
        "files": 0,
        "would_move": 0,
        "moved": 0,
        "no_model": 0,
        "inferred": 0,
        "duplicate_left": 0,
        "conflict": 0,
        "skipped_indexed": 0,
        "skipped_symlink": 0,
        "would_collide": 0,
        "interrupted_link": 0,
        "db_updated": 0,
        "healed": 0,
        "would_heal": 0,
        "heal_unverifiable": 0,
        "errors": 0,
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

class _Audit:
    """Per-file CSV writer, flushed AND fsynced after every row.

    Two rows per move: an ``intent`` before the syscall and a ``moved`` after
    it. That is what makes a kill survivable — a single row written after the
    move would leave a completed move with no undo record, and a single row
    written before it could not tell "moved" from "never started". `undo_refile`
    resolves an unconfirmed intent by looking at the disk.

    fsync (not just flush) because the failure this guards against includes
    power loss, where buffered-but-unwritten rows are exactly the ones
    describing the most recent moves. Appends to an existing file, so a resumed
    run keeps the earlier run's record.
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self._fh = None
        self._w = None
        if not path:
            return
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        self._fh = open(path, "a", newline="")
        self._w = csv.writer(self._fh)
        if not exists:
            self._w.writerow(AUDIT_FIELDS)
            self._sync()

    def _sync(self) -> None:
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass  # a filesystem that cannot fsync must not abort the run

    def row(self, action: str, source: str = "", destination: str = "",
            model: str = "", size: str = "", digest: str = "", reason: str = "") -> None:
        if self._w is None:
            return
        self._w.writerow([action, source, destination, model, size, digest, reason])
        self._sync()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._w = None


# ---------------------------------------------------------------------------
# the move primitive
# ---------------------------------------------------------------------------

def _fsync_dir(d: Path) -> None:
    """Persist a directory entry, so the new name survives a power loss."""
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _unlink_quietly(p: Path) -> None:
    try:
        os.unlink(str(p))
    except OSError:
        pass


def _same_inode(a: Path, b: Path) -> bool:
    """True when two names are the SAME file (one inode, two links).

    That is exactly the state a kill between `os.link` and `os.unlink` leaves.
    `(st_dev, st_ino)` decides it in two stat calls — no hashing, and unlike a
    hash comparison it is proof of identity rather than of equal content, so the
    source can be unlinked safely: the bytes stay reachable through the
    destination. Symlinks are excluded; `lstat` would compare the links, not
    their targets.
    """
    try:
        sa, sb = os.lstat(str(a)), os.lstat(str(b))
    except OSError:
        return False
    if stat_mod.S_ISLNK(sa.st_mode) or stat_mod.S_ISLNK(sb.st_mode):
        return False
    return sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino


def _move_file(src: Path, dst: Path) -> int:
    """Move `src` to `dst` WITHOUT the ability to overwrite. Returns dst's size.

    `os.link` is the whole point: it fails with EEXIST rather than clobbering,
    so the check-then-move race cannot destroy an original. `os.rename` and
    `shutil.copy2` both silently replace the destination, and the race is real —
    the nightly ingest writes into these same dated folders while this runs.
    A hardlink also preserves mtime and permissions for free (one inode), and a
    crash between the link and the unlink leaves both names on that one inode,
    which the next run reads as an identical-hash `duplicate_left`.

    EXDEV/EPERM/EMLINK/ENOTSUP mean hardlinks are unavailable, not that the name
    is taken; only those fall through to `_copy_exclusive`.
    """
    st = src.stat()
    try:
        os.link(str(src), str(dst))
    except FileExistsError:
        raise DestinationExists(errno.EEXIST, "destination exists", str(dst)) from None
    except OSError as exc:
        if exc.errno not in _LINK_UNSUPPORTED:
            raise
        _copy_exclusive(src, dst, st)
    else:
        _fsync_dir(dst.parent)
        os.unlink(str(src))
    out = os.stat(str(dst))
    if out.st_size != st.st_size:
        raise OSError(f"destination {dst} is {out.st_size} bytes, expected {st.st_size}")
    return out.st_size


def _copy_exclusive(src: Path, dst: Path, st: os.stat_result) -> None:
    """Cross-device fallback: copy into an O_EXCL fd, verify, fsync, then unlink.

    O_EXCL gives the copy path the same non-overwriting guarantee the hardlink
    has. Every failure removes the partial destination THIS call created (an
    ENOSPC/EIO truncation left among real photos would read as a permanent
    `conflict` on every later run) and leaves the source untouched. The
    destination and its directory are fsynced BEFORE the source is unlinked, so
    a power loss cannot leave the bytes only in page cache with the original
    already gone.
    """
    try:
        fd = os.open(str(dst), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise DestinationExists(errno.EEXIST, "destination exists", str(dst)) from None
    try:
        with open(str(src), "rb") as fsrc, os.fdopen(fd, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
            fdst.flush()
            os.fsync(fdst.fileno())
    except BaseException:
        _unlink_quietly(dst)
        raise
    try:
        if os.stat(str(dst)).st_size != st.st_size:
            raise OSError(f"cross-device copy of {src} is short; source kept")
        if file_hash(str(dst)) != file_hash(str(src)):
            raise OSError(f"cross-device copy of {src} did not verify; source kept")
        os.chmod(str(dst), stat_mod.S_IMODE(st.st_mode))
        os.utime(str(dst), (st.st_atime, st.st_mtime))
        _fsync_dir(dst.parent)
    except BaseException:
        _unlink_quietly(dst)
        raise
    os.unlink(str(src))


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

class _ReadOnlyDB:
    """A read-only stand-in for PhotoDB, used by dry runs.

    `PhotoDB(...)` opens read-write: on a mistyped `--db` it would CREATE an
    empty stub and run migrations against it, and a dry run must write nothing
    anywhere. The two path helpers are borrowed from PhotoDB itself rather than
    re-derived, so relative/absolute handling cannot drift.
    """

    relative_filepath = PhotoDB.relative_filepath
    resolve_filepath = PhotoDB.resolve_filepath

    def __init__(self, db_path: str, photo_root: Optional[str] = None):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        # The DB's OWN value, kept separately: `_preflight_root` has to compare
        # the root the stored paths were written against with the one this run
        # uses, and an explicit --photo-root would otherwise mask the mismatch.
        try:
            row = self.conn.execute(
                "SELECT value FROM schema_info WHERE key = 'photo_root'").fetchone()
        except sqlite3.Error:
            row = None
        self.db_photo_root = row["value"] if row else None
        # Same precedence as PhotoDB.__init__: arg > PHOTO_ROOT env > DB value.
        if photo_root:
            self.photo_root = str(Path(photo_root).resolve())
        elif os.environ.get("PHOTO_ROOT"):
            self.photo_root = str(Path(os.environ["PHOTO_ROOT"]).resolve())
        else:
            self.photo_root = self.db_photo_root

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _effective_root(db, photo_root_arg: Optional[str]) -> tuple[Path, str]:
    """Resolve the root and say where it came from, mirroring `PhotoDB.__init__`.

    Precedence is the one the rest of the CLI uses: the `--photo-root` argument,
    then the `PHOTO_ROOT` env var, then the DB's stored value. **The deployment
    this tool exists for stores none** — `schema_info` holds only
    `('version', N)` and the container supplies `PHOTO_ROOT=/photos` — so
    refusing on a missing stored root (which an earlier revision did) makes the
    tool unusable on the one system it was written for. `_ReadOnlyDB` follows
    the same precedence, so `relative_filepath` relativises against the
    effective root and the link lookups match the canonical relative paths the
    DB actually stores.
    """
    if photo_root_arg:
        return Path(photo_root_arg).expanduser().resolve(), "--photo-root"
    if os.environ.get("PHOTO_ROOT"):
        return (Path(os.environ["PHOTO_ROOT"]).expanduser().resolve(),
                "PHOTO_ROOT env")
    if db.db_photo_root:
        return Path(db.db_photo_root).resolve(), "the DB's stored photo_root"
    raise ValueError(
        "no photo root available: pass --photo-root, set PHOTO_ROOT, or store "
        "one in the database. Without it a stored path cannot be related to a "
        "file on disk and every DB link lookup would miss.")


def _preflight_root(db, root: Path) -> None:
    """Refuse unless the DB's `photo_root` really maps onto the root in use.

    This is the hole the spelling pre-flight CANNOT see. `relative_filepath`
    swallows a mismatch: `Path.relative_to` raises and it returns the ABSOLUTE
    path instead. So if the stored root is a symlink, a different mount spelling,
    or simply another directory, every stored path looks canonical, the
    pre-flight passes, and every single link lookup misses — an indexed file
    reads as unindexed and gets moved out from under its row. Measured: stored
    `2026/2026-06-19_unknown-camera/DSC01.JPG`, pre-flight green,
    `would_move=1 skipped_indexed=0`.

    A DB that stores NO root is not refused here — see `_effective_root`; the
    NAS is exactly that case. `_preflight_roundtrip` carries the proof instead,
    and it is mandatory.
    """
    if not db.db_photo_root:
        return
    stored = Path(db.db_photo_root)
    if stored.resolve() != root:
        raise ValueError(
            f"the database's photo root does not match the one in use:\n"
            f"    DB photo_root: {stored}  (resolves to {stored.resolve()})\n"
            f"    running under: {root}\n"
            f"Stored paths would not resolve against this root, so every DB link "
            f"lookup would miss and indexed files would be moved out from under "
            f"their rows. Point --photo-root at the DB's root (or fix the DB).")


def _sample_rows(db, n: int = 25) -> list[dict]:
    """Up to `n` rows spread across the table: lowest ids, highest ids, middle.

    Not `LIMIT 1` of the first row: paths written by different eras of the
    indexer live at different ends of the id range, and a mapping that holds for
    the oldest row can fail for the newest.
    """
    picks: dict[int, dict] = {}
    for sql in (
        "SELECT id, filepath FROM photos ORDER BY id LIMIT ?",
        "SELECT id, filepath FROM photos ORDER BY id DESC LIMIT ?",
        "SELECT id, filepath FROM photos ORDER BY id LIMIT ? "
        "OFFSET (SELECT COUNT(*) / 2 FROM photos)",
    ):
        for r in db.conn.execute(sql, (max(1, n // 3),)).fetchall():
            picks[r["id"]] = dict(r)
    return list(picks.values())[:n]


def _preflight_roundtrip(db, root: Path, root_source: str) -> int:
    """MANDATORY proof that the DB's paths and the root in use line up.

    This — not where the root happens to be configured — is what makes the link
    gate trustworthy. For each sampled row it runs the PRODUCTION lookup
    (`_resolve_link`, the exact call the gate makes, not a parallel
    re-implementation) on the absolute on-disk path and requires it to find that
    row back. A root that is a symlink, another mount spelling, or simply wrong
    makes `relative_filepath` return an absolute path and every lookup miss;
    this turns that from silent into a refusal.

    At least one sampled row must also exist on disk, which is what proves
    `root` is where the library actually lives. It is deliberately "at least
    one of up to 25", not "all": `_heal_folder` exists precisely because rows
    whose file has moved are a normal state, so a single stale row must not
    lock the operator out.

    An EMPTY photos table returns 0 rather than refusing — with no rows the gate
    cannot produce a false negative, so there is nothing to prove.

    Returns the number of rows verified (for the report header).
    """
    rows = _sample_rows(db)
    if not rows:
        return 0
    def _broken(detail: str) -> ValueError:
        return ValueError(
            f"path mapping between the DB and the photo root is broken: {detail}\n"
            f"    photo root in use: {root}  (from {root_source})\n"
            f"Every DB link lookup would miss, so indexed files would be moved "
            f"out from under their rows.")

    any_on_disk = False
    for r in rows:
        stored_fp = r["filepath"]
        abs_fp = str(root / stored_fp)
        if os.path.lexists(abs_fp):
            any_on_disk = True
        found, _ids = _resolve_link(db, db.relative_filepath(abs_fp), abs_fp, {}, {})
        if found is None or found["id"] != r["id"]:
            raise _broken(f"the production lookup for row {r['id']} "
                          f"({stored_fp!r}) does not find it back")
    if not any_on_disk:
        raise _broken(f"none of the {len(rows)} sampled photo rows has its file "
                      f"on disk under this root (checked e.g. "
                      f"{rows[0]['filepath']!r})")
    return len(rows)


def _preflight_paths(db) -> None:
    """Refuse to run while any stored path is spelled non-canonically.

    The link lookup matches paths as STRINGS. A row stored absolute, as
    `./2026/...`, with backslashes, or with a doubled slash would not match the
    canonical spelling this tool derives, so the file would look unindexed, be
    moved by default, and orphan its row — the same class of false negative as
    the `folder`-column gate this replaced. There is no override: a lookup that
    cannot be trusted is not a lookup.

    When no `photo_root` is configured the indexer stores absolute paths
    legitimately, so that one check is dropped in that case.
    """
    # `_preflight_root` has already guaranteed a photo_root, so an absolute
    # stored path is unambiguously non-canonical here.
    checks = ["filepath LIKE './%'", "filepath LIKE '%\\%'", "filepath LIKE '%//%'",
              "raw_filepath LIKE './%'", "raw_filepath LIKE '%\\%'",
              "raw_filepath LIKE '%//%'",
              "filepath LIKE '/%'", "raw_filepath LIKE '/%'"]
    rows = db.conn.execute(
        f"SELECT id, filepath, raw_filepath FROM photos WHERE {' OR '.join(checks)}"
    ).fetchall()
    if not rows:
        return
    samples = "\n".join(
        f"    [{r['id']}] {r['filepath']}"
        + (f"  (raw: {r['raw_filepath']})" if r["raw_filepath"] else "")
        for r in rows[:5])
    raise ValueError(
        f"{len(rows):,} photos row(s) store a non-canonical path (absolute, "
        f"'./', backslash, or '//'):\n{samples}\n"
        f"The DB link lookup matches paths as strings, so these rows could be "
        f"missed and their files moved out from under them. Normalise them "
        f"first (see db.remap_paths) — refile-unknown-camera will not run until "
        f"they are gone.")


def _db_links(db, rel_folders: list[str]) -> tuple[dict, dict]:
    """Map the DB's view of the target folders, keyed by EXACT filepath.

    Returns ``(by_path, raw_refs)``:

    * ``by_path``  — ``filepath -> photo row`` for every row whose path lies in
      one of these folders, found with a prefix RANGE scan on the UNIQUE
      ``filepath`` index. Deliberately NOT ``WHERE folder IN (...)``: the
      ``folder`` column is derived, and ``relocate-into-year-dirs`` (cli.py)
      and ``db.remap_paths`` used to ``UPDATE photos SET filepath`` and leave
      it stale. Every writer now goes through ``db.set_photo_filepath``, but
      a DB that ran those older versions can still carry stale values (until
      ``backfill-folders --force``). A stale ``folder`` would hide the row,
      the file would look unindexed, and it would be moved by default,
      orphaning the row — so the gate keeps trusting only ``filepath``.
    * ``raw_refs`` — ``filepath -> [photo_id, ...]`` for files named by some
      photo's ``raw_filepath``. ``index.py`` sets that from ``find_raw_pair``,
      which looks in the photo's OWN directory, so this fires when a JPEG and
      its RAW both landed in the unknown-camera folder. One sequential scan of
      the non-null column, once per run, matched by exact dirname.

    RAW/video companions have no row of their own (``ingest.py`` gates the DB
    path on ``is_photo = ext in INGEST_EXTENSIONS``), so ``by_path`` only ever
    matches JPEG/HEIC.
    """
    by_path: dict[str, dict] = {}
    raw_refs: dict[str, dict] = {}
    if not rel_folders:
        return by_path, raw_refs

    for rel in rel_folders:
        lo = rel + "/"
        hi = rel + "0"  # '/' is 0x2F, '0' is 0x30 — the exact prefix range
        for r in db.conn.execute(
            "SELECT id, filepath, filename, file_hash FROM photos "
            "WHERE filepath >= ? AND filepath < ?", (lo, hi)
        ).fetchall():
            by_path[r["filepath"]] = dict(r)

    # `raw_filepath` has NO index (checked: db.py creates none), so a per-file
    # lookup on it would be a full scan 9,600 times over. It is scanned once
    # here instead and keyed by ABSOLUTE path, which makes the per-file check
    # against this set independent of how the column happens to be spelled.
    folder_abs = {str(Path(db.resolve_filepath(rel))) for rel in rel_folders}
    for r in db.conn.execute(
        "SELECT id, raw_filepath FROM photos WHERE raw_filepath IS NOT NULL"
    ).fetchall():
        rp = r["raw_filepath"]
        ap = os.path.normpath(db.resolve_filepath(rp))
        if os.path.dirname(ap) in folder_abs:
            raw_refs.setdefault(ap, {"stored": rp, "ids": []})["ids"].append(r["id"])
    return by_path, raw_refs


def _resolve_link(db, rel_src: str, abs_src: str,
                  by_path: dict, raw_refs: dict) -> tuple[Optional[dict], list[int]]:
    """Is this file named by any photos row? Belt and braces.

    The up-front `by_path` scan is a prefix range over one spelling. This
    re-asks by EXACT filepath, both spellings, immediately before the move —
    one lookup on the UNIQUE index, ~9,600 times, which is free — so a row
    inserted since the scan, or stored under the other spelling, still blocks
    the move. `raw_filepath` cannot be re-queried per file (no index), so it is
    checked against the absolute-keyed set built once up front.
    """
    row = by_path.get(rel_src)
    entry = raw_refs.get(os.path.normpath(abs_src))
    ids = list(entry["ids"]) if entry else []
    if row is None:
        hit = db.conn.execute(
            "SELECT id, filepath, filename, file_hash FROM photos "
            "WHERE filepath IN (?, ?) LIMIT 1", (rel_src, abs_src)).fetchone()
        if hit is not None:
            row = dict(hit)
    return row, ids


def _heal_folder(db, folder: Path, year_dir: Path, date: str,
                 by_path: dict, raw_refs: dict, apply: bool,
                 audit: _Audit, counts: dict, notes: list) -> None:
    """Repair rows left pointing into this folder at a file that has moved.

    This is the recovery half of the move-then-write ordering used by
    ``include_indexed``: the file is relocated first and the row updated
    second, so a crash in between leaves a row naming a path that no longer
    exists — recoverable, unlike the reverse order, which would point a row at
    a file that was never created.

    A re-run heals it by CONTENT: the same basename in one of that date's
    sibling `YYYY-MM-DD_*` folders, with a matching ``photos.file_hash``.
    Anything that cannot be verified that way is counted ``heal_unverifiable``
    and reported for a human — including every stale ``raw_filepath``, which
    carries no hash at all, so an "only one candidate" repoint could silently
    attach the OTHER body's same-named RAW.

    A dry run never hashes: it reports how many rows would need healing.
    """
    siblings = [d for d in sorted(year_dir.iterdir())
                if d.is_dir() and d.name.startswith(date + "_") and d != folder]
    prefix = db.relative_filepath(str(folder))

    def _candidates(name: str) -> list[Path]:
        return [d / name for d in siblings if (d / name).is_file()]

    for rel, row in list(by_path.items()):
        if _folder_of(rel, os.path.basename(rel)) != prefix:
            continue
        if os.path.lexists(db.resolve_filepath(rel)):
            continue
        if not apply:
            counts["would_heal"] += 1
            continue
        if not row.get("file_hash"):
            counts["heal_unverifiable"] += 1
            notes.append(f"{rel}: row has no file_hash — repair by hand")
            audit.row("heal_unverifiable", rel,
                      reason="photos row has no file_hash to verify against")
            continue
        cands = [c for c in _candidates(os.path.basename(rel))
                 if file_hash(str(c)) == row["file_hash"]]
        if len(cands) != 1:
            counts["heal_unverifiable"] += 1
            notes.append(f"{rel}: {len(cands)} content matches — repair by hand")
            audit.row("heal_unverifiable", rel,
                      reason=f"{len(cands)} hash-matching candidates")
            continue
        new_rel = db.relative_filepath(str(cands[0]))
        try:
            set_photo_filepath(db.conn, row["id"], new_rel)
            db.conn.commit()
        except Exception as exc:
            counts["errors"] += 1
            audit.row("error", rel, new_rel, reason=f"heal failed: {exc}")
            continue
        by_path.pop(rel, None)
        counts["healed"] += 1
        audit.row("healed", rel, new_rel, reason=f"photo_id={row['id']}")

    for ap, entry in list(raw_refs.items()):
        if os.path.dirname(ap) != str(folder):
            continue
        if os.path.lexists(ap):
            continue
        if not apply:
            counts["would_heal"] += 1
            continue
        # Never auto-repoint: a raw_filepath has no stored hash, so "exactly one
        # candidate" would happily attach another body's identically-named RAW.
        counts["heal_unverifiable"] += 1
        notes.append(f"{entry['stored']}: stale raw_filepath on photo(s) "
                     f"{', '.join(str(i) for i in entry['ids'])} — repair by hand")
        audit.row("heal_unverifiable", entry["stored"],
                  reason="stale raw_filepath; no hash to verify a replacement")


# ---------------------------------------------------------------------------
# folder discovery
# ---------------------------------------------------------------------------

def _find_folders(photo_root: Path, only: Optional[Iterable[str]]) -> list[tuple[Path, Path, str]]:
    """Return sorted ``(year_dir, folder, date)`` for every target folder.

    `only` filters on the folder's basename (a full path is accepted too — only
    its name is compared), so an operator can do one small folder first.
    """
    wanted = {os.path.basename(str(o).rstrip("/")) for o in (only or ())}
    out: list[tuple[Path, Path, str]] = []
    for year_dir in sorted(photo_root.iterdir()):
        if not year_dir.is_dir() or not _YEAR_RE.match(year_dir.name):
            continue
        for d in sorted(year_dir.iterdir()):
            if not d.is_dir() or d.is_symlink():
                continue
            m = _FOLDER_RE.match(d.name)
            if not m:
                continue
            if wanted and d.name not in wanted:
                continue
            out.append((year_dir, d, m.group(1)))
    return out


def _infer_from_sibling(year_dir: Path, date: str, stem: str) -> Optional[str]:
    """Opt-in fallback: the one model folder for this date holding the same stem.

    Both gates from the brief must hold — the same-stem sibling exists in
    exactly ONE model folder for that date, AND no other model folder exists for
    that date. With a single body in play the pairing is unambiguous; with two,
    this returns None and the file stays put.
    """
    models = [d for d in sorted(year_dir.iterdir())
              if d.is_dir() and d.name.startswith(date + "_")
              and not d.name.endswith("_" + UNKNOWN_SUFFIX)]
    if len(models) != 1:
        return None
    holder = models[0]
    if not any(p.stem == stem for p in holder.iterdir() if p.is_file()):
        return None
    return holder.name[len(date) + 1:]


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def refile_unknown_camera(
    photo_root: Optional[str],
    db_path: str,
    apply: bool = False,
    audit_path: Optional[str] = None,
    only: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    include_indexed: bool = False,
    infer_from_sibling: bool = False,
    on_progress: Optional[ProgressFn] = None,
    sample_limit: int = 10,
) -> dict:
    """Move `*_unknown-camera` files onto the folder their own EXIF names.

    Dry run is the default: nothing is moved, nothing is written, no audit file
    is created, and the database is opened READ-ONLY. ``apply=True`` requires
    ``audit_path`` — the audit IS the undo — and takes ingest's sweep lock, so
    it cannot run against a library the nightly ingest is mid-sweep on.

    Only files at the TOP LEVEL of each folder are considered. A nested
    subdirectory is left untouched and reported: its layout carries meaning this
    tool cannot reproduce at the destination, and flattening it would invent
    collisions. That also means such a folder is never empty afterwards and is
    therefore never removed. Symlinks are never followed and never moved.

    Returns a stats dict::

        {"dry_run": bool, "photo_root": str, "audit_path": str|None,
         "folders": [ {name, path, files, by_model, nested_dirs, remaining,
                       removed, notes, **counts}, ... ],
         "totals": {...}, "samples": [(src, dst), ...],
         "skipped_undated": int}
    """
    if apply and not audit_path:
        raise ValueError(
            "refusing to --apply without --audit: the audit CSV is the only "
            "record that can reverse these moves")
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"database not found: {db_path} — refusing to create one "
            f"(check --db / PHOTOSEARCH_DB)")

    # Everything that can refuse is decided read-only, before the lock is taken
    # and before any file is touched. The context manager matters: an early
    # raise here used to leak the connection.
    with _ReadOnlyDB(db_path, photo_root) as ro:
        root, root_source = _effective_root(ro, photo_root)
        if not root.is_dir():
            raise FileNotFoundError(f"photo root not found: {root}")
        if audit_path:
            audit_abs = Path(audit_path).expanduser().resolve()
            if audit_abs == root or root in audit_abs.parents:
                raise ValueError(
                    f"refusing to write the audit inside the photo root ({root}): "
                    f"it would count as a leftover and block an empty folder's "
                    f"removal. Use the /data volume.")
        _preflight_root(ro, root)
        _preflight_paths(ro)
        checked = _preflight_roundtrip(ro, root, root_source)
        stored_note = ("DB stores none" if not ro.db_photo_root
                       else f"DB stores {ro.db_photo_root}")
        if not apply:
            # A dry run writes nothing, so it neither needs nor takes the mutex.
            with _no_lock():
                return _run(ro, root, False, None, only, limit, include_indexed,
                            infer_from_sibling, on_progress, sample_limit,
                            root_source, stored_note, checked)

    with _sweep_lock(db_path), PhotoDB(db_path, photo_root=str(root)) as db:
        return _run(db, root, True, audit_path, only, limit, include_indexed,
                    infer_from_sibling, on_progress, sample_limit,
                    root_source, stored_note, checked)


def _run(db, root: Path, apply: bool, audit_path: Optional[str],
         only, limit, include_indexed, infer_from_sibling,
         on_progress, sample_limit, root_source: str = "",
         stored_note: str = "", mapping_checked: int = 0) -> dict:
    targets = _find_folders(root, only)

    # Classify every folder's top level FIRST, so the DB lookup can ask about
    # the exact candidate paths rather than a folder column nothing maintains.
    plan = []
    for year_dir, folder, date in targets:
        nested_dirs: list[str] = []
        leftovers: list[str] = []
        candidates: list[Path] = []
        symlinks = 0
        for p in sorted(folder.iterdir()):
            if p.is_symlink():
                # Never move a link: the move would relocate the link and leave
                # its target, or (worse) act on a path outside this tree.
                leftovers.append(p.name)
                symlinks += 1
                continue
            if p.is_dir():
                nested_dirs.append(p.name)
                leftovers.append(p.name)
                continue
            if p.name.startswith("."):
                leftovers.append(p.name)  # .DS_Store / ._AppleDouble droppings
                continue
            if p.suffix.lower() in ALL_MEDIA_EXTENSIONS:
                candidates.append(p)
            else:
                leftovers.append(p.name)  # sidecars, .aae edits, etc.
        plan.append({"year_dir": year_dir, "folder": folder, "date": date,
                     "candidates": candidates, "nested_dirs": nested_dirs,
                     "leftovers": leftovers, "symlinks": symlinks})

    rel_folders = [db.relative_filepath(str(f)) for _y, f, _d in targets]
    by_path, raw_refs = _db_links(db, rel_folders)

    audit = _Audit(audit_path if apply else None)
    totals = _zero_counts()
    folders: list[dict] = []
    samples: list[tuple[str, str]] = []
    budget = limit if limit is not None else None

    # `_undated/unknown-camera` has no date to route by, so there is no
    # destination folder to compute. Reported, never touched.
    skipped_undated = 1 if (root / UNDATED_DIRNAME / UNKNOWN_SUFFIX).is_dir() else 0

    def _emit(payload: dict) -> None:
        if on_progress is not None:
            on_progress(payload)

    try:
        for fi, step in enumerate(plan):
            year_dir, folder, date = step["year_dir"], step["folder"], step["date"]
            _emit({"event": "folder", "name": folder.name, "index": fi,
                   "total": len(plan)})
            counts = _zero_counts()
            counts["skipped_symlink"] = step["symlinks"]
            by_model: dict[str, int] = {}
            notes: list[str] = []

            _heal_folder(db, folder, year_dir, date, by_path, raw_refs,
                         apply, audit, counts, notes)

            relocating: set[str] = set()
            for src in step["candidates"]:
                if budget is not None and budget <= 0:
                    break
                counts["files"] += 1
                # The audit records ABSOLUTE paths — it is a record of files on
                # disk, and must stay readable without knowing which photo_root
                # the run used. DB keys are the db-relative form.
                abs_src = str(src)
                rel_src = db.relative_filepath(abs_src)

                try:
                    meta = extract_exif(abs_src)
                except Exception:
                    meta = {}
                # The destination suffix must be EXACTLY what ingest would
                # produce today — imported, never re-implemented.
                suffix = _file_suffix(UNKNOWN_SUFFIX, UNKNOWN_SUFFIX, meta)
                inferred = False
                if suffix == UNKNOWN_SUFFIX and infer_from_sibling:
                    guess = _infer_from_sibling(year_dir, date, src.stem)
                    if guess:
                        suffix, inferred = guess, True

                if suffix == UNKNOWN_SUFFIX:
                    counts["no_model"] += 1
                    audit.row("no_model", abs_src, model="", size=str(_size(src)),
                              reason="EXIF names no usable camera model")
                    _emit({"event": "file", "action": "no_model", "path": rel_src})
                    continue

                by_model[suffix] = by_model.get(suffix, 0) + 1
                dest_dir = year_dir / f"{date}_{suffix}"
                dest = dest_dir / src.name
                abs_dest = str(dest)
                rel_dest = db.relative_filepath(abs_dest)

                # DB gate. A row (or a raw_filepath reference) makes the file
                # path-bearing state, not just bytes. Re-asked by exact path
                # here, not just read from the up-front scan — see _resolve_link.
                row, linked_ids = _resolve_link(db, rel_src, abs_src,
                                                by_path, raw_refs)
                if (row or linked_ids) and not include_indexed:
                    counts["skipped_indexed"] += 1
                    audit.row("skipped_indexed", abs_src, abs_dest, suffix,
                              str(_size(src)),
                              reason="has a photos row (or is a raw_filepath "
                                     "target); re-run with --include-indexed")
                    _emit({"event": "file", "action": "skipped_indexed",
                           "path": rel_src})
                    continue

                # lexists, not exists: a DANGLING symlink at the destination is
                # taken, not free, and `exists()` would call it free.
                interrupted = False
                if os.path.lexists(abs_dest):
                    # BEFORE any hashing: one inode under two names is our own
                    # kill between link and unlink, and (st_dev, st_ino) proves
                    # it in two stats. Treating it as a `duplicate_left` would
                    # strand the source forever and re-hash both names on every
                    # future run.
                    if _same_inode(src, dest):
                        counts["interrupted_link"] += 1
                        notes.append(f"{src.name}: interrupted move (one inode, "
                                     f"two names)"
                                     f"{'' if apply else ' — would be completed'}")
                        if not apply:
                            _emit({"event": "file", "action": "interrupted_link",
                                   "path": rel_src})
                            continue
                        interrupted = True
                    elif not apply:
                        # A dry run must not hash: repeated DSC names are the
                        # expected case here and hashing them would read
                        # gigabytes off a spinning disk to preview a no-op.
                        counts["would_collide"] += 1
                        ssz, dsz = _size(src), _size(dest)
                        notes.append(
                            f"{src.name}: destination taken "
                            f"({ssz:,} vs {dsz:,} bytes — "
                            + ("certainly a conflict)" if ssz != dsz
                               else "needs a hash at apply time)"))
                        _emit({"event": "file", "action": "would_collide",
                               "path": rel_src})
                        continue
                    else:
                        _collision(src, dest, abs_src, abs_dest, suffix,
                                   counts, audit, _emit, rel_src)
                        continue

                if len(samples) < sample_limit:
                    samples.append((rel_src, rel_dest))

                if not apply:
                    counts["would_move"] += 1
                    relocating.add(src.name)
                    if inferred:
                        counts["inferred"] += 1
                    if budget is not None:
                        budget -= 1
                    _emit({"event": "file", "action": "would_move",
                           "path": rel_src, "destination": rel_dest})
                    continue

                # Intent BEFORE the syscall, confirmation after it. A kill in
                # between leaves an intent undo_refile resolves from the disk.
                intent_bits = []
                if row is not None:
                    intent_bits.append(f"photo_id={row['id']}")
                if linked_ids:
                    intent_bits.append("raw_refs=" + "|".join(str(i) for i in linked_ids))
                if interrupted:
                    intent_bits.append("completed_interrupted_link")
                src_size = _size(src)
                audit.row("intent", abs_src, abs_dest, suffix, str(src_size),
                          reason=";".join(intent_bits))

                if interrupted:
                    # Finish the move the kill interrupted. The unlink needs
                    # PROOF that another name still holds the bytes, taken as
                    # late as possible: st_nlink >= 2 says so outright, and
                    # re-confirming the inode closes the check->unlink window.
                    # nlink also defeats a synthetic-inode false positive —
                    # SMB/NFS/FUSE can repeat st_ino while reporting nlink 1.
                    try:
                        nlink = os.lstat(abs_src).st_nlink
                    except OSError:
                        nlink = 1
                    if nlink < 2 or not _same_inode(src, dest):
                        counts["interrupted_link"] -= 1
                        audit.row("unlink_refused", abs_src, abs_dest, suffix,
                                  reason=f"st_nlink={nlink} or inode changed — no "
                                         f"proof another name holds the bytes; "
                                         f"treated as an ordinary collision")
                        _collision(src, dest, abs_src, abs_dest, suffix,
                                   counts, audit, _emit, rel_src)
                        continue
                    try:
                        os.unlink(abs_src)
                    except OSError as exc:
                        counts["errors"] += 1
                        audit.row("error", abs_src, abs_dest, suffix,
                                  reason=f"completing interrupted link failed: {exc}")
                        continue
                    size = _size(dest)
                    _after_move(db, counts, audit, by_path, raw_refs, row, linked_ids,
                                rel_src, rel_dest, abs_src, abs_dest, suffix, size,
                                src.name, inferred, ["completed_interrupted_link"])
                    relocating.add(src.name)
                    if budget is not None:
                        budget -= 1
                    _emit({"event": "file", "action": "moved", "path": rel_src,
                           "destination": rel_dest})
                    continue

                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    size = _move_file(src, dest)
                except DestinationExists:
                    # Something landed there between our check and the move —
                    # the nightly ingest, or another importer. Same handling as
                    # a pre-detected collision; never retried under a new name.
                    _collision(src, dest, abs_src, abs_dest, suffix,
                               counts, audit, _emit, rel_src)
                    continue
                except Exception as exc:
                    counts["errors"] += 1
                    audit.row("error", abs_src, abs_dest, suffix,
                              reason=f"move failed: {exc}")
                    _emit({"event": "file", "action": "error", "path": rel_src})
                    continue

                _after_move(db, counts, audit, by_path, raw_refs, row, linked_ids,
                            rel_src, rel_dest, abs_src, abs_dest, suffix, size,
                            src.name, inferred, [])
                relocating.add(src.name)
                if budget is not None:
                    budget -= 1
                _emit({"event": "file", "action": "moved", "path": rel_src,
                       "destination": rel_dest})

            # On apply this is the real listing; on a dry run it is the
            # prediction (everything that would NOT move), so the report can say
            # whether the folder would survive the run.
            leftovers = step["leftovers"] + [
                c.name for c in step["candidates"] if c.name not in relocating]
            remaining = (sorted(p.name for p in folder.iterdir())
                         if (apply and folder.is_dir()) else sorted(leftovers))
            removed = False
            if apply and not remaining:
                try:
                    folder.rmdir()  # rmdir only — fails safely if not empty
                    removed = True
                except OSError as exc:
                    audit.row("error", str(folder), reason=f"rmdir failed: {exc}")

            folders.append({
                "name": folder.name, "path": db.relative_filepath(str(folder)),
                "by_model": by_model, "nested_dirs": step["nested_dirs"],
                "remaining": remaining[:25], "removed": removed,
                "notes": notes[:10], **counts,
            })
            for k in totals:
                totals[k] += counts[k]
            if budget is not None and budget <= 0:
                break
    finally:
        audit.close()

    _emit({"event": "done", "totals": totals})
    return {
        "dry_run": not apply,
        "photo_root": str(root),
        "root_source": root_source,
        "root_note": stored_note,
        "mapping_checked": mapping_checked,
        "audit_path": audit_path if apply else None,
        "folders": folders,
        "totals": totals,
        "samples": samples,
        "skipped_undated": skipped_undated,
        "limit_reached": bool(budget is not None and budget <= 0),
    }


def _after_move(db, counts: dict, audit: _Audit, by_path: dict, raw_refs: dict,
                row: Optional[dict], linked_ids: list, rel_src: str, rel_dest: str,
                abs_src: str, abs_dest: str, suffix: str, size: int,
                name: str, inferred: bool, extra: list) -> None:
    """Point the DB at the new path and write the audit's confirmation row.

    Move first, write the row second: they cannot be one transaction, and this
    order leaves a row naming a gone file (which `_heal_folder` repairs on the
    next run) rather than a row naming a file that was never created. Shared by
    the normal move and by the completion of an interrupted hardlink so the two
    cannot drift.
    """
    reason_bits = list(extra)
    if row is not None:
        try:
            set_photo_filepath(db.conn, row["id"], rel_dest)
            db.conn.commit()
            counts["db_updated"] += 1
            reason_bits.append(f"photo_id={row['id']}")
            by_path.pop(rel_src, None)
        except Exception as exc:
            counts["errors"] += 1
            reason_bits.append(f"db update FAILED: {exc}")
    if linked_ids:
        try:
            for pid in linked_ids:
                db.conn.execute("UPDATE photos SET raw_filepath = ? WHERE id = ?",
                                (rel_dest, pid))
            db.conn.commit()
            counts["db_updated"] += 1
            reason_bits.append("raw_refs=" + "|".join(str(i) for i in linked_ids))
            raw_refs.pop(os.path.normpath(abs_src), None)
        except Exception as exc:
            counts["errors"] += 1
            reason_bits.append(f"raw_filepath update FAILED: {exc}")
    if inferred:
        counts["inferred"] += 1
        reason_bits.append("model inferred from same-stem sibling")
    counts["moved"] += 1
    audit.row("moved", abs_src, abs_dest, suffix, str(size),
              reason=";".join(reason_bits))


def _collision(src: Path, dest: Path, abs_src: str, abs_dest: str, suffix: str,
               counts: dict, audit: _Audit, emit, rel_src: str) -> None:
    """Record a taken destination as duplicate_left or conflict. Never writes.

    The ONLY place whole-file hashing happens. Hashing every file would mean
    reading the whole library off a spinning disk; a name collision is rare and
    is exactly the case where the bytes have to be compared.
    """
    try:
        digest = file_hash(str(src))
        same = dest.is_file() and digest == file_hash(str(dest))
    except Exception as exc:
        counts["errors"] += 1
        audit.row("error", abs_src, abs_dest, suffix, reason=f"hash failed: {exc}")
        emit({"event": "file", "action": "error", "path": rel_src})
        return
    if same:
        counts["duplicate_left"] += 1
        audit.row("duplicate_left", abs_src, abs_dest, suffix, str(_size(src)),
                  digest, "identical copy already at destination; source left "
                          "in place, delete by hand")
    else:
        counts["conflict"] += 1
        audit.row("conflict", abs_src, abs_dest, suffix, str(_size(src)), digest,
                  "different file already at destination; nothing overwritten, "
                  "nothing renamed")
    emit({"event": "file", "action": "duplicate_left" if same else "conflict",
          "path": rel_src})


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------

def _parse_reason(reason: str) -> tuple[Optional[int], list[int]]:
    photo_id: Optional[int] = None
    raw_ids: list[int] = []
    for bit in (reason or "").split(";"):
        bit = bit.strip()
        if bit.startswith("photo_id="):
            try:
                photo_id = int(bit.split("=", 1)[1])
            except ValueError:
                pass
        elif bit.startswith("raw_refs="):
            for tok in bit.split("=", 1)[1].split("|"):
                try:
                    raw_ids.append(int(tok))
                except ValueError:
                    pass
    return photo_id, raw_ids


def undo_refile(audit_path: str, db_path: str, apply: bool = False,
                on_progress: Optional[ProgressFn] = None) -> dict:
    """Reverse an audit CSV's moves, newest first.

    A `moved` row is restored only when the destination is still the file this
    tool put there — same size, and the same hash when one was recorded (a hash
    is recorded only for collision rows; hashing every move on the way out would
    mean reading the whole library) — and the source path is free. Anything else
    is `refused` and left exactly as it is.

    An `intent` with no matching `moved` is a crash window, resolved by looking
    at the disk rather than guessing: source still present means the move never
    happened (`no_op`); destination present at the recorded size with the source
    gone means it completed and is undoable.

    Any DB row this tool repointed (recorded in the audit's ``reason``) is
    pointed back, after the file has been restored. Restoring uses the same
    non-overwriting primitive as the forward move.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"database not found: {db_path} — refusing to create one "
            f"(check --db / PHOTOSEARCH_DB)")
    rows = []
    with open(audit_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("action") in ("moved", "intent"):
                rows.append(r)
    confirmed = {(r["source"], r["destination"]) for r in rows if r["action"] == "moved"}
    todo = [r for r in rows
            if r["action"] == "moved"
            or (r["source"], r["destination"]) not in confirmed]
    todo.reverse()

    stats = {"candidates": len(todo), "restored": 0, "would_restore": 0,
             "refused": 0, "no_op": 0, "errors": 0, "refusals": [],
             "dry_run": not apply}

    def _emit(payload: dict) -> None:
        if on_progress is not None:
            on_progress(payload)

    def _refuse(path: str, why: str) -> None:
        stats["refused"] += 1
        if len(stats["refusals"]) < 20:
            stats["refusals"].append((path, why))
        _emit({"event": "file", "action": "refused", "path": path, "reason": why})

    # An undo DRY RUN must not create a stub DB on a mistyped --db either.
    with (PhotoDB(db_path) if apply else _ReadOnlyDB(db_path)) as db:
        root = Path(db.photo_root or ".").resolve()
        for r in todo:
            src = root / r["source"]
            dst = root / r["destination"]

            if r["action"] == "intent":
                if os.path.lexists(str(src)):
                    stats["no_op"] += 1  # killed before the move — nothing to undo
                    continue
                if not dst.is_file():
                    _refuse(r["destination"], "unconfirmed move: neither path holds the file")
                    continue
                if r.get("size") and str(_size(dst)) != r["size"]:
                    _refuse(r["destination"], "unconfirmed move: destination size differs")
                    continue

            why = None
            if not dst.is_file():
                why = "destination no longer exists"
            elif os.path.lexists(str(src)):
                why = "source path is occupied"
            elif r.get("size") and str(_size(dst)) != r["size"]:
                why = "destination size changed since the move"
            elif r.get("hash"):
                try:
                    if file_hash(str(dst)) != r["hash"]:
                        why = "destination content changed since the move"
                except Exception as exc:
                    why = f"hash failed: {exc}"
            if why:
                _refuse(r["destination"], why)
                continue

            if not apply:
                stats["would_restore"] += 1
                continue

            try:
                src.parent.mkdir(parents=True, exist_ok=True)
                _move_file(dst, src)
            except Exception as exc:
                stats["errors"] += 1
                if len(stats["refusals"]) < 20:
                    stats["refusals"].append((r["destination"], f"restore failed: {exc}"))
                continue

            photo_id, raw_ids = _parse_reason(r.get("reason", ""))
            back = db.relative_filepath(str(src))
            try:
                if photo_id is not None:
                    set_photo_filepath(db.conn, photo_id, back)
                for pid in raw_ids:
                    db.conn.execute(
                        "UPDATE photos SET raw_filepath = ? WHERE id = ?", (back, pid))
                if photo_id is not None or raw_ids:
                    db.conn.commit()
            except Exception as exc:
                stats["errors"] += 1
                if len(stats["refusals"]) < 20:
                    stats["refusals"].append((r["source"], f"db restore failed: {exc}"))

            stats["restored"] += 1
            _emit({"event": "file", "action": "restored", "path": r["source"]})

    _emit({"event": "done", "stats": stats})
    return stats


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def render_report(stats: dict, sample_limit: int = 10) -> list[str]:
    """Readable per-folder table + grand totals, sized for ~25 folders."""
    t = stats["totals"]
    mode = "DRY RUN" if stats["dry_run"] else "APPLIED"
    src = stats.get("root_source") or "?"
    note = stats.get("root_note") or ""
    checked = stats.get("mapping_checked", 0)
    proof = (f"mapping verified on {checked} row{'s' if checked != 1 else ''}"
             if checked else "no photo rows — nothing to map")
    lines = [f"photo root: {stats['photo_root']} (from {src}; {note}) — {proof}",
             f"[{mode}]", ""]
    if not stats["folders"]:
        lines.append("No *_unknown-camera folders to process.")
    else:
        head = (f"{'folder':<34}{'files':>7}{'move':>7}{'nomdl':>7}"
                f"{'dup':>6}{'conf':>6}{'idxd':>6}{'err':>5}  models")
        lines.append(head)
        lines.append("-" * len(head))
        for f in stats["folders"]:
            moved = f["moved"] + f["would_move"]
            models = ", ".join(f"{m} {n}" for m, n in sorted(f["by_model"].items())) or "-"
            lines.append(
                f"{f['name']:<34}{f['files']:>7}{moved:>7}{f['no_model']:>7}"
                f"{f['duplicate_left']:>6}{f['conflict']:>6}"
                f"{f['skipped_indexed']:>6}{f['errors']:>5}  {models}")
            if f["nested_dirs"]:
                lines.append(f"    nested subfolders left untouched: "
                             f"{', '.join(f['nested_dirs'][:6])}")
            if f["skipped_symlink"]:
                lines.append(f"    {f['skipped_symlink']} symlink(s) skipped — "
                             f"links are never moved")
            for note in f.get("notes", []):
                lines.append(f"    NEEDS MANUAL REPAIR: {note}")
            if f["remaining"] and not f["removed"]:
                verb = "would remain" if stats["dry_run"] else "kept"
                lines.append(f"    folder {verb}, holding: "
                             f"{', '.join(f['remaining'][:6])}"
                             f"{' …' if len(f['remaining']) > 6 else ''}")
            elif f["removed"]:
                lines.append("    folder emptied and removed")
            elif stats["dry_run"]:
                lines.append("    folder would be emptied and removed")
        lines.append("-" * len(head))
        lines.append(
            f"{'TOTAL':<34}{t['files']:>7}{t['moved'] + t['would_move']:>7}"
            f"{t['no_model']:>7}{t['duplicate_left']:>6}{t['conflict']:>6}"
            f"{t['skipped_indexed']:>6}{t['errors']:>5}")
    lines.append("")
    if t["inferred"]:
        lines.append(f"  {t['inferred']} routed by same-stem sibling inference "
                     f"(--infer-from-sibling)")
    if t["skipped_symlink"]:
        lines.append(f"  {t['skipped_symlink']} symlink(s) skipped")
    if t["would_collide"]:
        lines.append(f"  {t['would_collide']} destination name(s) already taken — "
                     f"hashed at apply time, not now (see the per-folder notes)")
    if t["interrupted_link"]:
        verb = "completed" if not stats["dry_run"] else "would be completed"
        lines.append(f"  {t['interrupted_link']} interrupted move(s) {verb} "
                     f"(one inode under two names, from an earlier kill)")
    if t["db_updated"]:
        lines.append(f"  {t['db_updated']} photos rows repointed")
    if t["healed"]:
        lines.append(f"  {t['healed']} stale rows healed (file already at its destination)")
    if t["would_heal"]:
        lines.append(f"  {t['would_heal']} row(s) would need healing — their file "
                     f"is already gone from the source folder")
    if t["heal_unverifiable"]:
        lines.append(f"  {t['heal_unverifiable']} stale row(s) could NOT be verified "
                     f"— left alone, repair by hand (see above)")
    if stats.get("skipped_undated"):
        lines.append(f"  skipped {UNDATED_DIRNAME}/{UNKNOWN_SUFFIX} — no date to "
                     f"route by; sort it by hand")
    if stats.get("limit_reached"):
        lines.append("  --limit reached; re-run to continue")
    if stats["samples"]:
        lines.append("")
        lines.append("Sample moves:")
        for s, d in stats["samples"][:sample_limit]:
            lines.append(f"  {s}")
            lines.append(f"    -> {d}")
    if stats["dry_run"]:
        lines.append("")
        lines.append("Dry run — nothing moved, nothing written. Re-run with "
                     "--apply --audit PATH.")
    return lines
