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
* **Nothing is ever overwritten and nothing is ever deleted.** A name collision
  at the destination is resolved by hashing: identical content leaves the source
  alone (`duplicate_left`), different content leaves it alone too (`conflict`).
* **Every move is recorded before the next one starts**, so `undo_refile` can
  put the tree back from the audit CSV even after a crash.

IO shape matters: this runs on a 4-core NAS with spinning disks. `extract_exif`
reads the file header only (``exifread.process_file(..., details=False)``);
hashing is whole-file and therefore happens ONLY on a name collision, never as
part of the normal path.
"""

from __future__ import annotations

import csv
import errno
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional

from .db import PhotoDB, _folder_of
from .exif import extract_exif
from .index import file_hash
from .ingest import (
    ALL_MEDIA_EXTENSIONS,
    UNDATED_DIRNAME,
    _file_suffix,
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

ProgressFn = Callable[[dict], None]


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
        "db_updated": 0,
        "healed": 0,
        "would_heal": 0,
        "errors": 0,
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

class _Audit:
    """Per-file CSV writer, flushed after every row.

    Flushing costs one small write per file and buys the thing that makes this
    tool reversible: a crash mid-run still leaves a complete record of every
    move that actually happened. Appends to an existing file (a resumed run
    keeps the earlier run's record) rather than truncating it.
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
            self._fh.flush()

    def row(self, action: str, source: str = "", destination: str = "",
            model: str = "", size: str = "", digest: str = "", reason: str = "") -> None:
        if self._w is None:
            return
        self._w.writerow([action, source, destination, model, size, digest, reason])
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._w = None


# ---------------------------------------------------------------------------
# moving
# ---------------------------------------------------------------------------

def _move_file(src: Path, dst: Path) -> int:
    """Move `src` onto `dst`, preserving mtime. Returns the destination size.

    Same-filesystem is the expected case (both folders live under the one photo
    root), so this is a plain atomic `os.rename`. A cross-device layout (EXDEV)
    falls back to copy + hash verify + only then remove the source — the source
    is never unlinked until the copy has been proven byte-identical.
    """
    st = src.stat()
    try:
        os.rename(str(src), str(dst))
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(str(src), str(dst))
        if file_hash(str(dst)) != file_hash(str(src)):
            try:
                dst.unlink()
            except OSError:
                pass
            raise OSError(f"cross-device copy of {src} did not verify; source kept")
        src.unlink()
    out = dst.stat()
    if out.st_size != st.st_size:
        raise OSError(f"destination {dst} is {out.st_size} bytes, expected {st.st_size}")
    os.utime(str(dst), (st.st_atime, st.st_mtime))
    return out.st_size


# ---------------------------------------------------------------------------
# DB links
# ---------------------------------------------------------------------------

def _db_links(db: PhotoDB, rel_folders: list[str]) -> tuple[dict, dict]:
    """Map the DB's view of the target folders.

    Returns ``(by_path, raw_refs)``:

    * ``by_path``  — ``rel filepath -> photo row`` for photos whose own row sits
      in one of these folders. RAW/video companions have NO row (ingest gates
      the DB path on ``is_photo = ext in INGEST_EXTENSIONS``), so this only ever
      matches JPEG/HEIC.
    * ``raw_refs`` — ``rel filepath -> [photo_id, ...]`` for files named by some
      photo's ``raw_filepath``. ``index.py`` sets that from ``find_raw_pair``,
      which looks in the photo's OWN directory, so this can only fire when a
      JPEG and its RAW both landed in the unknown-camera folder. One sequential
      scan of the non-null column, once per run.
    """
    by_path: dict[str, dict] = {}
    raw_refs: dict[str, list[int]] = {}
    if not rel_folders:
        return by_path, raw_refs

    for chunk_start in range(0, len(rel_folders), 400):
        chunk = rel_folders[chunk_start:chunk_start + 400]
        ph = ",".join("?" * len(chunk))
        for r in db.conn.execute(
            f"SELECT id, filepath, filename, file_hash, folder FROM photos "
            f"WHERE folder IN ({ph})", chunk
        ).fetchall():
            by_path[r["filepath"]] = dict(r)

    folder_set = set(rel_folders)
    for r in db.conn.execute(
        "SELECT id, raw_filepath FROM photos WHERE raw_filepath IS NOT NULL"
    ).fetchall():
        rp = r["raw_filepath"]
        if _folder_of(rp, os.path.basename(rp)) in folder_set:
            raw_refs.setdefault(rp, []).append(r["id"])
    return by_path, raw_refs


def _heal_folder(db: PhotoDB, folder: Path, year_dir: Path, date: str,
                 by_path: dict, raw_refs: dict,
                 apply: bool, audit: _Audit, counts: dict) -> None:
    """Repair rows left pointing into this folder at a file that has moved.

    This is the recovery half of the move-then-write ordering used by
    ``include_indexed``: the file is relocated first and the row updated
    second, so a crash in between leaves a row naming a path that no longer
    exists — recoverable, unlike the reverse order, which would point a row at
    a file that was never created.

    A re-run heals it: for each row in this folder whose file is gone, look for
    the same basename in the date's sibling `YYYY-MM-DD_*` folders and require
    a ``file_hash`` match (for ``raw_filepath`` refs, which carry no hash,
    require exactly one candidate). Anything ambiguous is left for a human.
    """
    siblings = [d for d in sorted(year_dir.iterdir())
                if d.is_dir() and d.name.startswith(date + "_") and d != folder]
    if not siblings:
        return

    def _candidates(name: str) -> list[Path]:
        return [d / name for d in siblings if (d / name).is_file()]

    prefix = db.relative_filepath(str(folder))
    for rel, row in list(by_path.items()):
        if _folder_of(rel, os.path.basename(rel)) != prefix:
            continue
        if os.path.exists(db.resolve_filepath(rel)):
            continue
        name = os.path.basename(rel)
        cands = _candidates(name)
        if row.get("file_hash"):
            cands = [c for c in cands if file_hash(str(c)) == row["file_hash"]]
        elif len(cands) > 1:
            cands = []
        if len(cands) != 1:
            continue
        new_rel = db.relative_filepath(str(cands[0]))
        if not apply:
            counts["would_heal"] += 1
            continue
        try:
            db.conn.execute(
                "UPDATE photos SET filepath = ?, folder = ? WHERE id = ?",
                (new_rel, _folder_of(new_rel, name), row["id"]))
            db.conn.commit()
        except Exception as exc:
            counts["errors"] += 1
            audit.row("error", rel, new_rel, reason=f"heal failed: {exc}")
            continue
        by_path.pop(rel, None)
        counts["healed"] += 1
        audit.row("healed", rel, new_rel, reason=f"photo_id={row['id']}")

    for rel, ids in list(raw_refs.items()):
        if _folder_of(rel, os.path.basename(rel)) != prefix:
            continue
        if os.path.exists(db.resolve_filepath(rel)):
            continue
        cands = _candidates(os.path.basename(rel))
        if len(cands) != 1:
            continue
        new_rel = db.relative_filepath(str(cands[0]))
        if not apply:
            counts["would_heal"] += 1
            continue
        try:
            for pid in ids:
                db.conn.execute("UPDATE photos SET raw_filepath = ? WHERE id = ?",
                                (new_rel, pid))
            db.conn.commit()
        except Exception as exc:
            counts["errors"] += 1
            audit.row("error", rel, new_rel, reason=f"heal failed: {exc}")
            continue
        raw_refs.pop(rel, None)
        counts["healed"] += 1
        audit.row("healed", rel, new_rel,
                  reason="raw_refs=" + "|".join(str(i) for i in ids))


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
            if not d.is_dir():
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
    is created. ``apply=True`` requires ``audit_path`` — the audit IS the undo.

    Only files at the TOP LEVEL of each folder are considered. A nested
    subdirectory is left untouched and reported: its layout carries meaning this
    tool cannot reproduce at the destination, and flattening it would invent
    collisions. That also means such a folder is never empty afterwards and is
    therefore never removed.

    Returns a stats dict::

        {"dry_run": bool, "photo_root": str, "audit_path": str|None,
         "folders": [ {name, path, files, by_model, nested_dirs, remaining,
                       removed, **counts}, ... ],
         "totals": {...}, "samples": [(src, dst), ...],
         "skipped_undated": int}
    """
    if apply and not audit_path:
        raise ValueError(
            "refusing to --apply without --audit: the audit CSV is the only "
            "record that can reverse these moves")

    with PhotoDB(db_path, photo_root=photo_root) as db:
        root = Path(photo_root or db.photo_root or ".").resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"photo root not found: {root}")

        targets = _find_folders(root, only)
        # DB keys go through the DB's own helper, never a hand-rolled
        # relative_to(root): if no photo_root is configured the indexer stored
        # ABSOLUTE paths, and a mismatch here would make every indexed photo
        # look unindexed — the one failure mode that must not happen quietly.
        rel_folders = [db.relative_filepath(str(f)) for _y, f, _d in targets]
        by_path, raw_refs = _db_links(db, rel_folders)

        audit = _Audit(audit_path if apply else None)
        totals = _zero_counts()
        folders: list[dict] = []
        samples: list[tuple[str, str]] = []
        budget = limit if limit is not None else None

        # `_undated/unknown-camera` has no date to route by, so there is no
        # destination folder to compute. Reported, never touched.
        undated = root / UNDATED_DIRNAME / UNKNOWN_SUFFIX
        skipped_undated = 1 if undated.is_dir() else 0

        def _emit(payload: dict) -> None:
            if on_progress is not None:
                on_progress(payload)

        try:
            for fi, (year_dir, folder, date) in enumerate(targets):
                rel_folder = db.relative_filepath(str(folder))
                _emit({"event": "folder", "name": folder.name, "index": fi,
                       "total": len(targets)})
                counts = _zero_counts()
                by_model: dict[str, int] = {}
                nested_dirs: list[str] = []

                _heal_folder(db, folder, year_dir, date,
                             by_path, raw_refs, apply, audit, counts)

                # Names this run will NOT move, so a dry run can predict what
                # would be left behind (and therefore whether the folder would
                # be removed) instead of just listing everything still present.
                leftovers: list[str] = []
                entries = sorted(folder.iterdir())
                candidates: list[Path] = []
                for p in entries:
                    if p.is_dir():
                        nested_dirs.append(p.name)
                        leftovers.append(p.name)
                        continue
                    if p.name.startswith("."):
                        leftovers.append(p.name)  # .DS_Store / ._AppleDouble
                        continue
                    if p.suffix.lower() in ALL_MEDIA_EXTENSIONS:
                        candidates.append(p)
                    else:
                        leftovers.append(p.name)  # sidecars, .aae edits, etc.

                relocating: set[str] = set()
                for src in candidates:
                    if budget is not None and budget <= 0:
                        break
                    counts["files"] += 1
                    # The audit records ABSOLUTE paths — it is a record of files
                    # on disk, and must stay readable without knowing which
                    # photo_root the run used. DB keys are the db-relative form.
                    abs_src = str(src)
                    rel_src = db.relative_filepath(abs_src)

                    try:
                        meta = extract_exif(str(src))
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
                        audit.row("no_model", abs_src, model="",
                                  size=str(_size(src)),
                                  reason="EXIF names no usable camera model")
                        _emit({"event": "file", "action": "no_model", "path": rel_src})
                        continue

                    by_model[suffix] = by_model.get(suffix, 0) + 1
                    dest_dir = year_dir / f"{date}_{suffix}"
                    dest = dest_dir / src.name
                    abs_dest = str(dest)
                    rel_dest = db.relative_filepath(abs_dest)

                    # DB gate. A row (or a raw_filepath reference) makes the file
                    # path-bearing state, not just bytes.
                    linked_ids = list(raw_refs.get(rel_src, ()))
                    row = by_path.get(rel_src)
                    if (row or linked_ids) and not include_indexed:
                        counts["skipped_indexed"] += 1
                        audit.row("skipped_indexed", abs_src, abs_dest, suffix,
                                  str(_size(src)),
                                  reason="has a photos row (or is a raw_filepath "
                                         "target); re-run with --include-indexed")
                        _emit({"event": "file", "action": "skipped_indexed",
                               "path": rel_src})
                        continue

                    if dest.exists():
                        # The ONLY place whole-file hashing happens. Doing it for
                        # every file would mean reading the whole library off a
                        # spinning disk; a name collision is rare and is exactly
                        # the case where the bytes have to be compared.
                        try:
                            digest = file_hash(str(src))
                            same = digest == file_hash(str(dest))
                        except Exception as exc:
                            counts["errors"] += 1
                            audit.row("error", abs_src, abs_dest, suffix,
                                      reason=f"hash failed: {exc}")
                            continue
                        if same:
                            counts["duplicate_left"] += 1
                            audit.row("duplicate_left", abs_src, abs_dest, suffix,
                                      str(_size(src)), digest,
                                      "identical copy already at destination; "
                                      "source left in place, delete by hand")
                        else:
                            counts["conflict"] += 1
                            audit.row("conflict", abs_src, abs_dest, suffix,
                                      str(_size(src)), digest,
                                      "different file already at destination; "
                                      "nothing overwritten, nothing renamed")
                        _emit({"event": "file",
                               "action": "duplicate_left" if same else "conflict",
                               "path": rel_src})
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

                    try:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        size = _move_file(src, dest)
                    except Exception as exc:
                        counts["errors"] += 1
                        audit.row("error", abs_src, abs_dest, suffix,
                                  reason=f"move failed: {exc}")
                        _emit({"event": "file", "action": "error", "path": rel_src})
                        continue

                    # Move first, write the row second. See _heal_folder for why
                    # this order is the recoverable one.
                    reason_bits = []
                    if row is not None:
                        try:
                            db.conn.execute(
                                "UPDATE photos SET filepath = ?, folder = ? WHERE id = ?",
                                (rel_dest, _folder_of(rel_dest, src.name), row["id"]))
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
                                db.conn.execute(
                                    "UPDATE photos SET raw_filepath = ? WHERE id = ?",
                                    (rel_dest, pid))
                            db.conn.commit()
                            counts["db_updated"] += 1
                            reason_bits.append(
                                "raw_refs=" + "|".join(str(i) for i in linked_ids))
                            raw_refs.pop(rel_src, None)
                        except Exception as exc:
                            counts["errors"] += 1
                            reason_bits.append(f"raw_filepath update FAILED: {exc}")
                    if inferred:
                        counts["inferred"] += 1
                        reason_bits.append("model inferred from same-stem sibling")

                    counts["moved"] += 1
                    relocating.add(src.name)
                    if budget is not None:
                        budget -= 1
                    audit.row("moved", abs_src, abs_dest, suffix, str(size),
                              reason=";".join(reason_bits))
                    _emit({"event": "file", "action": "moved", "path": rel_src,
                           "destination": rel_dest})

                # On apply this is the real listing; on a dry run it is the
                # prediction (everything that would NOT move), so the report
                # can say whether the folder would survive the run.
                leftovers += [c.name for c in candidates if c.name not in relocating]
                remaining = (sorted(p.name for p in folder.iterdir())
                             if (apply and folder.is_dir()) else sorted(leftovers))
                removed = False
                if apply and not remaining:
                    try:
                        folder.rmdir()  # rmdir only — fails safely if not empty
                        removed = True
                    except OSError as exc:
                        audit.row("error", str(folder),
                                  reason=f"rmdir failed: {exc}")

                folders.append({
                    "name": folder.name, "path": rel_folder,
                    "by_model": by_model, "nested_dirs": nested_dirs,
                    "remaining": remaining[:25], "removed": removed, **counts,
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
            "audit_path": audit_path if apply else None,
            "folders": folders,
            "totals": totals,
            "samples": samples,
            "skipped_undated": skipped_undated,
            "limit_reached": bool(budget is not None and budget <= 0),
        }


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


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
                verify_hash: bool = False,
                on_progress: Optional[ProgressFn] = None) -> dict:
    """Reverse every `moved` row in an audit CSV, newest first.

    A row is restored only when the destination is still the file this tool put
    there — same size, and the same hash when one was recorded (a hash is only
    recorded for rows that hit a name collision, because hashing every file on
    the way out would mean reading the whole library). ``verify_hash=True``
    hashes every destination instead: slow on spinning disks, but it catches a
    same-size edit. The source path must also be free; anything else is
    ``refused`` and left exactly as it is.

    Any DB row this tool repointed (recorded in the audit's ``reason``) is
    pointed back, after the file has been restored.
    """
    rows = []
    with open(audit_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("action") == "moved":
                rows.append(r)
    rows.reverse()

    stats = {"candidates": len(rows), "restored": 0, "would_restore": 0,
             "refused": 0, "errors": 0, "refusals": [], "dry_run": not apply}

    def _emit(payload: dict) -> None:
        if on_progress is not None:
            on_progress(payload)

    with PhotoDB(db_path) as db:
        root = Path(db.photo_root or ".").resolve()
        for r in rows:
            src = root / r["source"]
            dst = root / r["destination"]
            why = None
            if not dst.is_file():
                why = "destination no longer exists"
            elif src.exists():
                why = "source path is occupied"
            elif r.get("size") and str(_size(dst)) != r["size"]:
                why = "destination size changed since the move"
            elif (r.get("hash") or verify_hash) and dst.is_file():
                try:
                    want = r.get("hash") or ""
                    got = file_hash(str(dst))
                    if want and got != want:
                        why = "destination content changed since the move"
                except Exception as exc:
                    why = f"hash failed: {exc}"
            if why:
                stats["refused"] += 1
                if len(stats["refusals"]) < 20:
                    stats["refusals"].append((r["destination"], why))
                _emit({"event": "file", "action": "refused",
                       "path": r["destination"], "reason": why})
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
                    db.conn.execute(
                        "UPDATE photos SET filepath = ?, folder = ? WHERE id = ?",
                        (back, _folder_of(back, src.name), photo_id))
                for pid in raw_ids:
                    db.conn.execute(
                        "UPDATE photos SET raw_filepath = ? WHERE id = ?",
                        (back, pid))
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
    lines = [f"Photo root: {stats['photo_root']}   [{mode}]", ""]
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
        lines.append(f"  {t['inferred']} routed by same-stem sibling inference (--infer-from-sibling)")
    if t["db_updated"]:
        lines.append(f"  {t['db_updated']} photos rows repointed")
    if t["healed"] or t["would_heal"]:
        lines.append(f"  {t['healed'] or t['would_heal']} stale rows healed "
                     f"(file already at its destination)")
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
