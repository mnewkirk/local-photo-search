"""``unwrap-zip-videos`` — extract ZIP-wrapped clips IN PLACE, on the NAS.

~58% of the 2016-17 "moov atom not found" videos are ZIP archives under a
video extension (an iOS Live-bundle export bug): ``IMG_1234.MOV`` starts with
``PK\\x03\\x04`` and wraps one playable ``IMG_1234.MOV``. The desktop scan
already unwraps them on the fly (``video_scan.unwrap_zip``); this tool fixes
the files themselves so every consumer (browser playback, ffprobe, the poster)
sees a real video. Member selection and caps are shared with the scan
(``photosearch/zip_video.py``), so the two cannot disagree on what is safe.

Per file (``--apply``):

1. validate the archive (exactly one safe video member, size/ratio caps);
2. extract the member to a temp file in the SAME directory, opened
   ``O_CREAT|O_EXCL``, sha256 computed while writing, fsync file + dir;
3. give the zip a second name ``<name>.zip`` with ``os.link`` — ``EEXIST``
   refuses the file (never overwrites, never renames to something else);
4. confirm the original is still the inode/size/mtime that was validated,
   then ``os.replace(temp, original)``: the original filename now holds the
   playable video, so ``filepath`` (and ``folder``) never change. The zip's
   mtime/atime and mode are copied onto the new file first — ingest and
   registration route by mtime.

The zip is NEVER deleted; it survives as ``<name>.zip``, which no walker picks
up (``.zip`` is in none of ``VIDEO_EXTENSIONS``, ``index.SUPPORTED_EXTENSIONS``
or ingest's ``ALL_MEDIA_EXTENSIONS``).

DB (``--apply`` only): a ``media_type='video'`` row at that filepath is reset
for RE-PROBE — ``video_meta.probe_version/probe_error/probe_json`` NULL and its
``video-probe`` ledger row deleted — and ``photos.file_hash`` is set to the
inner file's sha256 (the hash a scan of the file would now store; a scan of a
zip-wrapped row had stored the ZIP's hash). Scan results are KEPT: the scan of
a ``zip_wrapped`` row already ran on the extracted inner video, byte-identical
to what is on disk now, so re-scanning would redo the same work. The one
exception is a stored ``scan_error`` starting ``zip:`` (the scan refused the
wrapper): that is cleared so the scan re-runs on the real file. Every UPDATE
is guarded on the value it read.

Audit (mandatory for ``--apply``): CSV, fsynced per row — ``intent`` before the
link, ``done`` after the replace (inner size + sha256), ``refused`` when the
extraction itself refuses. ``undo_unwrap`` restores the zip under the original
name only when the current file there matches the recorded size + sha256.

Throttled for the N100 (the 2026-09-19 disk-starvation incident): nice(10) +
IDLE I/O class, a byte-rate cap on the copy (``mbps``), refuses to start while
an ingest sweep holds its lock, and ``--apply`` HOLDS that lock (as refile
does) so a sweep cannot start mid-run.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import zip_video as ZV

AUDIT_FIELDS = ["event", "ts", "path", "zip_path", "zip_size", "zip_mtime_ns",
                "member", "inner_size", "inner_sha256", "photo_id",
                "old_file_hash", "note"]
ZIP_SUFFIX = ".zip"
DEFAULT_MBPS = 30.0
SAMPLE_REFUSALS = 20
_YEAR_RE = re.compile(r"^\d{4}$")


class UnwrapRefused(Exception):
    """The run as a whole cannot start (bad audit path, ingest running...)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def zip_name(path: str) -> str:
    return path + ZIP_SUFFIX


def _temp_name(path: str, tag: str = "unwrap") -> str:
    d, base = os.path.split(path)
    return os.path.join(d, f".{base}.{tag}.tmp")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fsync_dir(d: str) -> None:
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _same_inode(a: str, b: str) -> bool:
    try:
        sa, sb = os.lstat(a), os.lstat(b)
    except OSError:
        return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def _year_of(rel: str) -> str:
    first = rel.replace("\\", "/").split("/", 1)[0]
    return first if _YEAR_RE.match(first) else "other"


def refusal_key(msg: str) -> str:
    """Bucket a ZipRefused message for the report."""
    m = msg.lower()
    for needle, key in (("video file(s)", "video_count"), ("unsafe member", "unsafe_name"),
                        ("encrypted", "encrypted"), ("bytes >", "oversize"),
                        ("exceeded", "oversize"), ("inflates suspiciously", "ratio"),
                        ("does not inflate", "not_inflatable"),
                        ("not a readable zip", "not_zip")):
        if needle in m:
            return key
    return "other"


class RateLimiter:
    """Byte-rate cap over a run: sleeps so cumulative bytes never run ahead of
    ``mbps`` MB/s. Never bursts to catch up after a slow stretch."""

    def __init__(self, mbps: Optional[float], clock=time.monotonic, sleep=time.sleep):
        self.rate = float(mbps) * 1e6 if mbps and mbps > 0 else None
        self.clock, self.sleep = clock, sleep
        self.t0 = clock()
        self.done = 0.0

    def consume(self, n: int) -> None:
        if self.rate is None:
            return
        self.done += n
        now = self.clock()
        due = self.t0 + self.done / self.rate
        if due > now:
            self.sleep(due - now)
        elif now - due > 1.0:   # fell behind (I/O, other work): re-anchor, no burst
            self.t0, self.done = now, 0.0


class _Audit:
    """CSV audit, flushed and fsynced after every row; appends to an existing
    file so a resumed run keeps the earlier record."""

    def __init__(self, path: str):
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        self._fh = open(path, "a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=AUDIT_FIELDS)
        if not exists:
            self._w.writeheader()
            self._sync()

    def _sync(self) -> None:
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass

    def row(self, event: str, **kw) -> None:
        rec = {k: "" for k in AUDIT_FIELDS}
        rec.update({k: ("" if v is None else v) for k, v in kw.items()})
        rec["event"], rec["ts"] = event, _now()
        self._w.writerow(rec)
        self._sync()

    def close(self) -> None:
        self._fh.close()


def _check_audit_path(audit_path: str, root: Path) -> None:
    a = Path(audit_path).expanduser().resolve()
    try:
        a.relative_to(root)
    except ValueError:
        return
    raise UnwrapRefused(
        f"audit path {a} is inside the photo root {root}: put it on /data "
        f"(an audit file among the photos would be walked as library content)")


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def _reset_row_for_reprobe(conn, stored: str, inner_sha: Optional[str]) -> dict:
    """Reset a video row whose file just became a real video. Guarded UPDATEs;
    commits. Returns ``{"photo_id", "old_file_hash", "scan_cleared"}`` or {}."""
    row = conn.execute("SELECT id, media_type, file_hash FROM photos WHERE filepath = ?",
                       (stored,)).fetchone()
    if row is None or row[1] != "video":
        return {}
    pid, old_hash = int(row[0]), row[2]
    out = {"photo_id": pid, "old_file_hash": old_hash, "scan_cleared": False}
    conn.execute("UPDATE photos SET file_hash = ? WHERE id = ? AND file_hash IS ?",
                 (inner_sha, pid, old_hash))
    vm = conn.execute("SELECT probe_version, probe_error, scan_error FROM video_meta "
                      "WHERE photo_id = ?", (pid,)).fetchone()
    if vm is not None:
        conn.execute(
            "UPDATE video_meta SET probe_version = NULL, probe_error = NULL, "
            "probe_json = NULL WHERE photo_id = ? AND probe_version IS ? "
            "AND probe_error IS ?", (pid, vm[0], vm[1]))
        if vm[2] and str(vm[2]).startswith("zip:"):
            cur = conn.execute(
                "UPDATE video_meta SET scan_version = NULL, scan_error = NULL "
                "WHERE photo_id = ? AND scan_error IS ?", (pid, vm[2]))
            out["scan_cleared"] = cur.rowcount > 0
    conn.execute("DELETE FROM worker_processed WHERE photo_id = ? "
                 "AND pass_type = 'video-probe'", (pid,))
    conn.commit()
    return out


def _needs_heal(conn, stored: str) -> bool:
    """An already-unwrapped file whose row still says ``zip_wrapped`` (a DB
    write that failed after the file was fixed)."""
    r = conn.execute(
        "SELECT 1 FROM photos p JOIN video_meta vm ON vm.photo_id = p.id "
        "WHERE p.filepath = ? AND p.media_type = 'video' "
        "AND vm.probe_error LIKE 'zip_wrapped%'", (stored,)).fetchone()
    return r is not None


# ---------------------------------------------------------------------------
# root / DB opening
# ---------------------------------------------------------------------------

def _resolve_root(db_path: str, photo_root: Optional[str]) -> Path:
    """``--photo-root`` > ``PHOTO_ROOT`` env > the DB's stored value — refile's
    precedence, read through refile's read-only stand-in."""
    from .refile import _ReadOnlyDB, _effective_root
    with _ReadOnlyDB(db_path, photo_root) as ro:
        root, _src = _effective_root(ro, photo_root)
    if not root.is_dir():
        raise FileNotFoundError(f"photo root not accessible: {root}")
    return root


# ---------------------------------------------------------------------------
# forward run
# ---------------------------------------------------------------------------

def _new_counts(root: Path, apply: bool) -> dict:
    return {"root": str(root), "dry_run": not apply, "videos_seen": 0, "zip_found": 0,
            "unwrapped": 0, "would_unwrap": 0, "already_unwrapped": 0,
            "beyond_limit": 0, "refused": {}, "refusal_samples": [], "errors": 0,
            "bytes_zip": 0, "bytes_inner": 0, "per_year": {}, "rows_reset": 0,
            "rows_healed": 0, "scan_cleared": 0, "db_errors": 0, "priority": None}


def _refuse(counts: dict, rel: str, key: str, msg: str) -> None:
    counts["refused"][key] = counts["refused"].get(key, 0) + 1
    if len(counts["refusal_samples"]) < SAMPLE_REFUSALS:
        counts["refusal_samples"].append({"path": rel, "reason": key, "detail": msg})


def unwrap_zip_videos(db_path: str, *, photo_root: Optional[str] = None,
                      folder: Optional[str] = None, year: Optional[str] = None,
                      limit: Optional[int] = None, apply: bool = False,
                      audit_path: Optional[str] = None, mbps: float = DEFAULT_MBPS,
                      lower_priority: bool = True,
                      on_progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Find and (with ``apply``) unwrap ZIP-wrapped videos under the photo root.

    Dry run (default) opens the DB ``mode=ro`` (only to resolve the root) and
    writes nothing: it validates each archive's central directory and reports
    what would be unwrapped. Raises ``ingest.IngestAlreadyRunning`` while an
    ingest sweep is running, ``UnwrapRefused`` for a missing/bad audit path.
    """
    from .ingest import IngestAlreadyRunning, _sweep_lock, sweep_lock_holder

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database not found: {db_path}")
    if folder and year:
        raise ValueError("--folder and --year are exclusive")
    if year:
        folder = str(year)
    root = _resolve_root(db_path, photo_root)
    if apply:
        if not audit_path:
            raise UnwrapRefused("--apply requires --audit PATH (the audit is the undo)")
        _check_audit_path(audit_path, root)
    holder = sweep_lock_holder(db_path)
    if holder:
        raise IngestAlreadyRunning(f"{holder} is running — try later")

    counts = _new_counts(root, apply)
    if lower_priority:
        from .sharpness_backfill import lower_priority as _lower
        counts["priority"] = _lower()

    if not apply:
        _walk(root, folder, limit, counts, None, None, None, on_progress)
        return counts

    from .db import PhotoDB
    with _sweep_lock(db_path, holder="unwrap-zip-videos"), PhotoDB(db_path, photo_root=str(root)) as db:
        audit = _Audit(audit_path)
        try:
            _walk(root, folder, limit, counts, db, audit, RateLimiter(mbps), on_progress)
        finally:
            audit.close()
    return counts


def _walk(root: Path, folder, limit, counts, db, audit, limiter, on_progress) -> None:
    from .video_registry import iter_video_files
    attempted = 0
    for abs_path in iter_video_files(str(root), folder):
        counts["videos_seen"] += 1
        rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
        if os.path.islink(abs_path):
            continue  # never followed; a link is not the file we would fix
        try:
            is_zip = ZV.has_zip_magic(abs_path)
        except OSError as e:
            counts["errors"] += 1
            _refuse(counts, rel, "read_error", str(e))
            continue
        if not is_zip:
            if os.path.lexists(zip_name(abs_path)):
                counts["already_unwrapped"] += 1
                if db is not None:
                    _heal(db, abs_path, rel, counts)
            continue
        counts["zip_found"] += 1
        y = _year_of(rel)
        pc = counts["per_year"].setdefault(y, {"zip_found": 0, "unwrapped": 0, "refused": 0})
        pc["zip_found"] += 1
        if limit is not None and attempted >= int(limit):
            counts["beyond_limit"] += 1
            continue
        attempted += 1
        ok = _one(abs_path, rel, counts, db, audit, limiter)
        if ok is True:
            pc["unwrapped"] += 1
        elif ok is False:
            pc["refused"] += 1
        if on_progress and attempted % 25 == 0:
            on_progress(dict(counts))


def _heal(db, abs_path: str, rel: str, counts: dict) -> None:
    from .db import PhotoDB
    stored = PhotoDB.relative_filepath(db, abs_path)
    try:
        if _needs_heal(db.conn, stored):
            # The inner hash is unknown without a full read: NULL is honest.
            if _reset_row_for_reprobe(db.conn, stored, None):
                counts["rows_healed"] += 1
    except Exception as e:  # noqa: BLE001 — a DB hiccup must not stop the walk
        counts["db_errors"] += 1
        _refuse(counts, rel, "db_error", str(e))


def _one(path: str, rel: str, counts: dict, db, audit, limiter) -> Optional[bool]:
    """Returns True unwrapped (or would), False refused, None error."""
    zpath = zip_name(path)
    st = os.lstat(path)
    resumed = False
    if os.path.lexists(zpath):
        if _same_inode(path, zpath):
            resumed = True   # a kill between link and replace: the zip is already kept
        else:
            _refuse(counts, rel, "zip_name_taken", f"{os.path.basename(zpath)} exists")
            return False
    try:
        zf = ZV.open_zip(path)
    except ZV.ZipRefused as e:
        _refuse(counts, rel, refusal_key(str(e)), str(e))
        return False
    except OSError as e:
        counts["errors"] += 1
        _refuse(counts, rel, "read_error", str(e))
        return None
    with zf:
        from .ingest import VIDEO_EXTENSIONS
        try:
            m, cap = ZV.select_video_member(zf, st.st_size, video_exts=VIDEO_EXTENSIONS)
        except ZV.ZipRefused as e:
            _refuse(counts, rel, refusal_key(str(e)), str(e))
            return False
        if db is None:
            counts["would_unwrap"] += 1
            counts["bytes_zip"] += st.st_size
            counts["bytes_inner"] += m.file_size
            return True
        return _apply_one(path, rel, st, zf, m, cap, resumed, counts, db, audit, limiter)


def _apply_one(path, rel, st, zf, m, cap, resumed, counts, db, audit, limiter) -> Optional[bool]:
    zpath = zip_name(path)
    d = os.path.dirname(path)
    temp = _temp_name(path)
    base = dict(path=path, zip_path=zpath, zip_size=st.st_size,
                zip_mtime_ns=st.st_mtime_ns, member=m.filename)
    if os.path.lexists(temp):
        if not stat.S_ISREG(os.lstat(temp).st_mode):
            _refuse(counts, rel, "temp_name_taken", temp)
            return False
        os.unlink(temp)   # our own artifact from a killed run; never the only copy
    audit.row("intent", **base, note="resumed_link" if resumed else "")

    h = hashlib.sha256()

    def on_chunk(chunk: bytes) -> None:
        h.update(chunk)
        limiter.consume(2 * len(chunk))   # read + write

    try:
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        _refuse(counts, rel, "temp_name_taken", temp)
        return False
    try:
        with os.fdopen(fd, "wb") as dst:
            n = ZV.copy_member(zf, m, dst, cap, on_chunk=on_chunk)
            dst.flush()
            os.fsync(dst.fileno())
    except ZV.ZipRefused as e:
        _unlink_quiet(temp)
        audit.row("refused", **base, note=str(e))
        _refuse(counts, rel, refusal_key(str(e)), str(e))
        return False
    except BaseException:
        _unlink_quiet(temp)
        raise
    sha = h.hexdigest()
    try:
        os.chmod(temp, stat.S_IMODE(st.st_mode))
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            try:
                os.chown(temp, st.st_uid, st.st_gid)
            except OSError:
                pass
        os.utime(temp, ns=(st.st_atime_ns, st.st_mtime_ns))
    except OSError as e:
        _unlink_quiet(temp)
        counts["errors"] += 1
        _refuse(counts, rel, "metadata_error", str(e))
        return None
    _fsync_dir(d)

    # The original must still be exactly what was validated.
    try:
        st2 = os.lstat(path)
    except OSError as e:
        _unlink_quiet(temp)
        _refuse(counts, rel, "changed_during_run", str(e))
        return False
    if (st2.st_dev, st2.st_ino, st2.st_size, st2.st_mtime_ns) != \
            (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns):
        _unlink_quiet(temp)
        _refuse(counts, rel, "changed_during_run", "original changed since validation")
        return False

    if not resumed:
        try:
            os.link(path, zpath)
        except FileExistsError:
            _unlink_quiet(temp)
            audit.row("refused", **base, note="zip name taken (EEXIST)")
            _refuse(counts, rel, "zip_name_taken", f"{os.path.basename(zpath)} exists")
            return False
        except OSError as e:
            _unlink_quiet(temp)
            counts["errors"] += 1
            audit.row("refused", **base, note=f"link failed: {e}")
            _refuse(counts, rel, "link_error", str(e))
            return None
        _fsync_dir(d)
    if not _same_inode(path, zpath):
        _unlink_quiet(temp)
        _refuse(counts, rel, "changed_during_run", "zip name is not the original inode")
        return False

    os.replace(temp, path)
    _fsync_dir(d)

    reset: dict = {}
    note = ""
    try:
        from .db import PhotoDB
        reset = _reset_row_for_reprobe(db.conn, PhotoDB.relative_filepath(db, path), sha)
    except Exception as e:  # noqa: BLE001 — the file is fixed; a re-run heals the row
        counts["db_errors"] += 1
        note = f"db_error: {e}"
        try:
            db.conn.rollback()
        except Exception:  # noqa: BLE001
            pass
    if reset:
        counts["rows_reset"] += 1
        counts["scan_cleared"] += int(bool(reset.get("scan_cleared")))
    audit.row("done", **base, inner_size=n, inner_sha256=sha,
              photo_id=reset.get("photo_id"), old_file_hash=reset.get("old_file_hash"),
              note=note)
    counts["unwrapped"] += 1
    counts["bytes_zip"] += st.st_size
    counts["bytes_inner"] += n
    return True


def _unlink_quiet(p: str) -> None:
    try:
        os.unlink(p)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------

def _sha_of_file(path: str, limiter: RateLimiter) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
            limiter.consume(len(b))
    return h.hexdigest()


def _sha_of_member(zpath: str, member: str, limiter: RateLimiter) -> tuple[int, str]:
    """Hash what the forward run would have written (an intent with no done
    row carries no digest): the member, streamed, nothing written."""
    from .ingest import VIDEO_EXTENSIONS
    h = hashlib.sha256()

    class _Null:
        def write(self, b):
            return len(b)

    with ZV.open_zip(zpath) as zf:
        m, cap = ZV.select_video_member(zf, os.path.getsize(zpath),
                                        video_exts=VIDEO_EXTENSIONS)
        if member and m.filename != member:
            raise ZV.ZipRefused(f"member {m.filename!r} != recorded {member!r}")

        def oc(b):
            h.update(b)
            limiter.consume(len(b))
        n = ZV.copy_member(zf, m, _Null(), cap, on_chunk=oc)
    return n, h.hexdigest()


def read_audit(audit_path: str) -> list[dict]:
    """One entry per path: the last ``intent`` and (if any) the ``done``
    after it."""
    with open(audit_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_path: dict[str, dict] = {}
    for r in rows:
        p = r.get("path") or ""
        if r.get("event") == "intent":
            by_path[p] = {"intent": r, "done": None}
        elif r.get("event") == "done" and p in by_path:
            by_path[p]["done"] = r
        elif r.get("event") == "refused" and p in by_path and by_path[p]["done"] is None:
            by_path.pop(p)   # extraction refused: nothing was linked or replaced
    return [dict(path=p, **v) for p, v in by_path.items()]


def undo_unwrap(audit_path: str, db_path: Optional[str], *, apply: bool = False,
                mbps: float = DEFAULT_MBPS, lower_priority: bool = True) -> dict:
    """Put the zip back under the original name for every audited unwrap.

    A row is restored only when the file now at the original path is the one
    this tool wrote — same size AND sha256 (the sha is checked on ``apply``;
    a dry run checks size only and reads nothing more) — and the ``.zip`` is
    still the recorded size. Restoring uses link + replace onto the verified
    file, then drops the extra ``.zip`` name only while ``st_nlink >= 2``
    proves the original name still holds the zip. Refused rows are left alone.
    DB rows are pointed back at "needs probe" and their old ``file_hash``."""
    from .ingest import IngestAlreadyRunning, _no_lock, _sweep_lock, sweep_lock_holder
    entries = read_audit(audit_path)
    res = {"dry_run": not apply, "entries": len(entries), "restored": 0,
           "would_restore": 0, "already_restored": 0, "interrupted_cleaned": 0,
           "refused": {}, "refusal_samples": [], "rows_reverted": 0, "db_errors": 0}
    if apply:
        if not db_path or not os.path.exists(db_path):
            raise FileNotFoundError(f"database not found: {db_path}")
        holder = sweep_lock_holder(db_path)
        if holder:
            raise IngestAlreadyRunning(f"{holder} is running — try later")
    if lower_priority:
        from .sharpness_backfill import lower_priority as _lower
        _lower()
    limiter = RateLimiter(mbps)
    db = None
    lock = (_sweep_lock(db_path, holder="unwrap-zip-videos-undo")
            if apply else _no_lock())
    with lock:
        if apply:
            from .db import PhotoDB
            db = PhotoDB(db_path)
        try:
            for e in entries:
                _undo_one(e, res, db, apply, limiter)
        finally:
            if db is not None:
                db.close()
    return res


def _undo_refuse(res, path, key, detail=""):
    res["refused"][key] = res["refused"].get(key, 0) + 1
    if len(res["refusal_samples"]) < SAMPLE_REFUSALS:
        res["refusal_samples"].append({"path": path, "reason": key, "detail": detail})


def _undo_one(e: dict, res: dict, db, apply: bool, limiter: RateLimiter) -> None:
    path = e["path"]
    intent, done = e["intent"], e["done"]
    zpath = intent.get("zip_path") or zip_name(path)
    if not os.path.lexists(zpath):
        try:
            if ZV.has_zip_magic(path):
                res["already_restored"] += 1   # or never started
                return
        except OSError:
            pass
        _undo_refuse(res, path, "zip_missing", zpath)
        return
    zst = os.lstat(zpath)
    if str(zst.st_size) != str(intent.get("zip_size")):
        _undo_refuse(res, path, "zip_changed", f"{zst.st_size} != {intent.get('zip_size')}")
        return
    d = os.path.dirname(path)

    if _same_inode(path, zpath):
        # Killed between link and replace: the original never changed; only
        # the extra .zip name needs to go.
        if not apply:
            res["would_restore"] += 1
            return
        if os.lstat(zpath).st_nlink >= 2 and _same_inode(path, zpath):
            os.unlink(zpath)
            _fsync_dir(d)
            res["interrupted_cleaned"] += 1
        else:
            _undo_refuse(res, path, "unlink_refused", "nlink < 2")
        return

    if not os.path.lexists(path):
        _undo_refuse(res, path, "original_missing")
        return
    cur = os.lstat(path)
    want_size = (done or {}).get("inner_size")
    want_sha = (done or {}).get("inner_sha256")
    if want_size and str(cur.st_size) != str(want_size):
        _undo_refuse(res, path, "size_mismatch", f"{cur.st_size} != {want_size}")
        return
    if not apply:
        res["would_restore"] += 1
        return
    try:
        if not want_sha:
            n, want_sha = _sha_of_member(zpath, intent.get("member") or "", limiter)
            if n != cur.st_size:
                _undo_refuse(res, path, "size_mismatch", f"{cur.st_size} != {n}")
                return
        if _sha_of_file(path, limiter) != want_sha:
            _undo_refuse(res, path, "sha_mismatch")
            return
    except (OSError, ZV.ZipRefused) as ex:
        _undo_refuse(res, path, "verify_error", str(ex))
        return

    tmp = _temp_name(path, "undo")
    try:
        os.link(zpath, tmp)
    except FileExistsError:
        _undo_refuse(res, path, "temp_name_taken", tmp)
        return
    cur2 = os.lstat(path)
    if (cur2.st_dev, cur2.st_ino, cur2.st_size, cur2.st_mtime_ns) != \
            (cur.st_dev, cur.st_ino, cur.st_size, cur.st_mtime_ns):
        _unlink_quiet(tmp)
        _undo_refuse(res, path, "changed_during_run")
        return
    os.replace(tmp, path)
    _fsync_dir(d)
    if _same_inode(path, zpath) and os.lstat(zpath).st_nlink >= 2:
        os.unlink(zpath)
        _fsync_dir(d)
    res["restored"] += 1

    pid = (done or {}).get("photo_id")
    if db is not None and pid:
        try:
            pid = int(pid)
            old = (done or {}).get("old_file_hash") or None
            db.conn.execute("UPDATE photos SET file_hash = ? WHERE id = ? AND file_hash IS ?",
                            (old, pid, want_sha))
            db.conn.execute("UPDATE video_meta SET probe_version = NULL, probe_error = NULL, "
                            "probe_json = NULL WHERE photo_id = ?", (pid,))
            db.conn.execute("DELETE FROM worker_processed WHERE photo_id = ? "
                            "AND pass_type = 'video-probe'", (pid,))
            db.conn.commit()
            res["rows_reverted"] += 1
        except Exception as ex:  # noqa: BLE001
            res["db_errors"] += 1
            _undo_refuse(res, path, "db_error", str(ex))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def render_report(c: dict) -> str:
    mode = "DRY RUN — nothing written" if c["dry_run"] else "APPLIED"
    lines = [f"unwrap-zip-videos ({mode})", f"  photo root: {c['root']}",
             f"  videos seen:        {c['videos_seen']:,}",
             f"  ZIP-wrapped found:  {c['zip_found']:,}"]
    if c["dry_run"]:
        lines.append(f"  would unwrap:       {c['would_unwrap']:,}")
    else:
        lines.append(f"  unwrapped:          {c['unwrapped']:,}")
    lines += [f"  already unwrapped:  {c['already_unwrapped']:,}",
              f"  beyond --limit:     {c['beyond_limit']:,}",
              f"  refused:            {sum(c['refused'].values()):,}"]
    for k, v in sorted(c["refused"].items(), key=lambda kv: -kv[1]):
        lines.append(f"      {k:<20} {v:,}")
    lines.append(f"  bytes: zip {_gb(c['bytes_zip'])} -> inner {_gb(c['bytes_inner'])}"
                 + (" (the .zip is kept: disk use grows by the inner size)"
                    if c["bytes_inner"] else ""))
    if not c["dry_run"]:
        lines.append(f"  DB rows reset for re-probe: {c['rows_reset']:,} "
                     f"(scan errors cleared {c['scan_cleared']:,}, healed "
                     f"{c['rows_healed']:,}, db errors {c['db_errors']:,})")
    if c["errors"]:
        lines.append(f"  errors: {c['errors']:,}")
    if c["per_year"]:
        lines.append("  per year (zip found / unwrapped|would / refused):")
        for y in sorted(c["per_year"]):
            p = c["per_year"][y]
            lines.append(f"      {y:<6} {p['zip_found']:>6,} {p['unwrapped']:>6,} "
                         f"{p['refused']:>6,}")
    if c["refusal_samples"]:
        lines.append("  sample refusals:")
        for s in c["refusal_samples"]:
            lines.append(f"      [{s['reason']}] {s['path']} — {s['detail']}")
    return "\n".join(lines)


def render_undo_report(r: dict) -> str:
    mode = "DRY RUN — nothing written (sha256 is verified on --apply)" \
        if r["dry_run"] else "APPLIED"
    lines = [f"unwrap-zip-videos --undo ({mode})",
             f"  audited unwraps:   {r['entries']:,}",
             f"  {'would restore' if r['dry_run'] else 'restored'}:"
             f"{'' if r['dry_run'] else '     '}     "
             f"{r['would_restore'] if r['dry_run'] else r['restored']:,}",
             f"  interrupted links cleaned: {r['interrupted_cleaned']:,}",
             f"  already restored / never started: {r['already_restored']:,}",
             f"  refused:           {sum(r['refused'].values()):,}"]
    for k, v in sorted(r["refused"].items(), key=lambda kv: -kv[1]):
        lines.append(f"      {k:<20} {v:,}")
    if not r["dry_run"]:
        lines.append(f"  DB rows reverted: {r['rows_reverted']:,} (db errors {r['db_errors']:,})")
    for s in r["refusal_samples"]:
        lines.append(f"      [{s['reason']}] {s['path']} {s['detail']}")
    return "\n".join(lines)
