"""Admin endpoints — deploy control surface for the /status page.

Exposes git pull / docker build / docker restart against the host's git repo
(mounted at REPO_DIR) and Docker daemon (via /var/run/docker.sock). Reads the
BUILD_SHA file written at image-build time so the UI can tell whether the
running container is on the latest commit.

Trust boundary: these endpoints assume the same operator audience as the rest
of /status — i.e. you're on your own LAN/tailnet. Mounting docker.sock gives
the container effective root on the host; only deploy this on a network you
control.
"""
import asyncio
import json
import logging
import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

# Mount points inside the container.
REPO_DIR = os.environ.get("PHOTOSEARCH_REPO_DIR", "/repo")
COMPOSE_FILE = os.environ.get("PHOTOSEARCH_COMPOSE_FILE", f"{REPO_DIR}/docker-compose.nas.yml")
COMPOSE_SERVICE = os.environ.get("PHOTOSEARCH_COMPOSE_SERVICE", "photosearch")
# The MCP server (M24a) runs as a sibling container that reuses the
# photosearch image, so it never needs its own build — only a recreate after
# the shared image is rebuilt.
MCP_SERVICE = os.environ.get("PHOTOSEARCH_COMPOSE_MCP_SERVICE", "photosearch-mcp")
# Compose project name — must match whatever the running containers were
# labeled with, or compose won't recognize them and will try to create
# fresh ones (hitting the explicit container_name conflicts). The compose
# file pins `name: photosearch` at the top level so this also matches a
# NAS-shell `docker compose` run from /volume1/docker/photosearch.
COMPOSE_PROJECT = os.environ.get("PHOTOSEARCH_COMPOSE_PROJECT", "photosearch")
# Host-side path to the repo. Restart spawns a *separate* helper container
# (running docker:cli) to do the actual `compose up`, so it survives the
# moment compose kills the running photosearch container — otherwise the
# subprocess running compose dies with its parent container right between
# "old container died" and "start new container", leaving the new one
# stranded in Created state. The helper needs its own /repo mount from
# the host, so we need the host path, not the in-container /repo path.
HOST_REPO_DIR = os.environ.get("HOST_REPO_DIR", "/volume1/docker/photosearch")

# Written at image-build time by the Dockerfile from the GIT_SHA build arg.
BUILD_SHA_FILE = Path("/app/BUILD_SHA")

# Serialize destructive ops: only one git-pull or build at a time.
_op_lock = threading.Lock()
# Separate lock for ingest-incoming so a long sweep doesn't block /version
# polling or deploy actions (they don't conflict), but two ingests can't race
# on the same _incoming/ files within this process.
_ingest_lock = threading.Lock()
# Light-index pass (EXIF + file-hash insert, no heavy passes). Its own guard so
# two library scans can't run at once, but it never blocks deploy/ingest.
_light_index_lock = threading.Lock()


def _native_repo_dir() -> str:
    """The project checkout dir — used when NOT running in the NAS container.

    The local replica (M26a) runs `cli.py serve` natively from the cloned
    repo, so REPO_DIR (=/repo, the docker mount) doesn't exist but the
    checkout two levels up from this file is a real git repo.
    """
    return str(Path(__file__).resolve().parent.parent)


def _active_repo_dir() -> Optional[str]:
    """The git repo to operate on: the docker mount if present, else the
    native checkout. None if neither is a git repo."""
    if Path(REPO_DIR, ".git").exists():
        return REPO_DIR
    nd = _native_repo_dir()
    if Path(nd, ".git").exists():
        return nd
    return None


def _deploy_mode() -> str:
    """How this instance is deployed, which decides the restart mechanism:

    - 'docker'  — running in the NAS container off the /repo mount; restart
      swaps the container via a helper container, build = docker compose build.
    - 'native'  — running `cli.py serve` directly off the checkout (the local
      replica); restart re-execs the process, no build step.
    - 'none'    — no git repo reachable at all (deploy controls unavailable).
    """
    if Path(REPO_DIR, ".git").exists():
        return "docker"
    if Path(_native_repo_dir(), ".git").exists():
        return "native"
    return "none"


def _run_git(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a git command in the active repo dir. Returns (rc, stdout, stderr).

    Uses -c safe.directory=* because the mounted repo is owned by the host
    user (UGOS / UID 1000ish), not the container's user, so git's hardened
    ownership check otherwise refuses to touch it. (Harmless for the native
    checkout, which is owned by the same user.)
    """
    cmd = ["git", "-c", "safe.directory=*", "-C", _active_repo_dir() or REPO_DIR] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return 127, "", "git not installed in this image"
    except subprocess.TimeoutExpired:
        return 124, "", f"git timed out after {timeout}s"


def _repo_available() -> bool:
    return _active_repo_dir() is not None


def _deployed_sha() -> Optional[str]:
    """The git SHA the *running* code was started from, or None if unknown.

    In docker mode this is baked into /app/BUILD_SHA at image-build time. In
    native mode there's no image, so we capture HEAD once at process startup
    (`_NATIVE_STARTUP_SHA`); after a `git pull` it stays pinned to the old
    commit until a restart re-execs the process, which is exactly the
    "running code is behind HEAD — restart to apply" signal the UI wants.
    """
    if _deploy_mode() == "native":
        return _NATIVE_STARTUP_SHA
    try:
        return BUILD_SHA_FILE.read_text().strip() or None
    except FileNotFoundError:
        return None
    except OSError:
        return None


# Captured once at import (process startup) so native mode can tell whether the
# running code is behind a subsequently-pulled HEAD. None in docker mode (we use
# the baked BUILD_SHA there) or if HEAD can't be read.
def _capture_startup_sha() -> Optional[str]:
    if _deploy_mode() != "native":
        return None
    rc, sha, _ = _run_git(["rev-parse", "HEAD"])
    return sha if rc == 0 else None


_NATIVE_STARTUP_SHA = _capture_startup_sha()


def _commit_info(ref: str) -> Optional[dict]:
    """Return {sha, sha_short, subject, date} for a git ref, or None."""
    rc, out, _ = _run_git(["log", "-1", "--format=%H%x00%h%x00%s%x00%cI", ref])
    if rc != 0 or not out:
        return None
    parts = out.split("\x00")
    if len(parts) != 4:
        return None
    return {"sha": parts[0], "sha_short": parts[1], "subject": parts[2], "date": parts[3]}


@router.get("/version")
def admin_version():
    """Current git state + deployed-image SHA + drift indicator.

    Does NOT fetch from the remote — call /git-fetch (or /git-pull) first to
    refresh `origin/main`. Cheap to poll; safe.
    """
    if not _repo_available():
        return {
            "available": False,
            "mode": "none",
            "reason": f"git repo not mounted at {REPO_DIR}",
            "deployed_sha": _deployed_sha(),
        }

    head = _commit_info("HEAD")
    upstream = _commit_info("@{upstream}")  # tracking branch (usually origin/main)

    # Working-tree cleanliness — empty porcelain output = clean. Returning
    # the file list (capped) lets the UI show *what* is dirty when the
    # operator wonders why the indicator hasn't gone green yet.
    rc, status_out, _ = _run_git(["status", "--porcelain"])
    dirty = (rc == 0 and bool(status_out))
    dirty_files = []
    if dirty:
        for line in status_out.splitlines()[:30]:
            dirty_files.append(line.rstrip())

    # Ahead / behind vs upstream
    ahead = behind = 0
    if head and upstream:
        rc, counts, _ = _run_git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])
        if rc == 0:
            try:
                a, b = counts.split()
                ahead, behind = int(a), int(b)
            except ValueError:
                pass

    deployed = _deployed_sha()
    deployed_matches_head = bool(deployed and head and deployed == head["sha"])

    return {
        "available": True,
        "mode": _deploy_mode(),
        "head": head,
        "upstream": upstream,
        "dirty": dirty,
        "dirty_files": dirty_files,
        "ahead": ahead,
        "behind": behind,
        "deployed_sha": deployed,
        "deployed_matches_head": deployed_matches_head,
        "compose_file": COMPOSE_FILE,
        "compose_service": COMPOSE_SERVICE,
    }


@router.post("/git-fetch")
def admin_git_fetch():
    """`git fetch origin` — refresh remote refs so /version reflects ahead/behind."""
    if not _repo_available():
        raise HTTPException(404, f"git repo not mounted at {REPO_DIR}")
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")
    try:
        rc, out, err = _run_git(["fetch", "origin"], timeout=60)
        if rc != 0:
            raise HTTPException(500, f"git fetch failed: {err or out}")
        return {"ok": True, "output": (out + "\n" + err).strip()}
    finally:
        _op_lock.release()


@router.post("/git-pull")
def admin_git_pull():
    """`git pull --ff-only origin <current-branch>` — refuse to merge."""
    if not _repo_available():
        raise HTTPException(404, f"git repo not mounted at {REPO_DIR}")
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")
    try:
        # Fast-forward only — if the branch can't fast-forward, fail loudly
        # rather than create a merge commit from inside the container.
        rc, out, err = _run_git(["pull", "--ff-only"], timeout=120)
        if rc != 0:
            combined = (err + "\n" + out).strip()
            raise HTTPException(500, f"git pull failed: {combined}")
        return {"ok": True, "output": (out + "\n" + err).strip()}
    finally:
        _op_lock.release()


async def _stream_subprocess(cmd: list[str], cwd: Optional[str] = None, env: Optional[dict] = None):
    """Run a subprocess and yield SSE events for each output line + a terminal event."""
    yield f"event: start\ndata: {json.dumps({'cmd': ' '.join(shlex.quote(c) for c in cmd)})}\n\n"
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError as e:
        yield f"event: fatal\ndata: {json.dumps({'error': str(e)})}\n\n"
        return
    try:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                text = line.decode("utf-8", errors="replace").rstrip()
            except Exception:
                text = repr(line)
            yield f"event: line\ndata: {json.dumps({'line': text})}\n\n"
        rc = await proc.wait()
        yield f"event: done\ndata: {json.dumps({'returncode': rc})}\n\n"
    except asyncio.CancelledError:
        proc.terminate()
        raise


@router.post("/docker-build")
async def admin_docker_build():
    """`docker compose build photosearch` — SSE stream of build output.

    Passes GIT_SHA via an explicit --build-arg (not just env-var expansion in
    the compose file — that occasionally evaluated to "unknown" even with the
    env set on the subprocess). The new image bakes /app/BUILD_SHA, and
    /version's deployed_matches_head flips true once a restart picks it up.
    """
    if _deploy_mode() == "native":
        raise HTTPException(
            400,
            "no build step in native/local-replica mode — Python changes apply "
            "on Restart (re-exec); the frontend is served straight from disk.",
        )
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")

    rc, sha, _ = _run_git(["rev-parse", "HEAD"])
    git_sha = sha if rc == 0 else "unknown"

    env = os.environ.copy()
    env["GIT_SHA"] = git_sha   # also exported for compose-file fallback expansion
    cmd = [
        "docker", "compose", "-p", COMPOSE_PROJECT, "-f", COMPOSE_FILE, "build",
        "--build-arg", f"GIT_SHA={git_sha}",
        COMPOSE_SERVICE,
    ]

    async def gen():
        try:
            # Emit the SHA up front so the user can see it land in BUILD_SHA.
            yield f"event: line\ndata: {json.dumps({'line': '[admin] passing GIT_SHA=' + git_sha[:12]})}\n\n"
            async for chunk in _stream_subprocess(cmd, cwd=REPO_DIR, env=env):
                yield chunk
        finally:
            _op_lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/ingest-incoming")
async def admin_ingest_incoming(dry_run: bool = False):
    """`docker compose run --rm photosearch ingest-incoming --no-colors` — SSE.

    Sweeps /photos/_incoming/<source>/ into the library (date routing, hash
    dedup, CLIP index) — the same job the daily cron runs. Launched in a
    throwaway sibling container (not in-process) so the heavy CLIP import
    doesn't run inside the web server, and `--no-deps` keeps it from touching
    ollama. Matches the `--no-colors` cadence of the cron (colors are not a
    worker pass; backfill later with `photosearch index <dir>`).

    Pass `?dry_run=true` to scan + report without moving anything.
    """
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "an ingest-incoming sweep is already running")

    cmd = [
        "docker", "compose", "-p", COMPOSE_PROJECT, "-f", COMPOSE_FILE,
        "run", "--rm", "--no-deps", COMPOSE_SERVICE,
        "ingest-incoming", "--no-colors",
    ]
    if dry_run:
        cmd.append("--dry-run")

    async def gen():
        try:
            async for chunk in _stream_subprocess(cmd, cwd=REPO_DIR, env=os.environ.copy()):
                yield chunk
        finally:
            _ingest_lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/light-index")
async def admin_light_index(directory: str = "/photos"):
    """`docker compose run --rm photosearch index <dir> --no-colors` — SSE.

    Light index pass: walk <dir> and insert EXIF + file-hash rows for any new
    photos. No CLIP, no colors, no faces/quality/describe, and --no-geocode (the
    reverse-geocode stage loads the GeoNames dataset and OOM-killed the run on
    the 8 GB N100 — place_names are backfilled by the maintenance sweep's geocode
    stage instead). Just the rows. This is the prerequisite that creates DB
    records so the worker fleet can then claim them for the heavy passes.
    Idempotent: only files not already in the DB are added, so it's safe to
    re-run on the whole library.

    Runs in a throwaway sibling container (not in-process, so the scan doesn't
    run inside the web server); `--no-deps` keeps it off ollama. The `index`
    entrypoint case maps `index <dir> --no-colors` to
    `cli.py index <dir> --db <DB> --no-colors`.

    `directory` must resolve under /photos (the library mount). Defaults to the
    whole library.
    """
    norm = os.path.normpath(directory)
    if norm != "/photos" and not norm.startswith("/photos/"):
        raise HTTPException(400, "directory must be under /photos")
    if not _light_index_lock.acquire(blocking=False):
        raise HTTPException(409, "a light index pass is already running")

    cmd = [
        "docker", "compose", "-p", COMPOSE_PROJECT, "-f", COMPOSE_FILE,
        "run", "--rm", "--no-deps", COMPOSE_SERVICE,
        "index", norm, "--no-colors", "--no-geocode",
    ]

    async def gen():
        try:
            async for chunk in _stream_subprocess(cmd, cwd=REPO_DIR, env=os.environ.copy()):
                yield chunk
        finally:
            _light_index_lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


def _replica_sync_script() -> str:
    """Path to sync-replica.sh — env override or repo-root default."""
    env = os.environ.get("PHOTOSEARCH_REPLICA_SYNC_SCRIPT")
    if env:
        return env
    return str(Path(__file__).resolve().parent.parent / "sync-replica.sh")


@router.get("/replica-status")
def admin_replica_status():
    """Replica freshness for the /status card (M26a).

    Reports the local replica's photo count + last-sync time (DB file mtime),
    and — if PHOTOSEARCH_NAS_URL is set — the source NAS's count so the UI can
    show drift. `replica_mode` is true when this instance proxies images from a
    NAS (i.e. it's running off a replica, not on the NAS itself).
    """
    import time
    nas_url = (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/")
    db_path = os.environ.get("PHOTOSEARCH_DB", "photo_index.db")

    local_count = None
    last_sync = None
    try:
        from .db import PhotoDB
        with PhotoDB(db_path) as db:
            local_count = db.conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"]
        mtime = os.path.getmtime(db_path)
        last_sync = {
            "epoch": int(mtime),
            "age_seconds": int(time.time() - mtime),
        }
    except Exception as e:
        logger.warning("replica-status local read failed: %s", e)

    nas_count = None
    if nas_url:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{nas_url}/api/stats", timeout=8) as r:
                nas_count = json.loads(r.read()).get("photos")
        except Exception as e:
            logger.info("replica-status NAS reach failed: %s", e)

    return {
        "replica_mode": bool(nas_url),
        "nas_url": nas_url or None,
        "db_path": db_path,
        "local_photos": local_count,
        "nas_photos": nas_count,
        "drift": (nas_count - local_count)
                 if (nas_count is not None and local_count is not None) else None,
        "last_sync": last_sync,
        "sync_script": _replica_sync_script(),
    }


@router.post("/replica-sync")
async def admin_replica_sync():
    """`bash sync-replica.sh` — SSE stream of a fresh replica pull (M26a).

    Pulls a consistent DB snapshot from the NAS (dump-db + cat-stream, atomic
    swap) so local search reflects the latest ingest. Requires ssh access to
    the NAS from this machine (replica mode is meant for a native local run).
    Thumbnails are not mirrored — image routes lazily proxy them from the NAS.
    """
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")

    script = _replica_sync_script()
    if not Path(script).exists():
        _op_lock.release()
        raise HTTPException(404, f"sync script not found: {script}")
    cmd = ["bash", script]

    async def gen():
        try:
            async for chunk in _stream_subprocess(cmd, env=os.environ.copy()):
                yield chunk
        finally:
            _op_lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/restart-mcp")
async def admin_restart_mcp():
    """`docker compose up -d --force-recreate --no-deps photosearch-mcp` — SSE.

    Recreates the MCP sibling container. It reuses the photosearch image, so
    it needs no build of its own — but after the shared image is rebuilt (the
    Build button), the MCP container keeps running the *old* image until it's
    recreated. This is that recreate.

    Unlike /restart, no helper container is needed: we're recreating a
    DIFFERENT container from the one serving this request, so the compose
    subprocess isn't in the PID namespace being torn down — a plain streamed
    subprocess completes normally.

    --force-recreate because the image tag (photosearch-photosearch) is
    unchanged after a rebuild, so compose otherwise sees no config change and
    skips the swap. --no-deps so it doesn't drag photosearch/ollama along.
    """
    if _deploy_mode() == "native":
        raise HTTPException(
            400, "no MCP sibling container in native/local-replica mode",
        )
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")

    cmd = [
        "docker", "compose", "-p", COMPOSE_PROJECT, "-f", COMPOSE_FILE,
        "up", "-d", "--force-recreate", "--no-deps", MCP_SERVICE,
    ]

    async def gen():
        try:
            async for chunk in _stream_subprocess(cmd, cwd=REPO_DIR, env=os.environ.copy()):
                yield chunk
        finally:
            _op_lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


def _restart_native():
    """Restart the local-replica process by re-execing it.

    The replica runs `cli.py serve` as a single uvicorn process (no docker, no
    workers), so the clean way to apply pulled Python changes is to replace the
    process image with the same argv+env. We do it from a daemon thread after a
    short delay so the HTTP response flushes first — the client then sees the
    listening socket drop (same UX as the docker swap) and auto-polls /version.
    """
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")
    try:
        import sys
        import time

        argv = [sys.executable] + sys.argv

        def _reexec():
            time.sleep(0.8)  # let the JSON response flush to the client
            logger.info("native restart: re-exec %s", " ".join(argv))
            try:
                os.execv(sys.executable, argv)
            except Exception as e:  # pragma: no cover — exec rarely returns
                logger.error("native restart re-exec failed: %s", e)

        threading.Thread(target=_reexec, daemon=True).start()
        return {
            "ok": True,
            "note": "restarting local replica (process re-exec) — this "
                    "connection will drop for a few seconds; the page auto-polls "
                    "for the new process.",
        }
    finally:
        _op_lock.release()


@router.post("/restart")
def admin_restart():
    """Restart the running app to apply pulled code.

    Native (local-replica) mode re-execs the `cli.py serve` process — see
    `_restart_native`. Docker (NAS) mode does the helper-container swap below.

    Docker path: trigger `docker compose up -d --no-deps photosearch` via a
    *separate* helper container so the compose process survives this container's
    death.

    Why a helper container instead of in-process subprocess: when compose
    SIGKILLs the running photosearch container after stop_grace_period, every
    process inside that container's PID namespace dies — including the
    subprocess that's running compose itself. That happens BETWEEN "old
    container died" and "start new container", leaving the new container
    stranded in Created state. setsid/nohup don't help — they're in the same
    PID namespace. A separate `docker run` container has its own lifecycle.

    Also extends active worker claims by 15 minutes before triggering the
    helper, so workers' in-flight batches survive the swap gap even if
    `renew-claim` calls 503/timeout during the brief downtime.

    --no-deps is required: without it, compose follows depends_on and tries to
    ensure ollama (and other deps) are also up. If those containers were
    created with different project labels, compose decides it needs to create
    them and hits a name conflict on `photosearch-ollama` etc.

    Returns immediately after launching the helper. The container then dies
    as docker swaps it for the fresh one, so the client sees a connection
    drop — that's expected.
    """
    if _deploy_mode() == "native":
        return _restart_native()
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")
    try:
        # Extend active claims so workers' in-flight batches survive the
        # restart gap. Best-effort: if the DB import or write fails we still
        # proceed with the restart — the worker would just lose some claims.
        extended = 0
        try:
            from .db import PhotoDB
            db_path = os.environ.get("PHOTOSEARCH_DB", "/data/photo_index.db")
            with PhotoDB(db_path) as db:
                cur = db.conn.execute(
                    "UPDATE worker_claims "
                    "SET expires_at = datetime('now', '+15 minutes') "
                    "WHERE expires_at > datetime('now')"
                )
                extended = cur.rowcount
                db.conn.commit()
        except Exception as e:
            logger.warning("claim TTL extension before restart failed: %s", e)

        # Launch helper: `docker run --rm -d docker:cli compose up -d ...`.
        # --detach so this `docker run` returns as soon as the helper is
        #   started (we don't want to wait for the swap to complete, and
        #   we can't anyway — we'd get killed mid-wait).
        # --rm so the helper container removes itself when compose finishes.
        # We mount docker.sock + the host repo, then run compose inside it.
        helper_cmd = [
            "docker", "run", "--rm", "--detach",
            "--name", "photosearch-restart-helper",
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "-v", f"{HOST_REPO_DIR}:/repo",
            "-w", "/repo",
            "docker:cli",
            "compose", "-p", COMPOSE_PROJECT, "-f", "docker-compose.nas.yml",
            "up", "-d", "--no-deps", COMPOSE_SERVICE,
        ]
        try:
            r = subprocess.run(helper_cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            raise HTTPException(500, "docker CLI not installed in this image")
        if r.returncode != 0:
            err = r.stderr.strip() or r.stdout.strip()
            # If a previous helper got wedged (name conflict), surface a
            # clear hint about it — clean up and retry are both manual.
            if "is already in use by container" in err:
                err += " — run `docker rm -f photosearch-restart-helper` and retry"
            raise HTTPException(500, f"restart helper launch failed: {err}")

        helper_id = r.stdout.strip()[:12]
        note = (f"restart helper {helper_id} launched in detached container; "
                "expect a connection drop while the swap completes")
        if extended:
            note += f" — extended TTL on {extended} active claim(s) to ride out the gap"
        return {"ok": True, "note": note, "helper_container": helper_id}
    finally:
        _op_lock.release()


# ---------------------------------------------------------------------------
# Re-run index passes (M28) — image-view "re-run this pass" + bulk re-queue.
# Compute on the desktop (LM Studio / GPU), write authoritative to the NAS,
# mirror the touched rows into the local replica DB. See photosearch/rerun.py.
# ---------------------------------------------------------------------------

from pydantic import BaseModel  # noqa: E402


class RerunRequest(BaseModel):
    photo_ids: list[int]
    passes: list[str]
    mode: str = "sync"  # "sync" = compute now in-process; "queue" = re-queue for the fleet


class MirrorRequest(BaseModel):
    photo_ids: list[int]


@router.post("/rerun-passes")
def admin_rerun_passes(req: RerunRequest):
    """Re-run index passes on specific photos.

    mode='sync'  — compute each (photo, pass) here via LM Studio / local models,
                   submit to the NAS, and mirror the result into the local DB.
                   Instant feedback; best for one photo from the image view.
    mode='queue' — clear the (photo, pass) state on the NAS so the worker fleet
                   re-processes it (poll /rerun-passes mirror or /mirror-photos
                   to pull results once a worker finishes).
    """
    from . import rerun
    if not req.photo_ids:
        raise HTTPException(400, "photo_ids is required (explicit scoping)")
    bad = [p for p in req.passes if p not in rerun.ALL_PASSES]
    if bad:
        raise HTTPException(400, f"unknown pass type(s): {', '.join(bad)}")
    if not req.passes:
        raise HTTPException(400, "passes is required")

    if req.mode == "queue":
        try:
            return {"mode": "queue", "result": rerun.requeue_passes(req.photo_ids, req.passes)}
        except Exception as e:
            raise HTTPException(500, f"re-queue failed: {e}")

    if req.mode != "sync":
        raise HTTPException(400, "mode must be 'sync' or 'queue'")

    # sync — compute each pass for each photo in dependency order (describe
    # before the text passes that read its output).
    order = [p for p in rerun.ALL_PASSES if p in req.passes]
    from . import web
    results, errors = [], []
    with web._get_db() as db:
        for pid in req.photo_ids:
            for pass_type in order:
                try:
                    results.append(rerun.run_pass_sync(db, pid, pass_type))
                except Exception as e:
                    errors.append({"photo_id": pid, "pass": pass_type, "error": str(e)})
    return {"mode": "sync", "results": results, "errors": errors}


@router.post("/mirror-photos")
def admin_mirror_photos(req: MirrorRequest):
    """Pull authoritative fields for the given photos from the NAS into the
    local replica DB (used to refresh after a queued re-run completes)."""
    from . import rerun, web
    if not req.photo_ids:
        raise HTTPException(400, "photo_ids is required")
    with web._get_db() as db:
        return rerun.mirror_photos(db, req.photo_ids)


# ---------------------------------------------------------------------------
# Worker fleet control (M28) — launch/stop/inspect the native worker fleet from
# /status. Workers point at the authoritative server (NAS in replica mode) and
# compute locally; LM Studio role models are passed through from this process's
# env (with sensible defaults when an OpenAI-compatible backend is configured).
# ---------------------------------------------------------------------------

# Fleet name so /status-launched workers are managed as their own group, leaving
# any hand-launched fleet untouched.
_UI_FLEET_NAME = "ui"


class WorkersStartRequest(BaseModel):
    passes: list[str]
    count: int = 2


def _run_workers_script() -> str:
    return str(Path(_native_repo_dir()) / "run-workers.sh")


def _fleet_server_url() -> str:
    """Authoritative server the fleet submits to — the NAS in replica mode,
    else this host's own server."""
    nas = (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/")
    return nas or "http://localhost:8000"


def _fleet_env() -> dict:
    """Env for run-workers.sh — inherits ours, filling LM Studio role models
    with defaults when PHOTOSEARCH_TEXT_LLM_URL is set so the LLM passes route
    to LM Studio out of the box."""
    env = os.environ.copy()
    if env.get("PHOTOSEARCH_TEXT_LLM_URL"):
        visual = env.get("PHOTOSEARCH_LLM_VISUAL_MODEL") or "qwen2.5-vl-7b-instruct"
        env.setdefault("PHOTOSEARCH_LLM_VISUAL_MODEL", visual)
        env.setdefault("PHOTOSEARCH_LLM_DESCRIBE_MODEL", visual)
        env.setdefault("PHOTOSEARCH_LLM_VERIFY_MODEL", visual)
        env.setdefault("PHOTOSEARCH_LLM_TEXT_MODEL", "llama-3.2-3b-instruct")
    return env


@router.post("/workers/start")
def admin_workers_start(req: WorkersStartRequest):
    """Launch a native worker fleet for the given passes (run-workers.sh)."""
    from . import rerun
    bad = [p for p in req.passes if p not in rerun.ALL_PASSES]
    if bad:
        raise HTTPException(400, f"unknown pass type(s): {', '.join(bad)}")
    if not req.passes:
        raise HTTPException(400, "passes is required")
    n = max(1, min(int(req.count), 8))
    script = _run_workers_script()
    if not Path(script).exists():
        raise HTTPException(404, f"run-workers.sh not found: {script}")
    cmd = ["bash", script, "--native", "--name", _UI_FLEET_NAME,
           "-s", _fleet_server_url(), "-p", ",".join(req.passes), "-n", str(n)]
    try:
        r = subprocess.run(cmd, cwd=_native_repo_dir(), env=_fleet_env(),
                           capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "worker launch timed out (still starting?)")
    if r.returncode != 0:
        raise HTTPException(500, f"worker launch failed: {(r.stderr or r.stdout)[-500:]}")
    return {"ok": True, "count": n, "passes": req.passes,
            "server": _fleet_server_url(), "output": (r.stdout or "")[-2000:]}


@router.post("/workers/stop")
def admin_workers_stop():
    """Stop the /status-launched worker fleet."""
    script = _run_workers_script()
    if not Path(script).exists():
        raise HTTPException(404, f"run-workers.sh not found: {script}")
    cmd = ["bash", script, "--native", "--name", _UI_FLEET_NAME, "--stop"]
    r = subprocess.run(cmd, cwd=_native_repo_dir(), env=os.environ.copy(),
                       capture_output=True, text=True, timeout=60)
    return {"ok": r.returncode == 0, "output": ((r.stdout or "") + (r.stderr or ""))[-2000:]}


@router.get("/workers/queue-status")
def admin_workers_queue_status():
    """Worker queue depth + active claims for the **authoritative** server.

    The Workers panel must show the queue the fleet actually pulls from. In
    replica mode that's the NAS (PHOTOSEARCH_NAS_URL) — the local replica DB is
    a fully-processed synced snapshot, so its own /api/worker/status reports ~0
    backlog even while the NAS has thousands of freshly-ingested photos awaiting
    clip/faces/etc. When no NAS is configured this host *is* authoritative, so
    fall through to the local queue.
    """
    nas = (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/")
    if nas:
        import requests
        try:
            # /api/worker/status runs per-pass count_unprocessed scans on the
            # NAS; on the N100 those spike past 8s under worker-fleet write load
            # (claim-batch/submit), surfacing a scary "could not reach
            # authoritative server" on the maintenance page for a box that's
            # actually up. This is a 5s background poll, so a generous read
            # timeout is harmless — prefer stale-but-shown over a false error.
            r = requests.get(f"{nas}/api/worker/status", timeout=30)
            r.raise_for_status()
            data = r.json()
            data["source"] = nas
            return data
        except requests.RequestException as exc:
            # Surface the failure rather than silently showing local zeros.
            return {"active_claims": [], "queue_depth": {}, "source": nas,
                    "error": f"could not reach authoritative server: {exc}"}
    from .worker_api import worker_status
    data = worker_status()
    data["source"] = "local"
    return data


@router.get("/workers/fleet-status")
def admin_workers_fleet_status():
    """Process-level status of the /status-launched fleet (run-workers.sh --status).

    Distinct from /api/worker/status, which reports queue depth + active claims.
    """
    script = _run_workers_script()
    if not Path(script).exists():
        return {"available": False, "output": "run-workers.sh not found"}
    cmd = ["bash", script, "--native", "--name", _UI_FLEET_NAME, "--status"]
    try:
        r = subprocess.run(cmd, cwd=_native_repo_dir(), env=os.environ.copy(),
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"available": True, "output": "(status timed out)"}
    return {"available": True, "server": _fleet_server_url(),
            "output": ((r.stdout or "") + (r.stderr or ""))[-4000:]}
