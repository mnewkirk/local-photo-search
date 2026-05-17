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

# Written at image-build time by the Dockerfile from the GIT_SHA build arg.
BUILD_SHA_FILE = Path("/app/BUILD_SHA")

# Serialize destructive ops: only one git-pull or build at a time.
_op_lock = threading.Lock()


def _run_git(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a git command in REPO_DIR. Returns (rc, stdout, stderr).

    Uses -c safe.directory=* because the mounted repo is owned by the host
    user (UGOS / UID 1000ish), not the container's user, so git's hardened
    ownership check otherwise refuses to touch it.
    """
    cmd = ["git", "-c", "safe.directory=*", "-C", REPO_DIR] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return 127, "", "git not installed in this image"
    except subprocess.TimeoutExpired:
        return 124, "", f"git timed out after {timeout}s"


def _repo_available() -> bool:
    return Path(REPO_DIR, ".git").exists()


def _deployed_sha() -> Optional[str]:
    """The git SHA the running container was built from, or None if unknown."""
    try:
        return BUILD_SHA_FILE.read_text().strip() or None
    except FileNotFoundError:
        return None
    except OSError:
        return None


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
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")

    rc, sha, _ = _run_git(["rev-parse", "HEAD"])
    git_sha = sha if rc == 0 else "unknown"

    env = os.environ.copy()
    env["GIT_SHA"] = git_sha   # also exported for compose-file fallback expansion
    cmd = [
        "docker", "compose", "-f", COMPOSE_FILE, "build",
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


@router.post("/restart")
def admin_restart():
    """`docker compose up -d photosearch` — recreates the service.

    Returns immediately. The container then dies as docker swaps it for the
    fresh one, so the client will see a connection drop — that's normal.
    """
    if not _op_lock.acquire(blocking=False):
        raise HTTPException(409, "another admin operation is in progress")
    try:
        cmd = ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d", COMPOSE_SERVICE]
        try:
            r = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            raise HTTPException(500, "docker CLI not installed in this image")
        if r.returncode != 0:
            raise HTTPException(500, f"docker compose up failed: {r.stderr.strip() or r.stdout.strip()}")
        return {"ok": True, "note": "container will be replaced; expect a connection drop"}
    finally:
        _op_lock.release()
