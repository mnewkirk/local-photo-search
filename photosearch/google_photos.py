"""Google Photos Library API integration.

Provides OAuth2 authentication + photo upload + album management.
Photos are uploaded without modifying originals — metadata is passed
via the API description field and Google Photos' native caption support.

Setup summary:
  1. Create a Google Cloud project, enable Photos Library API.
  2. Create OAuth 2.0 credentials (Web Application type).
  3. Add your server URL as an authorized redirect URI, e.g.:
       http://<nas-ip>:8000/api/google/callback
  4. Download the credentials JSON file (client_secret_*.json) and place
     it alongside the database file, or set GOOGLE_CLIENT_SECRET_FILE.

See GOOGLE_PHOTOS_SETUP.md for full step-by-step instructions.

API notes (as of 2025-2026):
  - Scope required: photoslibrary.appendonly
    (read/sharing scopes deprecated March 31, 2025)
  - Upload is a 2-step process: POST bytes → batchCreate media items
  - Description field (max 1000 chars) can be set on each media item
  - Albums can be created and photos added in one batchCreate call
  - No access to user's existing library — upload-only
"""

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Google OAuth2 / API endpoints
# ---------------------------------------------------------------------------
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# Photos Library API
PHOTOS_UPLOAD_URL = "https://photoslibrary.googleapis.com/v1/uploads"
PHOTOS_BATCH_CREATE_URL = "https://photoslibrary.googleapis.com/v1/mediaItems:batchCreate"
PHOTOS_ALBUMS_URL = "https://photoslibrary.googleapis.com/v1/albums"

# The only scope still supported for upload + album creation
SCOPES = ["https://www.googleapis.com/auth/photoslibrary.appendonly"]

# Default OAuth2 redirect URI — must match what's configured in Google Cloud Console.
# Overridable via GOOGLE_REDIRECT_URI env var or the server's detected host.
DEFAULT_REDIRECT_URI = "http://localhost:8000/api/google/callback"

# File names (stored alongside the SQLite DB)
_TOKEN_FILE = "google_photos_token.json"
_CLIENT_SECRET_FILE = "google_client_secret.json"

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _token_path(db_path: str) -> Path:
    return Path(db_path).parent / _TOKEN_FILE


def _client_secret_path(db_path: str) -> Path:
    env = os.environ.get("GOOGLE_CLIENT_SECRET_FILE")
    if env:
        return Path(env)
    return Path(db_path).parent / _CLIENT_SECRET_FILE


def load_client_secret(db_path: str) -> Optional[dict]:
    """Load and parse the Google OAuth2 client secret JSON.

    Supports both 'web' and 'installed' application types.
    Returns a flat dict with client_id, client_secret, redirect_uris, etc.
    Returns None if the file is missing or malformed.
    """
    path = _client_secret_path(db_path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        # Google's format nests credentials under 'web' or 'installed'
        if "web" in data:
            return data["web"]
        elif "installed" in data:
            return data["installed"]
        return data  # flat format (unusual)
    except Exception:
        return None


def load_token(db_path: str) -> Optional[dict]:
    """Load stored OAuth2 tokens from disk."""
    path = _token_path(db_path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_token(db_path: str, token: dict) -> None:
    """Persist OAuth2 tokens to disk (alongside the DB)."""
    path = _token_path(db_path)
    with open(path, "w") as f:
        json.dump(token, f, indent=2)


def clear_token(db_path: str) -> None:
    """Delete stored tokens (used for disconnect)."""
    path = _token_path(db_path)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# OAuth2 flow
# ---------------------------------------------------------------------------

def is_configured(db_path: str) -> bool:
    """Return True if a client secret file exists (setup has been done)."""
    return _client_secret_path(db_path).exists()


def is_authenticated(db_path: str) -> bool:
    """Return True if valid tokens are stored."""
    token = load_token(db_path)
    return token is not None and "access_token" in token


def get_authorization_url(db_path: str, redirect_uri: Optional[str] = None) -> str:
    """Return the Google OAuth2 authorization URL to send the user to.

    Args:
        db_path: Path to the database (used to find client_secret.json).
        redirect_uri: OAuth redirect URI. Defaults to DEFAULT_REDIRECT_URI or
                      GOOGLE_REDIRECT_URI env var.

    Returns:
        Full authorization URL string.
    """
    secret = load_client_secret(db_path)
    if not secret:
        raise RuntimeError(
            "Google client secret not found. "
            f"Place client_secret.json alongside the database, or set GOOGLE_CLIENT_SECRET_FILE. "
            "See GOOGLE_PHOTOS_SETUP.md for instructions."
        )

    client_id = secret["client_id"]
    redirect = redirect_uri or os.environ.get("GOOGLE_REDIRECT_URI", DEFAULT_REDIRECT_URI)

    params = {
        "client_id": client_id,
        "redirect_uri": redirect,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",   # request refresh token
        "prompt": "consent",         # always show consent screen to get refresh_token
    }
    # URL-encode each parameter value
    query = "&".join(
        f"{k}={requests.utils.quote(str(v), safe='')}"
        for k, v in params.items()
    )
    return f"{GOOGLE_AUTH_URL}?{query}"


def exchange_code(db_path: str, code: str, redirect_uri: Optional[str] = None) -> dict:
    """Exchange an authorization code for access + refresh tokens.

    Called from the OAuth callback endpoint. Saves the tokens to disk.

    Returns the token dict (includes access_token, refresh_token, expires_in).
    Raises requests.HTTPError on failure.
    """
    secret = load_client_secret(db_path)
    if not secret:
        raise RuntimeError("Google client secret not configured")

    redirect = redirect_uri or os.environ.get("GOOGLE_REDIRECT_URI", DEFAULT_REDIRECT_URI)

    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": secret["client_id"],
            "client_secret": secret["client_secret"],
            "redirect_uri": redirect,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json()
    token["obtained_at"] = time.time()
    save_token(db_path, token)
    return token


def refresh_access_token(db_path: str) -> Optional[str]:
    """Return a valid access token, refreshing if expired.

    Returns None if no token is stored.
    Raises RuntimeError if refresh fails or no refresh_token is available.
    """
    token = load_token(db_path)
    if not token:
        return None

    access_token = token.get("access_token")
    expires_in = token.get("expires_in", 3600)
    obtained_at = token.get("obtained_at", 0)

    # Refresh 5 minutes before expiry to avoid mid-upload failures
    needs_refresh = time.time() > obtained_at + expires_in - 300

    if needs_refresh:
        refresh_tok = token.get("refresh_token")
        if not refresh_tok:
            raise RuntimeError(
                "Access token expired and no refresh token available — please re-authorize."
            )

        secret = load_client_secret(db_path)
        if not secret:
            raise RuntimeError("Google client secret not configured")

        resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "refresh_token": refresh_tok,
                "client_id": secret["client_id"],
                "client_secret": secret["client_secret"],
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        resp.raise_for_status()
        new_token = resp.json()
        new_token["refresh_token"] = refresh_tok  # Google omits it on refresh
        new_token["obtained_at"] = time.time()
        save_token(db_path, new_token)
        access_token = new_token["access_token"]

    return access_token


def revoke_token(db_path: str) -> None:
    """Revoke the stored access/refresh token and delete it from disk."""
    token = load_token(db_path)
    if token:
        # Prefer to revoke the refresh token (longer-lived)
        tok = token.get("refresh_token") or token.get("access_token")
        if tok:
            try:
                requests.post(
                    GOOGLE_REVOKE_URL,
                    params={"token": tok},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=10,
                )
            except Exception:
                pass  # Best-effort revocation
    clear_token(db_path)


# ---------------------------------------------------------------------------
# Album management
# ---------------------------------------------------------------------------

def create_album(db_path: str, title: str) -> str:
    """Create a Google Photos album and return its ID.

    Args:
        db_path: DB path (for token lookup).
        title: Album title (truncated to 500 chars per API limit).

    Returns:
        The new album's ID string.
    """
    access_token = refresh_access_token(db_path)
    if not access_token:
        raise RuntimeError("Not authenticated with Google Photos — please authorize first.")

    resp = requests.post(
        PHOTOS_ALBUMS_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={"album": {"title": title[:500]}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def batch_add_to_album(db_path: str, album_id: str, media_item_ids: list[str]) -> dict:
    """Add existing media items (already in the user's library) to an album.

    This is a lightweight metadata-only call — no bytes are transferred.
    The API silently ignores items that are already in the album, so this
    is safe to call idempotently (e.g. to re-add photos that were removed
    from the album without knowing which specific ones are missing).

    Args:
        db_path: DB path (for token lookup).
        album_id: ID of the target album.
        media_item_ids: List of media item IDs to add (max 50 per call).

    Returns:
        Dict with keys:
            added   — number of items successfully added (or already present)
            errors  — list of (media_item_id, error_message) tuples
    """
    access_token = refresh_access_token(db_path)
    if not access_token:
        raise RuntimeError("Not authenticated with Google Photos — please authorize first.")

    BATCH = 50  # API limit per batchAddMediaItems call
    added = 0
    errors = []

    for i in range(0, len(media_item_ids), BATCH):
        chunk = media_item_ids[i:i + BATCH]
        resp = requests.post(
            f"{PHOTOS_ALBUMS_URL}/{album_id}:batchAddMediaItems",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"mediaItemIds": chunk},
            timeout=30,
        )
        if resp.status_code == 200:
            # Log the response body so we can diagnose silent failures
            # (e.g. batchAddMediaItems returns 200 but doesn't re-add
            # photos that were manually removed from the album via the
            # Google Photos UI).
            try:
                body = resp.json()
                if body:
                    log.warning("batchAddMediaItems 200 response body: %s", body)
                else:
                    log.info("batchAddMediaItems 200 — empty body (normal success)")
            except Exception:
                log.info("batchAddMediaItems 200 — no JSON body")
            added += len(chunk)
        else:
            try:
                detail = resp.json()
            except Exception:
                detail = {"message": resp.text}
            errors.append((chunk, detail.get("message") or str(resp.status_code)))

    return {"added": added, "errors": errors}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _upload_raw_bytes(access_token: str, file_path: str) -> str:
    """Step 1 of the 2-step Photos upload: POST raw bytes, get uploadToken.

    Uploads the file as-is (original JPEG). Large files work fine (API
    limit is effectively ~50 MB for reliable throughput).

    Returns:
        uploadToken string (valid for 24 hours).
    """
    filename = Path(file_path).name
    with open(file_path, "rb") as f:
        data = f.read()

    resp = requests.post(
        PHOTOS_UPLOAD_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-type": "application/octet-stream",
            "X-Goog-Upload-Content-Type": "image/jpeg",
            "X-Goog-Upload-File-Name": filename,
            "X-Goog-Upload-Protocol": "raw",
        },
        data=data,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.text.strip()


def upload_photos(
    db_path: str,
    photo_records: list[dict],
    album_id: Optional[str] = None,
    include_description: bool = True,
    progress_callback=None,
    begin_callback=None,
    bytes_done_callback=None,
) -> list[dict]:
    """Upload photos to Google Photos.

    This is the main upload function. It uploads each photo's original file
    to Google Photos. If the photo has a description in the database, it is
    set as the caption (unless include_description=False).

    The upload is done in batches of 50 (the API limit per batchCreate call).

    Args:
        db_path: Path to the SQLite database.
        photo_records: List of photo dicts. Each must have:
            - '_resolved_filepath' or 'filepath': path to the image file
            - 'filename': original filename
            - 'description' (optional): text description/caption
        album_id: If provided, all uploaded photos are added to this album.
        include_description: Whether to use the photo's description as caption.
        progress_callback: Optional callable(done, total, filename, status, error, media_item_id)
            called after each photo completes (success or error).
        begin_callback: Optional callable(queued, total, filename)
            called just before raw bytes are sent for a file.
        bytes_done_callback: Optional callable(queued, total, filename)
            called right after raw bytes finish uploading (before batchCreate).

    Returns:
        List of result dicts, one per photo:
            {'filename': str, 'status': 'uploaded'|'error', 'error': str|None,
             'media_item_id': str|None}
    """
    access_token = refresh_access_token(db_path)
    if not access_token:
        raise RuntimeError("Not authenticated with Google Photos — please authorize first.")

    results: list[dict] = []
    total = len(photo_records)
    done = 0

    # batchCreate supports up to 50 items per call.
    BATCH_SIZE = 50

    for batch_start in range(0, total, BATCH_SIZE):
        batch = photo_records[batch_start: batch_start + BATCH_SIZE]

        # ---- Step 1: Upload raw bytes for each photo ----
        # We upload sequentially (Photos API doesn't support parallel uploads
        # per token, and doing so risks 429s).
        upload_pairs: list[tuple[dict, str]] = []  # (photo_record, upload_token)

        for photo in batch:
            filepath = photo.get("_resolved_filepath") or photo.get("filepath", "")
            filename = photo.get("filename", Path(filepath).name)
            try:
                # Refresh token before each file in case it expired mid-upload
                access_token = refresh_access_token(db_path) or access_token
                if begin_callback:
                    begin_callback(batch_start + len(upload_pairs), total, filename)
                token = _upload_raw_bytes(access_token, filepath)
                upload_pairs.append((photo, token))
                if bytes_done_callback:
                    bytes_done_callback(batch_start + len(upload_pairs), total, filename)
            except InterruptedError:
                raise  # cancellation — don't swallow, let caller handle it
            except Exception as exc:
                done += 1
                results.append({
                    "filename": filename,
                    "status": "error",
                    "error": str(exc),
                    "media_item_id": None,
                })
                if progress_callback:
                    progress_callback(done, total, filename, "error", str(exc), None)

        if not upload_pairs:
            continue

        # ---- Step 2: batchCreate — create media items from upload tokens ----
        new_media_items = []
        for photo, token in upload_pairs:
            item: dict = {
                "simpleMediaItem": {
                    "uploadToken": token,
                    "fileName": photo.get("filename", ""),
                }
            }
            if include_description and photo.get("description"):
                # API max is 1000 chars; our descriptions are typically 100-200
                item["description"] = photo["description"][:1000]
            new_media_items.append(item)

        body: dict = {"newMediaItems": new_media_items}
        if album_id:
            body["albumId"] = album_id

        try:
            resp = requests.post(
                PHOTOS_BATCH_CREATE_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            api_results = resp.json().get("newMediaItemResults", [])

            for i, api_result in enumerate(api_results):
                photo = upload_pairs[i][0]
                filename = photo.get("filename", "")
                status_obj = api_result.get("status", {})
                media_item = api_result.get("mediaItem")
                done += 1
                # Google signals success either via status.message == "OK"
                # or by returning a populated mediaItem
                if media_item or (status_obj.get("message", "").upper() == "OK"):
                    result = {
                        "filename": filename,
                        "status": "uploaded",
                        "error": None,
                        "media_item_id": media_item.get("id") if media_item else None,
                    }
                    results.append(result)
                    if progress_callback:
                        progress_callback(done, total, filename, "uploaded", None,
                                          media_item.get("id") if media_item else None)
                else:
                    err_msg = status_obj.get("message") or "Upload failed"
                    result = {
                        "filename": filename,
                        "status": "error",
                        "error": err_msg,
                        "media_item_id": None,
                    }
                    results.append(result)
                    if progress_callback:
                        progress_callback(done, total, filename, "error", err_msg, None)

        except InterruptedError:
            raise  # cancellation — propagate immediately
        except Exception as exc:
            # batchCreate failed — mark all items in this batch as errors
            for photo, _ in upload_pairs:
                done += 1
                results.append({
                    "filename": photo.get("filename", ""),
                    "status": "error",
                    "error": str(exc),
                    "media_item_id": None,
                })
                if progress_callback:
                    try:
                        progress_callback(done, total, photo.get("filename", ""), "error", str(exc), None)
                    except InterruptedError:
                        raise

    return results
