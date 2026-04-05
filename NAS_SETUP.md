# Deploying to a UGREEN NAS

This guide walks through deploying local-photo-search on a UGREEN DXP4800 (Intel N100, 8 GB RAM) with remote access via Tailscale.

## Overview

The deployment runs three Docker containers:

- **photosearch** — the web UI and indexing engine
- **ollama** — serves the LLaVA vision model for generating photo descriptions
- **tailscale** — provides encrypted remote access from anywhere, without exposing ports to the internet

The UGREEN cloud link (`ug.link`) only serves the UGOS dashboard and can't route to custom containers, so Tailscale gives you a private URL that works from any device on your Tailscale network.


## Prerequisites

1. **SSH access to your NAS.** Enable SSH in UGOS under Settings > General > SSH. You'll run docker commands over SSH.

2. **Docker installed.** UGOS ships with Docker support. Verify with `docker --version` over SSH.

3. **A Tailscale account.** Sign up at [tailscale.com](https://tailscale.com) (free for personal use, up to 100 devices). Install Tailscale on your Mac/phone/other devices you'll access photos from.

4. **Photos on the NAS.** Copy your photo library to a shared folder. This guide assumes `/volume1/Photos` — adjust the path to match your setup.


## Step 1: Copy the project to the NAS

SSH into the NAS and clone the repo directly from GitHub. The UGOS restricted shell blocks `rsync` and `scp`, so cloning via Docker is the reliable approach:

```bash
ssh yournas
mkdir -p /volume1/docker/photosearch

# Clone using a temporary Alpine container (avoids needing git on the NAS)
docker run --rm -v /volume1/docker/photosearch:/repo alpine sh -c \
  "apk add -q git && git clone https://github.com/mattnewkirk/local-photo-search /repo"
```

To update later:

```bash
docker run --rm -v /volume1/docker/photosearch:/repo alpine sh -c \
  "apk add -q git && git config --global --add safe.directory /repo && git -C /repo pull"
```


## Step 2: Configure environment

SSH into the NAS and create your `.env` file:

```bash
ssh yournas
cd /volume1/docker/photosearch

cp .env.example .env
nano .env    # or vi
```

Set these values:

```env
# Path to your photos on the NAS
PHOTO_DIR=/volume1/Photos

# Tailscale hostname (this becomes http://photosearch:8000 on your tailnet)
TS_HOSTNAME=photosearch

# Optional: Tailscale auth key for headless setup
# Generate at https://login.tailscale.com/admin/settings/keys
# TS_AUTHKEY=tskey-auth-xxxxx

# Optional: HuggingFace token — avoids rate limits when downloading models
# Get one at https://huggingface.co/settings/tokens
# HF_TOKEN=hf_xxxxx
```

### About the Tailscale auth key

Without `TS_AUTHKEY`, you'll authenticate interactively on first start (the container logs will show a URL to visit). With an auth key, the container joins your tailnet automatically — useful if you want fully unattended startup after a NAS reboot.

To generate one: Tailscale admin console > Settings > Keys > Generate auth key. Check "Reusable" if you want it to survive container recreation.


## Step 3: Build and start

```bash
cd /volume1/docker/photosearch

# Build the photosearch image (takes a few minutes on the N100)
docker compose -f docker-compose.nas.yml build

# Start everything
docker compose -f docker-compose.nas.yml up -d
```

### Authenticate Tailscale (first time only)

If you didn't set `TS_AUTHKEY`:

```bash
docker logs photosearch-tailscale
```

Look for a line like:

```
To authenticate, visit: https://login.tailscale.com/a/abc123xyz
```

Open that URL in your browser and approve the device. Once connected, the container appears as `photosearch` in your Tailscale network.

While on the LAN, you can also access the web UI directly at `http://<nas-ip>:8000`.


## Step 4: Pull Ollama models

```bash
# The 7B model (~4 GB) — recommended for the N100's 8 GB RAM
docker compose -f docker-compose.nas.yml exec ollama ollama pull llava

# Verification model (~5 GB) — optional, enables hallucination detection
docker compose -f docker-compose.nas.yml exec ollama ollama pull minicpm-v
```

**RAM note:** The N100 has 8 GB RAM. `llava` (7B) uses about 4-5 GB during inference. `llava:13b` needs ~10 GB and will be too slow or fail on this hardware. Stick with the 7B model — descriptions are slightly less accurate but perfectly usable.

You can skip `minicpm-v` initially and add it later. Without it, the `verify` command won't work, but search and descriptions will be fine.


## Step 5: Index your photos

Indexing runs in passes — each pass adds a layer. Run them in this order so search works as quickly as possible.

### Pass 1: CLIP embeddings (required for search)

CLIP is the foundation. Nothing else works until photos have embeddings.

```bash
# Index one year at a time — easiest to monitor and restart if needed
nohup bash -c '
for year in 2026 2025 2024 2023; do
  echo "=== $year ==="
  docker compose -f /volume1/docker/photosearch/docker-compose.nas.yml run --rm \
    -e PYTHONUNBUFFERED=1 photosearch index /photos/$year --clip --no-colors
  echo "=== Done $year ==="
done
' > /tmp/clip.log 2>&1 &
echo "PID: $!"
```

Monitor progress:
```bash
tail -f /tmp/clip.log
```

**Timing on N100:** ~2 seconds per photo. Expect roughly 6–8 hours per 10,000 photos.

**Note:** Use `--no-colors` during the initial CLIP pass. Color extraction runs concurrently with the web server and causes SQLite lock errors on slower hardware. Run a separate color pass later.

### Pass 2: Face detection + quality scoring

Once CLIP is done for a year (or a folder), run faces and quality together:

```bash
nohup docker compose -f /volume1/docker/photosearch/docker-compose.nas.yml run --rm \
  -e PYTHONUNBUFFERED=1 photosearch index /photos/2026 --faces --quality --no-colors \
  > /tmp/faces2026.log 2>&1 &
```

**Timing on N100:**
- Face detection: ~3–5 seconds per photo
- Quality scoring: ~3–5 seconds per photo (runs in parallel)

### Pass 3: Face matching

After face detection completes, match detected faces to your registered reference persons:

```bash
docker compose -f docker-compose.nas.yml run --rm photosearch match-faces --temporal
```

This is fast (seconds to minutes). `--temporal` enables a second pass that uses timestamps to match faces that were too small or angled for direct matching.

### Pass 4: Photo stacking

Stacking detects burst/bracket groups using CLIP embeddings. Run per-folder or across the whole library:

```bash
# Stack a specific year
docker compose -f docker-compose.nas.yml run --rm photosearch \
    stack --directory /photos/2026

# Stack everything
docker compose -f docker-compose.nas.yml run --rm photosearch stack
```

This is fast — a few minutes for tens of thousands of photos.

### Pass 5: Descriptions (optional, very slow)

LLaVA descriptions are optional — search works without them, but they improve results for complex queries.

```bash
nohup docker compose -f /volume1/docker/photosearch/docker-compose.nas.yml run --rm \
  -e PYTHONUNBUFFERED=1 photosearch index /photos/2026 --describe \
  > /tmp/describe2026.log 2>&1 &
```

**Timing on N100:** ~60–90 seconds per photo with the 7B model. 10,000 photos ≈ 7–10 days.

### Monitoring progress

The web UI has a Status page at `http://photosearch:8000/status` showing CLIP coverage, face count, descriptions, and quality scores — with a Refresh button. No need to run CLI stats manually.

```bash
# Or via CLI
docker compose -f docker-compose.nas.yml run --rm photosearch stats
```


## Step 6: Register reference faces (optional)

To enable person search, add reference photos for each person you want to find:

```bash
# Copy reference photos to the NAS first
# Then register each person:
docker compose -f docker-compose.nas.yml run --rm photosearch \
    add-person "Alex" --photo /photos/references/alex.jpg

docker compose -f docker-compose.nas.yml run --rm photosearch \
    add-person "Jamie" --photo /photos/references/jamie1.jpg --photo /photos/references/jamie2.jpg
```

Reference photos should be clear, front-facing shots. Multiple reference photos per person improve matching accuracy. After adding references, run `match-faces` (Step 5, Pass 3) to apply them.


## Step 7: Access the web UI

From any device on your Tailscale network (once Tailscale is connected and CLIP indexing has run on at least some photos):

```
http://photosearch:8000
```

Or use the Tailscale IP directly:

```
http://100.x.y.z:8000
```

You can find the IP with:

```bash
docker compose -f docker-compose.nas.yml exec tailscale tailscale ip -4
```

This works from your Mac, phone (with Tailscale installed), or any other device on your tailnet — whether you're on your home LAN or halfway across the world.


## Deploying a database built on your Mac

If you've already indexed photos on your Mac and want to deploy that database to the NAS (instead of re-indexing), the paths stored in the database need to be remapped.

1. Copy your `photo_index.db` to the NAS:
   ```bash
   scp photo_index.db yournas:/volume1/docker/photosearch/data/
   ```

2. Add `REMAP_FROM` to the photosearch environment in `docker-compose.nas.yml`:
   ```yaml
   environment:
     - REMAP_FROM=/Users/mattnewkirk/Documents/Claude/Projects/Photo organization
   ```

3. Restart the container — paths are remapped automatically on startup:
   ```bash
   docker compose -f docker-compose.nas.yml up -d photosearch
   ```

4. Remove the `REMAP_FROM` line after the first successful start (it only needs to run once).


## Managing the deployment

```bash
# View logs
docker compose -f docker-compose.nas.yml logs -f photosearch
docker compose -f docker-compose.nas.yml logs -f ollama

# Stop everything
docker compose -f docker-compose.nas.yml down

# Restart after a config change
docker compose -f docker-compose.nas.yml up -d

# Update the code and rebuild
docker run --rm -v /volume1/docker/photosearch:/repo alpine sh -c \
  "apk add -q git && git config --global --add safe.directory /repo && git -C /repo pull"
docker compose -f docker-compose.nas.yml build photosearch
docker compose -f docker-compose.nas.yml up -d photosearch

# Run one-off CLI commands
docker compose -f docker-compose.nas.yml run --rm photosearch stats
docker compose -f docker-compose.nas.yml run --rm photosearch stack
docker compose -f docker-compose.nas.yml run --rm photosearch match-faces --temporal
docker compose -f docker-compose.nas.yml run --rm photosearch list-persons
```

### Running background indexing safely

Always use `nohup` with `PYTHONUNBUFFERED=1` so progress is visible and the job survives a disconnected SSH session:

```bash
nohup docker compose -f /volume1/docker/photosearch/docker-compose.nas.yml run --rm \
  -e PYTHONUNBUFFERED=1 photosearch index /photos/2026 --clip \
  > /tmp/index.log 2>&1 &
echo "PID: $!"

# Monitor
tail -f /tmp/index.log

# Check what's running
docker ps --filter name=photosearch-photosearch-run
```

To stop a background indexing job:
```bash
docker ps --filter name=photosearch-photosearch-run --format "{{.ID}}" | xargs docker stop
```


## Troubleshooting

**"Ollama connection refused"** — Ollama may still be starting. Check `docker compose -f docker-compose.nas.yml logs ollama` and wait for "Listening on 0.0.0.0:11434".

**Out of memory during indexing** — LLaVA 7B + CLIP + InsightFace together can spike to 6–7 GB. If the container is killed, try indexing in stages: first `--clip --faces --quality` (no LLaVA), then a separate pass with `--describe`.

**"database is locked" errors during indexing** — The web server and an indexing job are both writing to SQLite simultaneously. Use `--no-colors` during initial indexing passes (color extraction causes the most lock contention). You can run a dedicated color pass later when no other write jobs are active:
```bash
docker compose -f docker-compose.nas.yml run --rm photosearch index /photos/2026
```
(Without any flags, it only extracts colors for photos that don't have them yet.)

**Tailscale not connecting** — Check container logs: `docker logs photosearch-tailscale`. Common causes:
- Container was restarted individually by UGOS rather than via `docker compose` — the network config change won't take effect. Fix: `docker compose -f docker-compose.nas.yml down && docker compose -f docker-compose.nas.yml up -d`
- `/dev/net/tun` doesn't exist — run `modprobe tun`
- Auth token expired — check logs for an auth URL and re-authenticate

**CLIP indexing shows no progress / log appears frozen** — Normal during model loading (first run only). Once you see "Generating CLIP embeddings...", progress lines appear every ~5%. If it's truly stuck, check `docker ps` for duplicate index containers and stop the older ones.

**"No such command 'python'"** — The container entrypoint intercepts commands. Use `--entrypoint python` to run Python directly: `docker compose run --rm --entrypoint python photosearch cli.py --help`

**Slow search results** — First search after startup may be slow while CLIP loads into memory (~10–15 seconds on N100). Subsequent searches return in under a second.

**Photos not found** — Verify the mount path. From inside the container: `docker compose -f docker-compose.nas.yml exec photosearch ls /photos/` should show your photo folders.

**Can't SSH remotely (Tailscale down)** — If you have remote access to a UniFi or other managed router, add a temporary port forward for port 22 to the NAS IP. After reconnecting Tailscale, remove the port forward.
