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

From your Mac, copy the project to the NAS. You can use `scp`, `rsync`, or the UGOS file manager:

```bash
# From your Mac, in the project directory:
rsync -av --exclude='venv' --exclude='*.db' --exclude='__pycache__' \
    --exclude='node_modules' --exclude='thumbnails' --exclude='results' \
    ./ yournas:/volume1/docker/photosearch/
```

Or use the UGOS file manager to create `/volume1/docker/photosearch/` and upload the files.


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

```bash
# Full pipeline: CLIP embeddings + face detection + descriptions + quality scoring
docker compose -f docker-compose.nas.yml run --rm photosearch \
    index /photos --faces --describe --quality

# Or index without descriptions (much faster, no Ollama needed):
docker compose -f docker-compose.nas.yml run --rm photosearch \
    index /photos --faces --quality
```

**Timing expectations on the N100:**
- CLIP embeddings: ~5-10 seconds per photo (CPU-only, no GPU)
- Face detection: ~2-3 seconds per photo
- LLaVA descriptions (7B): ~60-90 seconds per photo
- Quality scoring: ~3-5 seconds per photo

For 1000 photos with full pipeline, expect 24-36 hours. You can run indexing in stages — it's safe to stop and restart. Already-indexed photos are skipped.

**Tip:** Index without `--describe` first to get search working quickly (CLIP alone takes ~2-3 hours for 1000 photos), then run a second pass with `--describe` overnight.


## Step 6: Access the web UI

From any device on your Tailscale network:

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

# Update the code (after pulling/copying new files)
docker compose -f docker-compose.nas.yml build photosearch
docker compose -f docker-compose.nas.yml up -d photosearch

# Run a one-off CLI command
docker compose -f docker-compose.nas.yml run --rm photosearch detect-stacks
docker compose -f docker-compose.nas.yml run --rm photosearch stats
```


## Troubleshooting

**"Ollama connection refused"** — Ollama may still be starting. Check `docker compose -f docker-compose.nas.yml logs ollama` and wait for "Listening on 0.0.0.0:11434".

**Out of memory during indexing** — LLaVA 7B + CLIP + InsightFace together can spike to 6-7 GB. If the container is killed, try indexing in stages: first `--faces --quality` (no LLaVA), then a separate pass with `--describe`.

**Tailscale not connecting** — Check that `/dev/net/tun` exists on the NAS. If not, you may need to load the tun kernel module: `modprobe tun`. Also verify the container has NET_ADMIN capability.

**Slow search results** — First search after startup may be slow while CLIP loads into memory. Subsequent searches should return in under a second.

**Photos not found** — Verify the mount path. From inside the container: `docker compose -f docker-compose.nas.yml exec photosearch ls /photos/` should show your photo folders.
