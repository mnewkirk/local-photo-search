#!/bin/bash
set -e

REPO_DIR="/volume1/docker/photosearch"

# Pull latest main from origin
# Use -c safe.directory to work around "dubious ownership" on UGOS
git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" pull origin main

# Print the new HEAD commit
git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" log -1 --oneline HEAD
