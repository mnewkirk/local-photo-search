#!/bin/bash
set -e

cd /volume1/docker/photosearch
git -c safe.directory=/volume1/docker/photosearch -c credential.helper='!gh auth git-credential' pull origin main
echo "New HEAD: $(git rev-parse HEAD)"
