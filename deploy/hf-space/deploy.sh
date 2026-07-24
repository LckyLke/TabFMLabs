#!/usr/bin/env bash
#
# Push the current checkout to a Hugging Face Docker Space.
#
#   ./deploy.sh <hf-user>/<space-name>       e.g. ./deploy.sh lukef/tabfm-studio-demo
#
# One-time setup on huggingface.co:
#   1. Create a Space → SDK "Docker" → blank template.
#   2. In Space settings, add variables:
#        MODEL_BACKEND=baseline        (or add secret TABPFN_TOKEN for TabPFN)
#        MAX_UPLOAD_BYTES=1000000      (demo upload cap, ~1 MB)
#        STUDIO_DB=/tmp/studio.db      (Spaces run as non-root; /tmp is writable)
#   3. Log in locally: `hf auth login` (or `huggingface-cli login`).
#
set -euo pipefail

SPACE="${1:?usage: ./deploy.sh <hf-user>/<space-name>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 "file://$ROOT" "$TMP/space"
cd "$TMP/space"
cp deploy/hf-space/README.md README.md   # Space card (frontmatter) replaces the repo README
rm -rf .git
git init -q -b main
git add -A
git commit -qm "Deploy TabFM Studio demo"
git push --force "https://huggingface.co/spaces/$SPACE" main

echo "Pushed. The Space builds automatically: https://huggingface.co/spaces/$SPACE"
