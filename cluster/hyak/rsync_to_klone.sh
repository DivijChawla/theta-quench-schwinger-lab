#!/bin/bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${REMOTE:-dc245@klone.hyak.uw.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/gscratch/scrubbed/dc245/theta_quench_magic_lab}"

rsync -az \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude ".pytest_cache" \
  --exclude "__pycache__" \
  --exclude ".DS_Store" \
  --exclude "outputs" \
  --exclude "artifacts/camera_ready" \
  "$LOCAL_ROOT"/ "$REMOTE:$REMOTE_ROOT/"

echo "Synced repo to $REMOTE:$REMOTE_ROOT"
