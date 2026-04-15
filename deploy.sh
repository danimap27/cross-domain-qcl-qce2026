#!/bin/bash
# deploy.sh — Sync project to Hercules cluster and optionally run manager.
#
# Usage:
#   bash deploy.sh                  # rsync only
#   bash deploy.sh --launch         # rsync + open SSH session
#
# Configure:
#   HERCULES_USER : your Hercules username
#   HERCULES_HOST : cluster hostname or SSH alias
#   REMOTE_DIR    : destination path on the cluster

HERCULES_USER="${HERCULES_USER:-dmarper2}"
HERCULES_HOST="${HERCULES_HOST:-hercules.spc.cica.es}"
REMOTE_DIR="${REMOTE_DIR:-~/cross-domain-qcl}"

LOCAL_DIR="$(dirname "$(realpath "$0")")"

echo "============================================================"
echo "  Deploying to ${HERCULES_USER}@${HERCULES_HOST}:${REMOTE_DIR}"
echo "============================================================"

rsync -avz --progress \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "data/raw" \
    --exclude "results" \
    --exclude "logs" \
    --exclude "paper/tables/*.tex" \
    "$LOCAL_DIR/" \
    "${HERCULES_USER}@${HERCULES_HOST}:${REMOTE_DIR}/"

echo ""
echo "[OK] Sync complete."

if [[ "$1" == "--launch" ]]; then
    echo "[INFO] Opening SSH session..."
    ssh "${HERCULES_USER}@${HERCULES_HOST}" -t "cd ${REMOTE_DIR} && bash"
fi
