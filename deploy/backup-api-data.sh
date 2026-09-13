#!/bin/sh
# deploy/backup-api-data.sh — nightly api_data volume backup
# (service/DEPLOYMENT.md §9.8).
#
# Everything stateful the service cannot regenerate lives in the
# `api_data` compose volume: the exchange log, the chart-spec store, the
# #217 per-UTC-day spend journal and the footprint aggregate ledger. The
# qdrant index is rebuildable from the corpus and is deliberately NOT
# backed up. Owner ruling 2026-09-13: no provider-level (Hetzner) backups
# — minimal cost, manual recovery accepted — so this on-box tarball plus
# the operator's workstation pull (§9.8) IS the whole backup story.
#
# Usage:  backup-api-data.sh [volume-name] [dest-dir]
#   volume-name  default climate-chat_api_data (compose prefixes the
#                volume with the project directory name — check
#                `docker volume ls` after first `up`)
#   dest-dir     default /root/backups
#
# Keeps the newest $KEEP dated tarballs; prints the tarball path on
# success so the cron line's log records what was written. Retention
# PURGES (90-day exchange log, 7-day rate-limit hashes) run in-process
# in the service (§7) — this script only copies bytes out.
set -eu

VOLUME="${1:-climate-chat_api_data}"
DEST="${2:-/root/backups}"
KEEP="${KEEP:-14}"

mkdir -p "$DEST"
STAMP="$(date -u +%F)"
OUT="$DEST/api-data-$STAMP.tar.gz"

# :ro mount — the backup can never write into the live volume. Alpine's
# tar is enough for a gzip'd copy; the container is removed on exit.
docker run --rm \
  -v "$VOLUME":/data:ro \
  -v "$DEST":/backup \
  alpine tar czf "/backup/api-data-$STAMP.tar.gz" -C /data .

# Prune to the newest $KEEP (files sort by mtime; -t is newest-first).
ls -1t "$DEST"/api-data-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm --

echo "$OUT"
