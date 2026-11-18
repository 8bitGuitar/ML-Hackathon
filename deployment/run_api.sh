#!/usr/bin/env bash
# Start the PathSnap API (laptop 1). Run from anywhere. Settings: deployment/config.py or PATHSNAP_* variables.
set -e
cd "$(dirname "$0")/.."
exec python api/api.py
