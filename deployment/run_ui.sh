#!/usr/bin/env bash
# Serve the evaluator UI (laptop 2) on http://localhost:${1:-8080}
# Open it with ?api=http://<laptop 1 IP>:8000 or set apiUrl in ui/config.js
set -e
cd "$(dirname "$0")/../ui"
exec python -m http.server "${1:-8080}" --bind 0.0.0.0
