#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

export PROXY_CONFIG_PATH="${PROXY_CONFIG_PATH:-$ROOT_DIR/models.yaml}"
export PROXY_REQUEST_LOG_PATH="${PROXY_REQUEST_LOG_PATH:-$ROOT_DIR/data/proxy_requests.jsonl}"
export PROXY_METRICS_PATH="${PROXY_METRICS_PATH:-$ROOT_DIR/data/proxy_metrics.json}"

mkdir -p "$(dirname "$PROXY_REQUEST_LOG_PATH")" "$(dirname "$PROXY_METRICS_PATH")"

exec "${PYTHON:-python3}" "$ROOT_DIR/search_proxy.py"
