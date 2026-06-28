#!/bin/bash
# Start TOEIC Campus Sprint backend (FastAPI :8090)
# Source key from /home/peter/.hermes/secrets/deepseek.env if present, else expect env var.
set -a
[ -f /home/peter/.hermes/secrets/deepseek.env ] && source /home/peter/.hermes/secrets/deepseek.env
set +a
cd /home/peter/standup-backend || exit 1
exec /home/peter/.hermes/hermes-agent/venv/bin/uvicorn main:app \
  --host 127.0.0.1 --port 8090 --workers 1 --log-level warning
