#!/usr/bin/env bash

set -e

# Check args
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <HURI_CONFIG_PATH> <AGENT_CONFIG_PATH> [CLEAN]"
  exit 1
fi

HURI_CONFIG="$1"
AGENT_CONFIG="$2"

LOG_DIR="./tmp/log"

if [[ " $* " == *" CLEAN "* ]]; then
  echo "Cleaning previous logs in ${LOG_DIR}"
  rm -rf "${LOG_DIR}"
fi

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
HURI_LOG="${LOG_DIR}/huri-${TIMESTAMP}.log"


# Run huri with output redirected
python -m src.launch_huri --config "$HURI_CONFIG" > "$HURI_LOG" 2>&1 &
HURI_PID=$!
echo "HURI started in background (PID=${HURI_PID}), logging to ${HURI_LOG}"

# Run agent
python -m src.launch_agent --config "$AGENT_CONFIG"

# Ensure HURI is killed on script exit (normal or Ctrl+C)
cleanup() {
  echo "Stopping HURI (PID=${HURI_PID})"
  kill "${HURI_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM