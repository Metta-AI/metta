#!/usr/bin/env bash
# Start Datadog agent as a background daemon in SkyPilot jobs
# This script is called during the run phase after environment setup

set -euo pipefail

AGENT_BINARY="/opt/datadog-agent/bin/agent/agent"

if [ ! -f "$AGENT_BINARY" ]; then
    echo "[DATADOG] Agent binary not found at $AGENT_BINARY, skipping startup"
    exit 0
fi

# Check if agent is already running
if pgrep -f "datadog-agent.*run" > /dev/null; then
    echo "[DATADOG] Agent is already running"
    exit 0
fi

echo "[DATADOG] Starting Datadog agent daemon..."

# Start agent in background, redirect output to logs
nohup "$AGENT_BINARY" run > /tmp/datadog-agent.log 2>&1 &
AGENT_PID=$!

# Wait a moment and verify it started
sleep 2

if ps -p "$AGENT_PID" > /dev/null; then
    echo "[DATADOG] Agent started successfully (PID: $AGENT_PID)"

    # Try to verify status (non-blocking)
    if "$AGENT_BINARY" status > /dev/null 2>&1; then
        echo "[DATADOG] Agent status: Running"
    else
        echo "[DATADOG] Agent started but status check failed (may need more time)"
    fi
else
    echo "[DATADOG] WARNING: Agent process died immediately after startup"
    echo "[DATADOG] Check logs: /tmp/datadog-agent.log"
    exit 1
fi

