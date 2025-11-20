#!/usr/bin/env bash
# Update Datadog log collection config with run-specific tags
# This must run in the run phase when METTA_RUN_ID and SKYPILOT_TASK_ID are available

set -e

CONF_D_DIR="/etc/datadog-agent/conf.d"
CUSTOM_LOGS_DIR="${CONF_D_DIR}/skypilot_training.d"
LOG_CONFIG_FILE="${CUSTOM_LOGS_DIR}/conf.yaml"

# Ensure directory exists
mkdir -p "$CUSTOM_LOGS_DIR"

# Build tags list for log configuration
LOG_TAGS=()
if [ -n "${METTA_RUN_ID:-}" ]; then
  LOG_TAGS+=("metta_run_id:${METTA_RUN_ID}")
fi
if [ -n "${SKYPILOT_TASK_ID:-}" ]; then
  LOG_TAGS+=("skypilot_task_id:${SKYPILOT_TASK_ID}")
fi
if [ -n "${SKYPILOT_NODE_RANK:-}" ]; then
  LOG_TAGS+=("node_rank:${SKYPILOT_NODE_RANK}")
fi
if [ -n "${SKYPILOT_NUM_NODES:-}" ]; then
  LOG_TAGS+=("num_nodes:${SKYPILOT_NUM_NODES}")
fi

# Format tags as YAML list (with real newlines, not literal '\n')
TAGS_YAML=""
if [ ${#LOG_TAGS[@]} -gt 0 ]; then
  TAGS_LINES=""
  for tag in "${LOG_TAGS[@]}"; do
    # Append a line like:       - "metta_run_id:run123"
    TAGS_LINES+=$'      - "'$tag$'"\n'
  done
  TAGS_YAML=$'    tags:\n'"${TAGS_LINES}"
fi

# Ensure log directory and files exist
TRAINING_LOG_DIR="/tmp/training_logs"
mkdir -p "$TRAINING_LOG_DIR"
chmod 777 "$TRAINING_LOG_DIR"

# We only use combined log now, as stdout/stderr are tee'd into it
for log_file in "training_combined.log"; do
  log_path="${TRAINING_LOG_DIR}/${log_file}"
  if [ ! -f "$log_path" ]; then
    touch "$log_path"
  fi
  chmod 666 "$log_path"
done

# Create/update log collection config
cat > "$LOG_CONFIG_FILE" <<EOF
# Custom log collection for SkyPilot jobs
# Updated during run phase with run-specific tags
logs:
  - type: file
    path: /tmp/datadog-agent.log
    service: datadog-agent
    source: datadog-agent
    sourcecategory: monitoring
${TAGS_YAML}
  - type: file
    path: /tmp/training_logs/training_combined.log
    service: skypilot-training
    source: training
    sourcecategory: application
${TAGS_YAML}
EOF

# Set proper permissions
chmod 644 "$LOG_CONFIG_FILE"

echo "[DATADOG] Updated log collection config at ${LOG_CONFIG_FILE}"
echo "[DATADOG] Tags: ${LOG_TAGS[*]}"
echo "[DATADOG] Config file size: $(stat -f%z "$LOG_CONFIG_FILE" 2>/dev/null || stat -c%s "$LOG_CONFIG_FILE" 2>/dev/null || echo "unknown") bytes"

