#!/usr/bin/env bash
# Update log config with tags

set -e

CONF_D_DIR="/etc/datadog-agent/conf.d"
CUSTOM_LOGS_DIR="${CONF_D_DIR}/skypilot_training.d"
LOG_CONFIG_FILE="${CUSTOM_LOGS_DIR}/conf.yaml"

# Create directory
mkdir -p "$CUSTOM_LOGS_DIR"

# Build tags list
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

# Format tags as YAML
TAGS_YAML=""
if [ ${#LOG_TAGS[@]} -gt 0 ]; then
  TAGS_LINES=""
  for tag in "${LOG_TAGS[@]}"; do
    TAGS_LINES+=$'      - "'$tag$'"\n'
  done
  TAGS_YAML=$'    tags:\n'"${TAGS_LINES}"
fi

# Create log directory
TRAINING_LOG_DIR="/tmp/training_logs"
mkdir -p "$TRAINING_LOG_DIR"
chmod 777 "$TRAINING_LOG_DIR"

# Create combined log file
for log_file in "training_combined.log"; do
  log_path="${TRAINING_LOG_DIR}/${log_file}"
  if [ ! -f "$log_path" ]; then
    touch "$log_path"
  fi
  chmod 666 "$log_path"
done

# Create log config
cat > "$LOG_CONFIG_FILE" <<EOF
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

# Set file permissions
chmod 644 "$LOG_CONFIG_FILE"

