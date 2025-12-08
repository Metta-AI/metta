#!/usr/bin/env bash
set -euo pipefail

# Get script directory and navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# 1. Generate JSON payloads from metric_schema.py
python3 -m devops.datadog.generate_dashboards

# 2. Push both dashboards to Datadog
python3 -m devops.datadog.push_dashboards
