#!/bin/bash
set -e

# Run a single test complexity level
# Usage: ./run-single-test.sh <complexity>

COMPLEXITY="${1:-basic}"
TEST_RESULTS_DIR="/home/metta/test-results"
TEST_LOGS_DIR="/home/metta/test-logs"
METTA_DIR="/home/metta/metta"

case "${COMPLEXITY}" in
    "basic"|"intermediate"|"advanced")
        echo "Running ${COMPLEXITY} test..."
        ;;
    *)
        echo "Invalid complexity level: ${COMPLEXITY}"
        echo "Valid options: basic, intermediate, advanced"
        exit 1
        ;;
esac

# Activate virtual environment
source "${METTA_DIR}/.venv/bin/activate"

# Set up environment
export DATA_DIR="${TEST_RESULTS_DIR}/${COMPLEXITY}/data"
export TEST_COMPLEXITY="${COMPLEXITY}"

# Create directories
mkdir -p "${TEST_RESULTS_DIR}/${COMPLEXITY}"
mkdir -p "${TEST_LOGS_DIR}"

echo "Running ${COMPLEXITY} sweep test..."

# Run the specific test
timeout 1800 python "${METTA_DIR}/tools/sweep_rollout.py" \
    --config-path="${METTA_DIR}/test-configs" \
    --config-name="test-sweep-${COMPLEXITY}" \
    sweep_name="test_${COMPLEXITY}_sweep" \
    runs_dir="${TEST_RESULTS_DIR}/${COMPLEXITY}/runs" \
    sweep_dir="${TEST_RESULTS_DIR}/${COMPLEXITY}/sweep" \
    data_dir="${TEST_RESULTS_DIR}/${COMPLEXITY}/data" \
    max_consecutive_failures=2 \
    rollout_retry_delay=5 \
    2>&1 | tee "${TEST_LOGS_DIR}/${COMPLEXITY}-single-test.log"

echo "${COMPLEXITY} test completed with exit code $?"
