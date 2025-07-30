#!/bin/bash
set -e

# Metta Sweep Test Runner
# Orchestrates distributed sweep testing with increasing complexity

echo "========================================"
echo "Metta Distributed Sweep Test Runner"
echo "========================================"

# Configuration
TEST_RESULTS_DIR="/home/metta/test-results"
TEST_LOGS_DIR="/home/metta/test-logs"
TEST_CONFIGS_DIR="/home/metta/metta/test-configs"
METTA_DIR="/home/metta/metta"

# Test configurations in order of complexity
TEST_CONFIGS=(
    "test-sweep-basic.yaml"
    "test-sweep-intermediate.yaml"
    "test-sweep-advanced.yaml"
)

# Test complexity levels
COMPLEXITY_LEVELS=("basic" "intermediate" "advanced")

# Activate virtual environment
source "${METTA_DIR}/.venv/bin/activate"

# Ensure test directories exist
mkdir -p "${TEST_RESULTS_DIR}"/{runs,sweep,data,logs,artifacts}
mkdir -p "${TEST_LOGS_DIR}"

# Initialize test results tracking
TEST_START_TIME=$(date +%s)
TEST_SUMMARY_FILE="${TEST_RESULTS_DIR}/test-summary.json"
OVERALL_SUCCESS=true

echo "Test start time: $(date)"
echo "Results directory: ${TEST_RESULTS_DIR}"
echo "Logs directory: ${TEST_LOGS_DIR}"

# Function to run a single sweep test
run_sweep_test() {
    local config_file="$1"
    local complexity="$2"
    local test_start_time=$(date +%s)

    echo "----------------------------------------"
    echo "Running test: ${config_file} (${complexity})"
    echo "----------------------------------------"

    # Create test-specific directories
    local test_run_dir="${TEST_RESULTS_DIR}/${complexity}"
    local test_log_file="${TEST_LOGS_DIR}/${complexity}-test.log"
    mkdir -p "${test_run_dir}"

    # Set up environment for this test
    export DATA_DIR="${test_run_dir}/data"
    export TEST_COMPLEXITY="${complexity}"

    # Clean up any previous state
    echo "Cleaning up previous test state..."
    cleanup_test_environment

    # Wait for distributed nodes to be ready
    echo "Waiting for distributed nodes..."
    wait_for_distributed_nodes

    local success=true
    local error_message=""

    # Run the distributed sweep test
    echo "Starting distributed sweep test..."
    if timeout 1800 run_distributed_sweep_test "${config_file}" "${test_run_dir}" "${test_log_file}"; then
        echo "✓ Test ${complexity} completed successfully"
    else
        local exit_code=$?
        echo "✗ Test ${complexity} failed with exit code ${exit_code}"
        success=false
        error_message="Test failed with exit code ${exit_code}"
        OVERALL_SUCCESS=false
    fi

    # Collect test metrics and artifacts
    collect_test_metrics "${complexity}" "${test_run_dir}" "${success}"

    # Generate test report
    local test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))

    generate_test_report "${complexity}" "${success}" "${test_duration}" "${error_message}"

    # Cleanup between tests
    cleanup_test_environment

    echo "Test ${complexity} completed in ${test_duration}s"
}

# Function to run distributed sweep test
run_distributed_sweep_test() {
    local config_file="$1"
    local test_run_dir="$2"
    local test_log_file="$3"

    echo "Running sweep with config: ${config_file}"
    echo "Output directory: ${test_run_dir}"
    echo "Log file: ${test_log_file}"

    # Run sweep on master node
    echo "Executing sweep on master node..."
    docker exec metta-sweep-master bash -c "
        source /home/metta/metta/.venv/bin/activate && \
        cd /home/metta/metta && \
        export DATA_DIR='${test_run_dir}/data' && \
        export PYTHONPATH=/home/metta/metta && \
        timeout 1200 python tools/sweep_rollout.py \
            --config-path=test-configs \
            --config-name=${config_file%%.yaml} \
            sweep_name=test_${TEST_COMPLEXITY}_sweep \
            runs_dir=${test_run_dir}/runs \
            sweep_dir=${test_run_dir}/sweep \
            data_dir=${test_run_dir}/data \
            max_consecutive_failures=2 \
            rollout_retry_delay=5
    " 2>&1 | tee "${test_log_file}"
}

# Function to wait for distributed nodes
wait_for_distributed_nodes() {
    echo "Checking distributed node health..."

    # Check master node
    if ! docker exec metta-sweep-master python3 -c "import torch; print('Master node ready')" > /dev/null 2>&1; then
        echo "Master node not ready"
        return 1
    fi

    # Check worker node
    if ! docker exec metta-sweep-worker python3 -c "import torch; print('Worker node ready')" > /dev/null 2>&1; then
        echo "Worker node not ready"
        return 1
    fi

    # Test network connectivity
    if ! docker exec metta-sweep-worker ping -c 1 sweep-master > /dev/null 2>&1; then
        echo "Network connectivity test failed"
        return 1
    fi

    echo "All distributed nodes are ready"
    return 0
}

# Function to cleanup test environment
cleanup_test_environment() {
    echo "Cleaning up test environment..."

    # Kill any hanging processes
    docker exec metta-sweep-master pkill -f "sweep_rollout" || true
    docker exec metta-sweep-worker pkill -f "sweep_rollout" || true

    # Clean up any distributed state
    docker exec metta-sweep-master bash -c "
        python3 -c 'import torch.distributed as dist; dist.destroy_process_group() if dist.is_initialized() else None' 2>/dev/null || true
    " || true

    # Wait for cleanup
    sleep 5
}

# Function to collect test metrics
collect_test_metrics() {
    local complexity="$1"
    local test_run_dir="$2"
    local success="$3"

    echo "Collecting test metrics for ${complexity}..."

    # Create metrics file
    local metrics_file="${test_run_dir}/metrics.json"

    # Collect basic metrics
    {
        echo "{"
        echo "  \"test_complexity\": \"${complexity}\","
        echo "  \"success\": ${success},"
        echo "  \"timestamp\": $(date +%s),"
        echo "  \"containers\": {"

        # Get container stats
        echo "    \"master\": {"
        docker stats --no-stream --format "table {{.CPUPerc}},{{.MemUsage}}" metta-sweep-master | tail -1 | \
        awk -F',' '{print "      \"cpu_percent\": \"" $1 "\", \"memory\": \"" $2 "\""}'
        echo "    },"
        echo "    \"worker\": {"
        docker stats --no-stream --format "table {{.CPUPerc}},{{.MemUsage}}" metta-sweep-worker | tail -1 | \
        awk -F',' '{print "      \"cpu_percent\": \"" $1 "\", \"memory\": \"" $2 "\""}'
        echo "    }"
        echo "  }"
        echo "}"
    } > "${metrics_file}"
}

# Function to generate test report
generate_test_report() {
    local complexity="$1"
    local success="$2"
    local duration="$3"
    local error_message="$4"

    echo "Generating test report for ${complexity}..."

    # Append to summary
    {
        echo "{"
        echo "  \"test\": \"${complexity}\","
        echo "  \"success\": ${success},"
        echo "  \"duration_seconds\": ${duration},"
        echo "  \"timestamp\": $(date +%s)"
        if [ -n "${error_message}" ]; then
            echo "  \"error\": \"${error_message}\""
        fi
        echo "},"
    } >> "${TEST_SUMMARY_FILE}.tmp"
}

# Function to finalize test summary
finalize_test_summary() {
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - TEST_START_TIME))

    echo "Finalizing test summary..."

    # Create final summary JSON
    {
        echo "{"
        echo "  \"overall_success\": ${OVERALL_SUCCESS},"
        echo "  \"total_duration_seconds\": ${total_duration},"
        echo "  \"start_time\": ${TEST_START_TIME},"
        echo "  \"end_time\": ${total_end_time},"
        echo "  \"tests\": ["

        # Add individual test results
        if [ -f "${TEST_SUMMARY_FILE}.tmp" ]; then
            sed '$ s/,$//' "${TEST_SUMMARY_FILE}.tmp"
        fi

        echo "  ]"
        echo "}"
    } > "${TEST_SUMMARY_FILE}"

    # Cleanup temp file
    rm -f "${TEST_SUMMARY_FILE}.tmp"
}

# Main execution
main() {
    echo "Starting distributed sweep test suite..."

    # Initialize summary file
    echo "" > "${TEST_SUMMARY_FILE}.tmp"

    # Run tests in order of increasing complexity
    for i in "${!TEST_CONFIGS[@]}"; do
        local config="${TEST_CONFIGS[$i]}"
        local complexity="${COMPLEXITY_LEVELS[$i]}"

        echo ""
        echo "========================================"
        echo "TEST $((i+1))/${#TEST_CONFIGS[@]}: ${complexity^^}"
        echo "========================================"

        run_sweep_test "${config}" "${complexity}"

        # Add delay between tests
        if [ $((i+1)) -lt ${#TEST_CONFIGS[@]} ]; then
            echo "Waiting 30 seconds before next test..."
            sleep 30
        fi
    done

    # Finalize results
    finalize_test_summary

    echo ""
    echo "========================================"
    echo "TEST SUITE COMPLETED"
    echo "========================================"
    echo "Overall success: ${OVERALL_SUCCESS}"
    echo "Results: ${TEST_SUMMARY_FILE}"
    echo "Logs: ${TEST_LOGS_DIR}"

    # Exit with appropriate code
    if [ "${OVERALL_SUCCESS}" = "true" ]; then
        echo "✓ All tests passed!"
        exit 0
    else
        echo "✗ Some tests failed!"
        exit 1
    fi
}

# Run main function
main "$@"
