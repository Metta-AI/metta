#!/bin/bash

# Learning Progress Arena Experiment Monitor
# This script helps monitor the runs and identify issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
USER=${USER:-$(whoami)}
EXPERIMENT_NAME="learning_progress_arena"
PROJECT="learning-progress-sweep"
ENTITY="metta-research"

# Function to check run status
check_run_status() {
    local run_name=$1

    log_info "Checking status of run: $run_name"

    # Get run status from wandb
    run_info=$(wandb runs list --project $PROJECT --entity $ENTITY | grep "$run_name" || echo "")

    if [ -z "$run_info" ]; then
        log_warning "Run '$run_name' not found in wandb"
        return 1
    fi

    # Extract status
    status=$(echo "$run_info" | awk '{print $NF}')

    case $status in
        "running")
            log_success "Run '$run_name' is running"
            return 0
            ;;
        "finished")
            log_success "Run '$run_name' completed successfully"
            return 0
            ;;
        "crashed"|"failed")
            log_error "Run '$run_name' failed"
            return 1
            ;;
        *)
            log_warning "Run '$run_name' has unknown status: $status"
            return 1
            ;;
    esac
}

# Function to get run metrics
get_run_metrics() {
    local run_name=$1

    log_info "Getting metrics for run: $run_name"

    # This would need wandb API access to get actual metrics
    # For now, we'll provide guidance on what to check

    log_info "Check these metrics in wandb for run '$run_name':"
    log_info "  - reward (should be increasing)"
    log_info "  - loss/value_loss (should be decreasing)"
    log_info "  - loss/policy_loss (should be stable)"
    log_info "  - lp/num_active_tasks (should be around 16)"
    log_info "  - lp/mean_sample_prob (should be > 0)"
    log_info "  - lp/task_success_rate (should improve)"
    log_info "  - system/gpu_memory_used (should be stable)"
}

# Function to check for common failure patterns
check_failure_patterns() {
    local run_name=$1

    log_info "Checking for common failure patterns in run: $run_name"

    # This would need access to run logs
    # For now, provide guidance on what to look for

    log_info "Look for these patterns in the run logs:"
    log_info "  - 'CUDA out of memory' (memory issue)"
    log_info "  - 'loss contains NaN' (training instability)"
    log_info "  - 'Failed to load checkpoint' (checkpoint issue)"
    log_info "  - 'Division by zero' (learning progress algorithm issue)"
    log_info "  - 'Invalid action' (environment issue)"
}

# Function to monitor all experiment runs
monitor_experiment_runs() {
    log_info "Monitoring Learning Progress Arena Experiment Runs"

    # Define expected run names
    runs=(
        "${USER}.${EXPERIMENT_NAME}.learning_progress"
        "${USER}.${EXPERIMENT_NAME}.random"
        "${USER}.${EXPERIMENT_NAME}.basic"
        "${USER}.${EXPERIMENT_NAME}.sweep"
    )

    log_info "Expected runs:"
    for run in "${runs[@]}"; do
        echo "  - $run"
    done
    echo

    # Check each run
    for run in "${runs[@]}"; do
        if check_run_status "$run"; then
            get_run_metrics "$run"
            check_failure_patterns "$run"
        fi
        echo
    done
}

# Function to check system resources
check_system_resources() {
    log_info "Checking system resources"

    # Check GPU memory
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r name used total util; do
            usage_pct=$((used * 100 / total))
            echo "  $name: ${used}MB / ${total}MB (${usage_pct}%) - GPU util: ${util}%"
        done
    else
        log_warning "nvidia-smi not available"
    fi

    # Check system memory
    echo "System Memory:"
    free -h | grep -E "Mem|Swap"

    # Check disk space
    echo "Disk Space:"
    df -h | grep -E "/$|/workspace|/home"
}

# Function to check recent wandb activity
check_wandb_activity() {
    log_info "Checking recent wandb activity"

    # List recent runs
    echo "Recent runs in project '$PROJECT':"
    wandb runs list --project $PROJECT --entity $ENTITY --limit 10

    echo
    echo "Runs with tags 'learning_progress_arena_comparison':"
    wandb runs list --project $PROJECT --entity $ENTITY | grep "learning_progress_arena_comparison" || echo "No runs found with this tag"
}

# Function to provide recovery suggestions
provide_recovery_suggestions() {
    log_info "Recovery Suggestions"

    echo "If runs are failing frequently, try these solutions:"
    echo
    echo "1. Memory Issues:"
    echo "   - Reduce batch_size: trainer.batch_size=262144"
    echo "   - Reduce num_workers: trainer.num_workers=2"
    echo "   - Increase minibatch_size: trainer.minibatch_size=32768"
    echo
    echo "2. Learning Progress Issues:"
    echo "   - Increase ema_timescale: ema_timescale=0.01"
    echo "   - Decrease progress_smoothing: progress_smoothing=0.01"
    echo "   - Increase sample_threshold: sample_threshold=20"
    echo
    echo "3. Checkpoint Issues:"
    echo "   - Increase checkpoint frequency: trainer.checkpoint.checkpoint_interval=25"
    echo "   - Check network connectivity for S3 uploads"
    echo
    echo "4. Training Stability:"
    echo "   - Reduce learning rate: trainer.optimizer.learning_rate=0.0001"
    echo "   - Increase gradient clipping: trainer.ppo.max_grad_norm=1.0"
    echo
    echo "5. Environment Issues:"
    echo "   - Test individual arena environments"
    echo "   - Check for invalid actions in environment logs"
}

# Function to create a quick test run
create_quick_test() {
    log_info "Creating quick test run"

    echo "Running a quick test with reduced parameters..."

    ./devops/skypilot/launch.py train \
        --gpus=1 \
        --nodes=1 \
        --no-spot \
        run="${USER}.${EXPERIMENT_NAME}.quick_test" \
        --config configs/user/learning_progress_experiment.yaml \
        trainer.total_timesteps=5_000_000 \
        trainer.num_workers=2 \
        trainer.batch_size=16384 \
        trainer.checkpoint.checkpoint_interval=25 \
        trainer.simulation.evaluate_interval=50 \
        ema_timescale=0.01 \
        progress_smoothing=0.01 \
        num_active_tasks=8 \
        sample_threshold=20

    if [ $? -eq 0 ]; then
        log_success "Quick test completed successfully"
    else
        log_error "Quick test failed"
    fi
}

# Main function
main() {
    log_info "Learning Progress Arena Experiment Monitor"
    log_info "This helps monitor runs and identify issues"

    echo "Choose an option:"
    echo "1. Monitor all experiment runs"
    echo "2. Check system resources"
    echo "3. Check wandb activity"
    echo "4. Provide recovery suggestions"
    echo "5. Create quick test run"
    echo "6. Run debug tests"
    echo "7. Exit"

    read -p "Enter your choice (1-7): " choice

    case $choice in
        1)
            monitor_experiment_runs
            ;;
        2)
            check_system_resources
            ;;
        3)
            check_wandb_activity
            ;;
        4)
            provide_recovery_suggestions
            ;;
        5)
            create_quick_test
            ;;
        6)
            ./recipes/test_learning_progress_debug.sh
            ;;
        7)
            log_info "Exiting..."
            exit 0
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Run the main function
main "$@"
