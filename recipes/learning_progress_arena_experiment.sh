#!/bin/bash

# Learning Progress Arena Experiment
# This script runs a complete experiment comparing learning progress curriculum
# against random curriculum and basic arena baselines.

set -e  # Exit on any error

# Configuration
USER=${USER:-$(whoami)}
EXPERIMENT_NAME="learning_progress_arena"
SWEEP_NAME="${USER}.${EXPERIMENT_NAME}.sweep"
LEARNING_PROGRESS_RUN="${USER}.${EXPERIMENT_NAME}.learning_progress"
RANDOM_CURRICULUM_RUN="${USER}.${EXPERIMENT_NAME}.random"
BASIC_ARENA_RUN="${USER}.${EXPERIMENT_NAME}.basic"

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

# Function to wait for a run to complete
wait_for_run() {
    local run_name=$1
    local max_wait_hours=48  # Maximum wait time in hours

    log_info "Waiting for run '$run_name' to complete..."

        # Check if run exists in wandb
    if ! wandb runs list --project learning-progress-sweep --entity metta-research | grep -q "$run_name"; then
        log_error "Run '$run_name' not found in wandb"
        return 1
    fi

    # Wait for completion
    local start_time=$(date +%s)
    while true; do
        # Check if run is still running
        if wandb runs list --project learning-progress-sweep --entity metta-research | grep "$run_name" | grep -q "running"; then
            local elapsed=$(( $(date +%s) - start_time ))
            local elapsed_hours=$(( elapsed / 3600 ))

            if [ $elapsed_hours -gt $max_wait_hours ]; then
                log_warning "Run '$run_name' has been running for more than $max_wait_hours hours. Continuing..."
                break
            fi

            log_info "Run '$run_name' still running... (elapsed: ${elapsed_hours}h)"
            sleep 300  # Check every 5 minutes
        else
            log_success "Run '$run_name' completed"
            break
        fi
    done
}

# Function to get the best hyperparameters from sweep
get_best_hyperparams() {
    local sweep_name=$1

    log_info "Getting best hyperparameters from sweep '$sweep_name'..."

    # This would need to be implemented based on your wandb API access
    # For now, we'll use default values and you can manually update them
    log_warning "Please manually check wandb for the best hyperparameters from sweep '$sweep_name'"
    log_warning "Update the hyperparameters in the launch commands below"

    # Example of how to extract best hyperparameters (you'll need to implement this)
    # BEST_EMA_TIMESCALE=$(wandb api get-best-param sweep_name ema_timescale)
    # BEST_PROGRESS_SMOOTHING=$(wandb api get-best-param sweep_name progress_smoothing)
    # etc.
}

# Main experiment flow
main() {
    log_info "Starting Learning Progress Arena Experiment"
    log_info "User: $USER"
    log_info "Experiment Name: $EXPERIMENT_NAME"

    # Step 1: Run hyperparameter sweep
    log_info "Step 1: Running hyperparameter sweep for learning progress curriculum..."

    ./devops/skypilot/launch.py sweep \
        --gpus=4 \
        --nodes=8 \
        --no-spot \
        run="$SWEEP_NAME" \
        --config configs/sweep_job_learning_progress.yaml \
        sweep_name="$SWEEP_NAME" \
        trainer.total_timesteps=1_000_000_000

    log_success "Hyperparameter sweep launched: $SWEEP_NAME"

    # Wait for sweep to complete
    log_info "Waiting for hyperparameter sweep to complete..."
    # Note: This is a simplified wait - you may need to implement proper sweep completion detection
    sleep 3600  # Wait 1 hour for sweep to start
    log_warning "Please monitor the sweep manually and proceed when it's complete"

    # Step 2: Get best hyperparameters (manual step for now)
    get_best_hyperparams "$SWEEP_NAME"

    # Step 3: Run the three comparison experiments
    log_info "Step 3: Running comparison experiments with best hyperparameters..."

    # 3a. Learning Progress Curriculum (with best hyperparameters)
    log_info "3a. Running Learning Progress Curriculum..."
    ./devops/skypilot/launch.py train \
        --gpus=4 \
        --nodes=8 \
        --no-spot \
        run="$LEARNING_PROGRESS_RUN" \
        --config configs/user/learning_progress_experiment.yaml \
        trainer.total_timesteps=1_000_000_000
        # Add best hyperparameters here after sweep completes:
        # ema_timescale=0.001 \
        # progress_smoothing=0.05 \
        # num_active_tasks=16 \
        # rand_task_rate=0.25 \
        # sample_threshold=10 \
        # memory=25

    # 3b. Random Curriculum Baseline
    log_info "3b. Running Random Curriculum Baseline..."
    ./devops/skypilot/launch.py train \
        --gpus=4 \
        --nodes=8 \
        --no-spot \
        run="$RANDOM_CURRICULUM_RUN" \
        --config configs/user/random_curriculum_experiment.yaml \
        trainer.total_timesteps=1_000_000_000

    # 3c. Basic Arena Baseline
    log_info "3c. Running Basic Arena Baseline..."
    ./devops/skypilot/launch.py train \
        --gpus=4 \
        --nodes=8 \
        --no-spot \
        run="$BASIC_ARENA_RUN" \
        --config configs/user/basic_arena_experiment.yaml \
        trainer.total_timesteps=1_000_000_000

    log_success "All experiments launched!"
    log_info "Experiment runs:"
    log_info "  - Learning Progress: $LEARNING_PROGRESS_RUN"
    log_info "  - Random Curriculum: $RANDOM_CURRICULUM_RUN"
    log_info "  - Basic Arena: $BASIC_ARENA_RUN"

    log_info "Monitor progress at: https://wandb.ai/metta-research/learning-progress-sweep"
    log_info "Experiment complete!"
}

# Run the main function
main "$@"
