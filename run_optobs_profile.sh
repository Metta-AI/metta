#!/usr/bin/env bash
set -euo pipefail

# Run optimized_obs A/B with torch profiler enabled and timer reporting.
# Usage: ./run_optobs_profile.sh [run_prefix] [total_timesteps] [profile_dir]

RUN_PREFIX="${1:-ab_optobs_prof}"
TOTAL_TIMESTEPS="${2:-1000000}"
PROFILE_DIR="${3:-./profiles}"

mkdir -p "${PROFILE_DIR}"

COMMON_ARGS=(
  ./tools/run.py train arena
  system.local_only=true
  wandb.enabled=false wandb.project=na wandb.entity=na
  trainer.minibatch_size=512
  trainer.batch_size=32768
  trainer.bptt_horizon=8
  "trainer.total_timesteps=${TOTAL_TIMESTEPS}"
  training_env.vectorization=serial
  training_env.num_workers=1
  training_env.async_factor=1
  training_env.forward_pass_minibatch_target_size=4096
  training_env.write_replays=false
  evaluator.evaluate_local=false
  evaluator.evaluate_remote=false
  evaluator.epoch_interval=0
  checkpointer.epoch_interval=1000
  torch_profiler.interval_epochs=1
  "torch_profiler.profile_dir=${PROFILE_DIR}"
  stats_server_uri=none
)

echo "Running baseline with torch profiler (optimized_obs OFF)..."
TORCH_PROFILER_FIRST_EPOCH=1 METTA_TIMER_REPORT=1 uv run "${COMMON_ARGS[@]}" "run=${RUN_PREFIX}_off"

echo "Running optimized with torch profiler (optimized_obs ON)..."
METTAGRID_OPTIMIZED_OBS=1 TORCH_PROFILER_FIRST_EPOCH=1 METTA_TIMER_REPORT=1 uv run "${COMMON_ARGS[@]}" "run=${RUN_PREFIX}_on"
