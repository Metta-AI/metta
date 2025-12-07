#!/usr/bin/env bash
set -euo pipefail

# Temporary helper to reproduce the optimized_obs A/B runs from the CLI.
# Usage: ./run_optobs_ab.sh [run_prefix] [total_timesteps]

RUN_PREFIX_BASE="${1:-ab_optobs}"
RUN_PREFIX="${RUN_PREFIX_BASE}_$(date +%s)"
TOTAL_TIMESTEPS="${2:-1000000}"

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
  torch_profiler.interval_epochs=0
  stats_server_uri=none
)

echo "Running baseline (optimized_obs OFF) run=${RUN_PREFIX}_off..."
METTA_TIMER_REPORT=1 uv run "${COMMON_ARGS[@]}" "run=${RUN_PREFIX}_off"

echo "Running optimized (optimized_obs ON via env) run=${RUN_PREFIX}_on..."
METTAGRID_OPTIMIZED_OBS=1 METTA_TIMER_REPORT=1 uv run "${COMMON_ARGS[@]}" "run=${RUN_PREFIX}_on"
