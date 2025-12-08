#!/usr/bin/env bash
set -euo pipefail

# Run A/B perf comparisons across observation implementations.
# Variants: main (origin/main), alt (copy of main to iterate on).
# Usage: ./run_obs_variants.sh [run_prefix] [total_timesteps]

RUN_PREFIX_BASE="${1:-obsvar}"
RUN_PREFIX="${RUN_PREFIX_BASE}_$(date +%s)"
TOTAL_TIMESTEPS="${2:-100000}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINDINGS_DIR="$ROOT_DIR/packages/mettagrid/cpp/bindings"

declare -A VARIANT_CPP=(
  [alt]="$BINDINGS_DIR/alt_mettagrid_c.cpp"
  [main]="$BINDINGS_DIR/main_mettagrid_c.cpp"
)

declare -A VARIANT_HPP=(
  [alt]="$BINDINGS_DIR/alt_mettagrid_c.hpp"
  [main]="$BINDINGS_DIR/main_mettagrid_c.hpp"
)

COMMON_ARGS=(
  ./tools/run.py train cogs_v_clips
  mission=machina_1.open_world
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

set_variant() {
  local variant="$1"
  local cpp_src="${VARIANT_CPP[$variant]}"
  local hpp_src="${VARIANT_HPP[$variant]}"
  if [[ ! -f "$cpp_src" || ! -f "$hpp_src" ]]; then
    echo "Missing sources for variant '$variant'" >&2
    exit 1
  fi
  ln -sf "$(basename "$cpp_src")" "$BINDINGS_DIR/mettagrid_c.cpp"
  ln -sf "$(basename "$hpp_src")" "$BINDINGS_DIR/mettagrid_c.hpp"
}

run_variant() {
  local variant="$1"
  local run_name="${RUN_PREFIX}_${variant}"
  echo "==> Switching to variant: $variant (run=${run_name})"
  set_variant "$variant"
  echo "Symlink now points to: $(readlink "$BINDINGS_DIR/mettagrid_c.cpp")"
  AWS_PROFILE= AWS_DEFAULT_PROFILE= uv sync --reinstall-package mettagrid
  AWS_PROFILE= AWS_DEFAULT_PROFILE= METTA_TIMER_REPORT=1 uv run "${COMMON_ARGS[@]}" "run=${run_name}"
}

for variant in alt main; do
  run_variant "$variant"
  echo
  echo "==> Completed variant: $variant"
  echo
  sleep 2
done

# Restore alt variant as default
set_variant alt

echo "All variants completed."
