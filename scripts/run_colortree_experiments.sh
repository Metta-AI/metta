#!/usr/bin/env bash
set -euo pipefail

# Ensure local toolchain (optional; adapt if you use uv wrappers elsewhere)
export PATH="/opt/homebrew/bin:$PATH"
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Optional dump mode: write commands to a file instead of running them
OUT_FILE="${DUMP_FILE:-}"
if [[ "${1:-}" == "--dump" ]]; then
  OUT_FILE="${2:-colortree_runs.txt}"
  shift 2
fi

# Load SkyPilot helpers (defines `lt`) if available
if [ -f devops/skypilot/setup_shell.sh ]; then
  # shellcheck disable=SC1091
  set +e
  source devops/skypilot/setup_shell.sh
  set -e
fi

# Common settings
SIM="colortree"
USER_OVR="+user=jacke"
# Fixed uses the single environment; random uses the curriculum
ENV_FIXED="/env/mettagrid/colortree_easy"
CURR_RANDOM="/env/mettagrid/curriculum/colortree_easy_random"

# Resolve a safe user name for run prefixes without relying on $USER being exported
RUN_USER="${USER:-$(id -un 2>/dev/null || whoami)}"

# Architectures: fast (config file) and recurrent (explicit target)
ARCHS=(fast recurrent)
BPTTS=(16 64)
MODES=(precise partial dense)

# Patterns for fixed curriculum (macOS /bin/bash compatible mapping)
get_seq_by_label() {
  case "$1" in
    1111)
      echo "[1,1,1,1]"
      ;;
    1010)
      echo "[1,0,1,0]"
      ;;
    *)
      echo "Unknown sequence label: $1" 1>&2
      return 1
      ;;
  esac
}

run_train() {
  local RUN_NAME="$1"; shift
  echo
  echo "=== Launch: $RUN_NAME ==="
  if [[ -n "$OUT_FILE" ]]; then
    local cmd=(./devops/skypilot/launch.py train --skip-git-check run="$RUN_NAME" "$@")
    {
      printf '# %s\n' "$RUN_NAME"
      printf '%q ' "${cmd[@]}"; printf '\n'
    } >> "$OUT_FILE"
    echo "(dump) wrote command to $OUT_FILE"
    return 0
  else
    if command -v lt >/dev/null 2>&1; then
      echo "lt --skip-git-check run=$RUN_NAME $*"
      set +e
      lt --skip-git-check run="$RUN_NAME" "$@"
      local job_exit=$?
      set -e
    else
      echo "./devops/skypilot/launch.py train --skip-git-check run=$RUN_NAME $*"
      set +e
      ./devops/skypilot/launch.py train --skip-git-check run="$RUN_NAME" "$@"
      local job_exit=$?
      set -e
    fi
    if [ ${job_exit:-0} -ne 0 ]; then
      echo "Job '$RUN_NAME' exited with code ${job_exit}. Continuing."
    fi
  fi
}

timestamp() {
  date +%Y%m%d_%H%M%S
}

# 1) Fixed patterns: 1111 and 1010 across BPTT, architectures, and reward modes
for ARCH in "${ARCHS[@]}"; do
  for MODE in "${MODES[@]}"; do
    for SEQ_LABEL in 1111 1010; do
      SEQ_STR=$(get_seq_by_label "$SEQ_LABEL")
      # Trial sequences must match target length to satisfy config validation
      TRIAL_SEQ_STR="[$SEQ_STR]"    # e.g., [[1,0,1,0]]

      for BPTT in "${BPTTS[@]}"; do
        STAMP=$(timestamp)
        RUN_NAME="$RUN_USER.colortree_${ARCH}_jacke_2color_${MODE}_${SEQ_LABEL}_bptt${BPTT}_easy_lstm_${STAMP}"

        if [ "$ARCH" = "fast" ]; then
          run_train "$RUN_NAME" \
            +trainer.env="$ENV_FIXED" \
            sim="$SIM" \
            $USER_OVR \
            agent=fast \
            trainer.bptt_horizon="$BPTT" \
            "+trainer.env_overrides.game.actions.color_tree.target_sequence=$SEQ_STR" \
            "+trainer.env_overrides.game.actions.color_tree.trial_sequences=$TRIAL_SEQ_STR" \
            +trainer.env_overrides.game.actions.color_tree.num_trials=1 \
            "+trainer.env_overrides.game.actions.color_tree.reward_mode=$MODE"
        else
          run_train "$RUN_NAME" \
            +trainer.env="$ENV_FIXED" \
            sim="$SIM" \
            $USER_OVR \
            agent._target_=metta.agent.external.lstm_transformer.Recurrent \
            trainer.bptt_horizon="$BPTT" \
            "+trainer.env_overrides.game.actions.color_tree.target_sequence=$SEQ_STR" \
            "+trainer.env_overrides.game.actions.color_tree.trial_sequences=$TRIAL_SEQ_STR" \
            +trainer.env_overrides.game.actions.color_tree.num_trials=1 \
            "+trainer.env_overrides.game.actions.color_tree.reward_mode=$MODE"
        fi
      done
    done
  done
done

# 2) Random curriculum across BPTT, architectures, and reward modes (no fixed sequence overrides)
for ARCH in "${ARCHS[@]}"; do
  for MODE in "${MODES[@]}"; do
    for BPTT in "${BPTTS[@]}"; do
      STAMP=$(timestamp)
      RUN_NAME="$RUN_USER.colortree_${ARCH}_jacke_2color_${MODE}_random_bptt${BPTT}_easy_lstm_${STAMP}"

      if [ "$ARCH" = "fast" ]; then
        run_train "$RUN_NAME" \
          trainer.curriculum="$CURR_RANDOM" \
          sim="$SIM" \
          $USER_OVR \
          agent=fast \
          trainer.bptt_horizon="$BPTT" \
          "+trainer.env_overrides.game.actions.color_tree.reward_mode=$MODE"
      else
        run_train "$RUN_NAME" \
          trainer.curriculum="$CURR_RANDOM" \
          sim="$SIM" \
          $USER_OVR \
          agent._target_=metta.agent.external.lstm_transformer.Recurrent \
          trainer.bptt_horizon="$BPTT" \
          "+trainer.env_overrides.game.actions.color_tree.reward_mode=$MODE"
      fi
    done
  done
done

printf "\nAll jobs launched.\n"


