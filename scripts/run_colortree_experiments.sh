#!/usr/bin/env bash
set -euo pipefail

# Ensure local toolchain (optional; adapt if you use uv wrappers elsewhere)
export PATH="/opt/homebrew/bin:$PATH"
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Load SkyPilot helpers (defines `lt`) if available
if [ -f devops/skypilot/setup_shell.sh ]; then
  # shellcheck disable=SC1091
  source devops/skypilot/setup_shell.sh
fi

# Common settings
SIM="colortree"
USER_OVR="+user=jacke"
CURR_FIXED="env/mettagrid/curriculum/colortree_easy_fixed"
CURR_RANDOM="env/mettagrid/curriculum/colortree_easy_random"

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
  if command -v lt >/dev/null 2>&1; then
    echo "lt run=$RUN_NAME $*"
    lt run="$RUN_NAME" "$@"
  else
    echo "./devops/skypilot/launch.py train run=$RUN_NAME $*"
    ./devops/skypilot/launch.py train run="$RUN_NAME" "$@"
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
        RUN_NAME="$USER.colortree_${ARCH}_jacke_2color_${MODE}_${SEQ_LABEL}_bptt${BPTT}_easy_lstm_${STAMP}"

        if [ "$ARCH" = "fast" ]; then
          run_train "$RUN_NAME" \
            trainer.curriculum="$CURR_FIXED" \
            sim="$SIM" \
            $USER_OVR \
            agent=fast \
            trainer.bptt_horizon="$BPTT" \
            "game.actions.color_tree.target_sequence=$SEQ_STR" \
            "game.actions.color_tree.trial_sequences=$TRIAL_SEQ_STR" \
            game.actions.color_tree.num_trials=1 \
            "game.actions.color_tree.reward_mode=$MODE"
        else
          run_train "$RUN_NAME" \
            trainer.curriculum="$CURR_FIXED" \
            sim="$SIM" \
            $USER_OVR \
            agent._target_=metta.agent.external.lstm_transformer.Recurrent \
            trainer.bptt_horizon="$BPTT" \
            "game.actions.color_tree.target_sequence=$SEQ_STR" \
            "game.actions.color_tree.trial_sequences=$TRIAL_SEQ_STR" \
            game.actions.color_tree.num_trials=1 \
            "game.actions.color_tree.reward_mode=$MODE"
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
      RUN_NAME="$USER.colortree_${ARCH}_jacke_2color_${MODE}_random_bptt${BPTT}_easy_lstm_${STAMP}"

      if [ "$ARCH" = "fast" ]; then
        run_train "$RUN_NAME" \
          trainer.curriculum="$CURR_RANDOM" \
          sim="$SIM" \
          $USER_OVR \
          agent=fast \
          trainer.bptt_horizon="$BPTT" \
          "game.actions.color_tree.reward_mode=$MODE"
      else
        run_train "$RUN_NAME" \
          trainer.curriculum="$CURR_RANDOM" \
          sim="$SIM" \
          $USER_OVR \
          agent._target_=metta.agent.external.lstm_transformer.Recurrent \
          trainer.bptt_horizon="$BPTT" \
          "game.actions.color_tree.reward_mode=$MODE"
      fi
    done
  done
done

echo "\nAll jobs launched."


