#!/bin/bash
set -eou pipefail

CONTAINER="${1}"

timeout 600 docker run --rm --gpus all "$CONTAINER" "
cd /workspace/metta && \
./tools/run.py experiments.recipes.arena.train \
  run=test_minimal \
                trainer.total_timesteps=10000 \
                trainer.checkpoint.checkpoint_interval=0 \
                evaluator.epoch_interval=0 2>&1 | tee /tmp/train.log && \
echo '=== Training completed, checking for success ===' && \
if grep -q 'Training complete!' /tmp/train.log && grep -q 'ksps' /tmp/train.log; then \
  echo '=== TRAINING VERIFIED SUCCESSFULLY ==='; \
  grep 'ksps' /tmp/train.log | tail -1; \
  exit 0; \
else \
  echo '=== ERROR: Training did not complete properly ==='; \
  tail -20 /tmp/train.log; \
  exit 1; \
fi
"

echo "Test passed for $CONTAINER"
