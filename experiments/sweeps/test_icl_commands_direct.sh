#!/bin/bash
# Direct test commands for ICL BPTT horizon experiments
# Copy and run these individually

# Quick test with minimal steps (add trainer.total_timesteps=10000 for fast testing)

# Test 1: Baseline (horizon=256, batch=2064384, minibatch=16384)
./tools/run.py experiments.recipes.icl_resource_chain.train \
  --args run=icl_test_h256 \
  --overrides trainer.bptt_horizon=256 trainer.batch_size=2064384 trainer.minibatch_size=16384 wandb.enabled=false trainer.total_timesteps=10000 \
  --dry-run

# Test 2: Small horizon (horizon=128, batch=4128768, minibatch=16384)
./tools/run.py experiments.recipes.icl_resource_chain.train \
  --args run=icl_test_h128 \
  --overrides trainer.bptt_horizon=128 trainer.batch_size=4128768 trainer.minibatch_size=16384 wandb.enabled=false trainer.total_timesteps=10000 \
  --dry-run

# Test 3: Large horizon (horizon=512, batch=1032192, minibatch=16384)
./tools/run.py experiments.recipes.icl_resource_chain.train \
  --args run=icl_test_h512 \
  --overrides trainer.bptt_horizon=512 trainer.batch_size=1032192 trainer.minibatch_size=16384 wandb.enabled=false trainer.total_timesteps=10000 \
  --dry-run

# Test 4: Very large horizon (horizon=1024, batch=524288, minibatch=16384)
./tools/run.py experiments.recipes.icl_resource_chain.train \
  --args run=icl_test_h1024 \
  --overrides trainer.bptt_horizon=1024 trainer.batch_size=524288 trainer.minibatch_size=16384 wandb.enabled=false trainer.total_timesteps=10000 \
  --dry-run

# Example with alignment parameters (placeholders; require trainer-side support)
# ./tools/run.py experiments.recipes.icl_resource_chain.train \
#   --args run=icl_test_align \
#   --overrides trainer.bptt_horizon=512 trainer.batch_size=1032192 trainer.minibatch_size=16384 trainer.episode_alignment=pack trainer.pack_episodes=2 wandb.enabled=false trainer.total_timesteps=10000 \
#   --dry-run
