#!/bin/bash
# Test commands for ICL BPTT horizon sweep
# Run these individually to verify they work before launching the full sweep

echo "Test commands for ICL BPTT horizon experiments"
echo "=============================================="
echo ""

# Test 1: Baseline configuration (256 horizon)
echo "Test 1: Baseline (horizon=256, batch=2064384)"
echo "./tools/run.py experiments.recipes.icl_resource_chain.train \\\n+  --args run=icl_test_h256 \\\n+  --overrides trainer.bptt_horizon=256 trainer.batch_size=2064384 trainer.minibatch_size=16384 wandb.enabled=false \\\n+  --dry-run"
echo ""

# Test 2: Small horizon (128)
echo "Test 2: Small horizon (horizon=128, batch=4128768)"
echo "./tools/run.py experiments.recipes.icl_resource_chain.train \\\n+  --args run=icl_test_h128 \\\n+  --overrides trainer.bptt_horizon=128 trainer.batch_size=4128768 trainer.minibatch_size=16384 wandb.enabled=false \\\n+  --dry-run"
echo ""

# Test 3: Large horizon (512)
echo "Test 3: Large horizon (horizon=512, batch=1032192)"
echo "./tools/run.py experiments.recipes.icl_resource_chain.train \\\n+  --args run=icl_test_h512 \\\n+  --overrides trainer.bptt_horizon=512 trainer.batch_size=1032192 trainer.minibatch_size=16384 wandb.enabled=false \\\n+  --dry-run"
echo ""

# Test 4: Very large horizon (1024)
echo "Test 4: Very large horizon (horizon=1024, batch=516096)"
echo "./tools/run.py experiments.recipes.icl_resource_chain.train \\\n+  --args run=icl_test_h1024 \\\n+  --overrides trainer.bptt_horizon=1024 trainer.batch_size=524288 trainer.minibatch_size=16384 wandb.enabled=false \\\n+  --dry-run"
echo ""

echo "=============================================="
echo "To run a test, copy and paste the command (without the echo quotes)"
echo ""
echo "You can also run with minimal steps for quick testing:"
echo "Add to --overrides: trainer.total_timesteps=10000"
echo ""
echo "For dry run (no actual training):"
echo "Use --dry-run; or add trainer.total_timesteps=0 to --overrides"
