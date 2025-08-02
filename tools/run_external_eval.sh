#!/bin/bash
# Script to evaluate external PyTorch policies using existing Metta tools

# Set default values
POLICY_PATH="${1:-train_dir/metta_7-23/metta.pt}"
SIMULATION_SUITE="${2:-laser_tag}"
USER_CONFIG="${3:-relh}"

echo "Evaluating external policy..."
echo "Policy: $POLICY_PATH"
echo "Simulation suite: $SIMULATION_SUITE"
echo "User config: $USER_CONFIG"
echo ""

# Option 1: Use the new standalone evaluation script
echo "=== Option 1: Using standalone evaluation script ==="
echo "Run this command:"
echo "python tools/eval_external_policy.py --policy-path $POLICY_PATH --simulation-suite $SIMULATION_SUITE --save-replays"
echo ""

# Option 2: Use the existing sim.py with your config
echo "=== Option 2: Using sim.py with user config ==="
echo "First, make sure your policy path is correct in configs/user/$USER_CONFIG.yaml"
echo "Then run:"
echo "python tools/sim.py --multirun hydra/launcher=local cmd=sim user=$USER_CONFIG sim_job.simulation_suite.name=$SIMULATION_SUITE"
echo ""

# Option 3: Direct evaluation with custom config
echo "=== Option 3: Direct evaluation command ==="
echo "python tools/sim.py cmd=sim user=$USER_CONFIG policy_uri=pytorch://$POLICY_PATH sim_job.simulation_suite.name=$SIMULATION_SUITE"
echo ""

# Option 4: Using your config for training task evaluation
echo "=== Option 4: Evaluate on training curriculum ==="
echo "python tools/sim.py cmd=sim user=$USER_CONFIG eval.policy_uri=pytorch://$POLICY_PATH"
