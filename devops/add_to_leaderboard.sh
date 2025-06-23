#!/bin/bash

# Usage function for better help messages
usage() {
  echo "Usage: $0 -r RUN_NAME [-w WANDB_PATH] [additional Hydra overrides]"
  echo "  -r RUN_NAME     Your run name (e.g., b.$USER.test_run)"
  echo "  -w WANDB_PATH   Optional: Full wandb path if different from auto-generated"
  echo ""
  echo "  Any additional arguments will be passed directly to the Python commands"
  echo "  Example: $0 -r b.$USER.test_run +hardware=macbook"
  exit 1
}

# Initialize variables
RUN_NAME=""
WANDB_PATH=""
ADDITIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -r | --run)
      RUN_NAME="$2"
      shift 2
      ;;
    -w | --wandb)
      WANDB_PATH="$2"
      shift 2
      ;;
    -h | --help)
      usage
      ;;
    *)
      # Collect additional arguments
      ADDITIONAL_ARGS="$ADDITIONAL_ARGS $1"
      shift
      ;;
  esac
done

# Check if run name is provided
if [ -z "$RUN_NAME" ]; then
  echo "Error: Run name is required"
  usage
fi

# Auto-generate wandb path if not provided
if [ -z "$WANDB_PATH" ]; then
  WANDB_PATH="wandb://run/$RUN_NAME"
fi

echo "Adding policy to eval leaderboard with run name: $RUN_NAME"
echo "Using policy URI: $WANDB_PATH"
if [ ! -z "$ADDITIONAL_ARGS" ]; then
  echo "Additional arguments: $ADDITIONAL_ARGS"
fi

# Step 1: Verifying policy exists on wandb
echo "Step 1: Verifying policy exists on wandb..."
# Add a check here if needed to verify the policy exists on wandb

# Step 2: Run the simulation
echo "Step 2: Running simulation..."
SIM_CMD="python3 -m tools.sim sim=navigation run=\"$RUN_NAME\" policy_uri=\"$WANDB_PATH\" +eval_db_uri=wandb://artifacts/navigation_db $ADDITIONAL_ARGS"
echo "Executing: $SIM_CMD"
eval $SIM_CMD

# Check if the sim was successful
if [ $? -ne 0 ]; then
  echo "Error: Simulation failed. Exiting."
  exit 1
fi

# Step 3: Analyze and update dashboard
echo "Step 3: Analyzing results and updating dashboard..."
ANALYZE_CMD="python3 -m tools.analyze run=analyze +eval_db_uri=wandb://artifacts/navigation_db analyzer.output_path=s3://softmax-public/policydash/dashboard.html +analyzer.num_output_policies=\"all\" $ADDITIONAL_ARGS"
echo "Executing: $ANALYZE_CMD"
eval $ANALYZE_CMD

if [ $? -ne 0 ]; then
  echo "Error: Analysis failed. Exiting."
  exit 1
fi

echo "Successfully added policy to leaderboard and updated dashboard!"
echo "Dashboard URL: https://softmax-public.s3.amazonaws.com/policydash/dashboard.html"
