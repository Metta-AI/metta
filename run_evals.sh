#!/bin/bash

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
  "b.daphne.navigation0"
)
MESSAGE="Running full sequence eval"
MAYBE_SMOKETEST=""

if [ "$1" = "smoketest" ]; then
  # This should be a policy that gets a known score, so we can check
  # that the eval is working.
  POLICIES=("b.daphne.navigation0:v12")
  MAYBE_SMOKETEST="+sim_job.maybe_smoketest=True"
  MESSAGE="Running smoketest eval"
fi

for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

  echo "$MESSAGE for policy $POLICY_URI"
  RANDOM_NUM=$((RANDOM % 1000))
  IDX="${IDX}_${RANDOM_NUM}"
  python3 -m tools.sim \
    sim=navigation \
    run=navigation$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    +eval_db_uri=wandb://artifacts/navigation_db \
    $MAYBE_SMOKETEST

done
