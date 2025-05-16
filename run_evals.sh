#!/bin/bash

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
  "b.daphne.navigation0"
)
MESSAGE="Running full sequence eval"
MAYBE_SMOKE_TEST=""

if [ "$1" = "smoke_test" ]; then
  # This should be a policy that gets a known score, so we can check
  # that the eval is working.
  POLICIES=("b.daphne.navigation0:v12")
  MAYBE_SMOKE_TEST="+sim_job.smoke_test=True +sim_job.smoke_test_min_reward=0.9 seed=31415926535"
  MESSAGE="Running smoke test eval"
elif [ -n "$1" ]; then
  echo "Invalid argument: $1"
  exit 1
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
    $MAYBE_SMOKE_TEST

done
