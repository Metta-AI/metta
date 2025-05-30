#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
  "sasmith_20250529_ablate_1_hot_types.3"
)
MESSAGE="Running full sequence eval"
MAYBE_SMOKE_TEST=""

if [ "$1" = "smoke_test" ]; then
  # If you're updating this smoke test:
  #   ... because you changed code in a way that invalidates old policies, please train a new policy
  #       that scores well enough on existing evals, and add it here.
  #   ... because you're adding a new eval family on which we can score well, please add a new policy
  #       that scores well on that eval family, and add it here.
  #   ... because you're adding a new eval family on which we can't score well, please add the new eval
  #       family after the smoke test terminates.
  POLICIES=("sasmith_20250529_ablate_1_hot_types.3")
  # We try to be as deterministic as possible, but this turns out to be less deterministic than we'd like.
  # Use device=cpu since we're probably on github. We should probably address this via
  # hardware=..., but for the most part this shouldn't matter for eval.
  MAYBE_SMOKE_TEST="+sim_job.smoke_test=True seed=31415 torch_deterministic=True device=cpu"
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

  if [ -n "$MAYBE_SMOKE_TEST" ]; then
    continue
  fi
  # Tests below this line aren't part of smoke tests, since we either
  # aren't scoring well enough to include them, or have some other reason.

  # We also don't want to upload to dashboards for these.

  python3 -m tools.sim \
    sim=memory \
    run=memory$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/memory_db \
    $MAYBE_SMOKE_TEST

  python3 -m tools.sim \
    sim=object_use \
    run=objectuse$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/objectuse_db \
    $MAYBE_SMOKE_TEST

done
