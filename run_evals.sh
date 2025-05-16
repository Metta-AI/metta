#!/bin/bash

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "daveey.dist.2x4"
    "navigation_training:v35"
    "b.daphne.navigation0"
    "daphne_navigation_train"
    "b.daphne.navigation1"
    "b.daphne.navigation3"
    "b.daphne.navigation4"
    "gd2_sharing24_03"
    "gd2_sharing24_06"
    "gd2_sharing_48"
    "gd2_sharing_24"
    "gd2_sharing48_03"
    "gd2_sharing48_06"
    "MRM_test_mettabox"
    "georged_48_no_sharing"
    "georged_24_no_sharing"
    "dd_object_use_easy2"
    "daphne.3object_use_no_colors"
    "daphne.3object_use_colors"
    "daphne.2object_use_colors_pretrained"
    "b.daphne.USER.navigation_before_refactor"

    "b.daphne.object_use_colored_converters"
    "b.daphne.object_use_onlyred"
    "b.daphne.object_use_colored_converters_ent0.05"
    "b.daphne.object_use_onlyred_ent0.05"

    "b.daphne.object_use_colored_converters2"
    "b.daphne.object_use_onlyred2"
    "b.daphne.object_use_colored_converters_ent0.052"
    "b.daphne.object_use_onlyred_ent0.052"

    "b.georgedeane.george_sequence_no_increment"
    "b.georgedeane.george_sequence_incremental"
    "george_sequence_incremental"
    "george2_multienv_noincrement"
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
  POLICIES=("b.daphne.navigation0:v12")
  # Use a fixed seed to tests are deterministic. It's okay to change this seed if you have
  # a reason -- we just don't want tests to fail spuriously.
  MAYBE_SMOKE_TEST="+sim_job.smoke_test=True seed=31415"
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
