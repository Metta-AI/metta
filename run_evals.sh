#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
<<<<<<< HEAD
#   "b.daphne.terrain_prioritized_styles_pretrained_r"
#   "b.daphne.terrain_prioritized_styles2"
#   "terrain_prioritized_styles_pretrained_mpmc"
#   "terrain_prioritized_styles_pretrained"
#   "b.terrain_prioritized_styles_nb"
#   "b.terrain_prioritized_styles_pretrained_nb"
#   "b.terrain_prioritized_styles"
#   "b.terrain_prioritized_styles_pretrained"
#   "b.georgedeane.terrain_multienv"
#   "b.daphne.terrain_multienv_3_no_blocks3"
#   "terrain_multienv_3_single_agent"
#   "b.daphne.terrain_multienv_prioritized_multienv_cylinders2"
#   "b.daphne.terrain_multienv_prioritized_multienv_cylinders"
#   "b.georgedeane.terrain_massive_empty_world_pretrained"
#   "b.georgedeane.terrain_extra_hard:v1"
#   "b.daphne.terrain_varied_cyl_lab_pretrained"
#   "b.daphne.terrain_prioritized_styles"
#   "b.daphne.terrain_prioritized_styles_pretrained"
#   "george_memory_pretrained"
#   "b.daphne.terrain_multiagent_48_norewardsharing"
#   "b.daphne.terrain_multiagent_24_norewardsharing"
#   "b.daphne.terrain_multiagent_24_rewardsharing"
#   "b.daphne.terrain_multiagent_48_rewardsharing"
  "objectuse_no_colors"
  "george_sequence_varied"
  "george_sequence_meta"
  "george3_multienv_noincrement"
  "george_sequence_incremental"
  "george_multienv_incremental"
=======
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
>>>>>>> ac18c213577a8394c8c1e2897b1cfb89b749b4ed
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
  # Use a fixed seed so tests are deterministic. It's okay to change this seed if you have
  # a reason -- we just don't want tests to fail spuriously.
  # Use device=cpu since we're probably on github. We should probably address this via
  # hardware=..., but for the most part this shouldn't matter for eval.
  MAYBE_SMOKE_TEST="+sim_job.smoke_test=True seed=31415 device=cpu"
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
    sim=simple_sequence \
    run=navigation$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/simple_sequence \

  python3 -m tools.sim \
    sim=extended_sequence \
    run=extended_sequence$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/extended_sequence \


  python3 -m tools.sim \
    sim=navigation \
    run=navigation$IDX \
    policy_uri=wandb://run/$POLICY_URI \
<<<<<<< HEAD
    sim_job.stats_db_uri=wandb://stats/navigation_new \
    

    python3 -m tools.dashboard +eval_db_uri=wandb://stats/simple_sequence run=simpleseq ++dashboard.output_path=s3://softmax-public/policydash/simpleseq.html \
    python3 -m tools.dashboard +eval_db_uri=wandb://stats/extended_sequence run=extended_sequence ++dashboard.output_path=s3://softmax-public/policydash/extended_sequence.html \
    python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_new run=navigation_new ++dashboard.output_path=s3://softmax-public/policydash/navigation_new.html \

=======
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

>>>>>>> ac18c213577a8394c8c1e2897b1cfb89b749b4ed
done
