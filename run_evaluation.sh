#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
  # "daphne.3.navsequence.all.sky"
  "daphned.all.bucketed.sky.06-18"
  "daphned.navsequence.bucketed.sky.06-18"
  "daphned.object_use.bucketed.sky.06-18"
  "daphned.navigation.bucketed.sky.06-18"
  "daphne.all.bucketed.sky"
  "daphne.navigation.bucketed.skypilot"
  "daphne.navsequence.all2.withbackchain.sky"
)

for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

  echo "Running full sequence eval for policy $POLICY_URI"
  RANDOM_NUM=$((RANDOM % 1000))
  IDX="${IDX}_${RANDOM_NUM}"
  python3 -m tools.sim \
    sim=navigation \
    run=navigation$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/navigation_db2 \
    device=cpu

  python3 -m tools.sim \
    sim=memory \
    run=memory$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/memory_db2 \
    device=cpu \

  python3 -m tools.sim \
    sim=object_use \
    run=objectuse$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/objectuse_db2 \
    device=cpu \

  python3 -m tools.sim \
    sim=nav_sequence \
    run=nav_sequence$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/nav_sequence_db2 \
    device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db2 run=navigation_db ++dashboard.output_path=s3://softmax-public/policydash/navigation.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/memory_db2 run=memory_db ++dashboard.output_path=s3://softmax-public/policydash/memory.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/objectuse_db2 run=objectuse_db ++dashboard.output_path=s3://softmax-public/policydash/objectuse.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/nav_sequence_db2 run=nav_sequence_db ++dashboard.output_path=s3://softmax-public/policydash/navsequence.html

done
