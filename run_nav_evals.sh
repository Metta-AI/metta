#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
  "daphne_navigation_bucketed"
  "daphne_nav_bucketed"
  "daphne.optimize_nav_aws"
  "daphne.optimize_nav"
  "dd_navigation_curriculum"
  "dd.nav_optimized"
  "dd.nav_optimized_bucket"
  "daphne.navopt_devbox"
  "daphne.navbucketedopt_devbox"
  "dd.navbucketed_sparser"
  "dd.navbucketed_2"
  "daphne.navnoterrain"


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
    sim_job.stats_db_uri=wandb://stats/navigation_db \
    device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db run=navigation_db ++dashboard.output_path=s3://softmax-public/policydash/navigation.html

done
