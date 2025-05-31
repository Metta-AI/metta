#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "b.daphne.navsequence_curriculum"
    "b.daphne.navigation_curriculum"
    "daphne_navseq_curriculum_SA"
    "b.daphne.navsequence.sweep.r.0"
    "b.daphne.multiagent_curriculum"
    "b.daphne.objectuse_sweep.r.3"
    "b.daphne.objectuse_curriculum"
    "b.daphne.navigation_sweep.r.0"
    "b.daphne.navsequence_sweep_sa.r.0"
    )


#TOKEN POLICIES
  #  "daphne.token.object_use"
  #  "daphne.token.navsequence"
 #   "daphne.token.nav"

for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/stats_db \
        device=cpu \

    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/stats_db \
        device=cpu \

    python3 -m tools.sim \
        sim=object_use \
        run=objectuse$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/stats_db \
        device=cpu \


    python3 -m tools.sim \
        sim=nav_sequence \
        run=nav_sequence$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/stats_db \
        device=cpu \

   python3 -m tools.sim \
        sim=multiagent \
        run=multi_agent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/stats_db \
        device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/stats_db run=navigation_db2 ++dashboard.output_path=s3://softmax-public/policydash/results.html

done
