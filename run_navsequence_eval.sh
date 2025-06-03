#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "dd.2objectuse_curriculum"
    "dd_navigation_curriculum"
    "dd_navsequence_memory_pretrained"
    "dd_navsequence_memory"
    "dd_navsequence_all_pretrained"
    "dd_navsequence_all"
    "dd_multiagent"
    "dd_multiagent_pretrained"
    "b.dd.curriculum_all"
    "dd.curriculum_ALL"
    "dd.curriculum_all_reg0.1"
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
    # python3 -m tools.sim \
    #     sim=navigation \
    #     run=navigation$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/stats_db \
    #     device=cpu \


    # python3 -m tools.sim \
    #     sim=memory \
    #     run=memory$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/stats_db \
    #     device=cpu \

    # python3 -m tools.sim \
    #     sim=object_use \
    #     run=objectuse$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/stats_db \
    #     device=cpu \


    python3 -m tools.sim \
        sim=nav_sequence \
        run=nav_sequence$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/stats_db \
        # device=cpu \

   python3 -m tools.sim \
        sim=multiagent \
        run=multi_agent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/stats_db \
        # device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/stats_db run=makestatsdb ++dashboard.output_path=s3://softmax-public/policydash/results.html

done
