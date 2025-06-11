#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "dd_all_tokenized_sweep.r.1"
    "dd_all_tokenized_sweep.r.0"
    "gd_pure_seq_backchain_tokenized"
    "gd_backchain_full_extended_tokenized2"
    "gd_backchain_scratch_hard_tokenized"
    "gd_backchain_in_context_tokenized"
    "gd_pure_mem_backchain_tokenized"
    "gd_kitchen_hard_pretrained_tokenized"
    "dd2_curriculum_all_tokenized"
    "dd3_all_tokenized"
    "dd_curriculum_navigation_tokenized"
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
        sim_job.stats_db_uri=wandb://stats/navstatsdb \
    #     # device=cpu \


    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/memstatsdb \
        # device=cpu \

    python3 -m tools.sim \
        sim=object_use \
        run=objectuse$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/objusestatsdb \
        # device=cpu \


    python3 -m tools.sim \
        sim=nav_sequence \
        run=nav_sequence$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/navseqstatsdb \
        # device=cpu \

  #  python3 -m tools.sim \
  #       sim=multiagent \
  #       run=multi_agent$IDX \
  #       policy_uri=wandb://run/$POLICY_URI \
  #       sim_job.stats_db_uri=wandb://stats/stats_db1 \
  # #       # device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navstatsdb run=makestatsdbnav ++dashboard.output_path=s3://softmax-public/policydash/navstatsdb.html
  python3 -m tools.dashboard +eval_db_uri=wandb://stats/memstatsdb run=makestatsdbmem ++dashboard.output_path=s3://softmax-public/policydash/memstatsdb.html
  python3 -m tools.dashboard +eval_db_uri=wandb://stats/objusestatsdb run=makestatsdbobj ++dashboard.output_path=s3://softmax-public/policydash/objusestatsdb.html
  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navseqstatsdb run=makestatsdbnavseq ++dashboard.output_path=s3://softmax-public/policydash/navseqstatsdb.html

done
