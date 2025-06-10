#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "b.rwalters.0605.nav.pr811.00"
    )


#TOKEN POLICIES
  #  "daphne.token.object_use"
  #  "daphne.token.navsequence"
 #   "daphne.token.nav"
    # "dd3_all_tokenized"
    # "dd2_curriculum_all_tokenized"
    # "dd_curriculum_navigation_tokenized"
for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/test \
        device=cpu \


    # python3 -m tools.sim \
    #     sim=memory \
    #     run=memory$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/memstatsdb \
    #     # device=cpu \

    # python3 -m tools.sim \
    #     sim=object_use \
    #     run=objectuse$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/objusestatsdb \
    #     # device=cpu \


    # python3 -m tools.sim \
    #     sim=nav_sequence \
    #     run=nav_sequence$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     sim_job.stats_db_uri=wandb://stats/navseqstatsdb \
    #     # device=cpu \

  #  python3 -m tools.sim \
  #       sim=multiagent \
  #       run=multi_agent$IDX \
  #       policy_uri=wandb://run/$POLICY_URI \
  #       sim_job.stats_db_uri=wandb://stats/stats_db1 \
  # #       # device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/test run=makestatsdbtest ++dashboard.output_path=s3://softmax-public/policydash/test.html
#   python3 -m tools.dashboard +eval_db_uri=wandb://stats/memstatsdb run=makestatsdbmem ++dashboard.output_path=s3://softmax-public/policydash/memstatsdb.html
#   python3 -m tools.dashboard +eval_db_uri=wandb://stats/objusestatsdb run=makestatsdbobj ++dashboard.output_path=s3://softmax-public/policydash/objusestatsdb.html
#   python3 -m tools.dashboard +eval_db_uri=wandb://stats/navseqstatsdb run=makestatsdbnavseq ++dashboard.output_path=s3://softmax-public/policydash/navseqstatsdb.html

done
