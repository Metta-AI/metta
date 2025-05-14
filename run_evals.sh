#!/bin/bash

POLICIES=(
    "b.daphne.navigation0"
    # "b.daphne.navigation1"
    # "b.daphne.navigation4"
    # "b.daphne.navigation3"
    # "b.daphne.navigation4"
)

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"

    python3 -m tools.sim \
        run=navigation$IDX \
        sim=navigation \
        sim_job.replay_dir="s3://softmax-public/replays/evals" \
        sim_job.policy_uris=\[wandb://run/$POLICY_URI\] \
        sim_job.stats_db_uri="wandb://stats/pasha_wanderout_db" \
        +eval_db_uri=wandb://stats/pasha_wanderout_db \
        ++device=cpu

    python3 -m tools.dashboard +eval_db_uri=wandb://stats/pasha_wanderout_db run=pashanavigation ++dashboard.output_path=s3://softmax-public/policydash/pasha_wanderout_new.html
done