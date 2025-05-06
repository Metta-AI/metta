#!/bin/bash

# Define the list of policy URIs
POLICIES=(
<<<<<<< Updated upstream
    "b.daphne.navigation_multiagent_24_rewardsharing_maxinv"
    "b.daphne.navigation_multiagent_48_rewardsharing_maxinv"
    "b.daphne.navigation_multiagent_24_norewardsharing_maxinv"
    "b.daphne.navigation_multiagent_48_norewardsharing_maxinv"
    "daphne.navigation"
    "b.daphne.navigation"
    "b.daphne.navigation2"
=======
    "daphne.navigation:v127"
>>>>>>> Stashed changes
)

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
<<<<<<< Updated upstream
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/test_navigation \
        ++sim_job.selector_type=top \
        # ++sim_job.metric=navigation_score
=======
        policy_uri=wandb://run/daphne.navigation:v127 \
        +eval_db_uri=wandb://artifacts/test_navigation_main
>>>>>>> Stashed changes

    # python3 -m tools.sim \
    #     sim=multiagent \
    #     run=multiagent$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
<<<<<<< Updated upstream
    #     +eval_db_uri=wandb://artifacts/multiagent_db \
    #     ++sim_job.selector_type=top \
    #     ++sim_job.metric=multiagent_score

=======
    #     +eval_db_uri=wandb://artifacts/multiagent_db
>>>>>>> Stashed changes

    # python3 -m tools.sim \
    #     sim=memory \
    #     run=memory$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
<<<<<<< Updated upstream
    #     +eval_db_uri=wandb://artifacts/memory_db \
    #     ++sim_job.selector_type=top \
    #     ++sim_job.metric=memory_score

    # python3 -m tools.sim \
    #     sim=cards \
    #     run=cards$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     +eval_db_uri=wandb://artifacts/cards_db \
    #     ++sim_job.selector_type=latest

    # python3 -m tools.sim \
    #     sim=object_use \
    #     run=object_use$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     +eval_db_uri=wandb://artifacts/object_use_db \
    #     ++sim_job.selector_type=latest
=======
    #     +eval_db_uri=wandb://artifacts/memory_db
>>>>>>> Stashed changes
done
