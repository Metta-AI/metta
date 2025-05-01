#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "b.daphne.terrain_prioritized_styles_pretrained_r"
    "terrain_prioritized_styles_pretrained"
    "terrain_prioritized_styles_pretrained_mpmc"
    "b.terrain_prioritized_styles_pretrained_nb"
    "b.terrain_prioritized_styles"
    "b.georgedeane.terrain_multienv"
     "terrain_multienv_3_single_agent"
	 "b.daphne.terrain_multienv_prioritized_multienv_cylinders"
     "b.georgedeane.terrain_extra_hard:v1"
     "b.daphne.terrain_prioritized_styles"
     "b.daphne.terrain_prioritized_styles_pretrained"
     "george_memory_pretrained"
     "b.daphne.terrain_multiagent_48_norewardsharing"
     "b.daphne.terrain_multiagent_24_norewardsharing"
     "b.daphne.terrain_multiagent_24_rewardsharing"
     "b.daphne.terrain_multiagent_48_rewardsharing"
     "b.daphne.terrain_multiagent_48_norewardsharing_maxinv"
     "b.daphne.terrain_multiagent_24_norewardsharing_maxinv"
     "b.daphne.terrain_multiagent_24_rewardsharing_maxinv"
     "b.daphne.terrain_multiagent_48_rewardsharing_maxinv"
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
        +eval_db_uri=wandb://artifacts/navigation_db \
        ++sim_job.selector_type=top \
        ++sim_job.metric=navigation_score

    python3 -m tools.sim \
        sim=multiagent \
        run=multiagent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/multiagent_db \
        ++sim_job.selector_type=top \
        ++sim_job.metric=multiagent_score


    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/memory_db \
        ++sim_job.selector_type=top \
        ++sim_job.metric=memory_score

    python3 -m tools.sim \
        sim=cards \
        run=cards$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/cards_db \
        ++sim_job.selector_type=latest
done
