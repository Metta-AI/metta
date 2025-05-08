#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "b.daphne.multiagent_mix3"
    "b.daphne.multiagent_c3"
    "b.daphne.multiagent_nc3"
    "b.daphne.multiagent_nc1"
    "b.daphne.multiagent_mix1"
    "b.daphne.multiagent_c1"
    "george_sharing_48"
    "george_sharing48_06"
    "george_sharing48_03"
    "george_48_no_sharing"
    "b.daphne.navigation4"
    "b.daphne.navigation1"
    "b.daphne.navigation3"
    "b.daphne.navigation0"
    "b.daphne.navigation5"
)

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"


    python3 -m tools.sim \
        sim=multiagent \
        run=multiagent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/multiagent_db \
        sim.num_envs=20 \
        sim.num_episodes=20

    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/memory_db \

    python3 -m tools.sim \
        sim=cards \
        run=cards$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/cards_db \

    python3 -m tools.sim \
        sim=object_use \
        run=object_use$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/object_use_db \
        sim.num_envs=20 \
        sim.num_episodes=20

    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/navigation_db \
        sim.num_envs=20 \
        sim.num_episodes=20
done
