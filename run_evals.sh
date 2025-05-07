#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "multiagent_objects_nocolors2"
    "multienv_nocolors2"
    "multiagent_colors"
    "multiagent_objects"
    "george_24_nosharing_pretrained"
    "b.daphne.multiagent_mixed_noinitialheart"
    "b.daphne.multiagent_nc_pretrained"
    "b.daphne.multiagent_c_pretrained"
    "b.daphne.multiagent_mix_pretrained"
    "b.daphne.multiagent_nc"
    "b.daphne.multiagent_c"
    "b.daphne.multiagent_mix"
    "george_sharing_24"
    "george_sharing_24_sharing_06"
    "george_sharing_24_sharing03"
    "george_24_no_sharing"
    "george_sharing_48_range_pretrained"
    "george_sharing48_sharing06_pretrained"
    "george_sharing_24_range_pretrained"
    "george_sharing24_sharing06_pretrained"
    "george_sharing24_sharing03_pretrained"
    "george_48_no_sharing_prterained"
    "george_24_no_sharing_pretrained"
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
done
