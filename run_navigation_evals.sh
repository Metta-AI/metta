#!/bin/bash

# Define the list of policy URIs
POLICIES=(
  "b.daphne.varied_terrain_lessactions"
  "b.daphne.terrain_multienv_singleA_lessactions"
)


for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        eval=navigation \
        run=navigation$IDX \
        eval.policy_uri=wandb://run/$POLICY_URI \
        eval_db_uri=wandb://artifacts/navigation_db \
        # device=cpu
done
