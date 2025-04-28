#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "b.georgedeane.terrain_multienv"
     "b.daphne.terrain_multienv_3_no_blocks3"
     "terrain_multienv_3_single_agent"
	 "b.daphne.terrain_multienv_prioritized_multienv_cylinders2"
	 "b.daphne.terrain_multienv_prioritized_multienv_cylinders"
     "b.georgedeane.terrain_massive_empty_world_pretrained"
     "b.georgedeane.terrain_extra_hard:v1"
     "b.daphne.terrain_varied_cyl_lab_pretrained"
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
done
