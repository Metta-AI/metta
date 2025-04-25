#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    # "navigation_infinite_cooldown_sweep"
    # "navigation_infinite_cooldown_sweep.r.0"
    # "b.daveey.t.8.rdr9.3"
    # "b.daveey.t.4.rdr9.3"
    # "b.daveey.t.8.rdr9.mb2.1"
    # "b.daphne.navigation_terrain_training"
    # "b.georgedeane.terrain_newmaps_pretrained"
    # "b.georgedeane.terrain_newmaps"
    # "b.daphne.terrain_newmaps2"
    # "terrain_training_newmaps2"
    # "b.daphne.navigation_terrain_training_v2"
    # "b.georgedeane.navigation_terrain_training_v2"
    # "terrain_training_multienv2"
    # "b.daphne.cylinder_run"
    # "b.georgedeane.terrain_multienv_labyrinth_pretrained_DR"
    # "b.daphne.terrain_multienv_april18"
    # "b.daphne.terrain_varied_cyl_lab"
    "b.georgedeane.terrain_multienv"
    # "b.daphne.terrain_varied_cyl_lab_pretrained"
    # "b.daphne.terrain_multienv_altar_no_resets"
    # "terrain_multienv_3_single_agent"
    # "b.daphne.terrain_multienv_singleA_kitchensink"
    # "b.daphne.terrain_multienv_singleA_withgenerators"
    # "b.daphne.terrain_multienv_singleA_altar_resets"
     "b.daphne.terrain_multienv_singleA_50hearts"
     "b.georgedeane.terrain_multienv_uniform"
    "b.georgedeane.terrain_multienv_fromscratch"
    # "b.daphne.terrain_multienv_kitchensinkwc"
    # "b.daphne.terrain_multienv_prioritized_multienv_cylinders"
    "b.daphne.terrain_multienv_prioritized_multienv_cylinders2"
    # "b.daphne.terrain_multienv_prioritized_george_maps"
    "b.georgedeane.terrain_multienv_unstable_pretrained_mb3"
    # "b.georgedeane.terrain_multienv_homogenous_pretrained"
    "b.georgedeane.terrain_multienv_stable_pretrained_mb4"
    # "b.daphne.terrain_multienv_3_no_blocks3"
     "b.daphne.terrain_multienv_3_no_blocks4"
     "terrain_multienv_3_single_agent"
	 "b.daphne.terrain_multienv_prioritized_multienv_cylinders2"
	 "b.daphne.terrain_multienv_prioritized_multienv_cylinders"
	 "b.daphne.terrain_varied_cyl_lab_pretrained"
	 "b.georgedeane.terrain_multienv_unstable_pretrained_mb3"

    #  "b.georgedeane.terrain_massive_empty_world_pretrained"
    #  "b.georgedeane.terrain_easy_world_pretrained"
    #  "b.georgedeane.terrain_memory_world_pretrained"
    #  "b.georgedeane.terrain_extra_hard"
    #  "b.georgedeane.terrain_multienv_stable_pretrained_mb4"
 )

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/memory_db \
        device=cpu
done

