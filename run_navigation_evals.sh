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
    "b.georgedeane.terrain_multienv"
    # "terrain_training_mapfromfile"
    # "b.daphne.terrain_multienv_muon"
    # "terrain_training_multienv_muon"
    # "b.daphne.terrain_10k_maps2"
    # "b.georgedeane.terrain_newmaps_pretrained:v41"
    # "b.daphne.terrain_newmaps"
    # "terrain_training_10kmaps_april18"
    # "b.daphne.terrain_10k_maps_april18"
    # "b.daphne.terrain_multienv_april18"
    # "terrain_training_multienv_april18"
    # "b.daphne.terrain_10k_maps2"
    # "b.daphne.terrain_multienv_april18_sweep.r.2"
    # "b.daphne.terrain_multienv_april18_sweep.r.3"
    # "terrain_multienv_with_labyrinth2"
    # "b.daphne.varied_terrain_sweep.r.2"
    # "b.daphne.varied_terrain_sweep.r.1"
    "b.daphne.terrain_varied_cyl_lab_pretrained"
    # "b.daphne.terrain_varied_cyl_lab"
    # "b.georgedeane.terrain_multienv_pretrained_varied"
    # "terrain_multienv_with_labyrinth"
    # "b.daphne.terrain_multienv_april18_sweep.r.6"
    "b.georgedeane.terrain_multienv_labyrinth_pretrained_DR"
    "b.daphne.terrain_varied_cyl_lab"
    # "b.daphne.terrain_multienv_altar_resets"
    "b.daphne.terrain_multienv_altar_resets_withgenerators"
    # "terrain_multienv_resets"
    "b.daphne.terrain_multienv_3_singleagent"
    "b.daphne.terrain_multienv_april18_sweep.r.7"
    "b.daphne.terrain_multienv_altar_no_resets"
    "b.daphne.terrain_multienv_singleA_kitchensink"
    "b.daphne.terrain_multienv_singleA_altar_resets"
    "b.daphne.terrain_multienv_singleA_withgenerators"
    "b.georgedeane.terrain_multienv_unsable_pretrained_mb"
    "b.georgedeane.terrain_multienv_uniform"
    "b.daphne.terrain_multienv_prioritized_georges_maps"
    "b.georgdeane.terrain_multienv_stable_pretrained"
    "b.georgedeane.terrain_multienv_uniform_pretrained"
    "b.georgedeane.terrain_multienv_fromscratch"
    "b.daphne.terrain_multienv_prioritized_singleA_50hearts"
    "b.daphne.terrain_multienv_singleA_50hearts_with_cylinders"
    "b.daphne.terrain_multienv_singleA_50hearts"
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
        eval_db_uri=wandb://artifacts/navigation_db2 \
        # device=cpu
done
