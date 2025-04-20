#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "navigation_infinite_cooldown_sweep"
    "navigation_infinite_cooldown_sweep.r.0"
    "b.daveey.t.8.rdr9.3"
    "b.daveey.t.4.rdr9.3"
    "b.daveey.t.8.rdr9.mb2.1"
    "daveey.t.1.pi.dpm"
    "b.daveey.t.64.dr90.1"
    "b.daveey.t.8.rdr9.sb"
    "b.daphne.navigation_training_onlyhearts.r.1"
    "b.daphne.navigation_training_onlyhearts.r.0"
    "navigation_training_suite_onlyhearts"
    "b.daphne.navigation_terrain_training"
    "b.georgedeane.terrain_newmaps_pretrained"
    "b.georgedeane.terrain_newmaps"
    "b.daphne.terrain_newmaps2"
    "terrain_training_newmaps2"
    "b.daphne.navigation_terrain_training_v2"
    "b.georgedeane.navigation_terrain_training_v2"
    "terrain_training_multienv2"
    "b.daphne.cylinder_run"
    "b.georgedeane.terrain_cylinderstyle_finetune"
    "b.daphne.terrain_newstyle_pretrained"
    "b.georgedeane.terrain_cylinderstyle_fromscratch"
    "b.georgedeane.terrain_multienv"
    "terrain_training_mapfromfile"
    "b.daphne.terrain_multienv_muon"
    "terrain_training_multienv_muon"
    "b.daphne.terrain_10k_maps2"
    "b.georgedeane.terrain_newmaps_pretrained:v41"
    "b.daphne.terrain_newmaps"
    "b.daphne.multienv_mettagrid.r.5"
    "b.daphne.terrain_multienv_april18_sweep.r.2"
    "b.daphne.varied_terrain_sweep.r.0"
    "b.daphne.terrain_10k_maps_april18_"
    "terrain_training_10kmaps_april18"
    "b.daphne.terrain_10k_maps_april18"
    "b.daphne.terrain_multienv_april18"
    "terrain_training_multienv_april18"
    "b.daphne.terrain_10k_maps2"
    "b.daveey.s.1.dr.muon.short.r.111"
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
