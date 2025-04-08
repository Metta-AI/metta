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
    "navigation_onlyhearts_ent0.1"
    "b.daphne.navigation_training_onlyhearts.r.0"
    "navigation_training_suite_onlyhearts"
    "navigation_training_varied_obs"
    "navigation_training_simple"
)


for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}
    IDX=$((i + 1))

    echo "Running full sequence eval for policy $POLICY_URI"
    python3 -m tools.eval \
        eval=navigation \
        run=navigation_evaluation$IDX \
        eval.policy_uri=wandb://run/$POLICY_URI \
        eval_db_uri=wandb://artifacts/navigation_db
done
