#!/bin/bash

# Define the list of policy URIs
POLICIES_HEARTS=(
    "navigation_infinite_cooldown_sweep"
    "navigation_infinite_cooldown_sweep.r.0"
    "b.daveey.t.8.rdr9.3"
    "b.daveey.t.4.rdr9.3"
    "b.daveey.t.8.rdr9.mb2.1"
    "daveey.t.1.pi.dpm"
    "b.daveey.t.64.dr90.1"
    "b.daveey.t.8.rdr9.sb"
)

POLICIES_FULL_SEQUENCE=(
    "b.daphne.navigation_varied_obstacle_shapes_pretrained.r.1"
    "b.daphne.navigation_varied_obstacle_shapes.r.0"
    "navigation_infinite_cooldown_sparser_pretrained.r.0"
    "navigation_infinite_cooldown_sparser.r.0"
    "navigation_poisson_sparser_pretrained.r.6"
    "b.daveey.t.8.rdr9.3"
    "b.daveey.t.4.rdr9.3"
    "b.daveey.t.8.rdr9.mb2.1"
    "daveey.t.1.pi.dpm"
    "b.daveey.t.64.dr90.1"
    "b.daveey.t.8.rdr9.sb"
)

# Loop through the policies and run evaluations
for i in "${!POLICIES_HEARTS[@]}"; do
    POLICY_URI=${POLICIES_HEARTS[$i]}
    IDX=$((i + 1))

    echo "Running only hearts eval for policy $POLICY_URI"
    python3 -m tools.eval \
        eval=navigation_evals_onlyheart \
        run=navigation_eval_onlyhearts$IDX \
        eval.policy_uri=wandb://run/$POLICY_URI \
        eval_db_uri=wandb://artifacts/navigation_evaldb_only_hearts
done

for i in "${!POLICIES_FULL_SEQUENCE[@]}"; do
    POLICY_URI=${POLICIES_FULL_SEQUENCE[$i]}
    IDX=$((i + 1))

    echo "Running full sequence eval for policy $POLICY_URI"
    python3 -m tools.eval \
        eval=navigation_evals_fullsequence \
        run=navigation_eval_fullsequence$IDX \
        eval.policy_uri=wandb://run/$POLICY_URI \
        eval_db_uri=wandb://artifacts/navigation_evaldb_full_sequence
done
