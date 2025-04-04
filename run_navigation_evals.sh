#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "b.daphne.navigation_varied_obstacle_shapes_pretrained.r.1"
    "b.daphne.navigation_varied_obstacle_shapes.r.0"
    "navigation_poisson_sparser.r.2"
    "navigation_infinite_cooldown_sparser_pretrained.r.0"
    "navigation_infinite_cooldown_sparser.r.0"
    "navigation_poisson_sparser_pretrained.r.6"
)

# Loop through the policies and run evaluations
for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}
    IDX=$((i + 1))

    echo "Running only hearts eval for policy $POLICY_URI"
    python3 -m tools.eval \
        eval=navigation_evals_onlyheart \
        run=navigation_eval_onlyhearts$IDX \
        eval.policy_uri=wandb://run/$POLICY_URI \
        eval_db_uri=wandb://artifacts/navigation_evaldb_onlyhearts

    echo "Running full sequence eval for policy $POLICY_URI"
    python3 -m tools.eval \
        eval=navigation_evals_fullsequence \
        run=navigation_evals_fullsequence$IDX \
        eval.policy_uri=wandb://run/$POLICY_URI \
        eval_db_uri=wandb://artifacts/navigation_evaldb_fullsequence
done
