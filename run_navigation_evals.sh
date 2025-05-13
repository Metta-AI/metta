#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "b.daphne.navigation0"
    "b.daphne.navigation4"
    "b.daphne.navigation1"
    "b.daphne.navigation3"
    "navigation_training"
    "daphne_navigation_train"
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
        +eval_db_uri=wandb://stats/navigation_db_after_refactor \

done

python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db_after_refactor run=multiagentdaphne ++dashboard.output_path=s3://softmax-public/policydash/navigation_db_after_refactor.html \
