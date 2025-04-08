#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "b.daveey.t.8.rdr9.3"
    "b.daveey.t.4.rdr9.3"
    "b.daveey.t.8.rdr9.mb2.1"
    "daveey.t.1.pi.dpm"
    "b.daveey.t.64.dr90.1"
    "b.daveey.t.8.rdr9.sb"
)


for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}
    IDX=$((i + 1))

    echo "Running full sequence eval for policy $POLICY_URI"
    python3 -m tools.eval \
        eval=object_use \
        run=object_use_evaluation$IDX \
        eval.policy_uri=wandb://run/$POLICY_URI \
        eval_db_uri=wandb://artifacts/object_use_db
done
