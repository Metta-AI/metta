#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "b.daphne.object_use_training_no_colors_pretrained3"
    "b.daphne.object_use_training_no_colors3"
    "b.daphne.object_use_training_colors2"
    "b.daphne.object_use_training_colors_pretrained2"
)


for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    echo "Running full sequence eval for policy $POLICY_URI"
    python3 -m tools.sim \
        sim=object_use \
        run=objectuse$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/object_use_db_2 \

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/object_use_db_2 run=objectuseanalyze ++analyzer.output_path=s3://softmax-public/policydash/objectuse_2.html \

done
