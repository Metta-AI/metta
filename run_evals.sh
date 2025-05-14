#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "daveey.dist.2x4"
    "navigation_training:v35"
    "b.daphne.navigation0"
    "daphne_navigation_train"
    "b.daphne.navigation1"
    "b.daphne.navigation3"
    "b.daphne.navigation4"
    "gd2_sharing24_03"
    "gd2_sharing24_06"
    "gd2_sharing_48"
    "gd2_sharing_24"
    "gd2_sharing48_03"
    "gd2_sharing48_06"
    "MRM_test_mettabox"
    "georged_48_no_sharing"
    "georged_24_no_sharing"
    "dd_object_use_easy2"
    "daphne.3object_use_no_colors"
    "daphne.3object_use_colors"
    "daphne.2object_use_colors_pretrained"
    "b.daphne.USER.navigation_before_refactor"
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
        +eval_db_uri=wandb://artifacts/navigation_db_main \

    python3 -m tools.sim \
        sim=multiagent \
        run=multiagent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/multiagent_db_main \
        ++device=cpu

    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/memory_db_main \
        ++device=cpu

    python3 -m tools.sim \
        sim=objectuse \
        run=objectuse$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/objectuse_db_main \
        ++device=cpu
done

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/navigation_db_main run=navigation_db_main ++analyzer.output_path=s3://softmax-public/policydash/navigation_main.html

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/multiagent_db_main run=multiagent_db_main ++analyzer.output_path=s3://softmax-public/policydash/multiagent_main.html

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/memory_db_main run=memory_db_main ++analyzer.output_path=s3://softmax-public/policydash/memory_main.html

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/objectuse_db_main run=objectuse_db_main ++analyzer.output_path=s3://softmax-public/policydash/objectuse_main.html
