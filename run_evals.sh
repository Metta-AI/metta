#!/bin/bash

# Define the list of policy URIs
"""
MISSING

georged_sharing48_06 - check num_agents
georged_sharing24_06 - unexpected keyword teams in terain_from_numpy
georged_sharing_24_range_pretrained - unexpected keyword teams in terain_from_numpy
georged_sharing24_06_pretrained - unexpected keyword teams in terain_from_numpy
"""


POLICIES=(
    # "b.daphne.navigation0"
    # "b.daphne.navigation1"
    # "b.daphne.navigation4"
    # "b.daphne.navigation3"
    # "b.daphne.navigation4"

    "daphne.2object_use_colors_pretrained"
    "daphne.2object_use_colors"
    "daphne.2object_use_no_colors_pretrained"
    "daphne.2object_use_no_colors"
    "dd_object_use_easy"

    "gd2_sharing24_06"
    "gd2_24_no_sharing"
    "gd2_sharing24_03"
    "gd2_sharing_24"
    "gd2_48_no_sharing"
    "gd2_sharing48_03"
    "gd2_sharing48_06"
    "gd2_sharing_48"

    "b.daphne.navigation_training"
    "b.daphne.simple_training"

    # "gd_sharing_24_range_pretrained"
    # "gd_sharing24_06_pretrained"
    # "gd_sharing48_06_pretrained"
    # "gd_24_nosharing_pretrained"
    # "gd_48_nosharing_pretrained"
    # "gd_sharing_48_range_pretrained"

    # "gd_sharing24_03_pretrained"
    # "gd_sharing48_03_preteained"
    ""


)

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"


    python3 -m tools.sim \
        sim=multiagent \
        run=multiagent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://stats/multiagent_db \


    python3 -m tools.sim \
        sim=object_use \
        run=object_use$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://stats/object_use_db \


    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://stats/memory_db \

    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://stats/navigation_db \


python3 -m tools.dashboard +eval_db_uri=wandb://stats/multiagent_db run=multiagentdaphne ++dashboard.output_path=s3://softmax-public/policydash/multiagent.html \

python3 -m tools.dashboard +eval_db_uri=wandb://stats/object_use_db run=objectusedaphne ++dashboard.output_path=s3://softmax-public/policydash/object_use.html \

python3 -m tools.dashboard +eval_db_uri=wandb://stats/memory_db run=memorydpahne ++dashboard.output_path=s3://softmax-public/policydash/memory.html \

python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db run=navigationdaphne ++dashboard.output_path=s3://softmax-public/policydash/navigation2.html \

done
