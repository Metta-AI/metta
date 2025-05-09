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
    "daphne_object_use_training_colors"
    "b.daphne.object_use_training_colors2"
    "b.daphne.object_use_training_colors_pretrained2"
    "b.daphne.object_use_training_no_colors3"
    "b.daphne.object_use_training_no_colors_pretrained3"

    "georged_sharing_48"
    "georged_24_no_sharing"
    "georged_48_no_sharing"
    "georged_sharing48_06_pretrained"

    "georged_24_nosharing_pretrained"
    "georged_48_nosharing_pretrained"
    "georged_sharing_48_range_pretrained"
    "georged_sharing_24_03"
    "georged_sharing24_03_pretrained"
    "georged_sharing48_03_pretrained"


    "georged_sharing_48_06_pretrained"
    "georged_48_no_sharing"
    "georged_extended_sequence_pretrained"
    "georged_extended_sequence"


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

python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db run=navigationdaphne ++dashboard.output_path=s3://softmax-public/policydash/navigation.html \

done
