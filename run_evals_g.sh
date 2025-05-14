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

    "gd2_sharing_48"
    "gd2_sharing48_06"
    "gd2_sharing48_03"
    "gd2_sharing24_03"
    "gd2_sharing_24"
    "gd2_sharing24_06"

)

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"

    DB=wandb://stats/navigation_sharing_db


    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=$DB \

python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_sharing_db run=run_nav_sharing ++dashboard.output_path=s3://softmax-public/policydash/nav_sharing.html \

done
