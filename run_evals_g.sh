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

    ""

)

for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"

    DB=wandb://stats/navigation_sharing_db


    python3 -m tools.sim \
        sim=multiagent \
        run=multiagent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=$DB \

python3 -m tools.dashboard +eval_db_uri=$DB run=multiagentrugeorge2 ++dashboard.output_path=s3://softmax-public/policydash/navigation_sharing.html \

done
