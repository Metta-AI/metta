#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "breadcrumb_daphne_multiagent_training"
    "b.daphne.multiagent_training_pretrained"
    "b.daphne.multiagent_training2"
    "onlyred_daphne_multiagent_training"
    "b.daphne.multiagent_training_8agents_onlyred"
    "b.daphne.multiagent_training_4agents_onlyred"
    "b.daphne.multiagent_training_4agents"
    "4agents_daphne_multiagent_training"
    "8agents_daphne_multiagent_training"
    "b.daphne.multiagent_training"
    "b.daphne.multiagent_training_8agents"
    "b.daphne.objectuse_training_breadcrumb"
    "b.daphne.objectuse_training_breadcrumb_sm"
    "b.daphne.objectuse_training_breadcrumb_pretrained_sm"
    "b.daphne.objectuse_training_breadcrumb_pretrained"
    "b.daphne.multiagent_training_breadcrumb"
    "b.daphne.multiagent_training_breadcrumb_pretrained"
    "b.daphne.multiagent_training_breadcrumb_sm"
    "b.daphne.multiagent_training_breadcrumb_pretrained_sm"
)
#!/bin/bash



for i in "${!POLICIES[@]}"; do
    POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/navigation_db \
        # device=cpu \

    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/memory_db \
        # device=cpu \

    python3 -m tools.sim \
        sim=object_use \
        run=objectuse$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/objectuse_db \
        # device=cpu \

    python3 -m tools.sim \
        sim=nav_sequence \
        run=nav_sequence$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/nav_sequence_db \
        # device=cpu \


    python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db run=navigation_db ++dashboard.output_path=s3://softmax-public/policydash/navigation.html \

    python3 -m tools.dashboard +eval_db_uri=wandb://stats/memory_db run=memory_db ++dashboard.output_path=s3://softmax-public/policydash/memory.html \

    python3 -m tools.dashboard +eval_db_uri=wandb://stats/objectuse_db run=objectuse_db ++dashboard.output_path=s3://softmax-public/policydash/objectuse.html \

    python3 -m tools.dashboard +eval_db_uri=wandb://stats/nav_sequence_db run=nav_sequence_db ++dashboard.output_path=s3://softmax-public/policydash/nav_sequence.html \

done
