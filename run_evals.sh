#!/bin/bash

# Define the list of policy URIs
POLICIES=(
    "daphne_object_use_training_colors"
    "b.daphne.object_use_training_colors2"
    "b.daphne.object_use_training_colors_pretrained2"
    "b.daphne.object_use_training_no_colors3"
    "b.daphne.object_use_training_no_colors_pretrained3"
    "georged_sharing48"
    "georged_sharing48_03_pretrained"
    "georged_sharing48_nosharing_pretrained"
    "georged_24_no_sharing"
    "georged_24_nosharing_pretrained"
    "georged_sharing_48_range_pretrained"
    "georged_sharing_48_06_pretrained"
    "georged_sharing_24_03_pretrained"
    "georged_sharing_24_03"
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
        +eval_db_uri=wandb://artifacts/multiagent_db \
        # sim.num_envs=20 \
        # sim.num_episodes=20

    # python3 -m tools.sim \
    #     sim=cards \
    #     run=cards$IDX \
    #     policy_uri=wandb://run/$POLICY_URI \
    #     +eval_db_uri=wandb://artifacts/cards_db \

    python3 -m tools.sim \
        sim=object_use \
        run=object_use$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/object_use_db \
        # sim.num_envs=20 \
        # sim.num_episodes=20 \

    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/memory_db \

    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        +eval_db_uri=wandb://artifacts/navigation_db \
        # sim.num_envs=20 \
        # sim.num_episodes=20 \

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/multiagent_db run=multiagentrun4 ++analyzer.output_path=s3://softmax-public/policydash/multiagent.html \

# python3 -m tools.analyze +eval_db_uri=wandb://artifacts/cards_db run=cardsrun3 ++analyzer.output_path=s3://softmax-public/policydash/cards.html \

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/object_use_db run=object_userun4 ++analyzer.output_path=s3://softmax-public/policydash/object_use.html \

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/memory_db run=multiagentrun4 ++analyzer.output_path=s3://softmax-public/policydash/memory.html \

python3 -m tools.analyze +eval_db_uri=wandb://artifacts/navigation_db run=navigationrun4 ++analyzer.output_path=s3://softmax-public/policydash/navigation.html \

done
