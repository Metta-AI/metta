#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "b.daphne.objectuse"
    "b.daphne.navigation_objectuse_sp"
    "b.daphne.multiagent_8"
    "b.daphne.npc_simple"
    "objectuse"
    "navigation"
    "b.daphne.objectuse_sp"
    "b.daphne.navigation_sp"
    "b.daphne.multiagent_8_smaller_rooms"
    "b.daphne.objectuse_less_initial_items"
    "b.daphne.objectuse_smaller_rooms"
    "b.daphne.multiagent_8_less_initial_items"
    "b.daphne.multiagent_8_smaller_rooms2"
    "b.georgedeane.navseq_training"
    "b.georgedeane.navseq_training_sp"
    # "b.georgedeane.navseq_training_sp_heartmax15"
    "b.george.multiagent"
    "b.george.multiagent_rewardsharing"
    "b.george.multiagent_mixed"
    "b.george.multiagent_rewardsharing_pretrained"
    "b.george.multiagent_mixed_pretrained"
    "b.george.multiagent_pretrained"
    "b.george.navsequence_mem_pretrained"
    "b.george.navsequence_all"
    "b.george.navsequence_sequence_pretrained"
    "b.george.navsequence_mem"
    "b.george_navsequence_all"
    "b.george_multienv_sequence"
    "mrazo_cooperation_two-room-coord_v06"
    "mrazo_cooperation_two-room-coord_v05"
    "mrazo_memory_varied-terrain_v05"


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

   python3 -m tools.sim \
        sim=multiagent \
        run=multi_agent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/multi_agent_db \
        # device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db run=navigation_db2 ++dashboard.output_path=s3://softmax-public/policydash/navigation.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/memory_db run=memory_db2 ++dashboard.output_path=s3://softmax-public/policydash/memory.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/objectuse_db run=objectuse_db2 ++dashboard.output_path=s3://softmax-public/policydash/objectuse.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/nav_sequence_db run=nav_sequence_db2 ++dashboard.output_path=s3://softmax-public/policydash/nav_sequence.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/multi_agent_db run=multi_agent_db2 ++dashboard.output_path=s3://softmax-public/policydash/multiagent.html

done
