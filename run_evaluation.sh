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
    "b.georgedeane.navseq_training_sp_heartmax15"
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
    "george_navsequence_all"
    "george_multienv_sequence"
)
#!/bin/bash

    # "daphne_objectuse_allobjs_multienv:v94"
    # "b.daphne.object_use_multienv2:v65"
    # "training_regular_envset_nb:v76"
    # "daphne_objectuse_bigandsmall:v67"
    # "navigation_training:v35"
    # "training_regular_envset"
    # "training_prioritized_envset"
    # "b.daphne.navigation_prioritized_envset"
    # "b.daphne.navigation_regular_envset"
    # "b.daphne.objectuse_prioritized_envset"
    # "b.daphne.objectuse_regular_envset"
    # "b.daphne.prioritized_envset"
    # "b.daphne.uniform_envset_nb"
    # "b.daphne.regular_envset_nb"
    # "b.daphne.prioritized_envset_nb"
    # "b.daphne.regular_envset"
    # "training_regular_envset_nb"
    # "training_uniform_envset_nb"



for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/navigation_db2 \
        # device=cpu \

    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/memory_db2 \
        # device=cpu \

    python3 -m tools.sim \
        sim=object_use \
        run=objectuse$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/objectuse_db2 \
        # device=cpu \


    python3 -m tools.sim \
        sim=nav_sequence \
        run=nav_sequence$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/nav_sequence_db2 \
        # device=cpu \

   python3 -m tools.sim \
        sim=multiagent \
        run=multi_agent$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/multi_agent_db2 \
        # device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_db2 run=navigation_db2 ++dashboard.output_path=s3://softmax-public/policydash/navigation.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/memory_db2 run=memory_db2 ++dashboard.output_path=s3://softmax-public/policydash/memory.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/objectuse_db2 run=objectuse_db2 ++dashboard.output_path=s3://softmax-public/policydash/objectuse.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/nav_sequence_db2 run=nav_sequence_db2 ++dashboard.output_path=s3://softmax-public/policydash/nav_sequence.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/multi_agent_db2 run=multi_agent_db2 ++dashboard.output_path=s3://softmax-public/policydash/multiagent.html

done
