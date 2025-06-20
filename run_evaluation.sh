#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    # "dd.2objectuse_curriculum"
    # "dd_navigation_curriculum"
    # "dd_navsequence_memory_pretrained"
    # "dd_navsequence_memory"
    # "dd_navsequence_all_pretrained"
    # "dd_navsequence_all"
    # "dd_multiagent"
    # "dd_multiagent_pretrained"
    # "dd.curriculum_ALL"
    # "dd_curriculum_all_reg0.1"
    # "b.dd.curriculum_all"
    # "dd.navsequencemem.smallinventory"
    # "dd.navsequence_seq.smallinventory"
    # "dd.navsequence_all.smallinventory"
    # "gd.1.kitchensink"
    # "b.gd.easysequence"
    # "gd.backchain_kitchen2"
    # "gd2_backchain_kitchen"
    # "gd_backchain3"
    # "gd_sequence_stripped3"
    # "gd_backchain_seq3"
    # "gd_backchain_none3"
    # "gd_all2"
    # "gd_all"
    # "gd_backchain_mem_pretrained"
    # "gd_sequence_strippedt"
    # "gd1.easysequence"
    # "gd_pure_mem_backchain"
    # "gd_pure_seq_backchain"
    # "gd_kitchen_hard_pretrained"
    # "gd_backchain_full_extended"
    # "gd_backchain_in_context"
    # "gd_backchain_scratch_hard"
    # "georgedeane.nav_memory_sequence.baseline.06-19.2"
    # "georgedeane.object_use.baseline.06-19.2"
    # "georgedeane.sequence.baseline.06-19.2"
    # "georgedeane.nav_memory_sequence.baseline.06-19.2"
    # "george.deane.navigation.baseline.06-19.2"
    # "georgedeane.object_use.baseline.06-19.2"
    # "georgedeane.sequence.baseline.06-19.2"
    # "daphne.progressive.06-19"
    # "daphnedemekas.nav_memory_sequence.bucketed.06-19.2"
    # "daphnedemekas.navigation.bucketed.06-19.2"
    # "daphnedemekas.sequence.bucketed.06-19.2"
    # "daphnedemekas.object_use.bucketed.06-19.2"
    # "daphne.progressive.metta1.06-19"
    # "daphne.navigation.bucketed.mettabox2.06-19"
    # "daphne.navmemorysequence.bucketed.metta3.06-19"
    "george.memory_training_variednew"
    "george.extendedsequencenew"
    "georgedeane.mem_mettascope_georgenew"
    "george.extendedsequence.sky2"
    "georgedeane.mem_mettascope_george_init2"






    "george.extendedsequence.metta1"
    "georgedeane.memory_training_varied2"
    "georgedeane.extended_sequence"
    "georgedeane.mem_relh_init"

    "georgedeane.mem_daphne_init"
    "georgedeane.nav_scratch"
    "georgedeane.mem_pretrained5"
    "georgedeane.mem_minimal"
    "georgedeane.mem_general_pretrained"
    "georgedeane.memory_scratch"
    "georgedeane.memory_pretrained"
    "georgedeane.memory_general"
    "georgedeane.nav_scratch"

    # "george.memory_training_variednew"
    # "george.memory_training_varied.init"






    )


#TOKEN POLICIES
  #  "daphne.token.object_use"
  #  "daphne.token.navsequence"
 #   "daphne.token.nav"
    # "dd3_all_tokenized"
    # "dd2_curriculum_all_tokenized"
    # "dd_curriculum_navigation_tokenized"
for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

    echo "Running full sequence eval for policy $POLICY_URI"
    RANDOM_NUM=$((RANDOM % 1000))
    IDX="${IDX}_${RANDOM_NUM}"
    python3 -m tools.sim \
        sim=navigation \
        run=navigation$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/navstatsdbgd \
        device=cpu \


    python3 -m tools.sim \
        sim=memory \
        run=memory$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/memstatsdbgd \
        device=cpu \

    python3 -m tools.sim \
        sim=object_use \
        run=objectuse$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/objusestatsdbgd \
        device=cpu \


    python3 -m tools.sim \
        sim=nav_sequence \
        run=nav_sequence$IDX \
        policy_uri=wandb://run/$POLICY_URI \
        sim_job.stats_db_uri=wandb://stats/navseqstatsdbgd \
        device=cpu \

  #  python3 -m tools.sim \
  #       sim=multiagent \
  #       run=multi_agent$IDX \
  #       policy_uri=wandb://run/$POLICY_URI \
  #       sim_job.stats_db_uri=wandb://stats/stats_db1 \
  # #       # device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navstatsdbgd run=makestatsdbnav ++dashboard.output_path=s3://softmax-public/policydash/navstatsdb.html
  python3 -m tools.dashboard +eval_db_uri=wandb://stats/memstatsdbgd run=makestatsdbmem ++dashboard.output_path=s3://softmax-public/policydash/memstatsdb.html
  python3 -m tools.dashboard +eval_db_uri=wandb://stats/objusestatsdbgd run=makestatsdbobj ++dashboard.output_path=s3://softmax-public/policydash/objusestatsdb.html
  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navseqstatsdbgd run=makestatsdbnavseq ++dashboard.output_path=s3://softmax-public/policydash/navseqstatsdb.html

done
