#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
  "georgedeane.sky.nav_backchain_mem"
  "georgdeane.sky.nav_backchain"
  "georgedeane.sky.nav_mem_pretrained"
  "georgedeane.sky.nav_backchain_mem_pretrained"
  "georgedeane.sky.nav_navsequence_backchain"
  "georgedeane.sky.nav_backchain_mem"
  "gd_all2"
  "gd_backchain_mem_pretrained:v18"
  "gd_all"
  # "dd.2objectuse_curriculum"
  # "dd_navigation_curriculum"
  "dd_navsequence_memory_pretrained"
  "dd_navsequence_all_pretrained"
  "dd_navsequence_all"
  # "dd_multiagent"
  # "dd_multiagent_pretrained"
  "dd.curriculum_ALL"
  "dd_curriculum_all_reg0.1"
  "b.dd.curriculum_all"
  "dd.navsequencemem.smallinventory"
  "dd.navsequence_seq.smallinventory"
  "dd.navsequence_all.smallinventory"
  "gd.1.kitchensink"
  "gd.backchain_kitchen2"
  "gd2_backchain_kitchen"
  "gd_sequence_stripped3"
  "gd_backchain_none3"
  "gd_all2"
  "gd_all"
  "gd_backchain_mem_pretrained"
  "gd_sequence_strippedt"
  "gd_pure_mem_backchain"
  "gd_pure_seq_backchain"
  "gd_kitchen_hard_pretrained"
  "gd_backchain_full_extended"
  "gd_backchain_in_context"
  "gd_backchain_scratch_hard"
)


for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

  echo "Running full sequence eval for policy $POLICY_URI"
  RANDOM_NUM=$((RANDOM % 1000))
  IDX="${IDX}_${RANDOM_NUM}"
  python3 -m tools.sim \
    sim=memory \
    run=memory$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/memory_db \
    device=cpu \

  python3 -m tools.sim \
    sim=nav_sequence \
    run=navsequence$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/navsequence_db \
    device=cpu \

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/memory_db  run=memory_db ++dashboard.output_path=s3://softmax-public/policydash/memory.html

  python3 -m tools.dashboard +eval_db_uri=wandb://stats/navsequence_db run=navsequence_db ++dashboard.output_path=s3://softmax-public/policydash/navsequence.html

done
