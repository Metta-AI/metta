#!/bin/bash

# Define the list of policy URIs
POLICIES=(
#   "b.daphne.terrain_prioritized_styles_pretrained_r"
#   "b.daphne.terrain_prioritized_styles2"
#   "terrain_prioritized_styles_pretrained_mpmc"
#   "terrain_prioritized_styles_pretrained"
#   "b.terrain_prioritized_styles_nb"
#   "b.terrain_prioritized_styles_pretrained_nb"
#   "b.terrain_prioritized_styles"
#   "b.terrain_prioritized_styles_pretrained"
#   "b.georgedeane.terrain_multienv"
#   "b.daphne.terrain_multienv_3_no_blocks3"
#   "terrain_multienv_3_single_agent"
#   "b.daphne.terrain_multienv_prioritized_multienv_cylinders2"
#   "b.daphne.terrain_multienv_prioritized_multienv_cylinders"
#   "b.georgedeane.terrain_massive_empty_world_pretrained"
#   "b.georgedeane.terrain_extra_hard:v1"
#   "b.daphne.terrain_varied_cyl_lab_pretrained"
#   "b.daphne.terrain_prioritized_styles"
#   "b.daphne.terrain_prioritized_styles_pretrained"
#   "george_memory_pretrained"
#   "b.daphne.terrain_multiagent_48_norewardsharing"
#   "b.daphne.terrain_multiagent_24_norewardsharing"
#   "b.daphne.terrain_multiagent_24_rewardsharing"
#   "b.daphne.terrain_multiagent_48_rewardsharing"
  "objectuse_no_colors"
  "george_sequence_varied"
  "george_sequence_meta"
  "george3_multienv_noincrement"
  "george_sequence_incremental"
  "george_multienv_incremental"
)

for i in "${!POLICIES[@]}"; do
  POLICY_URI=${POLICIES[$i]}

  echo "Running full sequence eval for policy $POLICY_URI"
  RANDOM_NUM=$((RANDOM % 1000))
  IDX="${IDX}_${RANDOM_NUM}"
  python3 -m tools.sim \
    sim=simple_sequence \
    run=navigation$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/simple_sequence \

  python3 -m tools.sim \
    sim=extended_sequence \
    run=extended_sequence$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/extended_sequence \


  python3 -m tools.sim \
    sim=navigation \
    run=navigation$IDX \
    policy_uri=wandb://run/$POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/navigation_new \
    

    python3 -m tools.dashboard +eval_db_uri=wandb://stats/simple_sequence run=simpleseq ++dashboard.output_path=s3://softmax-public/policydash/simpleseq.html \
    python3 -m tools.dashboard +eval_db_uri=wandb://stats/extended_sequence run=extended_sequence ++dashboard.output_path=s3://softmax-public/policydash/extended_sequence.html \
    python3 -m tools.dashboard +eval_db_uri=wandb://stats/navigation_new run=navigation_new ++dashboard.output_path=s3://softmax-public/policydash/navigation_new.html \

done
