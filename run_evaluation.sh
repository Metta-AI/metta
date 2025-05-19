#!/bin/bash

set -e

# Define the list of policy URIs to evaluate on a normal run.
POLICIES=(
    "b.georgedeane.george_sequence_no_increment"
    "b.georgedeane.george_sequence_incremental"
    "george_sequence_incremental"
    "george2_multienv_noincrement"

    "objectuse_nocolors"

    "george_sequence_varied"
    "george3_multienv_noincrement"

    "daphne_objectuse_allobjs_multienv"
    "daphne_objectuse_allobjs"
    "b.daphne.object_use_mulitenv_pretrained"
    "b.daphne.object_use_all_easy"
    "b.daphne.object_use_multienv"
    "b.daphne.object_use_multienv2"
)
#!/bin/bash


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
