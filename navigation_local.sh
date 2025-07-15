#!/bin/bash

# Local navigation training script for use with tools/train.py

python3 tools/train.py \
run="$USER.navigation.heart_only_counts_if_winning_LOCAL.$(date +%m-%d)" \
trainer.curriculum=env/mettagrid/curriculum/navigation/prioritize_regressed \
+trainer.env_overrides.game.num_agents=4 \
sim=navigation \
+USER=greg \
"$@"
