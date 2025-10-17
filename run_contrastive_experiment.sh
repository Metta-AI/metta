#!/bin/bash

BASE_RUN_NAME="tasha.10.17.shaped_PPOhypers_vit"

PAIRED_SEEDS=(67 134 201 34 672)

for i in "${!PAIRED_SEEDS[@]}"; do
    SEED=${PAIRED_SEEDS[$i]}
    PAIR_NUM=$((i + 1))

    echo "=== Paired Comparison $PAIR_NUM: Seed $SEED ==="

    # With contrastive loss
    RUN_NAME="${BASE_RUN_NAME}.contrastive.seed${SEED}"
    echo "  Launching WITH contrastive: $RUN_NAME"
    ./devops/skypilot/launch.py experiments.recipes.arena_basic_easy_shaped.train \
        --gpus 4 \
        run=$RUN_NAME \
        system.seed=$SEED \
        training_env.seed=$SEED \
        trainer.losses.enable_contrastive=true

    # Without contrastive loss
    RUN_NAME="${BASE_RUN_NAME}.no_contrastive.seed${SEED}"
    echo "  Launching WITHOUT contrastive: $RUN_NAME"
    ./devops/skypilot/launch.py experiments.recipes.arena_basic_easy_shaped.train \
        --gpus 4 \
        run=$RUN_NAME \
        system.seed=$SEED \
        training_env.seed=$SEED \
        trainer.losses.enable_contrastive=false

    echo ""
done

echo "All 54 jobs launched! (27 seed pairs)"
echo "Monitor with: sky jobs queue"
