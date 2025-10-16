#!/bin/bash

# Note: Using heavily reduced batch sizes for 4 GPU ViT setup to avoid OOM

BASE_RUN_NAME="tasha.10.15.shaped_hypers_vit_2"

PAIRED_SEEDS=(42 123 456)

for i in "${!PAIRED_SEEDS[@]}"; do
    SEED=${PAIRED_SEEDS[$i]}
    PAIR_NUM=$((i + 1))

    echo "=== Paired Comparison $PAIR_NUM: Seed $SEED ==="

    # With contrastive loss
    RUN_NAME="${BASE_RUN_NAME}.contrastive.seed${SEED}"
    echo "  Launching WITH contrastive: $RUN_NAME"
    ./devops/skypilot/launch.py experiments.recipes.arena_basic_easy_shaped.train \
        run=$RUN_NAME \
        system.seed=$SEED \
        training_env.seed=$SEED \
        trainer.losses.enable_contrastive=true \
        trainer.batch_size=65536 \
        trainer.minibatch_size=2048 \
        --gpus 4 \
        --max-runtime-hours 8

    # Without contrastive loss
    RUN_NAME="${BASE_RUN_NAME}.no_contrastive.seed${SEED}"
    echo "  Launching WITHOUT contrastive: $RUN_NAME"
    ./devops/skypilot/launch.py experiments.recipes.arena_basic_easy_shaped.train \
        run=$RUN_NAME \
        system.seed=$SEED \
        training_env.seed=$SEED \
        trainer.losses.enable_contrastive=false \
        trainer.batch_size=65536 \
        trainer.minibatch_size=2048 \
        --gpus 4 \
        --max-runtime-hours 8

    echo ""
done

echo "All 6 jobs launched!"
echo "Monitor with: sky jobs queue"
