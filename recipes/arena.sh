#!/bin/bash
./devops/skypilot/launch.py train \
--gpus=8 \
--nodes=4 \
--no-spot \
run=$USER.recipes.arena.8x4 \
trainer.curriculum=/env/mettagrid/arena/basic_easy_shaped \
trainer.simulation.evaluate_interval=50

