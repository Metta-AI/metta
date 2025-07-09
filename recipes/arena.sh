
./devops/skypilot/launch.py train \
--gpus=1 \
--nodes=1 \
--no-spot \
run=$USER.recipes.arena.8x4 \
trainer.curriculum=/env/mettagrid/arena/basic_easy_shaped \
trainer.simulation.evaluate_interval=50 \

