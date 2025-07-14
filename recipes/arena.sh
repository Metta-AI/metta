
./devops/skypilot/launch.py train \
--gpus=1 \
--nodes=1 \
--no-spot \
run=$USER.recipes.arena.1x1.$(date +%m-%d) \
trainer.curriculum=/env/mettagrid/arena/basic_easy_shaped \
trainer.simulation.evaluate_interval=50 \
