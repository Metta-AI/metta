
./devops/skypilot/launch.py train \
--gpus=4 \
--nodes=8 \
--no-spot \
run=$USER.recipes.arena.8x4.$(date +%m-%d) \
+USER=greg \
trainer.curriculum=/env/mettagrid/arena/basic_easy_shaped \
trainer.simulation.evaluate_interval=50 \
