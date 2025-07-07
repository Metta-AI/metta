
./devops/skypilot/launch.py train \
--gpus=4 \
--nodes=8 \
run=$USER.arena.baseline.8x4 \
trainer.curriculum=/env/mettagrid/arena/basic_easy_shaped \
trainer.simulation.evaluate_interval=50
