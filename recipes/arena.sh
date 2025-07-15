
./devops/skypilot/launch.py train \
--gpus=1 \
--nodes=1 \
--no-spot \
run=$USER.recipes.arena.8x4.$(date +%m-%d) \
trainer.curriculum=/env/mettagrid/curriculum/arena/learning_progress \
trainer.optimizer.learning_rate=0.0045 \
trainer.optimizer.type=muon \
trainer.simulation.evaluate_interval=50
