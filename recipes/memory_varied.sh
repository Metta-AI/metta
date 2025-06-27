#MEMORY-VARIED
./devops/skypilot/launch.py train run=$USER.memory_varied.baseline  trainer.curriculum=env/mettagrid/curriculum/memory_training_varied --no-spot --gpus=1 --skip-git-check \
