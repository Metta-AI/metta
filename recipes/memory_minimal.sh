#MINIMAL-MEMORY
./devops/skypilot/launch.py train run=USER.minimal_memory.baseline  trainer.curriculum=env/mettagrid/curriculum/minimal_memory --no-spot --gpus=4 --skip-git-check \
