
#NAV-MEMORY-SEQUENCE
./devops/skypilot/launch.py train run=$USER.nav_memory_sequence.baseline trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --gpus=4 --skip-git-check \
