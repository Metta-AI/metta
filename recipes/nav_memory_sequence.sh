
#NAV-MEMORY-SEQUENCE
./devops/skypilot/launch.py train run=$USER.recipes.nav_memory_sequence trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --gpus=4 --skip-git-check \
