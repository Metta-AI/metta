#NAVIGATION
./devops/skypilot/launch.py train run=USER.navigation.baseline  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \

#LEARNING THE SEQUENCE
./devops/skypilot/launch.py train run=USER.sequence.baseline  trainer.curriculum=env/mettagrid/curriculum/sequence --no-spot --gpus=4 --skip-git-check \

#OBJECT USE
./devops/skypilot/launch.py train run=USER.object_use.baseline  trainer.curriculum=env/mettagrid/curriculum/object_use --no-spot --gpus=4 --skip-git-check \

#NAV-MEMORY-SEQUENCE
./devops/skypilot/launch.py train run=USER.nav_memory_sequence.baseline  trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --no-spot --gpus=4 --skip-git-check \
