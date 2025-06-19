#NAVIGATION
./devops/skypilot/launch.py train run=georgdeane.navigation.baseline.06-19  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \

#LEARNING THE SEQUENCE
./devops/skypilot/launch.py train run=georgdeane.sequence.baseline.06-19  trainer.curriculum=env/mettagrid/curriculum/sequence --no-spot --gpus=4 --skip-git-check \

#OBJECT USE
./devops/skypilot/launch.py train run=georgdeane.object_use.baseline.06-19  trainer.curriculum=env/mettagrid/curriculum/object_use --no-spot --gpus=4 --skip-git-check \

#NAV-MEMORY-SEQUENCE
./devops/skypilot/launch.py train run=georgdeane.nav_memory_sequence.baseline.06-19  trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --no-spot --gpus=4 --skip-git-check \
