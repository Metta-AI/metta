#NAVIGATION
./devops/skypilot/launch.py train run=georgdeane.navigation.baseline.06-20  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \

#LEARNING THE SEQUENCE
./devops/skypilot/launch.py train run=georgdeane.sequence.baseline.06-20  trainer.curriculum=env/mettagrid/curriculum/sequence --no-spot --gpus=4 --skip-git-check \

#OBJECT USE
./devops/skypilot/launch.py train run=georgdeane.object_use.baseline.06-20  trainer.curriculum=env/mettagrid/curriculum/object_use --no-spot --gpus=4 --skip-git-check \

#NAV-MEMORY-SEQUENCE
./devops/skypilot/launch.py train run=georgdeane.nav_memory_sequence.baseline.06-20  trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --no-spot --gpus=4 --skip-git-check \


# ./devops/skypilot/launch.py train run=daphne.progressive.06-20 trainer.curriculum=env/mettagrid/curriculum/progressive --no-spot --gpus=4 --skip-git-check \

#SWEEPS


# #NAVIGATION
# ./devops/skypilot/launch.py sweep run=daphne.navigation.sweep.06-19.2  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \

# #LEARNING THE SEQUENCE
# ./devops/skypilot/launch.py sweep run=daphne.sequence.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/sequence --no-spot --gpus=4 --skip-git-check \

# #OBJECT USE
# # ./devops/skypilot/launch.py sweep run=daphne.object_use.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/object_use --no-spot --gpus=4 --skip-git-check \

# #NAV-MEMORY-SEQUENCE
# ./devops/skypilot/launch.py sweep run=daphne.nav_memory_sequence.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --no-spot --gpus=4 --skip-git-check \
