#NAVIGATION
./devops/skypilot/launch.py train run=daphnedemekas.navigation.bucketed.06-20  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \

#LEARNING THE SEQUENCE
./devops/skypilot/launch.py train run=daphne.demekas.sequence.bucketed.06-20  trainer.curriculum=env/mettagrid/curriculum/sequence --no-spot --gpus=4 --skip-git-check \

#OBJECT USE
./devops/skypilot/launch.py train run=daphnedemekas.object_use.bucketed.06-20  trainer.curriculum=env/mettagrid/curriculum/object_use --no-spot --gpus=4 --skip-git-check \

#NAV-MEMORY-SEQUENCE
./devops/skypilot/launch.py train run=daphne.demekas.nav_memory_sequence.bucketed.06-20  trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --no-spot --gpus=4 --skip-git-check \



#SWEEPS


# #NAVIGATION
# ./devops/skypilot/launch.py sweep run=daphne.navigation.sweep.06-19.2  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \

# #LEARNING THE SEQUENCE
# ./devops/skypilot/launch.py sweep run=daphne.sequence.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/sequence --no-spot --gpus=4 --skip-git-check \

# #OBJECT USE
# # ./devops/skypilot/launch.py sweep run=daphne.object_use.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/object_use --no-spot --gpus=4 --skip-git-check \

# #NAV-MEMORY-SEQUENCE
# ./devops/skypilot/launch.py sweep run=daphne.nav_memory_sequence.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --no-spot --gpus=4 --skip-git-check \
