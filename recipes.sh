#NAVIGATION
<<<<<<< HEAD
./devops/skypilot/launch.py train run=georgdeane.navigation.baseline.128tokens.06-19  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \
=======
./devops/skypilot/launch.py train run=george.deane.navigation.baseline.06-19  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \
>>>>>>> e31115c2bb0142add9fadced24a03d87085e2d07

#LEARNING THE SEQUENCE
./devops/skypilot/launch.py train run=georgdeane.sequence.baseline.128tokens.06-19.2  trainer.curriculum=env/mettagrid/curriculum/sequence --no-spot --gpus=4 --skip-git-check \

#OBJECT USE
./devops/skypilot/launch.py train run=georgdeane.object_use.baseline.128tokens.06-19  trainer.curriculum=env/mettagrid/curriculum/object_use --no-spot --gpus=4 --skip-git-check \

#NAV-MEMORY-SEQUENCE
./devops/skypilot/launch.py train run=georgdeane.nav_memory_sequence.baseline.128tokens.06-19  trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --no-spot --gpus=4 --skip-git-check \
<<<<<<< HEAD



#SWEEPS


# #NAVIGATION
# ./devops/skypilot/launch.py sweep run=daphne.navigation.sweep.06-19.2  trainer.curriculum=env/mettagrid/curriculum/navigation --no-spot --gpus=4 --skip-git-check \

# #LEARNING THE SEQUENCE
# ./devops/skypilot/launch.py sweep run=daphne.sequence.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/sequence --no-spot --gpus=4 --skip-git-check \

# #OBJECT USE
# # ./devops/skypilot/launch.py sweep run=daphne.object_use.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/object_use --no-spot --gpus=4 --skip-git-check \

# #NAV-MEMORY-SEQUENCE
# ./devops/skypilot/launch.py sweep run=daphne.nav_memory_sequence.sweep.06-19  trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --no-spot --gpus=4 --skip-git-check \
=======
>>>>>>> e31115c2bb0142add9fadced24a03d87085e2d07
