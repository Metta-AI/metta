#NAVIGATION
./devops/skypilot/launch.py train run=$USER.navigation.baseline  trainer.curriculum=env/mettagrid/curriculum/navigation --gpus=4 --skip-git-check \
