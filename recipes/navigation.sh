#NAVIGATION
./devops/skypilot/launch.py train run=$USER.recipes.navigation  trainer.curriculum=env/mettagrid/curriculum/navigation/low_reward --gpus=4 --skip-git-check \
