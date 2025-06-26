#NAVIGATION

#Best recipe for navigation
#For baseline to compare, see wandb run: #TODO
./devops/skypilot/launch.py train run=$USER.navigation.bucketed.06-25  trainer.curriculum=env/mettagrid/curriculum/navigation/bucketed --skip-git-check --gpus=1 +trainer.env_overrides.game.num_agents=4 \
