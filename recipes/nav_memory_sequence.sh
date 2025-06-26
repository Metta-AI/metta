
#NAV-MEMORY-SEQUENCE

#1) finetune off of object use
./devops/skypilot/launch.py train run=daphne.nav_memory_sequence.object_use_finetuned.06-25 trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --gpus=1 --skip-git-check trainer.initial_policy.uri=wandb://run/daphne.object_use.all.nodesync.metta4.06-25 \


#2) finetune off of navigatoin
./devops/skypilot/launch.py train run=daphne.nav_memory_sequence.navigation_finetuned.06-25 trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --gpus=1 --skip-git-check trainer.initial_policy.uri=wandb://run/daphne.navigation.low_reward.1gpu.4agents.06-25 \

#3) from scratch
./devops/skypilot/launch.py train run=$USER.nav_memory_sequence.from_scratch.06-25 trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence --gpus=1 --skip-git-check \
