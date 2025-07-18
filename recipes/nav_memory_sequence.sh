#NAV-MEMORY-SEQUENCE
#These are still experimental, and we may need to use a progressive curriculum to train them
#For now, our best results are achieved by finetuning

#1) finetune off of object use
#baseline to compare: https://wandb.ai/metta-research/metta/runs/daphne.nav_memory_sequence.object_use_finetuned.06-25?nw=nwuserdaphned
./devops/skypilot/launch.py train \
run=$USER.nav_memory_sequence.object_use_finetuned.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence \
--gpus=1 \
trainer.initial_policy.uri=wandb://run/daphne.object_use.06-25 \
sim=all \
"$@"


#2) finetune off of navigation
#baseline to compare: https://wandb.ai/metta-research/metta/runs/daphne.moretime.nav_memory_sequence.navigation_finetuned.06-25?nw=nwuserdaphned
./devops/skypilot/launch.py train \
run=$USER.nav_memory_sequence.navigation_finetuned.$(date +%m-%d) \
trainer=recipe_trainer \
trainer.curriculum=env/mettagrid/curriculum/nav_memory_sequence \
--gpus=1 \
trainer.initial_policy.uri=wandb://run/daphne.navigation.low_reward.1gpu.4agents.06-25 \
sim=all \
"$@"
