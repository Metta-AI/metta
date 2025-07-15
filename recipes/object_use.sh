#Object Use Recipe
#For baseline to compare, see wandb run: https://wandb.ai/metta-research/metta/runs/daphne.object_use.06-25?nw=nwuserdaphned
#We expect this policy to achieve ~60% on the object use evals, in particular it should learn the full mine-generator-altar sequence,
#but it may not learn lasery/shooting/any combat related objects
./devops/skypilot/launch.py train \
run=$USER.object_use.baseline.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/object_use \
--gpus=1 \
sim=object_use \
"$@"
