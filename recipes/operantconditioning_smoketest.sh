
./devops/skypilot/launch.py train \
run=$USER.operantconditioning.smoke_test.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/smoke_test \
sim=operant_condition_smoketest \
"$@"
