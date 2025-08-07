#2 converter smoke test
./devops/skypilot/launch.py train \
run=$USER.operantconditioning.smoke_test.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/smoke_tests/2_converters \
sim=operant_condition_smoketest \
"$@"

#3 converter smoke test
./devops/skypilot/launch.py train \
run=$USER.operantconditioning.smoke_test.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/smoke_tests/3_converters \
sim=operant_condition_smoketest \
"$@"

#4 converter smoke test
./devops/skypilot/launch.py train \
run=$USER.operantconditioning.smoke_test.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/smoke_tests/4_converters \
sim=operant_condition_smoketest \
"$@"

#all converter smoke test
./devops/skypilot/launch.py train \
run=$USER.operantconditioning.smoke_test.$(date +%m-%d) \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/smoke_tests/all \
sim=operant_condition_smoketest \
"$@"
