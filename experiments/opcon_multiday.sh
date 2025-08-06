
#SINGLE TRIAL SMOKE TEST

./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_1m.random.2.1trial.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_1m \
+trainer.env_overrides.num_trials=1 \
+trainer.env_overrides.game.max_steps=32 \
trainer.bptt_horizon=32 \
trainer.batch_size=524288 \


./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_2m.random.2.1trial.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_2m \
+trainer.env_overrides.num_trials=1 \
+trainer.env_overrides.game.max_steps=32 \
trainer.bptt_horizon=32 \
trainer.batch_size=524288 \


./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_3m.random.2.1trial.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_3m \
+trainer.env_overrides.num_trials=1 \
+trainer.env_overrides.game.max_steps=32 \
trainer.bptt_horizon=32 \
trainer.batch_size=524288 \


#MULTI-DAY 8 TRIALS

./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_1m.random.2.8trials.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_1m \
+trainer.env_overrides.num_trials=8 \
+trainer.env_overrides.game.max_steps=16 \
trainer.bptt_horizon=128 \
trainer.batch_size=1032192 \

./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_2m.random.2.8trials.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_2m \
+trainer.env_overrides.num_trials=8 \
+trainer.env_overrides.game.max_steps=16 \
trainer.bptt_horizon=128 \
trainer.batch_size=1032192 \


./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_3m.random.2.8trials.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_3m \
+trainer.env_overrides.num_trials=8 \
+trainer.env_overrides.game.max_steps=16 \
trainer.bptt_horizon=128 \
trainer.batch_size=1032192 \


#MULTI-DAY 16 TRIALS


./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_1m.random.2.16trials.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_1m \
+trainer.env_overrides.num_trials=16 \
+trainer.env_overrides.game.max_steps=16 \
trainer.bptt_horizon=256 \
trainer.batch_size=2064384 \


./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_2m.random.2.16trials.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_2m \
+trainer.env_overrides.num_trials=16 \
+trainer.env_overrides.game.max_steps=16 \
trainer.bptt_horizon=256 \
trainer.batch_size=2064384 \


./devops/skypilot/launch.py train \
run=george.operantconditioning.multiday_3m.random.2.16trials.08-05 \
trainer.curriculum=env/mettagrid/curriculum/operant_conditioning/multiday_3m \
+trainer.env_overrides.num_trials=16 \
+trainer.env_overrides.game.max_steps=16 \
trainer.bptt_horizon=256 \
trainer.batch_size=2064384 \
