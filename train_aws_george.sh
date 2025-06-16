
python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.extended_sequence_pretrained --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/object_use/training/extended_sequence +trainer.initial_policy.uri=wandb://run/b.daphne.navigation3:v12 --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.extended_sequence --git-branch=multiagent-training wandb.enabled=true wandb.track=true trainer.env=env/mettagrid/object_use/training/extended_sequence --skip-validation \

