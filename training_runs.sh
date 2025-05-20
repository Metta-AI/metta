python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.object_use_multienv_pretrained --git-branch=daphne-evaluation trainer.env=env/mettagrid/object_use/training/multienv --skip-validation trainer.initial_policy.uri=wandb://run/objectuse_no_colors  \

python -m devops.aws.batch.launch_task --cmd=train --run=b.$USER.object_use_bigandsmall_pretrained --git-branch=daphne-evaluation trainer.env=env/mettagrid/object_use/training/multienv --skip-validation trainer.initial_policy.uri=wandb://run/daphne_objectuse_allobjs_multienv:v94 \
