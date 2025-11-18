uv run launch.py recipes.experiment.arena.train run=perfcheck_$(date +%Y%m%d_%H%M%S) \
>   trainer.total_timesteps=100000000 \
>   --gpus 1 --skip-git-check 

RESULT https://skypilot-api.softmax-research.net/dashboard/jobs/8155

uv run launch.py recipes.experiment.arena.train run=perfcheck_multi_gpu \
  trainer.total_timesteps=100000000 \
  --gpus 4 \
  --nodes 1 --skip-git-check

## parallel performance testing




uv run launch.py recipes.experiment.arena.evaluate policy_uris='s3://softmax-public/policies/local.nishadsingh.20251114.124019/local.nishadsingh.20251114.124019:v74.mpt' \ 
  --copies 3 --skip-git-check --gpus 2 --nodes 1 --max-runtime-hours 2