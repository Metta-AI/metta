cmd.sh info <job_id>
cmd.sh logs <job_id>
cmd.sh stop <job_id>

cmd.sh jobs <job_queue>="metta-jq" --max=10
cmd.sh compute <compute_environment>="all"

# Launch a new job
cmd.sh launch --run RUN_ID --cmd COMMAND [options]

# Launch command options
# --run RUN             The run id (required)
# --cmd {train,sweep,evolve}  The command to run (required)
# --git-branch BRANCH   The git branch to use (default: current commit)
# --git-commit COMMIT   The git commit to use (default: current commit)
# --gpus GPUS           Total number of GPUs to use (default: 4)
# --node-gpus NODE_GPUS GPUs per node (default: 4)
# --copies COPIES       Number of job copies to submit (default: 1)
# --job-queue QUEUE     AWS Batch job queue to use (default: metta-jq)

# Examples:
# Launch a training job with 4 GPUs on a single node
cmd.sh launch --run my_run --cmd train

# Launch a training job with 8 GPUs across 2 nodes
cmd.sh launch --run my_run --cmd train --gpus 8 --node-gpus 4

# Launch a training job with a specific git branch
cmd.sh launch --run my_run --cmd train --git-branch my-branch
