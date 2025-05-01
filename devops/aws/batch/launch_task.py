import argparse
import netrc
import os
import random
import string
import sys

import boto3

from metta.util.colorama import blue, bold, cyan, green, red, use_colors, yellow
from metta.util.git import (
    get_branch_commit,
    get_current_branch,
    get_current_commit,
    has_unstaged_changes,
    is_commit_pushed,
)


def get_specs():
    return {
        1: {"node_gpus": 1, "node_ram_gb": 50, "gpu_cpus": 8},
        4: {"node_gpus": 4, "node_ram_gb": 150, "gpu_cpus": 12},
        8: {"node_gpus": 8, "node_ram_gb": 300, "gpu_cpus": 24},
    }


def container_config(args, task_args, job_name):
    try:
        netrc_info = netrc.netrc(os.path.expanduser("~/.netrc"))
        auth_data = netrc_info.authenticators("api.wandb.ai")
        if auth_data is None:
            raise ValueError("No api.wandb.ai entry found in .netrc file")
        wandb_key = auth_data[2]  # Index 2 contains the password/token
        if not wandb_key:
            raise ValueError("WANDB_API_KEY not found in .netrc file")
    except Exception as e:
        print(red(f"Error reading WANDB_API_KEY: {e}"))
        sys.exit(1)

    vcpus_per_gpu = args.gpu_cpus
    total_vcpus = vcpus_per_gpu * args.node_gpus
    memory_mb = args.node_ram_gb * 1024

    # Determine the git reference to use (commit or branch)
    git_ref = args.git_commit if args.git_commit else args.git_branch

    env_vars = [
        {"name": "HYDRA_FULL_ERROR", "value": "1"},
        {"name": "WANDB_API_KEY", "value": wandb_key},
        {"name": "WANDB_SILENT", "value": "true"},
        {"name": "COLOR_LOGGING", "value": "false"},
        {"name": "WANDB_HOST", "value": job_name},
        {"name": "METTA_HOST", "value": job_name},
        {"name": "JOB_NAME", "value": job_name},
        {"name": "METTA_USER", "value": os.environ.get("USER", "unknown")},
        {"name": "NCCL_DEBUG", "value": "ERROR"},
        {"name": "NCCL_IGNORE_DISABLED_P2P", "value": "1"},
        {"name": "NCCL_IB_DISABLE", "value": "1"},
        {"name": "NCCL_P2P_DISABLE", "value": "1"},
        {"name": "RUN_ID", "value": args.run},
        {"name": "HARDWARE", "value": "aws"},
        {"name": "MASTER_PORT", "value": str(random.randint(10000, 65535))},
        {"name": "CMD", "value": args.cmd},
        {"name": "NUM_GPUS", "value": str(args.node_gpus)},
        {"name": "NUM_WORKERS", "value": str(vcpus_per_gpu)},
        {"name": "GIT_REF", "value": git_ref},
        {"name": "TASK_ARGS", "value": " ".join(task_args)},
    ]

    entrypoint_cmd = [
        "git fetch",
        f"git checkout {git_ref}",
        "./devops/aws/batch/entrypoint.sh",
    ]

    print(f"Resources: {args.num_nodes} nodes, {args.node_gpus} GPUs, {total_vcpus} vCPUs, {args.node_ram_gb}GB RAM")

    return {
        "command": ["; ".join(entrypoint_cmd)],
        "environment": env_vars,
        "resourceRequirements": [
            {"type": "GPU", "value": str(args.node_gpus)},
            {"type": "VCPU", "value": str(total_vcpus)},
            {"type": "MEMORY", "value": str(memory_mb)},
        ],
    }


def validate_batch_job(args, task_args, job_name, job_queue, job_definition, request):
    critical_files = [
        "./devops/aws/batch/entrypoint.sh",
        f"./devops/{args.cmd}.sh",
        "requirements.txt",
    ]

    if args.cmd == "train":
        critical_files.append("tools/train.py")
    elif args.cmd == "sweep":
        critical_files.append("tools/sweep.py")
    elif args.cmd == "evolve":
        critical_files.append("tools/evolve.py")

    # Check for configuration files mentioned in task arguments
    for task_arg in task_args:
        # Check for agent configuration
        if task_arg.startswith("agent="):
            agent_value = task_arg.split("=", 1)[1]
            critical_files.append(f"./configs/agent/{agent_value}.yaml")

        # Check for trainer configuration
        elif task_arg.startswith("trainer="):
            trainer_value = task_arg.split("=", 1)[1]
            critical_files.append(f"./configs/trainer/{trainer_value}.yaml")

        # Check for environment configuration
        elif task_arg.startswith("trainer.env="):
            env_value = task_arg.split("=", 1)[1]
            critical_files.append(f"./configs/{env_value}.yaml")

        # Check for evaluation configuration
        elif task_arg.startswith("trainer.eval="):
            eval_value = task_arg.split("=", 1)[1]
            critical_files.append(f"./configs/eval/{eval_value}.yaml")

    divider = "=" * 40
    print(f"\n{divider}")
    all_files_exist = True
    for file in critical_files:
        if not os.path.exists(file):
            all_files_exist = False
            print(red(f"❌ Missing required file: {file}"))
        else:
            print(green(f"✅ Found: {file}"))

    if not all_files_exist:
        print(red("\nOne or more critical files are missing. Validation failed."))
        sys.exit(1)
    print(green("All critical files found. Validation successful."))

    if has_unstaged_changes() and not args.skip_validation:
        print(red("Error: You have unstaged changes in your repository."))
        print(red("These changes will not be reflected in the AWS environment."))
        print(yellow("Commit or stash your changes, or use --skip-validation to bypass this check."))
        return False

    # Get the git reference
    git_ref = args.git_commit if args.git_commit else args.git_branch

    # Display job details
    print(f"\n{divider}")
    print(bold("Job will be submitted with the following details:"))
    print(f"{divider}")

    print(f"Job Name: {job_name}")
    print(f"Job Queue: {job_queue}")
    print(f"Job Definition: {job_definition}")
    if args.num_nodes > 1:
        print(f"Number of Nodes: {args.num_nodes}")
    print(f"Number of GPUs per Node: {args.node_gpus}")
    print(f"Total GPUs: {args.gpus}")
    print(f"vCPUs per GPU: {args.gpu_cpus}")
    print(f"RAM per Node: {args.node_ram_gb} GB")
    print(f"Git Reference: {git_ref}")
    print(f"{'-' * 40}")
    print(f"Command: {args.cmd}")
    if task_args:
        print(yellow("\nTask Arguments:"))
        for i, arg in enumerate(task_args):
            print(f"  {i + 1}. {cyan(arg)}")
    print(f"\n{divider}")

    # Check if we should proceed
    if args.dry_run:
        print(bold("DRY RUN - No job will be submitted"))
        print(f"{divider}")
        return False

    if not args.skip_validation:
        response = input("Should we send this job to AWS? (Y/n): ")
        if response.lower() not in ["", "y", "yes"]:
            print(yellow("Job submission cancelled by user."))
            return False

    return True


def submit_batch_job(args, task_args):
    session_kwargs = {"region_name": "us-east-1"}
    if args.profile:
        session_kwargs["profile_name"] = args.profile

    session = boto3.Session(**session_kwargs)
    batch = session.client("batch")

    random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    job_name = args.job_name if args.job_name else args.run.replace(".", "_") + "_" + random_id
    job_queue = args.job_queue
    job_definition = "metta-batch-train-jd"

    request = {
        "jobName": job_name,
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "containerOverrides": container_config(args, task_args, job_name),
    }

    if args.num_nodes > 1:
        job_definition = "metta-batch-dist-train"
        print(f"Using multi-node job definition: {job_definition} for {args.num_nodes} nodes")
        request["jobDefinition"] = job_definition
        request["nodeOverrides"] = {
            "nodePropertyOverrides": [
                {"targetNodes": "0:", "containerOverrides": container_config(args, task_args, job_name)}
            ],
            "numNodes": args.num_nodes,
        }
        del request["containerOverrides"]

    # Validate job and get user confirmation
    should_submit = validate_batch_job(args, task_args, job_name, job_queue, job_definition, request)

    if not should_submit:
        return

    # Submit the job
    response = batch.submit_job(**request)

    job_id = response["jobId"]
    job_url = f"https://us-east-1.console.aws.amazon.com/batch/v2/home?region=us-east-1#/jobs/detail/{job_id}"

    print(f"Submitted job {job_name} to queue {job_queue} with job ID {green(bold(job_id))}")
    print(blue(bold(job_url)))

    if task_args:
        print(yellow("\nTask Arguments:"))
        for i, arg in enumerate(task_args):
            print(f"  {i + 1}. {cyan(arg)}")


def main():
    parser = argparse.ArgumentParser(description="Launch an AWS Batch task with a wandb key.")
    parser.add_argument("--cluster", default="metta")
    parser.add_argument("--run", required=True)
    parser.add_argument("--job-name", help="The job name. If not specified, will use run id with random suffix.")
    parser.add_argument(
        "--cmd", required=True, choices=["train", "sweep", "evolve", "sandbox"], help="The command to run."
    )

    # Git reference group - default is to use current commit
    git_group = parser.add_mutually_exclusive_group()
    git_group.add_argument("--git-branch", help="Use the HEAD of a specific git branch instead of current commit")
    git_group.add_argument("--git-commit", help="Use a specific git commit hash")

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--node-gpus", type=int, default=1)
    parser.add_argument("--gpu-cpus", type=int)
    parser.add_argument("--node-ram-gb", type=int)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--profile", default="stem")
    parser.add_argument("--job-queue", default="metta-jq")
    parser.add_argument("--skip-push-check", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="DEPRECATED: Show job details without submitting")
    parser.add_argument("--skip-validation", action="store_true", help="Skip confirmation prompt")
    args, task_args = parser.parse_known_args()

    use_colors(sys.stdout.isatty() and not args.no_color)

    specs = get_specs()

    args.num_nodes = max(1, args.gpus // args.node_gpus)
    if args.node_ram_gb is None:
        args.node_ram_gb = specs[args.node_gpus]["node_ram_gb"]
    if args.gpu_cpus is None:
        args.gpu_cpus = specs[args.node_gpus]["gpu_cpus"]

    # Use current commit hash by default if neither commit nor branch is specified
    if not args.git_commit and not args.git_branch:
        args.git_commit = get_current_commit()
        if not args.git_commit:
            print(red("Error: Could not get current commit hash. Ensure you're in a git repository."))
            sys.exit(1)
        print(green(f"Using current commit: {args.git_commit}"))

    # If a branch is specified, look up its current commit on origin
    elif args.git_branch:
        original_branch_ref = args.git_branch
        # Ensure branch has origin/ prefix if not already
        if not args.git_branch.startswith("origin/") and not args.git_branch.startswith("refs/"):
            args.git_branch = f"origin/{args.git_branch}"

        # Get the current commit for this branch
        branch_commit = get_branch_commit(args.git_branch)
        if not branch_commit:
            print(red(f"Error: Failed to get commit for branch '{args.git_branch}'"))
            sys.exit(1)

        print(green(f"Using commit {branch_commit} from branch {original_branch_ref}"))
        args.git_commit = branch_commit

        # Let user know we're using the commit, not the branch reference
        print(yellow("Converting branch reference to specific commit hash for stability."))

        # Check if we're on a different branch locally (just as a warning)
        current_branch = get_current_branch()
        clean_branch_name = original_branch_ref.replace("origin/", "")
        if current_branch and current_branch != clean_branch_name and not args.skip_validation:
            print(
                yellow(
                    f"Note: You are currently on branch '{current_branch}' but submitted "
                    f"a job based on '{original_branch_ref}'."
                )
            )

    # Validate that the commit is pushed
    if not args.skip_push_check and not is_commit_pushed(args.git_commit):
        print(red(f"Error: Git commit {args.git_commit} has not been pushed."))
        print("Please push your commit or use --skip-push-check to bypass.")
        sys.exit(1)

    for _ in range(args.copies):
        submit_batch_job(args, task_args)


if __name__ == "__main__":
    main()
