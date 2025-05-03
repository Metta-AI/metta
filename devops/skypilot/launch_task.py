import argparse
import netrc
import os
import sys

import sky

from metta.util.colorama import bold, cyan, green, red, use_colors, yellow
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


def validate_task(args, task_args):
    critical_files = [
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
        if task_arg.startswith("agent="):
            agent_value = task_arg.split("=", 1)[1]
            critical_files.append(f"./configs/agent/{agent_value}.yaml")
        elif task_arg.startswith("trainer="):
            trainer_value = task_arg.split("=", 1)[1]
            critical_files.append(f"./configs/trainer/{trainer_value}.yaml")
        elif task_arg.startswith("trainer.env="):
            env_value = task_arg.split("=", 1)[1]
            critical_files.append(f"./configs/{env_value}.yaml")
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
        print(red("These changes will not be reflected in the SkyPilot environment."))
        print(yellow("Commit or stash your changes, or use --skip-validation to bypass this check."))
        return False

    # Get the git reference
    git_ref = args.git_commit if args.git_commit else args.git_branch

    # Display task details
    print(f"\n{divider}")
    print(bold("Task will be submitted with the following details:"))
    print(f"{divider}")

    print(f"Task Name: {args.run}")
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
        print(bold("DRY RUN - No task will be submitted"))
        print(f"{divider}")
        return False

    if not args.skip_validation:
        response = input("Should we submit this task to SkyPilot? (Y/n): ")
        if response.lower() not in ["", "y", "yes"]:
            print(yellow("Task submission cancelled by user."))
            return False

    return True


def submit_skypilot_task(args, task_args):
    try:
        netrc_info = netrc.netrc(os.path.expanduser("~/.netrc"))
        auth_data = netrc_info.authenticators("api.wandb.ai")
        if auth_data is None:
            raise ValueError("No api.wandb.ai entry found in .netrc file")
        wandb_key = auth_data[2]
        if not wandb_key:
            raise ValueError("WANDB_API_KEY not found in .netrc file")
    except Exception as e:
        print(red(f"Error reading WANDB_API_KEY: {e}"))
        sys.exit(1)

    # Create task
    task = sky.Task.from_yaml("devops/skypilot/config/task.yaml")

    # Set environment variables
    task.update_envs(
        {
            "WANDB_API_KEY": wandb_key,
            "WANDB_HOST": args.run,
            "METTA_HOST": args.run,
            "JOB_NAME": args.run,
            "METTA_USER": os.environ.get("USER", "unknown"),
            "RUN_ID": args.run,
            "CMD": args.cmd,
            "GIT_REF": args.git_commit if args.git_commit else args.git_branch,
            "TASK_ARGS": " ".join(task_args),
        }
    )

    # Update resources
    specs = get_specs()
    if args.node_ram_gb is None:
        args.node_ram_gb = specs[args.node_gpus]["node_ram_gb"]
    if args.gpu_cpus is None:
        args.gpu_cpus = specs[args.node_gpus]["gpu_cpus"]

    task.set_resources(
        {
            "cloud": "aws",
            "region": "us-east-1",
            "instance_type": "g6.4xlarge",
            "use_spot": True,
            "num_nodes": args.num_nodes,
            "accelerators": {"A10G": args.node_gpus},
        }
    )

    # Validate task and get user confirmation
    should_submit = validate_task(args, task_args)

    if not should_submit:
        return

    # Submit the task
    sky.launch(task, cluster_name=args.run)

    print(f"Submitted task {args.run} with ID {green(bold(args.run))}")

    if task_args:
        print(yellow("\nTask Arguments:"))
        for i, arg in enumerate(task_args):
            print(f"  {i + 1}. {cyan(arg)}")


def main():
    parser = argparse.ArgumentParser(description="Launch a SkyPilot task with a wandb key.")
    parser.add_argument("--cluster", default="metta")
    parser.add_argument("--run", required=True)
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
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show task details without submitting")
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
        if not args.git_branch.startswith("origin/") and not args.git_branch.startswith("refs/"):
            args.git_branch = f"origin/{args.git_branch}"

        branch_commit = get_branch_commit(args.git_branch)
        if not branch_commit:
            print(red(f"Error: Failed to get commit for branch '{args.git_branch}'"))
            sys.exit(1)

        print(green(f"Using commit {branch_commit} from branch {original_branch_ref}"))
        args.git_commit = branch_commit
        print(yellow("Converting branch reference to specific commit hash for stability."))

        current_branch = get_current_branch()
        clean_branch_name = original_branch_ref.replace("origin/", "")
        if current_branch and current_branch != clean_branch_name and not args.skip_validation:
            print(
                yellow(
                    f"Note: You are currently on branch '{current_branch}' but submitted "
                    f"a task based on '{original_branch_ref}'."
                )
            )

    # Validate that the commit is pushed
    if not args.skip_validation and not is_commit_pushed(args.git_commit):
        print(red(f"Error: Git commit {args.git_commit} has not been pushed."))
        print("Please push your commit or use --skip-validation to bypass.")
        sys.exit(1)

    for _ in range(args.copies):
        submit_skypilot_task(args, task_args)


if __name__ == "__main__":
    main()
