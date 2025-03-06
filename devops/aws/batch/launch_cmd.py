#!/usr/bin/env python3
"""
AWS Batch Launch Command

This script provides a command-line interface for launching AWS Batch jobs
using the launch_task module.

Usage:
    launch_cmd.py --run RUN_ID --cmd COMMAND [options]
"""

import sys
import os

try:
    from devops.aws.batch import launch_task
except ImportError as e:
    print(f"Error: Could not import devops.aws.batch.launch_task: {str(e)}")
    print("Please ensure that the module is available in your Python path")
    sys.exit(1)

def main():
    """Main entry point for the AWS Batch Launch CLI."""
    # Just call the main function from launch_task if it exists
    if hasattr(launch_task, 'main'):
        launch_task.main()
    else:
        # Otherwise, run the code from the module's __main__ section
        import argparse

        parser = argparse.ArgumentParser(description='Launch an AWS Batch task with a wandb key.')
        parser.add_argument('--cluster', default="metta", help='The name of the ECS cluster.')
        parser.add_argument('--run', required=True, help='The run id.')
        parser.add_argument('--cmd', required=True, choices=["train", "sweep", "evolve"], help='The command to run.')
        parser.add_argument('--git-branch', default=None, help='The git branch to use for the task. If not specified, will use the current commit.')
        parser.add_argument('--git-commit', default=None, help='The git commit to use for the task. If not specified, will use the current commit.')
        parser.add_argument('--mettagrid-branch', default=None, help='The mettagrid branch to use for the task. If not specified, will use the current commit.')
        parser.add_argument('--mettagrid-commit', default=None, help='The mettagrid commit to use for the task. If not specified, will use the current commit.')
        parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs per node to use for the task.')
        parser.add_argument('--node-gpus', type=int, default=4, help='Number of GPUs per node to use for the task.')
        parser.add_argument('--gpu-cpus', type=int, default=6, help='Number of CPUs per GPU (vCPUs will be 2x this value).')
        parser.add_argument('--cpu-ram-gb', type=int, default=20, help='RAM per node in GB.')
        parser.add_argument('--copies', type=int, default=1, help='Number of job copies to submit.')
        parser.add_argument('--profile', default="stem", help='AWS profile to use. If not specified, uses the default profile.')
        parser.add_argument('--job-queue', default="metta-jq", help='AWS Batch job queue to use.')
        parser.add_argument('--skip-push-check', action='store_true', help='Skip checking if commits have been pushed.')
        args, task_args = parser.parse_known_args()

        args.num_nodes = max(1, args.gpus // args.node_gpus)

        # Set default commit values if not specified
        if args.git_branch is None and args.git_commit is None:
            args.git_commit = launch_task.get_current_commit()

        if args.mettagrid_branch is None and args.mettagrid_commit is None:
            args.mettagrid_commit = launch_task.get_current_commit("deps/mettagrid")

        # Submit the job
        for i in range(args.copies):
            launch_task.submit_batch_job(args, task_args)

if __name__ == '__main__':
    main()
