#!/usr/bin/env python3
"""
AWS Batch Command Line Interface

This script provides a command-line interface for interacting with AWS Batch resources.
It supports operations on job queues, compute environments, and jobs.

Usage:
    cmd.py [resource_type] [id] [command] [options]
    cmd.py [command] [id] [options]

Resource Types:
    - job-queue (jq): AWS Batch job queues
    - compute-environment (ce): AWS Batch compute environments
    - job (j): AWS Batch jobs
    - jobs: List jobs in the default queue (metta-jq)

Commands:
    - list (l): List resources (default if not specified)
    - info (d): Get detailed information about a resource
    - logs (ls): Get logs for a job
    - stop (s): Stop a job or compute environment
    - launch (st): Launch a job
"""

import sys
import argparse
from job_queue import list_job_queues, get_job_queue_info
from compute_environment import list_compute_environments, get_compute_environment_info, stop_compute_environment
from job import list_jobs, get_job_info, stop_job, launch_job
from job_logs import get_job_logs

# Define the valid commands and resource types
VALID_COMMANDS = ['list', 'l', 'info', 'd', 'logs', 'ls', 'stop', 's', 'launch', 'st']
VALID_RESOURCE_TYPES = ['job-queue', 'jq', 'compute-environment', 'ce', 'job', 'j', 'jobs', 'compute']

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='AWS Batch CLI')

    # First positional argument could be either a resource type or a command
    parser.add_argument('arg1', nargs='?', default='list',
                        help='Resource type or command')

    # Second positional argument could be either an ID or a command
    parser.add_argument('arg2', nargs='?', default=None,
                        help='Resource ID or command')

    # Third positional argument could be a command if the first two are resource type and ID
    parser.add_argument('arg3', nargs='?', default=None,
                        help='Command (if arg1 is resource type and arg2 is ID)')

    # Additional options
    parser.add_argument('--queue', '-q', help='Job queue name (default: metta-jq for job commands)')
    parser.add_argument('--max', '-m', type=int, default=5, help='Maximum number of items to return (default: 5)')
    parser.add_argument('--tail', '-t', action='store_true', help='Tail logs')
    parser.add_argument('--attempt', '-a', type=int, help='Job attempt index')
    parser.add_argument('--node', '-n', type=int, help='Node index for multi-node jobs')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    return parser.parse_args()

def normalize_command(cmd):
    """Normalize command aliases to their full form."""
    if cmd == 'l':
        return 'list'
    elif cmd == 'd':
        return 'info'
    elif cmd == 'ls':
        return 'logs'
    elif cmd == 's':
        return 'stop'
    elif cmd == 'st':
        return 'launch'
    return cmd

def normalize_resource_type(res_type):
    """Normalize resource type aliases to their full form."""
    if res_type == 'jq':
        return 'job-queue'
    elif res_type == 'ce':
        return 'compute-environment'
    elif res_type == 'j':
        return 'job'
    elif res_type == 'compute':
        return 'compute-environment'
    return res_type

def handle_compute_command(resource_id=None, command=None):
    """Handle the 'compute' command."""
    if command == 'info' or (resource_id and resource_id != 'all' and command != 'list'):
        # Get detailed info about a specific compute environment, including its instances
        get_compute_environment_info(resource_id)
    else:
        # List all compute environments with status, instance types, and number of instances
        list_compute_environments()

def main():
    """Main entry point for the AWS Batch CLI."""
    args = parse_args()

    # Special case for 'compute' command
    if args.arg1 == 'compute' or args.arg1 == 'c':
        if args.arg2 and args.arg2 != 'all':
            # If a specific compute environment is provided, get detailed info about it
            get_compute_environment_info(args.arg2)
        else:
            # Otherwise, list all compute environments
            list_compute_environments()
        return

    # Determine if we're using the new simplified syntax or the old syntax
    # New syntax: cmd.sh info <job_id>
    # Old syntax: cmd.sh job <job_id> info

    resource_type = None
    resource_id = None
    command = None

    # Check if arg1 is a command
    if args.arg1 in VALID_COMMANDS or normalize_command(args.arg1) in VALID_COMMANDS:
        # New syntax: cmd.sh <command> <id>
        command = normalize_command(args.arg1)
        resource_id = args.arg2

        # For info, logs, and stop commands, assume it's a job if no resource type is specified
        if command in ['info', 'logs', 'stop']:
            resource_type = 'job'
        # For list command with no ID, we need to determine what to list
        elif command == 'list' and not resource_id:
            # Default to listing jobs in the default queue
            resource_type = 'jobs'
        # For launch command, assume it's a job
        elif command == 'launch':
            resource_type = 'job'
    # Check if arg1 is a resource type
    elif args.arg1 in VALID_RESOURCE_TYPES or normalize_resource_type(args.arg1) in VALID_RESOURCE_TYPES:
        # Old syntax: cmd.sh <resource_type> <id> <command>
        resource_type = normalize_resource_type(args.arg1)
        resource_id = args.arg2
        command = normalize_command(args.arg3) if args.arg3 else 'list'
    else:
        # If arg1 is neither a command nor a resource type, assume it's a job ID for info
        resource_type = 'job'
        resource_id = args.arg1
        command = 'info'

    # Special case for 'jobs' resource type
    if resource_type == 'jobs':
        # If resource_id is provided, it might be a job queue name
        job_queue = resource_id if resource_id else 'metta-jq'
        list_jobs(job_queue=job_queue, max_jobs=args.max)
        return

    # Execute the appropriate command based on resource type and command
    if resource_type == 'job-queue':
        if command == 'list':
            list_job_queues()
        elif command == 'info':
            if not resource_id:
                print("Error: Job queue ID is required for info command")
                sys.exit(1)
            get_job_queue_info(resource_id, max_jobs=args.max)
        else:
            print(f"Error: Command '{command}' is not supported for job queues")
            sys.exit(1)

    elif resource_type == 'compute-environment':
        if command == 'list':
            list_compute_environments()
        elif command == 'info':
            if not resource_id:
                print("Error: Compute environment ID is required for info command")
                sys.exit(1)
            get_compute_environment_info(resource_id)
        elif command == 'stop':
            if not resource_id:
                print("Error: Compute environment ID is required for stop command")
                sys.exit(1)
            stop_compute_environment(resource_id)
        else:
            print(f"Error: Command '{command}' is not supported for compute environments")
            sys.exit(1)

    elif resource_type == 'job':
        # Set default job queue to metta-jq if not specified
        job_queue = args.queue if args.queue else 'metta-jq'

        if command == 'list':
            list_jobs(job_queue=job_queue, max_jobs=args.max)
        elif command == 'info':
            if not resource_id:
                print("Error: Job ID is required for info command")
                sys.exit(1)

            # Try to get job info
            job = get_job_info(resource_id)

            # If job not found, try to check if it's a compute environment or job queue
            if job is None:
                print(f"Checking if '{resource_id}' is a compute environment or job queue...")

                # Check if it's a compute environment
                try:
                    ce_info = get_compute_environment_info(resource_id)
                    if ce_info is not None:
                        return
                except Exception:
                    pass

                # Check if it's a job queue
                try:
                    jq_info = get_job_queue_info(resource_id, max_jobs=args.max)
                    if jq_info is not None:
                        return
                except Exception:
                    pass

                print(f"'{resource_id}' was not found as a job, compute environment, or job queue.")

        elif command == 'logs':
            if not resource_id:
                print("Error: Job ID is required for logs command")
                sys.exit(1)
            get_job_logs(resource_id, attempt_index=args.attempt, node_index=args.node,
                         tail=args.tail, debug=args.debug)
        elif command == 'stop':
            if not resource_id:
                print("Error: Job ID is required for stop command")
                sys.exit(1)
            stop_job(resource_id)
        elif command == 'launch':
            launch_job(job_queue=job_queue)
        else:
            print(f"Error: Command '{command}' is not supported for jobs")
            sys.exit(1)

if __name__ == '__main__':
    main()
