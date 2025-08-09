#!/usr/bin/env -S uv run
import argparse
import copy
import shlex
import sys

import sky

from devops.skypilot.utils import (
    check_config_files,
    check_git_state,
    display_job_summary,
    launch_task,
    set_task_secrets,
)
from metta.common.util.cli import get_user_confirmation
from metta.common.util.fs import cd_repo_root
from metta.common.util.git import get_current_commit, validate_git_ref
from metta.common.util.text_styles import red


def patch_task(
    task: sky.Task,
    cpus: int | None,
    gpus: int | None,
    nodes: int | None,
    no_spot: bool = False,
    timeout_hours: float | None = None,
) -> sky.Task:
    overrides = {}
    if cpus:
        overrides["cpus"] = cpus
    if overrides:
        task.set_resources_override(overrides)
    if nodes:
        task.num_nodes = nodes

    new_resources_list = list(task.resources)

    if gpus:
        new_resources_list = []
        for res in list(task.resources):
            if not isinstance(res.accelerators, dict):
                # shouldn't happen with our current config
                raise Exception(f"Unexpected accelerator type: {res.accelerators}, {type(res.accelerators)}")

            patched_accelerators = copy.deepcopy(res.accelerators)
            patched_accelerators = {gpu_type: gpus for gpu_type in patched_accelerators.keys()}
            new_resources = res.copy(accelerators=patched_accelerators)
            new_resources_list.append(new_resources)

    if no_spot:
        new_resources_list = [res.copy(use_spot=False) for res in new_resources_list]

    if gpus or no_spot:
        task.set_resources(type(task.resources)(new_resources_list))

    # Add timeout configuration if specified
    if timeout_hours is not None:
        current_run_script = task.run or ""
        # Construct the command parts
        # timeout utility takes DURATION COMMAND [ARG]...
        # Here, COMMAND is 'bash', and its ARGs are '-c' and the script itself.
        timeout_command_parts = [
            "timeout",
            f"{timeout_hours}h",  # Use 'h' suffix for hours, timeout supports floats
            "bash",
            "-c",
            current_run_script,
        ]
        # shlex.join will correctly quote each part, especially current_run_script,
        # ensuring it's passed as a single argument to bash -c.
        task.run = shlex.join(timeout_command_parts)

    return task


def main():
    # To match other usage patterns we want to specify the run ID with `run=foo`` somewhere in the args
    # A named argument with argparse would end up as `--run=foo` which is not quite right
    run_id = None
    filtered_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("run="):
            run_id = arg[4:]  # Remove 'run=' prefix
        else:
            filtered_args.append(arg)

    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="Command to run")
    parser.add_argument("--git-ref", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-spot", action="store_true", help="Disable spot instances")
    parser.add_argument("--copies", type=int, default=1, help="Number of identical job copies to launch")
    parser.add_argument(
        "--heartbeat-timeout-seconds",
        type=int,
        default=600,
        help="Automatically terminate the job if no heartbeat signal is received for this many seconds",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help="Maximum job runtime in hours before automatic termination (supports decimals, e.g., 1.5 = 90 minutes)",
    )
    parser.add_argument("--skip-git-check", action="store_true", help="Skip git state validation")
    parser.add_argument("-c", "--confirm", action="store_true", help="Show confirmation prompt")
    parser.add_argument(
        "--github-token", type=str, default=None, help="GitHub token for posting status updates (repo:status scope)"
    )

    (args, cmd_args) = parser.parse_known_args(filtered_args)

    if run_id is None:
        parser.error("run= parameter is required")

    cd_repo_root()

    # check that the parsed args.git_ref provides a valid commit hash
    if args.git_ref:
        commit_hash = validate_git_ref(args.git_ref)
        if not commit_hash:
            print(red(f"❌ Invalid git reference: '{args.git_ref}'"))
            sys.exit(1)
    else:
        commit_hash = get_current_commit()

        # check that the commit has been pushed and there are no staged changes
        if not args.skip_git_check:
            error_message = check_git_state(commit_hash)
            if error_message:
                print(error_message)
                print("  - Skip check: add --skip-git-check flag")
                sys.exit(1)

    # check that the files referenced in the cmd exist
    if not check_config_files(cmd_args):
        sys.exit(1)

    assert commit_hash

    task = sky.Task.from_yaml("./devops/skypilot/config/sk_train.yaml")
    task = task.update_envs(
        dict(
            METTA_RUN_ID=run_id,
            METTA_CMD=args.cmd,
            METTA_CMD_ARGS=" ".join(cmd_args),
            METTA_GIT_REF=commit_hash,
            HEARTBEAT_TIMEOUT=args.heartbeat_timeout_seconds,
            GITHUB_TOKEN=args.github_token or "",
        )
    )
    task.name = run_id
    task.validate_name()

    task = patch_task(
        task,
        cpus=args.cpus,
        gpus=args.gpus,
        nodes=args.nodes,
        no_spot=args.no_spot,
        timeout_hours=args.max_runtime_hours,
    )
    set_task_secrets(task)

    if args.confirm:
        extra_details = {}
        if args.copies > 1:
            extra_details["copies"] = args.copies

        display_job_summary(
            job_name=run_id,
            cmd=args.cmd,
            task_args=cmd_args,
            commit_hash=commit_hash,
            git_ref=args.git_ref,
            timeout_hours=args.max_runtime_hours,
            task=task,
            **extra_details,
        )
        if not get_user_confirmation("Should we launch this task?"):
            sys.exit(0)

    # Launch the task(s)
    if args.copies == 1:
        launch_task(task, dry_run=args.dry_run)
    else:
        for _ in range(1, args.copies + 1):
            copy_task = copy.deepcopy(task)
            copy_task = copy_task.update_envs({"METTA_RUN_ID": run_id})
            copy_task.name = run_id
            copy_task.validate_name()
            launch_task(copy_task, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
