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
    get_user_confirmation,
    launch_task,
)
from metta.util.fs import cd_repo_root
from metta.util.git import get_current_commit


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
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="Command to run")
    parser.add_argument("run", help="Run ID")
    parser.add_argument("--git-ref", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-spot", action="store_true", help="Disable spot instances")
    parser.add_argument("--copies", type=int, default=1, help="Number of identical job copies to launch")
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=None,
        help="Automatically terminate the job after this many hours (supports decimals, e.g., 1.5 for 90 minutes)",
    )
    parser.add_argument("--skip-git-check", action="store_true", help="Skip git state validation")
    parser.add_argument("--confirm", action="store_true", help="Show confirmation prompt")
    parser.add_argument(
        "--id_suffix",
        dest="id_suffix",
        type=str,
        default=None,
        help="Suffix to append to run ID (e.g., --id_suffix=0 makes 'base_name.0')",
    )
    (args, cmd_args) = parser.parse_known_args()

    cd_repo_root()

    git_ref = args.git_ref
    if not git_ref:
        git_ref = get_current_commit()

    if not args.skip_git_check and not check_git_state(git_ref):
        sys.exit(1)

    if not check_config_files(cmd_args):
        sys.exit(1)

    run_id = args.run
    if args.id_suffix:
        run_id = f"{args.run}.{args.id_suffix}"

    task = sky.Task.from_yaml("./devops/skypilot/config/sk_train.yaml")
    task = task.update_envs(
        dict(
            METTA_RUN_ID=run_id,
            METTA_CMD=args.cmd,
            METTA_CMD_ARGS=" ".join(cmd_args),
            METTA_GIT_REF=git_ref,
        )
    )
    task.name = run_id
    task.validate_name()

    task = patch_task(
        task, cpus=args.cpus, gpus=args.gpus, nodes=args.nodes, no_spot=args.no_spot, timeout_hours=args.timeout_hours
    )

    if args.confirm:
        extra_details = {}
        if args.copies > 1:
            extra_details["copies"] = args.copies

        display_job_summary(
            job_name=run_id,
            cmd=args.cmd,
            task_args=cmd_args,
            git_ref=git_ref,
            timeout_hours=args.timeout_hours,
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
