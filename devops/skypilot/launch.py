#!/usr/bin/env python3

import argparse
import copy
import re
import subprocess
import sys

import sky


def blue(text: str):
    BLUE = "\033[1;34m"
    RESET = "\033[0;0m"

    return f"{BLUE}{text}{RESET}"


def print_tip(text: str):
    print(blue(text), file=sys.stderr)


def patch_task(task: sky.Task, cpus: int | None, gpus: int | None, nodes: int | None) -> sky.Task:
    overrides = {}
    if cpus:
        overrides["cpus"] = cpus
    if overrides:
        task.set_resources_override(overrides)
    if nodes:
        task.num_nodes = nodes

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

        task.set_resources(type(task.resources)(new_resources_list))

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
    (args, cmd_args) = parser.parse_known_args()

    git_ref = args.git_ref
    if not git_ref:
        git_ref = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    task = sky.Task.from_yaml("./devops/skypilot/config/sk_train.yaml")
    task = task.update_envs(
        dict(
            METTA_RUN_ID=args.run,
            METTA_CMD=args.cmd,
            METTA_CMD_ARGS=" ".join(cmd_args),
            METTA_GIT_REF=git_ref,
        )
    )

    task = patch_task(task, cpus=args.cpus, gpus=args.gpus, nodes=args.nodes)

    if args.dry_run:
        print_tip("DRY RUN.")
        print_tip("Tip: Pipe this command to `| yq -P .` to get the pretty yaml config.\n")
        print(task.to_yaml_config())
        return

    request_id = sky.jobs.launch(task)
    (job_id, _) = sky.get(request_id)

    server_url = sky.server.common.get_server_url()
    # strip username and password from server_url
    server_url = re.sub("https://.*@", "https://", server_url)
    job_url = f"{server_url}/dashboard/jobs/{job_id}"
    print("\nJob submitted successfully!")

    # Note: direct urls don't work in skypilot dashboard yet, this always opens clusters list.
    # Hopefully this will be fixed soon.
    print(f"Open {blue(job_url)} to track your job.")
    print("To sign in, use credentials from your ~/.skypilot/config.yaml file.")


if __name__ == "__main__":
    main()
