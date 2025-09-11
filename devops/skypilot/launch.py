#!/usr/bin/env -S uv run
import argparse
import copy
import json
import logging
import subprocess
import sys

import sky
import yaml

import gitta as git
from devops.skypilot.utils.job_helpers import (
    check_config_files,
    check_git_state,
    display_job_summary,
    launch_task,
    set_task_secrets,
)
from metta.common.util.cli import get_user_confirmation
from metta.common.util.fs import cd_repo_root
from metta.common.util.text_styles import red
from metta.tools.utils.auto_config import auto_run_name

logger = logging.getLogger("launch.py")


def _validate_run_tool(module_path: str, run_id: str, filtered_args: list, overrides: list) -> None:
    """Validate that run.py can successfully create a tool config with the given arguments."""
    # Build the run.py command
    run_cmd = ["uv", "run", "--active", "tools/run.py", module_path, "--dry-run"]

    # Add args if provided (run= is already included in filtered_args)
    if filtered_args:
        run_cmd.extend(["--args"] + filtered_args)

    # Add overrides if provided
    if overrides:
        run_cmd.extend(["--overrides"] + overrides)

    output = ""
    success = False
    try:
        # Run the validation command
        output = subprocess.check_output(run_cmd, text=True)
        success = True
        print("[VALIDATION] ✅ Configuration validation successful")
    except AssertionError as e:
        print(red("[VALIDATION] ❌ Configuration validation failed"))
        print(red(f"[VALIDATION] {str(e)}"))
    except FileNotFoundError as e:
        print(red("[VALIDATION] ❌ Could not find run.py or uv command"))
        print(red(f"[VALIDATION] {str(e)}"))

    with open("/tmp/run_cmd.txt", "w") as f:
        f.write(output)
        logger.info("[VALIDATION] Output saved to /tmp/run_cmd.txt")

    if not success:
        sys.exit(1)


def patch_task(
    task: sky.Task,
    cpus: int | None,
    gpus: int | None,
    nodes: int | None,
    no_spot: bool = False,
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

    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module_path", help="Module path to run (e.g., experiments.recipes.arena.train)")
    parser.add_argument("--run", type=str, default=None, help="Run ID for the job")
    parser.add_argument("--args", nargs="*", default=[], help="Arguments to pass to the module")
    parser.add_argument("--overrides", nargs="*", default=[], help="Overrides to apply to the config")
    parser.add_argument("--git-ref", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Show job summary without launching")
    parser.add_argument(
        "--dump-config",
        choices=["json", "yaml", "pretty"],
        help="Dump task configuration in specified format and exit",
    )
    parser.add_argument("--no-spot", action="store_true", help="Disable spot instances")
    parser.add_argument("--copies", type=int, default=1, help="Number of identical job copies to launch")
    parser.add_argument(
        "-hb",
        "--heartbeat-timeout-seconds",
        type=int,
        default=300,
        help="Automatically terminate the job if no heartbeat signal is received for this many seconds",
    )
    parser.add_argument(
        "-t",
        "--max-runtime-hours",
        type=float,
        default=None,
        help="Maximum job runtime in hours before automatic termination (supports decimals, e.g., 1.5 = 90 minutes)",
    )
    parser.add_argument("--skip-git-check", action="store_true", help="Skip git state validation")
    parser.add_argument("-c", "--confirm", action="store_true", help="Show confirmation prompt")
    parser.add_argument(
        "--github-pat", type=str, default=None, help="GitHub PAT token for posting status updates (repo scope)"
    )
    parser.add_argument(
        "--discord-webhook-url", type=str, default=None, help="Discord webhook URL for status update channel"
    )
    parser.add_argument(
        "--run-ci-tests",
        action="store_true",
        help="Run NCCL and job restart tests",
    )

    args = parser.parse_args()

    # Handle run ID - it can come from --run flag or from args as run=value
    run_id = args.run
    filtered_args = []

    # Check if run= is in the args and extract it
    for arg in args.args:
        if arg.startswith("run="):
            if run_id is None:  # Only use if --run wasn't specified
                run_id = arg[4:]  # Remove 'run=' prefix
            # Don't add run= to filtered_args - we'll add it back later
        else:
            filtered_args.append(arg)

    # If run is still not specified, error out
    if run_id is None:
        run_id = auto_run_name()
        logger.info(f"Using auto-generated run ID: {run_id}")
        logger.info("To specify a run ID, use --run=foo or pass run=foo in --args")

    # Always add run= to the filtered args so it gets passed to run.py
    filtered_args.append(f"run={run_id}")

    cd_repo_root()

    # check that the parsed args.git_ref provides a valid commit hash
    if args.git_ref:
        commit_hash = git.validate_git_ref(args.git_ref)
        if not commit_hash:
            print(red(f"❌ Invalid git reference: '{args.git_ref}'"))
            sys.exit(1)
    else:
        commit_hash = git.get_current_commit()

        # check that the commit has been pushed and there are no staged changes
        if not args.skip_git_check:
            error_message = check_git_state(commit_hash)
            if error_message:
                print(error_message)
                print("  - Skip check: add --skip-git-check flag")
                sys.exit(1)

    # check that the files referenced in the module path exist
    # Convert module path to file path for validation
    module_file_path = args.module_path.replace(".", "/") + ".py"
    if not check_config_files([module_file_path]):
        print(red(f"❌ Module path '{args.module_path}' does not exist (looking for {module_file_path})"))
        sys.exit(1)

    assert commit_hash

    # Validate the run.py tool configuration early to catch errors before setting up the task
    _validate_run_tool(args.module_path, run_id, filtered_args, args.overrides)

    task = sky.Task.from_yaml("./devops/skypilot/config/skypilot_run.yaml")

    # Prepare environment variables including status parameters
    env_updates = dict(
        METTA_RUN_ID=run_id,
        METTA_MODULE_PATH=args.module_path,
        METTA_ARGS=" ".join(filtered_args),
        METTA_OVERRIDES=" ".join(args.overrides),
        METTA_GIT_REF=commit_hash,
        HEARTBEAT_TIMEOUT=args.heartbeat_timeout_seconds,
        GITHUB_PAT=args.github_pat,
        MAX_RUNTIME_HOURS=args.max_runtime_hours,
        DISCORD_WEBHOOK_URL=args.discord_webhook_url,
        TEST_JOB_RESTART="true" if args.run_ci_tests else "false",
        TEST_NCCL="true" if args.run_ci_tests else "false",
    )

    env_updates = {k: v for k, v in env_updates.items() if v is not None}
    task = task.update_envs(env_updates)
    task.name = run_id
    task.validate_name()

    task = patch_task(
        task,
        cpus=args.cpus,
        gpus=args.gpus,
        nodes=args.nodes,
        no_spot=args.no_spot,
    )
    set_task_secrets(task)

    # Handle --dump-config option
    if args.dump_config:
        task_config = task.to_yaml_config()

        # Check if it's already a dict or a YAML string
        if isinstance(task_config, dict):
            config_dict = task_config
        else:
            config_dict = yaml.safe_load(task_config)

        if args.dump_config == "json":
            print(json.dumps(config_dict, indent=2))
        elif args.dump_config == "yaml":
            # Output raw YAML (compact form)
            if isinstance(task_config, dict):
                print(yaml.dump(config_dict))
            else:
                print(task_config)
        elif args.dump_config == "pretty":
            # Pretty print the YAML
            print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))

    extra_details = {}
    if args.copies > 1:
        extra_details["copies"] = args.copies

    display_job_summary(
        job_name=run_id,
        cmd=f"{args.module_path} (args: {filtered_args}, overrides: {args.overrides})",
        task_args=[],  # We're showing args differently now
        commit_hash=commit_hash,
        git_ref=args.git_ref,
        timeout_hours=args.max_runtime_hours,
        task=task,
        **extra_details,
    )

    # For --dry-run, just exit after showing summary
    if args.dry_run:
        print(red("Dry run: exiting"))
        sys.exit(0)

    # For --confirm, ask for confirmation
    if args.confirm and not get_user_confirmation("Should we launch this task?"):
        sys.exit(0)

    # Launch the task(s)
    if args.copies == 1:
        launch_task(task)
    else:
        for _ in range(1, args.copies + 1):
            copy_task = copy.deepcopy(task)
            copy_task = copy_task.update_envs({"METTA_RUN_ID": run_id})
            copy_task.name = run_id
            copy_task.validate_name()
            launch_task(copy_task)


if __name__ == "__main__":
    main()
