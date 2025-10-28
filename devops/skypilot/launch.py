#!/usr/bin/env -S uv run
import argparse
import copy
import json
import logging
import re
import subprocess
import sys

import sky
import yaml

import gitta as git
from devops.skypilot.utils.job_helpers import (
    check_git_state,
    display_job_summary,
    launch_task,
    set_task_secrets,
)
from metta.common.tool.tool_path import validate_module_path
from metta.common.util.cli import get_user_confirmation
from metta.common.util.fs import cd_repo_root
from metta.common.util.text_styles import red
from metta.tools.utils.auto_config import auto_run_name

logger = logging.getLogger("launch.py")


def _validate_sky_cluster_name(run_name: str) -> bool:
    """Validate that we will meet Sky's cluster naming requirements.

    Sky requires cluster names to:
    - Start with a letter (a-z or A-Z)
    - Contain only letters, numbers, dashes, underscores, or dots
    - End with a letter or number
    """
    # Sky's regex pattern: [a-zA-Z]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?
    pattern = r"^[a-zA-Z]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?$"
    valid = bool(re.match(pattern, run_name))

    if not valid:
        print(red(f"[VALIDATION] ❌ Invalid run name: '{run_name}'"))
        print("Sky cluster names must:")
        print("  - Start with a letter (not a number)")
        print("  - Contain only letters, numbers, dashes, underscores, or dots")
        print("  - End with a letter or number")
        print()

    return valid


def _extract_module_from_command(command: str) -> str | None:
    """Extract module path from command if it uses tools/run.py pattern.

    Returns module path if found, None otherwise.
    """
    # Match patterns like: "uv run ./tools/run.py arena.train ..." or "tools/run.py module.path ..."
    import re

    pattern = r"(?:uv\s+run\s+)?(?:\./)?tools/run\.py\s+([a-zA-Z0-9_.]+)"
    match = re.search(pattern, command)
    if match:
        return match.group(1)
    return None


def _validate_command(command: str) -> None:
    """Validate command by attempting dry-run if it uses tools/run.py.

    If the command uses tools/run.py with a module, validate the module.
    Otherwise, skip validation (arbitrary commands are allowed).
    """
    module_path = _extract_module_from_command(command)

    if not module_path:
        print("[VALIDATION] ⏭️  Skipping validation (command doesn't use tools/run.py)")
        return

    # Validate the module
    if not validate_module_path(module_path):
        sys.exit(1)

    # Attempt dry-run validation
    run_cmd = ["uv", "run", "--active", "tools/run.py", module_path, "--dry-run"]

    # Extract any additional args from command (after the module)
    import re

    pattern = r"(?:uv\s+run\s+)?(?:\./)?tools/run\.py\s+[a-zA-Z0-9_.]+\s+(.*)"
    match = re.search(pattern, command)
    if match:
        args_str = match.group(1).strip()
        if args_str:
            import shlex

            run_cmd.extend(shlex.split(args_str))

    try:
        subprocess.run(run_cmd, capture_output=True, text=True, check=True)
        print(f"[VALIDATION] ✅ Module '{module_path}' validated successfully")
    except subprocess.CalledProcessError as e:
        print(red(f"[VALIDATION] ❌ Module '{module_path}' validation failed"))
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(red(e.stderr))
        sys.exit(1)
    except FileNotFoundError:
        print(red("[VALIDATION] ❌ Could not find run.py or uv command"))
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using --tool flag (recommended for tools/run.py):
  %(prog)s --tool arena.train run=test_123 trainer.total_timesteps=100000 --gpus 2

  # Arbitrary command:
  %(prog)s "pytest tests/rl/" --gpus 1

  # Both styles support resource flags:
  %(prog)s --tool arena.train run=foo --gpus 2 --nodes 4
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        help="Command to execute on remote cluster (required if --tool not used). "
        "Will be wrapped with devops/run.sh for torchrun setup if it uses tools/run.py.",
    )

    parser.add_argument(
        "--tool",
        type=str,
        help="Run a tool from tools/run.py. This auto-prefixes 'uv run ./tools/run.py' and validates the module. "
        "Example: --tool arena.train run=foo trainer.steps=1000",
    )

    # Launch-specific flags
    parser.add_argument("--run", type=str, default=None, help="Run ID for the job")
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

    # Parse known args to handle --tool with remaining args
    args, remaining_args = parser.parse_known_args()

    # Validate that either --tool or command is provided (not both, not neither)
    if args.tool and args.command:
        parser.error("Cannot specify both --tool and command. Use one or the other.")
    if not args.tool and not args.command:
        parser.error("Must specify either --tool or command")

    # Build command string
    if args.tool:
        # Using --tool: validate module and build command
        if not validate_module_path(args.tool):
            sys.exit(1)

        # Combine tool module with remaining args
        cmd_parts = [args.tool] + remaining_args
        command = f"uv run ./tools/run.py {' '.join(cmd_parts)}"
    else:
        # Using direct command
        command = args.command

    # Extract run ID from --run flag or from command string
    run_id = args.run

    # Try to extract run= from command if not provided via flag
    if run_id is None:
        import re

        # Match run=value (handling various formats: run=foo, run="foo bar", etc.)
        match = re.search(r"run=([^\s]+)", command)
        if match:
            run_id = match.group(1)

    # Generate auto run ID if still not found
    if run_id is None:
        run_id = auto_run_name()
        logger.info(f"Using auto-generated run ID: {run_id}")
        logger.info("To specify a run ID, add 'run=foo' to your command or use --run flag")

    # Ensure command includes run= (append if missing)
    if "run=" not in command:
        command = f"{command} run={run_id}"

    cd_repo_root()

    # check that the parsed args.git_ref provides a valid commit hash
    if args.git_ref:
        commit_hash = git.resolve_git_ref(args.git_ref)
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

    assert commit_hash

    # Validate command (skip if we used --tool, already validated)
    if not args.tool:
        _validate_command(command)
    else:
        print(f"[VALIDATION] ✅ Module '{args.tool}' validated successfully")

    # Validate the run name for Sky cluster naming requirements
    if not _validate_sky_cluster_name(run_id):
        sys.exit(1)

    task = sky.Task.from_yaml("./devops/skypilot/config/skypilot_run.yaml")

    # Prepare environment variables including status parameters
    env_updates = dict(
        METTA_RUN_ID=run_id,
        METTA_CMD=command,
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
        cmd=command,
        task_args=[],
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
