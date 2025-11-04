#!/usr/bin/env -S uv run
# ruff: noqa: E402
# ^ Imports must come after warnings.filterwarnings() to suppress Pydantic warnings from SkyPilot

import argparse
import copy
import json
import logging
import os
import re
import subprocess
import sys
import uuid
import warnings
from typing import Any

# Suppress Pydantic warnings from SkyPilot dependencies before importing sky
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import sky
import yaml

import gitta as git
from devops.skypilot.utils.job_helpers import (
    check_git_state,
    display_job_summary,
    launch_task,
    open_job_log_from_request_id,
    set_task_secrets,
)
from metta.common.tool.tool_path import parse_two_token_syntax, validate_module_path
from metta.common.util.cli import get_user_confirmation
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.common.util.fs import cd_repo_root
from metta.common.util.text_styles import red, yellow
from metta.tools.utils.auto_config import auto_run_name

logger = logging.getLogger(__name__)

# Known tool modules that we explicitly validate
KNOWN_TOOL_MODULES = {
    "devops.skypilot.tools.nccl",
    "devops.skypilot.tools.restart_test",
    "devops.skypilot.tools.run",
}


def _validate_tool_module(tool_module: str) -> bool:
    """Validate tool module, with warnings for unknown tools.

    For known tools (like devops.skypilot.tools.nccl), performs strict validation.
    For unknown tools, logs a warning and allows them to pass (assumed valid).
    """
    # Check if this is a known tool that requires strict validation
    if tool_module in KNOWN_TOOL_MODULES:
        # Known tool - validate strictly
        try:
            if not validate_module_path(tool_module):
                print(red(f"[VALIDATION] ❌ Known tool module failed validation: '{tool_module}'"))
                return False
            return True
        except Exception as e:
            print(red(f"[VALIDATION] ❌ Known tool module validation error: '{tool_module}'"))
            print(red(f"             Error: {e}"))
            return False

    # Unknown tool - try to validate but warn and allow if validation fails
    try:
        if validate_module_path(tool_module):
            # Validation passed
            return True
    except Exception:
        # Validation failed due to exception
        pass

    # Validation failed for unknown tool - warn but allow
    print(yellow(f"[VALIDATION] ⚠️  Unrecognized tool module: '{tool_module}'"))
    print(yellow("             Assuming valid and continuing..."))
    print(yellow("             If this is a new standard tool, add it to KNOWN_TOOL_MODULES in launch.py"))
    return True


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
        print(red(f"[VALIDATION] ❌ Invalid run name: '{run_name}'"), flush=True)
        print("Sky cluster names must:", flush=True)
        print("  - Start with a letter (not a number)", flush=True)
        print("  - Contain only letters, numbers, dashes, underscores, or dots", flush=True)
        print("  - End with a letter or number", flush=True)
        print(flush=True)

    return valid


def _validate_run_tool(module_path: str, run_id: str, tool_args: list) -> bool:
    """Validate that run.py can successfully create a tool config with the given arguments.

    Returns:
        True if validation succeeds, False otherwise
    """
    # Build the run.py command
    run_cmd = ["uv", "run", "--active", "tools/run.py", module_path, "--dry-run"]

    # Add args if provided
    if tool_args:
        run_cmd.extend(tool_args)

    # Always include run= for validation
    if not any(arg.startswith("run=") for arg in tool_args):
        run_cmd.append(f"run={run_id}")

    try:
        subprocess.run(run_cmd, capture_output=True, text=True, check=True)
        print("[VALIDATION] ✅ Configuration validation successful")
        return True
    except subprocess.CalledProcessError as e:
        print(red("[VALIDATION] ❌ Configuration validation failed"), flush=True)
        if e.stdout:
            print(e.stdout, flush=True)
        if e.stderr:
            print(red(e.stderr), flush=True)
        return False
    except FileNotFoundError:
        print(red("[VALIDATION] ❌ Could not find run.py or uv command"), flush=True)
        return False


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


def main() -> int:
    """Main entry point for launch.py.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using --tool flag (recommended for tools/run.py):
  %(prog)s --tool arena.train run=test_123 trainer.total_timesteps=100000 --gpus 2

  # Two-token syntax with --tool:
  %(prog)s --tool train arena run=test_123 trainer.total_timesteps=100000 --gpus 2

  # Arbitrary command:
  %(prog)s "pytest tests/rl/" --gpus 1

  # Both styles support resource flags:
  %(prog)s --tool arena.train run=foo --gpus 2 --nodes 4
        """,
    )

    # First, we need to separate launch flags from tool args
    # We'll parse known args only, allowing unknown ones to be passed as tool args
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
        "Supports both single-token (arena.train) and two-token (train arena) syntax. "
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
    parser.add_argument("-jl", "--job-log", action="store_true", help="Open job log after launch")
    parser.add_argument("--verbose", action="store_true", help="Print detailed request IDs and log instructions")

    # Parse known args to handle --tool with remaining args
    args, remaining_args = parser.parse_known_args()

    # When using --tool, argparse may assign positional args to 'command'
    # Treat those as tool args instead
    if args.tool and args.command:
        remaining_args = [args.command] + remaining_args
        args.command = None

    # Validate that either --tool or command is provided
    if not args.tool and not args.command:
        parser.error("Must specify either --tool or command")

    # NORMALIZATION: If command looks like a module path, treat it as --tool
    # This allows both `launch.py arena.train` and `launch.py --tool arena.train` to work the same
    if not args.tool and args.command:
        # Check if command looks like a module path:
        # - Contains a dot (e.g., arena.train)
        # - OR first word + remaining_args[0] could be two-token syntax
        command_parts = args.command.split()
        first_word = command_parts[0] if command_parts else args.command

        # Check if it looks like a module path (has dots and no spaces in first token)
        if "." in first_word and " " not in first_word:
            # Normalize to --tool
            args.tool = first_word
            # Everything after the module path becomes tool args
            remaining_args = command_parts[1:] + remaining_args
            args.command = None
        # Check if it could be two-token syntax (e.g., "train arena")
        elif remaining_args and not first_word.startswith("-"):
            # Try to resolve as two-token syntax
            second_token = remaining_args[0] if remaining_args[0] and not remaining_args[0].startswith("-") else None
            if second_token:
                resolved_path, args_consumed = parse_two_token_syntax(first_word, second_token)
                # If it resolved to something different, it's likely two-token syntax
                if resolved_path != first_word and args_consumed > 0:
                    # Normalize to --tool
                    args.tool = first_word
                    # Don't consume args yet, let the --tool handler do it below
                    args.command = None

    # Build command string
    if args.tool:
        # Using --tool: handle two-token syntax if applicable
        module_path = args.tool
        second_token = remaining_args[0] if remaining_args else None
        resolved_path, args_consumed = parse_two_token_syntax(module_path, second_token)
        module_path = resolved_path
        tool_args = remaining_args[args_consumed:]  # Skip consumed args

        # Validate module
        if not _validate_tool_module(module_path):
            sys.exit(1)

        # Combine tool module with remaining args
        cmd_parts = [module_path] + tool_args
        command = f"uv run ./tools/run.py {' '.join(cmd_parts)}"
        use_torchrun = True
    else:
        # Using direct command
        command = args.command
        tool_args = []
        module_path = None
        # Auto-detect if command uses tools/run.py
        use_torchrun = "tools/run.py" in command

    # Extract run ID from --run flag, tool_args, or command string
    run_id = args.run

    # Try to extract run= from tool_args if using --tool
    if run_id is None and args.tool:
        for arg in tool_args:
            if arg.startswith("run="):
                run_id = arg.split("=", 1)[1]
                break

    # Try to extract run= from command if not provided via flag
    if run_id is None:
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
            print(red(f"❌ Invalid git reference: '{args.git_ref}'"), flush=True)
            return 1
    else:
        commit_hash = git.get_current_commit()

        # check that the commit has been pushed and there are no staged changes
        if not args.skip_git_check:
            error_message = check_git_state(commit_hash)
            if error_message:
                print(error_message, flush=True)
                print("  - Skip check: add --skip-git-check flag", flush=True)
                return 1

    assert commit_hash

    # Validate the run.py tool configuration early if using --tool
    if args.tool and module_path:
        if not _validate_run_tool(module_path, run_id, tool_args):
            return 1

    # Validate the provided run name
    if not _validate_sky_cluster_name(run_id):
        return 1

    # Load the job configuration
    task = sky.Task.from_yaml("./devops/skypilot/recipes/job.yaml")

    # Prepare environment variables including status parameters
    env_updates = dict(
        METTA_RUN_ID=run_id,
        METTA_CMD=command,
        METTA_USE_TORCHRUN="true" if use_torchrun else "false",
        METTA_GIT_REF=commit_hash or "main",
        HEARTBEAT_TIMEOUT=args.heartbeat_timeout_seconds,
        MAX_RUNTIME_HOURS=args.max_runtime_hours,
        DISCORD_WEBHOOK_URL=args.discord_webhook_url,  # enables discord notification
        GITHUB_PAT=args.github_pat,  # enables github status update
        WANDB_PROJECT=os.environ.get("WANDB_PROJECT", METTA_WANDB_PROJECT),
        WANDB_ENTITY=os.environ.get("WANDB_ENTITY", METTA_WANDB_ENTITY),
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
        return 0

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
        return 0

    # For --confirm, ask for confirmation
    if args.confirm and not get_user_confirmation("Should we launch this task?"):
        return 0

    # Set secrets only when actually launching (not for dry-run or dump-config)
    set_task_secrets(task)

    # Launch the task(s)
    def prepare_task(base_task: sky.Task, env_updates: dict[str, Any], run_id: str, copies: int) -> sky.Task:
        prepared_task = copy.deepcopy(base_task).update_envs(env_updates)
        if copies > 1:
            suffix = f"_{uuid.uuid4().hex[:6]}"
            max_name_length = 63
            trimmed_base = run_id[: max_name_length - len(suffix)]
            prepared_task.name = f"{trimmed_base}{suffix}"
        else:
            prepared_task.name = run_id
        prepared_task.validate_name()
        return prepared_task

    request_ids = [
        launch_task(prepare_task(task, env_updates, run_id, args.copies), verbose=args.verbose)
        for _ in range(args.copies)
    ]

    if args.job_log:
        open_job_log_from_request_id(request_ids[0])

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)
