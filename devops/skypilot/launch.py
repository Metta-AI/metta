#!/usr/bin/env -S uv run
# ruff: noqa: E402
# ^ Imports must come after warnings.filterwarnings() to suppress Pydantic warnings from SkyPilot

import copy
import json
import logging
import re
import subprocess
import warnings
from typing import Annotated, Any, Literal, Optional

import typer
from sky import exceptions
from sky.utils.common_utils import check_cluster_name_is_valid

from metta.common.util.log_config import init_logging

# Suppress Pydantic warnings from SkyPilot dependencies before importing sky
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import sky
import yaml

import gitta as git
from devops.skypilot.utils.job_helpers import (
    display_task_summary,
    launch_task,
    set_task_secrets,
)
from metta.common.tool.tool_path import parse_two_token_syntax, validate_module_path
from metta.common.util.cli import get_user_confirmation
from metta.common.util.fs import cd_repo_root
from metta.common.util.text_styles import red
from metta.tools.utils.auto_config import auto_run_name

logger = logging.getLogger("devops.skypilot.launch")


def _validate_sky_task_name(run_name: str) -> bool:
    """Validate that we will meet Sky's task naming requirements."""
    pattern = r"^[a-zA-Z]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?$"
    valid = bool(re.match(pattern, run_name))

    try:
        check_cluster_name_is_valid(run_name)
    except exceptions.InvalidClusterNameError:
        print(red(f"[VALIDATION] ❌ Invalid run name: '{run_name}'"))
        print("Sky task names must:")
        print("  - Start with a letter (not a number)")
        print("  - Contain only letters, numbers, dashes, underscores, or dots")
        print("  - End with a letter or number")
        print()

    return valid


def _validate_run_tool(module_path: str, args: list) -> bool:
    """Validate that run.py can successfully create a tool config with the given arguments.

    Returns:
        True if validation succeeds, False otherwise
    """
    # Build the run.py command
    run_cmd = ["uv", "run", "--active", "tools/run.py", module_path, "--dry-run"]

    # Add args if provided
    if args:
        run_cmd.extend(args)
    try:
        subprocess.run(run_cmd, capture_output=True, text=True, check=True)
        print("[VALIDATION] ✅ Configuration validation successful")
        return True
    except subprocess.CalledProcessError as e:
        print(red("[VALIDATION] ❌ Configuration validation failed"))
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(red(e.stderr))
        return False
    except FileNotFoundError:
        print(red("[VALIDATION] ❌ Could not find run.py or uv command"))
        return False


def check_git_state(commit_hash: str) -> str | None:
    error_lines: list[str] = []

    has_changes, status_output = git.has_uncommitted_changes()
    if has_changes:
        error_lines.append(red("❌ You have uncommitted changes that won't be reflected in the cloud job."))
        error_lines.append("Options:")
        error_lines.append("  - Commit: git add . && git commit -m 'your message'")
        error_lines.append("  - Stash: git stash")
        error_lines.append("\nDebug:\n" + status_output)
        return "\n".join(error_lines)

    if not git.is_commit_pushed(commit_hash):
        commit_display = commit_hash[:8]
        error_lines.append(
            red(f"❌ Commit {commit_display} hasn't been pushed and won't be reflected in the cloud job.")
        )
        error_lines.append("Options:")
        error_lines.append("  - Push: git push")
        return "\n".join(error_lines)

    return None


def patch_task(
    task: sky.Task,
    cpus: int | None,
    gpus: int | None,
    nodes: int | None,
    no_spot: bool = False,
) -> None:
    overrides: dict[str, Any] = {}
    if cpus:
        overrides["cpus"] = cpus
    if overrides:
        task.set_resources_override(overrides)
    if nodes:
        task.num_nodes = nodes

    new_resources_list: list[sky.Resources] = list(task.resources)

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


app = typer.Typer(
    rich_markup_mode="rich",
)


@app.command()
def main(
    module_path: Annotated[
        str,
        typer.Argument(
            help="Module path to run (e.g., arena.train or recipes.experiment.arena.train, "
            "or two-token syntax like 'train arena'). "
            "Any arguments following the module path will be passed to the tool."
        ),
    ],
    # Launch-specific flags
    run_id: Annotated[Optional[str], typer.Option("--run", help="Run ID for the job")] = None,
    git_ref: Optional[str] = None,
    gpus: Optional[int] = None,
    nodes: Optional[int] = None,
    cpus: Optional[int] = None,
    dry_run: Annotated[bool, typer.Option(help="Show job summary without launching")] = False,
    dump_config: Annotated[
        Optional[Literal["json", "yaml"]],
        typer.Option(help="Dump task configuration in specified format and exit"),
    ] = None,
    spot: Annotated[bool, typer.Option(help="Use spot instances")] = True,
    copies: Annotated[int, typer.Option(help="Number of identical job copies to launch")] = 1,
    heartbeat_timeout_seconds: Annotated[
        int,
        typer.Option(
            "--heartbeat-timeout-seconds",
            "--hb",
            help="Automatically terminate the job if no heartbeat signal is received for this many seconds",
        ),
    ] = 300,
    max_runtime_hours: Annotated[
        Optional[float],
        typer.Option(
            "-t",
            "--max-runtime-hours",
            help="Maximum job runtime in hours before automatic termination (supports decimals, e.g., 1.5 = 90 minutes)",
        ),
    ] = None,
    skip_git_check: Annotated[bool, typer.Option(help="Skip git state validation and GitHub API calls")] = False,
    confirm: Annotated[bool, typer.Option(help="Show confirmation prompt")] = False,
    github_pat: Annotated[
        Optional[str], typer.Option(help="GitHub PAT token for posting status updates (repo scope)")
    ] = None,
    discord_webhook_url: Annotated[
        Optional[str], typer.Option(help="Discord webhook URL for status update channel")
    ] = None,
    run_ci_tests: Annotated[bool, typer.Option(help="Run NCCL and job restart tests")] = False,
    tool_args: Annotated[
        Optional[list[str]], typer.Argument(help="Tool arguments. Will be passed to tools/run.py.")
    ] = None,
):
    """

    Main entry point for launch.py.

    [bold green]Examples:[/bold green]

    [bold yellow]Basic (single-token):[/bold yellow]
    launch.py arena.train run=test_123 trainer.total_timesteps=100000

    [bold yellow]Basic (two-token syntax):[/bold yellow]
    launch.py train arena run=test_123 trainer.total_timesteps=100000

    [bold yellow]Mix of launch flags and tool args:[/bold yellow]
    launch.py arena.train --gpus 2 --nodes 4 -- run=test_123 trainer.steps=1000
    launch.py train arena --gpus 2 --nodes 4 -- run=test_123 trainer.steps=1000
    """

    # First, we need to separate launch flags from tool args
    # We'll parse known args only, allowing unknown ones to be passed as tool args

    # Handle two-token syntax (e.g., 'train arena' → 'arena.train')
    tool_args = tool_args or []
    second_token = tool_args[0] if tool_args else None
    resolved_path, args_consumed = parse_two_token_syntax(module_path, second_token)
    module_path = resolved_path
    tool_args = tool_args[args_consumed:]  # Skip consumed args

    # Handle run ID extraction
    filtered_args: list[str] = []

    for arg in tool_args:
        if arg.startswith("run="):
            # Extract the run ID
            new_run_id = arg[4:]
            if run_id is not None and new_run_id != run_id:
                raise ValueError(f"Conflicting run IDs specified: '{run_id}' and '{new_run_id}'")
            run_id = new_run_id
        else:
            filtered_args.append(arg)

    if run_id is None:
        run_id = auto_run_name()
        logger.info(f"Using auto-generated run ID: {run_id}")
        logger.info("To specify a run ID pass run=foo")

    filtered_args.append(f"run={run_id}")

    cd_repo_root()

    # check that the parsed args.git_ref provides a valid commit hash
    if git_ref:
        commit_hash = git.resolve_git_ref(git_ref)
        if not commit_hash:
            print(red(f"❌ Invalid git reference: '{git_ref}'"))
            raise typer.Exit(1)
    else:
        commit_hash = git.get_current_commit()

        # check that the commit has been pushed and there are no staged changes
        if not skip_git_check:
            error_message = check_git_state(commit_hash)
            if error_message:
                print(error_message)
                print("  - Skip check: add --skip-git-check flag")
                raise typer.Exit(1)

    # Validate module path (supports shorthand like 'arena.train' or two-token 'train arena')
    if not validate_module_path(module_path):
        print(f"❌ Invalid module path: '{module_path}'")
        print("Module path should be like 'arena.train' or 'recipes.experiment.arena.train'")
        raise typer.Exit(1)

    assert commit_hash

    # Validate the run.py tool configuration early to catch errors before setting up the task
    if not _validate_run_tool(module_path, filtered_args):
        raise typer.Exit(1)

    # Validate the provided run name
    if not _validate_sky_task_name(run_id):
        raise typer.Exit(1)

    task = sky.Task.from_yaml("./devops/skypilot/config/skypilot_run.yaml")
    task._user_specified_yaml = None

    # Prepare environment variables including status parameters
    env_updates: dict[str, str] = dict(
        METTA_RUN_ID=run_id,
        METTA_MODULE_PATH=module_path,
        METTA_ARGS=" ".join(filtered_args),
        METTA_GIT_REF=commit_hash,
        TEST_JOB_RESTART="true" if run_ci_tests else "false",
        TEST_NCCL="true" if run_ci_tests else "false",
    )
    if heartbeat_timeout_seconds:
        env_updates["HEARTBEAT_TIMEOUT"] = str(heartbeat_timeout_seconds)
    if max_runtime_hours:
        env_updates["MAX_RUNTIME_HOURS"] = str(max_runtime_hours)

    task = task.update_envs(env_updates)
    task.name = run_id
    task.validate_name()

    patch_task(
        task,
        cpus=cpus,
        gpus=gpus,
        nodes=nodes,
        no_spot=not spot,
    )

    # Handle --dump-config option
    if dump_config:
        config_dict = task.to_yaml_config()

        if dump_config == "json":
            print(json.dumps(config_dict, indent=2))
        elif dump_config == "yaml":
            # Pretty print the YAML
            print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
        return

    display_task_summary(
        task=task,
        commit_hash=commit_hash,
        git_ref=git_ref,
        skip_github=skip_git_check,
        copies=copies,
    )

    # For --dry-run, just exit after showing summary
    if dry_run:
        print(red("Dry run: exiting"))
        return

    # For --confirm, ask for confirmation
    if confirm and not get_user_confirmation("Should we launch this task?"):
        return

    # Set secrets only when actually launching (not for dry-run or dump-config)
    set_task_secrets(task)
    task.update_secrets(
        dict(
            DISCORD_WEBHOOK_URL=discord_webhook_url or "",
            GITHUB_PAT=github_pat or "",
        )
    )

    # Launch the task(s)
    for _ in range(copies):
        launch_task(task)

    return


if __name__ == "__main__":
    init_logging()
    app()
