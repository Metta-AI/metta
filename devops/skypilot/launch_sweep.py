#!/usr/bin/env -S uv run
import argparse
import logging
import shlex
import sys

import sky

import gitta as git
from devops.skypilot.launch import _validate_run_tool, _validate_sky_cluster_name
from devops.skypilot.utils.job_helpers import check_git_state, display_job_summary, launch_task, set_task_secrets
from metta.common.tool.tool_path import validate_module_path
from metta.common.util.fs import cd_repo_root
from metta.common.util.text_styles import red
from metta.tools.utils.auto_config import auto_run_name

logger = logging.getLogger("launch_sweep.py")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch a SkyPilot job for Ray sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s metta.tools.ray_sweep.RaySweepTool sweep_config.sweep_id=test
  %(prog)s experiments.recipes.my_recipe.sweep --gpus 4 --nodes 2 -- ray_address=...
        """,
    )

    parser.add_argument("module_path", help="Tool module path (e.g. metta.tools.ray_sweep.RaySweepTool)")
    parser.add_argument("--run", type=str, default=None, help="Run ID (defaults to auto generated)")
    parser.add_argument("--git-ref", type=str, default=None, help="Git commit or branch to launch")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--ray-head-port", type=int, default=6379, help="Port for Ray head")
    parser.add_argument("--ray-client-port", type=int, default=10001, help="Ray client port")
    parser.add_argument("--skip-git-check", action="store_true", help="Skip clean git tree check")
    parser.add_argument("--dry-run", action="store_true", help="Show summary and exit")
    parser.add_argument("--wandb-project", type=str, default="sweeps", help="WandB project name")
    parser.add_argument(
        "--heartbeat-timeout-seconds",
        type=int,
        default=300,
        help="Per-node heartbeat timeout in seconds (default: 300)",
    )
    return parser.parse_known_args()


def build_task(
    args: argparse.Namespace,
    run_id: str,
    tool_args: list[str],
    commit_hash: str,
) -> sky.Task:
    task = sky.Task.from_yaml("./devops/skypilot/config/sweep.yaml")

    env_updates = {
        "SWEEP_RUN_ID": run_id,
        "SWEEP_MODULE_PATH": args.module_path,
        "SWEEP_MODULE_ARGS": " ".join(shlex.quote(arg) for arg in tool_args),
        "RAY_HEAD_PORT": args.ray_head_port,
        "RAY_CLIENT_PORT": args.ray_client_port,
        "METTA_GIT_REF": commit_hash,
        "METTA_RUN_ID": run_id,
        "METTA_MODULE_PATH": args.module_path,
        "METTA_ARGS": " ".join(tool_args),
        "WANDB_PROJECT": args.wandb_project,
        "HEARTBEAT_TIMEOUT": args.heartbeat_timeout_seconds,
    }
    task = task.update_envs(env_updates)

    if args.gpus:
        task.set_resources_override({"accelerators": f"L4:{args.gpus}"})

    task.num_nodes = args.nodes
    task.name = run_id
    task.validate_name()

    # Note: set_task_secrets() is called just before launch, not here
    return task


def main() -> None:
    args, tool_args = parse_args()

    cd_repo_root()

    run_id = args.run or auto_run_name()
    logger.info("Using run id: %s", run_id)

    if not validate_module_path(args.module_path):
        sys.exit(1)

    filtered_tool_args = list(tool_args)
    filtered_tool_args.append(f"run={run_id}")

    _validate_run_tool(args.module_path, run_id, filtered_tool_args)

    if not _validate_sky_cluster_name(run_id):
        sys.exit(1)

    if args.git_ref:
        commit_hash = git.resolve_git_ref(args.git_ref)
        if not commit_hash:
            print(red(f"❌ Invalid git reference: '{args.git_ref}'"))
            sys.exit(1)
    else:
        commit_hash = git.get_current_commit()
        if not args.skip_git_check:
            error_message = check_git_state(commit_hash)
            if error_message:
                print(error_message)
                print("  - Skip check: add --skip-git-check flag")
                sys.exit(1)

    task = build_task(args, run_id, filtered_tool_args, commit_hash)

    display_job_summary(
        job_name=run_id,
        cmd=f"{args.module_path} (args: {filtered_tool_args})",
        task_args=[],
        commit_hash=commit_hash,
        git_ref=args.git_ref,
        timeout_hours=None,
        task=task,
    )

    if args.dry_run:
        print(red("Dry run: exiting"))
        return

    # Set secrets only when actually launching (not for dry-run)
    set_task_secrets(task)
    launch_task(task)


if __name__ == "__main__":
    main()
