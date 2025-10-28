#!/usr/bin/env -S uv run
import argparse
import copy
import json
import logging
import re
import shlex
import subprocess
import sys
import textwrap

import sky
import yaml
from sky import clouds as sky_clouds
from sky.backends import CloudVmRayBackend

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


def _validate_run_tool(module_path: str, run_id: str, filtered_args: list) -> None:
    """Validate that run.py can successfully create a tool config with the given arguments."""
    # Build the run.py command
    run_cmd = ["uv", "run", "--active", "tools/run.py", module_path, "--dry-run"]

    # Add args if provided (run= is already included in filtered_args)
    if filtered_args:
        run_cmd.extend(filtered_args)
    try:
        subprocess.run(run_cmd, capture_output=True, text=True, check=True)
        print("[VALIDATION] ✅ Configuration validation successful")
    except subprocess.CalledProcessError as e:
        print(red("[VALIDATION] ❌ Configuration validation failed"))
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


def _build_ray_launch_task(
    *,
    base_task: sky.Task,
    run_id: str,
    commit_hash: str,
    args: argparse.Namespace,
    controller_cmd: str,
) -> sky.Task:
    """Create a Ray-ready Sky task that starts Ray cluster and runs controller on head node."""
    ray_task = sky.Task(
        setup=base_task.setup,
        envs=base_task.envs,
        secrets=base_task.secrets,
        file_mounts=base_task.file_mounts,
        storage_mounts=base_task.storage_mounts,
        workdir=base_task.workdir,
    )

    # Combined script: start Ray + run controller on head, just Ray on workers
    ray_port = args.ray_port
    combined_script = textwrap.dedent(
        f"""
        echo "=========================================="
        echo "[STEP 1] Minimal Ray Sweep Test Starting"
        echo "=========================================="
        echo "Hostname: $(hostname)"
        echo "Date: $(date)"
        echo "PWD: $PWD"
        echo "USER: $USER"
        echo "SKYPILOT_NODE_RANK: ${{SKYPILOT_NODE_RANK:-0}}"
        echo "SKYPILOT_NODE_IPS: ${{SKYPILOT_NODE_IPS:-NOT SET}}"

        echo ""
        echo "[STEP 2] Sleeping for 5 minutes to allow SSH access..."
        echo "You can SSH in now with: sky ssh <cluster-name>"
        echo "Sleeping until: $(date -d '+5 minutes' 2>/dev/null || date)"
        echo ""

        sleep 300

        echo ""
        echo "[STEP 3] Sleep complete, exiting successfully"
        echo "==========================================="
        exit 0

        # ORIGINAL CODE COMMENTED OUT FOR INCREMENTAL TESTING:
        # set -euo pipefail
        # set -x
        # cd /workspace/metta
        # if [ -f .venv/bin/activate ]; then
        #     . .venv/bin/activate
        # fi
        # ...etc

        if [[ "${{SKYPILOT_NODE_RANK:-0}}" == "0" ]]; then
            echo "[Ray Sweep] Starting head node on port {ray_port}"
            echo "[Ray Sweep] PWD: $(pwd)"
            echo "[Ray Sweep] USER: $(whoami)"
            echo "[Ray Sweep] Free memory: $(free -h | grep Mem)"

            # Check if Ray is running before attempting to stop
            echo "[Ray Sweep] Checking for existing Ray processes..."
            if ray status >/dev/null 2>&1; then
                echo "[Ray Sweep] Existing Ray cluster found, stopping gracefully first..."
                ray stop || echo "[Ray Sweep] Graceful stop failed, trying force stop..."
                sleep 2
            fi

            # Final cleanup with force flag
            echo "[Ray Sweep] Ensuring all Ray processes are stopped..."
            ray stop --force 2>&1 | grep -v "No such file or directory" || true
            sleep 3

            echo "[Ray Sweep] Starting Ray head with memory limits..."
            ray start --head \
              --port {ray_port} \
              --disable-usage-stats \
              --dashboard-host=0.0.0.0 \
              --object-store-memory=100000000 \
              --num-gpus={args.ray_gpus_per_node} \
              --verbose || {{
                echo "[Ray Sweep] ERROR: Ray start failed with exit code $?"
                echo "[Ray Sweep] Checking Ray logs..."
                ls -la /tmp/ray/ 2>/dev/null || echo "No /tmp/ray directory"
                tail -100 /tmp/ray/session_latest/logs/raylet.err 2>/dev/null || echo "No raylet.err"
                tail -100 /tmp/ray/session_latest/logs/raylet.out 2>/dev/null || echo "No raylet.out"
                exit 1
            }}

            sleep 5

            echo "[Ray Sweep] Ray head started successfully"
            ray status

            # Wait for worker nodes to connect
            echo "[Ray Sweep] Waiting for worker nodes to join..."
            sleep 15

            echo "[Ray Sweep] Current cluster state:"
            ray status

            # Run sweep controller on head node
            echo "[Ray Sweep] Running controller: {controller_cmd}"
            {controller_cmd}

            # After sweep completes, stop Ray on head
            echo "[Ray Sweep] Sweep complete, stopping Ray head"
            ray stop
        else
            echo "[Ray Sweep] Worker node ${{SKYPILOT_NODE_RANK}} waiting for head node to be ready..."

            # Wait for head node to start Ray and be reachable
            # Poll until Ray head is accepting connections (max 2 minutes)
            max_attempts=24  # 24 * 5s = 120s timeout
            attempt=0
            while [ $attempt -lt $max_attempts ]; do
                if nc -z -w 5 $HEAD_IP {ray_port} 2>/dev/null; then
                    echo "[Ray Sweep] Head node is reachable on port {ray_port}"
                    sleep 5  # Extra grace period for Ray to fully initialize
                    break
                fi
                attempt=$((attempt + 1))
                echo "[Ray Sweep] Waiting for head node... (attempt $attempt/$max_attempts)"
                sleep 5
            done

            if [ $attempt -eq $max_attempts ]; then
                echo "[Ray Sweep] ERROR: Head node did not become reachable within 120 seconds"
                exit 1
            fi

            echo "[Ray Sweep] Starting worker node ${{SKYPILOT_NODE_RANK}} -> $HEAD_IP:{ray_port}"

            # Clean up any existing Ray processes (same as head node)
            echo "[Ray Sweep] Checking for existing Ray processes on worker..."
            if ray status >/dev/null 2>&1; then
                echo "[Ray Sweep] Existing Ray found, stopping gracefully..."
                ray stop || echo "[Ray Sweep] Graceful stop failed, trying force stop..."
                sleep 2
            fi
            ray stop --force 2>&1 | grep -v "No such file or directory" || true
            sleep 2

            echo "[Ray Sweep] Executing: ray start --address $HEAD_IP:{ray_port} --disable-usage-stats --object-store-memory=100000000 --num-gpus={args.ray_gpus_per_node} --block"
            if ! ray start --address "$HEAD_IP:{ray_port}" --disable-usage-stats \
                --object-store-memory=100000000 \
                --num-gpus={args.ray_gpus_per_node} \
                --block; then
                echo "[Ray Sweep] ERROR: Failed to start Ray worker node"
                exit 1
            fi
        fi
        """
    ).strip()
    ray_task.run = combined_script
    ray_task.num_nodes = args.ray_num_nodes
    ray_task.name = run_id
    ray_task.validate_name()

    accelerator = args.ray_accelerator
    resources = sky.Resources(
        cloud=sky_clouds.AWS(),
        accelerators={accelerator: args.ray_gpus_per_node},
        cpus=args.ray_cpus_per_node,
        use_spot=not args.no_spot,
        image_id="docker:metta:latest",
        job_recovery={"strategy": "EAGER_NEXT_REGION", "max_restarts_on_errors": 20},
    )
    ray_task.set_resources(resources)

    env_updates = dict(base_task.envs or {})
    env_updates.update(
        {
            "METTA_RUN_ID": run_id,
            "METTA_GIT_REF": commit_hash,
            "METTA_MODULE_PATH": args.module_path or "",
            "METTA_ARGS": "",
            "RAY_PORT": str(ray_port),
        }
    )

    # Add WandB project if specified
    if args.wandb_project:
        env_updates["WANDB_PROJECT"] = args.wandb_project

    ray_task = ray_task.update_envs(env_updates)
    return ray_task


def launch_ray_sweep(
    *,
    args: argparse.Namespace,
    run_id: str,
    commit_hash: str,
    ray_args: list[str],
) -> None:
    """Launch a Ray cluster and execute the sweep controller in a single stage."""
    if args.copies != 1:
        print(red("Ray sweeps do not support --copies; launch one sweep per command."))
        sys.exit(1)

    # Build controller command first (needed for task construction)
    controller_args: list[str] = [
        "--ray-address",
        f"ray://127.0.0.1:{args.ray_port}",
        "--experiment-id",
        run_id,
    ]
    if args.module_path:
        controller_args.extend(["--module-path", args.module_path])
    if args.ray_num_samples is not None:
        controller_args.extend(["--num-samples", str(args.ray_num_samples)])

    # Filter ray_args to remove flags we're already handling
    # This prevents duplicate --module-path, --num-samples, etc.
    filtered_ray_args = []
    skip_next = False
    skip_flags = {"--module-path", "--num-samples", "--ray-address", "--experiment-id"}
    for i, arg in enumerate(ray_args):
        if skip_next:
            skip_next = False
            continue
        if arg in skip_flags:
            skip_next = True  # Skip this flag and its value
            continue
        filtered_ray_args.append(arg)

    controller_args.extend(filtered_ray_args)

    controller_cmd_parts = ["uv", "run", "python", "metta/sweep/ray/ray_controller.py", *controller_args]
    controller_cmd = shlex.join(controller_cmd_parts)

    # Build Ray task with combined bootstrap + controller execution
    base_task = sky.Task.from_yaml("./devops/skypilot/config/skypilot_run.yaml")
    ray_task = _build_ray_launch_task(
        base_task=base_task,
        run_id=run_id,
        commit_hash=commit_hash,
        args=args,
        controller_cmd=controller_cmd,
    )
    set_task_secrets(ray_task)

    display_job_summary(
        job_name=run_id,
        cmd=f"Ray sweep controller: {controller_cmd}",
        task_args=ray_args,
        commit_hash=commit_hash,
        git_ref=args.git_ref,
        timeout_hours=args.max_runtime_hours,
        task=ray_task,
        nodes=args.ray_num_nodes,
        gpus_per_node=args.ray_gpus_per_node,
    )

    if args.dry_run:
        print(red("Dry run: exiting"))
        sys.exit(0)

    if args.confirm and not get_user_confirmation("Should we launch this Ray sweep?"):
        sys.exit(0)

    backend = CloudVmRayBackend()

    print("[RAY] Launching Ray cluster...")
    print(f"[RAY] Cluster name: {run_id}")
    print(f"[RAY] Controller command: {controller_cmd}")
    sky.launch(
        ray_task,
        cluster_name=run_id,
        backend=backend,
        retry_until_up=True,
    )

    print()
    print("[RAY] ✓ Cluster launched successfully!")
    print("[RAY] The sweep is now running on the cluster in the background.")
    print()
    print("[RAY] To monitor progress:")
    print(f"[RAY]   - SSH into cluster: sky ssh {run_id}")
    print(f"[RAY]   - View logs: sky logs {run_id}")
    print(f"[RAY]   - Check status: sky status {run_id}")
    print()
    print("[RAY] When complete, stop the cluster:")
    print(f"[RAY]   sky down {run_id}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic:
  %(prog)s arena.train run=test_123 trainer.total_timesteps=100000

  # Mix of launch flags and tool args:
  %(prog)s arena.train --gpus 2 --nodes 4 -- run=test_123 trainer.steps=1000
        """,
    )

    # First, we need to separate launch flags from tool args
    # We'll parse known args only, allowing unknown ones to be passed as tool args
    parser.add_argument(
        "module_path",
        nargs="?",
        default=None,
        help="Module path to run (e.g., arena.train or experiments.recipes.arena.train). "
        "Any arguments following the module path will be passed to the tool.",
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
    parser.add_argument(
        "--ray-sweep",
        action="store_true",
        help="Launch a Ray sweep instead of a single run.",
    )
    parser.add_argument(
        "--ray-num-nodes",
        type=int,
        default=4,
        help="Number of Ray nodes (default: 4).",
    )
    parser.add_argument(
        "--ray-gpus-per-node",
        type=int,
        default=4,
        help="GPUs per node for Ray sweep (default: 4).",
    )
    parser.add_argument(
        "--ray-accelerator",
        type=str,
        default="A10G",
        help="GPU accelerator type for Ray nodes (default: A10G).",
    )
    parser.add_argument(
        "--ray-cpus-per-node",
        type=int,
        default=32,
        help="CPUs per node for Ray sweep (default: 32).",
    )
    parser.add_argument(
        "--ray-port",
        type=int,
        default=6379,
        help="Port used for the Ray head node (default: 6379).",
    )
    parser.add_argument(
        "--ray-num-samples",
        type=int,
        default=None,
        help="Number of Ray Tune samples to run (defaults to sweep_config.max_trials).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name for Ray sweep (sets WANDB_PROJECT env var on cluster).",
    )

    # Use parse_known_args to handle both launch flags and tool args
    args, tool_args = parser.parse_known_args()

    # Handle run ID extraction
    run_id = args.run
    filtered_args = []

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

    ray_args = filtered_args.copy()

    if not args.ray_sweep:
        filtered_args.append(f"run={run_id}")

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

    if not _validate_sky_cluster_name(run_id):
        sys.exit(1)

    if args.ray_sweep:
        if args.module_path and not validate_module_path(args.module_path):
            sys.exit(1)
        launch_ray_sweep(
            args=args,
            run_id=run_id,
            commit_hash=commit_hash,
            ray_args=ray_args,
        )
        return

    if not args.module_path:
        parser.error("module_path is required unless --ray-sweep is specified.")

    # Validate module path (supports shorthand like 'arena.train')
    if not validate_module_path(args.module_path):
        sys.exit(1)

    assert commit_hash

    # Validate the run.py tool configuration early to catch errors before setting up the task
    _validate_run_tool(args.module_path, run_id, filtered_args)

    task = sky.Task.from_yaml("./devops/skypilot/config/skypilot_run.yaml")

    # Prepare environment variables including status parameters
    env_updates = dict(
        METTA_RUN_ID=run_id,
        METTA_MODULE_PATH=args.module_path,
        METTA_ARGS=" ".join(filtered_args),
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
        cmd=f"{args.module_path} (args: {filtered_args})",
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
