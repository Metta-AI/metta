import json
import os
import subprocess

import yaml

from metta.common.util.collections import remove_none_values
from metta.common.util.fs import get_repo_root


def load_available_environments() -> list[str]:
    config_path = os.path.join(get_repo_root(), "configs", "sim", "all.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    environments = []
    if "simulations" in config:
        for sim_config in config["simulations"].values():
            if "env" in sim_config:
                env_path = sim_config["env"]
                environments.append(env_path)
    return environments


def launch_training(
    run_name: str,
    num_gpus: int | None = None,
    num_cpus: int | None = None,
    no_spot: bool | None = None,
    curriculum: str | None = None,
    git_ref: str | None = None,
    skip_git_check: bool | None = None,
    additional_args: list[str] | None = None,
    dry_run: bool | None = None,
    wandb_tags: list[str] | None = None,
) -> dict:
    if curriculum and curriculum not in load_available_environments():
        raise ValueError(
            f"Curriculum {curriculum} not found. Available environments: {load_available_environments()}"
        )

    cmd_args = remove_none_values(
        {
            "gpu": num_gpus,
            "cpu": num_cpus,
            "no_spot": no_spot,
            "git_ref": git_ref,
            "skip_git_check": skip_git_check,
            "dry_run": dry_run,
        }
    )

    cmd = [
        "./devops/skypilot/launch.py",
        "train",
        f"run={run_name}",
        *[f"--{k}={v}" for k, v in cmd_args.items()],
    ]

    if curriculum:
        cmd.append(f"trainer.curriculum={curriculum}")
    if wandb_tags:
        cmd.append(f"+wandb.tags={json.dumps(wandb_tags)}")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching training job: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    result = {
        "job_id": None,
        "job_name": run_name,
        "success": False,
        "command": " ".join(cmd),
        "output": [],
    }

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=get_repo_root(),
        )

        for line in process.stdout or []:
            result["output"].append(line.strip())
            print(line, end="")
            if "Job ID:" in line or "sky-" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("sky-") and "-" in part[4:]:
                        result["job_id"] = part

        process.wait()
        result["success"] = process.returncode == 0

        if result["success"]:
            print("\n✓ Job launched successfully!")
            if result["job_id"]:
                print(f"Job ID: {result['job_id']}")
        else:
            print(f"\n✗ Launch failed with return code: {process.returncode}")

    except Exception as e:
        print(f"\n✗ Error launching job: {str(e)}")
        result["output"].append(f"Error: {str(e)}")

    return result
