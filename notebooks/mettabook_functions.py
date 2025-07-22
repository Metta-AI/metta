import os
import re
import subprocess

import pandas as pd
import wandb
import yaml
from IPython.display import IFrame, display
from wandb.apis.public.runs import Run

from metta.common.util.collections import remove_none_values


def get_run(run_name: str, entity: str = "metta-research", project: str = "metta") -> Run | None:
    """
    Get a W&B run object by name.

    Args:
        run_name: W&B run name
        api: Optional W&B API instance (will create one if not provided)

    Returns:
        W&B Run object or None if error
    """
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"Error connecting to W&B: {str(e)}")
        print("Make sure you are connected to W&B: `metta status`")
        return None

    try:
        return api.run(f"{entity}/{project}/{run_name}")
    except Exception as e:
        print(f"Error getting run {run_name}: {str(e)}")
        return None


def _load_available_environments():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "sim", "all.yaml")
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
) -> dict:
    """
    Launch a training job on SkyPilot.

    Args:
        run_name: Name for the training run
        num_gpus: Number of GPUs to request
        num_cpus: Number of CPUs to request
        use_spot: Whether to use spot instances
        curriculum: Curriculum to use (defaults to first available)
        additional_args: Additional training arguments

    Returns:
        dict with keys: 'job_id', 'job_name', 'success', 'command', 'output'
    """
    if curriculum and curriculum not in _load_available_environments():
        raise ValueError(f"Curriculum {curriculum} not found. Available environments: {_load_available_environments()}")

    cmd_args = remove_none_values(
        {
            "gpu": num_gpus,
            "cpu": num_cpus,
            "no_spot": no_spot,
            "git_ref": git_ref,
            "skip_git_check": skip_git_check,
            "trainer.curriculum": curriculum,
            "dry_run": dry_run,
        }
    )

    cmd = [
        "./devops/skypilot/launch.py",
        "train",
        f"run={run_name}",
        *[f"--{k}={v}" for k, v in cmd_args.items()],
    ]

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
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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


def fetch_metrics(run_names: list[str], samples: int = 1000) -> dict[str, pd.DataFrame]:
    """
    Fetch metrics from W&B for multiple runs.

    Args:
        run_names: List of W&B run names
        last_n_points: Number of most recent points to fetch
        sample_rate: Sample every Nth data point (1 = all data)

    Returns:
        Dictionary mapping run_name -> metrics DataFrame
    """
    metrics_dfs = {}

    for run_name in run_names:
        run = get_run(run_name)
        if run is None:
            continue

        print(f"Fetching metrics for {run_name}: {run.state}, {run.created_at}\n{run.url}...")

        try:
            metrics_df: pd.DataFrame = run.history(samples=samples, pandas=True)  # type: ignore
            metrics_dfs[run_name] = metrics_df
            print(f"  Fetched {len(metrics_df)} data points.")

            if len(metrics_df) > 0 and "overview/reward" in metrics_df.columns:
                print(
                    f"  Reward: mean={metrics_df['overview/reward'].mean():.4f}, "
                    f"max={metrics_df['overview/reward'].max():.4f}"
                )
            print(f"  Access with `metrics_dfs['{run_name}']`")
            print("")

        except Exception as e:
            print(f"  Error: {str(e)}")
    return metrics_dfs


def show_replay(run_name: str, step: str | int = "last", width: int = 1000, height: int = 600) -> None:
    """
    Display a replay viewer for a specific run and step.

    Args:
        run_name: W&B run name
        step: "last" for most recent, "first" for earliest, or specific step number
        width: IFrame width in pixels
        height: IFrame height in pixels
    """
    run = get_run(run_name)
    if run is None:
        return

    replay_urls = _fetch_replay_urls_for_run(run)

    if not replay_urls:
        print(f"No replays found for {run_name}")
        return

    # Select the requested replay
    if step == "last":
        selected = replay_urls[-1]
    elif step == "first":
        selected = replay_urls[0]
    else:
        # Find replay closest to requested step
        target_step = int(step)
        selected = min(replay_urls, key=lambda r: abs(r["step"] - target_step))
        if selected["step"] != target_step:
            print(f"Note: Requested step {target_step}, showing closest available step {selected['step']}")

    print(f"Loading MettaScope viewer for {run_name} at step {selected['step']:,}...")
    print(f"\nDirect link: {selected['url']}")
    display(IFrame(src=selected["url"], width=width, height=height))


def get_available_replays(run_name: str) -> list[dict]:
    """
    Get list of available replay steps for a run.

    Args:
        run_name: W&B run name

    Returns:
        List of dicts with keys: 'step', 'url', 'label'
    """
    run = get_run(run_name)
    if run is None:
        return []

    return _fetch_replay_urls_for_run(run)


def _fetch_replay_urls_for_run(run) -> list[dict]:
    """Fetch replay URLs for a single W&B run."""
    files = run.files()
    replay_urls = []

    # Filter for replay HTML files
    replay_files = [f for f in files if "media/html/replays/link_" in f.name and f.name.endswith(".html")]

    if not replay_files:
        return []

    # Sort by step number
    def get_step_from_filename(file):
        match = re.search(r"link_(\d+)_", file.name)
        return int(match.group(1)) if match else 0

    replay_files.sort(key=get_step_from_filename)

    # Process files (limit to avoid too many)
    max_files = min(20, len(replay_files))
    recent_files = replay_files[-max_files:]

    for file in recent_files:
        try:
            # Download and read the HTML file
            with file.download(replace=True, root="/tmp") as f:
                content = f.read()
            match = re.search(r'<a[^>]+href="([^"]+)"', content)
            if match:
                href = match.group(1)
                if href:
                    step = get_step_from_filename(file)
                    replay_urls.append({"step": step, "url": href, "filename": file.name, "label": f"Step {step:,}"})
        except Exception:
            pass

    return replay_urls
