#!/usr/bin/env python
"""
FastAPI backend for Sweep Dashboard
Serves real data from WandB and Sky jobs
"""

import os
import subprocess
import sys
import warnings

# Add the metta module to path
sys.path.append(os.path.abspath("../.."))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import the same utilities used in the Dash version
from metta.sweep.wandb_utils import deep_clean, get_active_sweep_runs, get_sweep_runs
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration (can be overridden via environment variables)
DEFAULT_ENTITY = os.getenv("WANDB_ENTITY", "metta-research")
DEFAULT_PROJECT = os.getenv("WANDB_PROJECT", "metta")
DEFAULT_HOURLY_COST = float(os.getenv("HOURLY_COST", "4.6"))
MAX_OBSERVATIONS = int(os.getenv("MAX_OBSERVATIONS", "1000"))


class SweepConfig(BaseModel):
    entity: str = DEFAULT_ENTITY
    project: str = DEFAULT_PROJECT
    sweep_name: str
    max_observations: int = MAX_OBSERVATIONS
    hourly_cost: float = DEFAULT_HOURLY_COST


class LaunchWorkerRequest(BaseModel):
    num_gpus: int
    sweep_name: str


def flatten_nested_dict(d, parent_key="", sep="."):
    """Recursively flatten a nested dictionary structure."""
    items = []

    if not isinstance(d, dict):
        return {parent_key: d} if parent_key else {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def extract_observations_to_dict(observations):
    """Convert protein observations to a list of dictionaries."""
    all_rows = []

    for obs in observations:
        if obs.get("is_failure", False):
            continue

        row_data = {}

        # Get suggestion
        suggestion = obs.get("suggestion", {})

        # Flatten the suggestion dictionary
        if isinstance(suggestion, dict) and suggestion:
            flattened = flatten_nested_dict(suggestion)
            row_data.update(flattened)

        # Add metrics
        row_data["score"] = obs.get("objective", np.nan)
        row_data["cost"] = obs.get("cost", np.nan)
        row_data["runtime"] = obs.get(
            "time_total", obs.get("cost", 0) * 3600.0
        )  # Convert hours to seconds if needed
        row_data["timestamp"] = obs.get("timestamp", obs.get("created_at", ""))
        row_data["run_name"] = obs.get("run_name", "")
        row_data["run_id"] = obs.get("run_id", "")
        row_data["status"] = (
            "finished" if not obs.get("is_failure", False) else "failed"
        )

        all_rows.append(row_data)

    return all_rows


@app.get("/api/sweeps")
async def get_available_sweeps():
    """Get list of available sweeps"""
    # In production, this would query WandB API for available sweeps
    # For now, return some example sweep names
    return {
        "sweeps": [
            "protein-optimization-2024",
            "hyperparameter-sweep-v3",
            "architecture-search-latest",
        ]
    }


@app.get("/api/sweeps/{sweep_name}")
async def get_sweep_data(
    sweep_name: str,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    max_observations: int = MAX_OBSERVATIONS,
):
    """Get sweep data from WandB"""
    try:
        print(f"Loading sweep data for {sweep_name}")
        print(f"Entity: {entity}, Project: {project}")

        # Get completed runs with observations from WandB
        runs = get_sweep_runs(sweep_name=sweep_name, entity=entity, project=project)
        print(f"Loaded {len(runs)} completed runs")

        # Get active runs separately
        active_runs_raw = get_active_sweep_runs(
            sweep_name=sweep_name, entity=entity, project=project
        )
        print(f"Loaded {len(active_runs_raw)} active runs")

        # Debug: Uncomment to see active run structure
        if active_runs_raw and False:  # Set to True to enable debug
            print("\n=== DEBUG: First Active Run Structure ===")
            first_run = active_runs_raw[0]
            if hasattr(first_run, "summary"):
                print(f"ID: {first_run.id}")
                print(f"Name: {first_run.name}")
                print(f"Runtime: {first_run.summary.get('_runtime', 0)} seconds")
                print(f"Step: {first_run.summary.get('_step', 0)}")
                print(f"Timestamp: {first_run.summary.get('_timestamp', 0)}")
                import time

                print(f"Current time: {time.time()}")
                print(
                    f"Total timesteps: {first_run.config.get('trainer', {}).get('total_timesteps', 0)}"
                )

                # Check history for heartbeat info
                if hasattr(first_run, "history"):
                    print("Has history: Yes")
                    try:
                        # Get last few history entries
                        history_df = first_run.history(samples=5)
                        if not history_df.empty:
                            print(
                                f"Last history timestamp: {history_df['_timestamp'].iloc[-1] if '_timestamp' in history_df else 'N/A'}"
                            )
                    except Exception:
                        pass

                # Check for other time-related fields
                if hasattr(first_run, "heartbeat_at"):
                    print(f"Heartbeat at: {first_run.heartbeat_at}")
                if hasattr(first_run, "updated_at"):
                    print(f"Updated at: {first_run.updated_at}")
            print("=== END DEBUG ===\n")

        # Process active runs for the active runs table
        active_runs = []
        for run in active_runs_raw:
            try:
                # Handle WandB Run objects directly
                if hasattr(run, "summary"):  # It's a WandB Run object
                    # Get runtime from _runtime or _wandb.runtime
                    runtime_seconds = run.summary.get(
                        "_runtime", run.summary.get("_wandb", {}).get("runtime", 0)
                    )

                    # Get current timesteps from _step
                    timesteps = run.summary.get("_step", 0)

                    # Get total timesteps from config
                    total_timesteps = run.config.get("trainer", {}).get(
                        "total_timesteps", 1000000
                    )

                    # Try to find a score metric - look for common patterns
                    score = 0.0
                    # Check for various score patterns
                    for key in [
                        "score",
                        "eval/score",
                        "protein.objective",
                        "objective",
                        "reward",
                        "eval/reward",
                    ]:
                        if key in run.summary:
                            score = run.summary.get(key, 0)
                            break

                    # Calculate cost based on runtime and hourly rate
                    # Assuming $4.6/hour as default (can be overridden)
                    hours_run = runtime_seconds / 3600.0
                    cost = hours_run * DEFAULT_HOURLY_COST

                    # Also check if there's an explicit cost metric
                    if "cost.accrued" in run.summary:
                        cost = run.summary.get("cost.accrued")
                    elif "cost.total" in run.summary:
                        cost = run.summary.get("cost.total")

                    # Calculate seconds since last update
                    import time
                    from datetime import datetime

                    seconds_since_update = None

                    # Try multiple sources for last update time
                    # 1. First try heartbeat_at (most reliable)
                    if hasattr(run, "heartbeat_at") and run.heartbeat_at:
                        try:
                            # Parse ISO format datetime
                            heartbeat_dt = datetime.fromisoformat(
                                run.heartbeat_at.replace("Z", "+00:00")
                            )
                            current_dt = datetime.utcnow()
                            seconds_since_update = int(
                                (current_dt - heartbeat_dt).total_seconds()
                            )
                        except Exception as e:
                            print(f"Failed to parse heartbeat_at: {e}")

                    # 2. If no heartbeat_at, try using _timestamp
                    if seconds_since_update is None:
                        last_update_timestamp = run.summary.get("_timestamp", 0)
                        if last_update_timestamp and last_update_timestamp > 0:
                            current_time = time.time()
                            # Just use the difference directly - the timestamps appear to be in the same epoch
                            seconds_since_update = int(
                                abs(current_time - last_update_timestamp)
                            )

                            # If the difference is very small or negative, it's probably accurate
                            if (
                                seconds_since_update > 86400 * 7
                            ):  # More than a week seems wrong
                                seconds_since_update = None

                    active_run_data = {
                        "run_id": run.id,
                        "run_name": run.name,
                        "state": "running",
                        "created_at": run.created_at,
                        "runtime_seconds": int(runtime_seconds),
                        "timesteps": int(timesteps) if timesteps else 0,
                        "total_timesteps": int(total_timesteps)
                        if total_timesteps
                        else 1000000,
                        "score": float(score) if score else 0.0,
                        "cost": float(cost) if cost else 0.0,
                        "progress": 0.0,
                        "seconds_since_update": seconds_since_update,
                    }

                    # Calculate progress
                    if active_run_data["total_timesteps"] > 0:
                        active_run_data["progress"] = min(
                            100.0,
                            (
                                active_run_data["timesteps"]
                                / active_run_data["total_timesteps"]
                            )
                            * 100,
                        )

                    active_runs.append(active_run_data)

                elif isinstance(run, dict):  # It's already a dict/JSON
                    runtime_seconds = (
                        run.get("summary", {}).get("_wandb", {}).get("runtime", 0)
                    )
                    timesteps = run.get("summary", {}).get(
                        "timestep", run.get("summary", {}).get("global_step", 0)
                    )
                    total_timesteps = (
                        run.get("config", {})
                        .get("trainer", {})
                        .get("total_timesteps", 1000000)
                    )
                    score = run.get("summary", {}).get(
                        "score", run.get("summary", {}).get("protein.objective", 0)
                    )
                    cost = run.get("summary", {}).get(
                        "cost.accrued", run.get("summary", {}).get("cost.total", 0)
                    )

                    active_run_data = {
                        "run_id": run.get("id", ""),
                        "run_name": run.get("name", ""),
                        "state": "running",
                        "created_at": run.get("created_at", ""),
                        "runtime_seconds": runtime_seconds,
                        "timesteps": int(timesteps) if timesteps else 0,
                        "total_timesteps": int(total_timesteps)
                        if total_timesteps
                        else 1000000,
                        "score": float(score) if score else 0.0,
                        "cost": float(cost) if cost else 0.0,
                        "progress": 0.0,
                    }

                    # Calculate progress
                    if active_run_data["total_timesteps"] > 0:
                        active_run_data["progress"] = min(
                            100.0,
                            (
                                active_run_data["timesteps"]
                                / active_run_data["total_timesteps"]
                            )
                            * 100,
                        )

                    active_runs.append(active_run_data)

            except Exception as e:
                print(f"Warning: Failed to process active run: {e}")
                continue

        # Process completed runs for observations
        observations = []
        for run in runs[:max_observations]:
            # Only process runs with observations
            protein_obs = run.summary.get("protein_observation")
            protein_suggestion = run.summary.get("protein_suggestion")

            if protein_obs:
                obs = deep_clean(protein_obs)
                if "suggestion" not in obs and protein_suggestion:
                    obs["suggestion"] = deep_clean(protein_suggestion)
                obs["time_total"] = run.summary.get(
                    "time.total", run.summary.get("_wandb", {}).get("runtime", 0)
                )
                obs["timestamp"] = run.created_at
                obs["run_name"] = run.name
                obs["run_id"] = run.id
                obs["state"] = run.state
                obs["timesteps"] = run.summary.get(
                    "timestep", run.summary.get("global_step", 0)
                )
                observations.append(obs)
            elif protein_suggestion:
                obs = {
                    "suggestion": deep_clean(protein_suggestion),
                    "objective": run.summary.get(
                        "score", run.summary.get("protein.objective", np.nan)
                    ),
                    "cost": run.summary.get(
                        "cost.accrued", run.summary.get("cost.total", 0)
                    ),
                    "time_total": run.summary.get(
                        "time.total", run.summary.get("_wandb", {}).get("runtime", 0)
                    ),
                    "is_failure": run.state != "finished",
                    "timestamp": run.created_at,
                    "run_name": run.name,
                    "run_id": run.id,
                    "state": run.state,
                    "timesteps": run.summary.get(
                        "timestep", run.summary.get("global_step", 0)
                    ),
                }
                observations.append(obs)

        # Convert to list of dicts
        runs_data = extract_observations_to_dict(observations)

        # If no completed runs but we have active runs, still return active runs
        if not runs_data and active_runs:
            return {
                "runs": [],
                "activeRuns": active_runs,
                "totalRuns": 0,
                "bestScore": 0,
                "totalCost": 0,
                "avgRuntime": 0,
                "parameters": [],
            }
        elif not runs_data:
            return {
                "runs": [],
                "activeRuns": [],
                "totalRuns": 0,
                "bestScore": 0,
                "totalCost": 0,
                "avgRuntime": 0,
                "parameters": [],
            }

        # Calculate summary statistics
        scores = [r["score"] for r in runs_data if not np.isnan(r.get("score", np.nan))]
        costs = [r["cost"] for r in runs_data if not np.isnan(r.get("cost", np.nan))]
        runtimes = [
            r["runtime"] for r in runs_data if not np.isnan(r.get("runtime", np.nan))
        ]

        # Get parameter names
        param_names = set()
        for run in runs_data:
            for key in run.keys():
                if key.startswith("trainer."):
                    param_names.add(key)

        # Format runs for frontend
        formatted_runs = []
        for run in runs_data:
            # Extract parameters
            parameters = {}
            for key in run.keys():
                if key.startswith("trainer."):
                    parameters[key] = run[key]

            formatted_runs.append(
                {
                    "id": run.get("run_id", ""),
                    "name": run.get("run_name", ""),
                    "score": float(run.get("score", 0))
                    if not np.isnan(run.get("score", np.nan))
                    else 0,
                    "cost": float(run.get("cost", 0))
                    if not np.isnan(run.get("cost", np.nan))
                    else 0,
                    "runtime": float(run.get("runtime", 0))
                    if not np.isnan(run.get("runtime", np.nan))
                    else 0,
                    "timestamp": run.get("timestamp", ""),
                    "parameters": parameters,
                    "status": run.get("status", "finished"),
                }
            )

        # Log summary
        print(
            f"Returning {len(active_runs)} active runs and {len(formatted_runs)} completed runs"
        )

        return {
            "runs": formatted_runs,
            "activeRuns": active_runs,
            "totalRuns": len(formatted_runs),
            "activeRunsCount": len(active_runs),
            "bestScore": max(scores) if scores else 0,
            "totalCost": sum(costs) if costs else 0,
            "avgRuntime": sum(runtimes) / len(runtimes) if runtimes else 0,
            "parameters": list(param_names),
        }

    except Exception as e:
        print(f"Error loading sweep data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sky-jobs")
async def get_sky_jobs():
    """Get Sky jobs status"""
    try:
        result = subprocess.run(
            ["sky", "jobs", "queue", "-s"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500, detail=f"Sky command failed: {result.stderr}"
            )

        # Parse the output
        lines = result.stdout.strip().split("\n")
        jobs = []

        for line in lines:
            if line.startswith("ID") or "In progress tasks:" in line:
                continue
            if line.strip() and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 10:
                    job_id = parts[0]
                    name = parts[2]

                    # Parse resources
                    resource_parts = []
                    for i in range(3, len(parts)):
                        resource_parts.append(parts[i])
                        if "]" in parts[i]:
                            break
                    resources = " ".join(resource_parts)

                    # Parse status (look for known status values)
                    status = "UNKNOWN"
                    for part in reversed(parts):
                        if part in [
                            "RUNNING",
                            "SUCCEEDED",
                            "FAILED",
                            "PENDING",
                            "CANCELLED",
                        ]:
                            status = part
                            break

                    jobs.append(
                        {
                            "id": job_id,
                            "name": name,
                            "resources": resources,
                            "submitted": "N/A",  # Would need more parsing
                            "totalDuration": "N/A",
                            "jobDuration": "N/A",
                            "status": status,
                        }
                    )

        return jobs

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Sky command timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Sky CLI not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sky-jobs/{job_id}/cancel")
async def cancel_sky_job(job_id: str):
    """Cancel a Sky job"""
    try:
        result = subprocess.run(
            ["sky", "jobs", "cancel", "-y", str(job_id)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500, detail=f"Failed to cancel job: {result.stderr}"
            )

        return {"status": "success", "message": f"Job {job_id} cancelled"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sky-jobs/launch")
async def launch_worker(request: LaunchWorkerRequest):
    """Launch a new sweep worker"""
    try:
        metta_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        cmd = (
            f"cd {metta_root} && "
            f"source .venv/bin/activate && "
            f"./devops/skypilot/launch.py --no-spot --gpus={request.num_gpus} sweep run={request.sweep_name}"
        )

        # Run in background
        subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=metta_root,
            executable="/bin/bash",
        )

        return {
            "status": "success",
            "message": f"Launching worker with {request.num_gpus} GPUs",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("Starting Sweep Dashboard Backend...")
    print(f"Default entity: {DEFAULT_ENTITY}")
    print(f"Default project: {DEFAULT_PROJECT}")
    print("Server running at http://localhost:8000")
    print("\nTo use custom settings, set environment variables:")
    print("  WANDB_ENTITY=your-entity")
    print("  WANDB_PROJECT=your-project")
    print("  HOURLY_COST=4.6")

    uvicorn.run(app, host="0.0.0.0", port=8000)
