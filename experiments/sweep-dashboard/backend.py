#!/usr/bin/env python
"""
FastAPI backend for Sweep Dashboard
Serves real data from WandB and Sky jobs
"""

import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone

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
async def get_available_sweeps(
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
):
    """Get list of available sweeps from WandB"""
    try:
        from metta.sweep.wandb_utils import get_all_sweep_groups

        # Get all sweep groups from WandB
        sweeps = get_all_sweep_groups(entity=entity, project=project, max_runs=500)

        # If no sweeps found, return some defaults for testing
        if not sweeps:
            sweeps = [
                "protein-optimization-2024",
                "hyperparameter-sweep-v3",
                "architecture-search-latest",
            ]

        return {
            "sweeps": sweeps,
            "count": len(sweeps),
            "entity": entity,
            "project": project,
        }
    except Exception as e:
        print(f"Error fetching sweep groups: {e}")
        # Fallback to example sweeps if WandB API fails
        return {
            "sweeps": [
                "protein-optimization-2024",
                "hyperparameter-sweep-v3",
                "architecture-search-latest",
            ],
            "count": 3,
            "error": str(e),
        }


@app.get("/api/sweeps/{sweep_name}/confidence")
async def get_model_confidence(
    sweep_name: str,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
):
    """Get model confidence metrics and convergence statistics"""
    try:
        from metta.sweep.protein import Protein
        import torch

        print(f"Loading confidence metrics for {sweep_name}")

        # Get runs with observations
        runs = get_sweep_runs(sweep_name=sweep_name, entity=entity, project=project)

        if len(runs) < 5:
            return {
                "error": "Not enough data",
                "message": "Need at least 5 observations for confidence metrics",
            }

        # Extract observations with run metadata
        observations = []
        for run in runs:
            protein_obs = run.summary.get("protein_observation")
            if protein_obs:
                obs = deep_clean(protein_obs)
                # Try different possible score keys in order of preference
                score = (
                    obs.get("objective")
                    or run.summary.get("reward")
                    or run.summary.get("score")
                    or run.summary.get("objective")
                    or 0
                )
                obs["objective"] = score
                obs["cost"] = obs.get("cost", run.summary.get("cost.accrued", 0))
                # Use run's created_at timestamp for proper chronological ordering
                obs["created_at"] = run.created_at
                obs["run_name"] = run.name
                observations.append(obs)

        if len(observations) < 5:
            return {
                "error": "Not enough valid observations",
                "message": "Need at least 5 valid observations for confidence metrics",
            }

        # Calculate running best (for convergence plot)
        # Sort observations by run creation time to ensure chronological order
        observations_sorted = sorted(
            observations, key=lambda x: x.get("created_at", "")
        )
        scores = [obs.get("objective", 0) for obs in observations_sorted]

        # Debug output
        print(f"Confidence metrics: Found {len(scores)} scores")
        if scores:
            print(f"  Score range: {min(scores):.4f} to {max(scores):.4f}")
            print(f"  First 5 scores: {scores[:5]}")

        # Calculate actual running statistics
        running_best = []
        running_mean = []
        running_std = []

        for i in range(len(scores)):
            scores_so_far = scores[: i + 1]
            # Running best is the maximum seen up to this point
            running_best.append(float(max(scores_so_far)))
            # Running mean of all scores so far
            running_mean.append(float(np.mean(scores_so_far)))
            # Running std (0 for first observation)
            running_std.append(float(np.std(scores_so_far)) if i > 0 else 0.0)

        # Calculate convergence metrics
        convergence_window = min(10, len(scores) // 2)
        if len(scores) > convergence_window:
            recent_scores = scores[-convergence_window:]
            convergence_std = float(np.std(recent_scores))
            convergence_slope = float(
                np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            )
        else:
            convergence_std = float(np.std(scores))
            convergence_slope = 0.0

        # Initialize Protein to get GP predictions (simplified)
        try:
            # Debug: Check observation structure
            if observations and len(observations) > 0:
                print(f"Sample observation keys: {observations[0].keys()}")
                if "suggestion" in observations[0]:
                    print(f"Sample suggestion: {observations[0]['suggestion']}")

            # Check if we have suggestions to work with
            has_suggestions = any("suggestion" in obs for obs in observations)

            if not has_suggestions:
                print("No observations have suggestions, skipping GP metrics")
                raise ValueError("No suggestions found in observations")

            # Create a config that matches the actual parameter structure
            # First, find parameters that exist in ALL observations
            all_param_sets = []
            for obs in observations:
                if "suggestion" in obs and obs["suggestion"]:
                    suggestion = obs["suggestion"]
                    param_set = set()

                    # Flatten the suggestion to get all parameters (use / separator like pufferlib)
                    def extract_params(d, prefix=""):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                extract_params(v, prefix + k + "/")
                            elif isinstance(v, (int, float)):
                                param_key = prefix + k
                                param_set.add(param_key)

                    extract_params(suggestion)
                    all_param_sets.append(param_set)

            # Only use parameters that appear in ALL observations
            if all_param_sets:
                common_params = set.intersection(*all_param_sets)
            else:
                common_params = set()

            print(
                f"Found {len(common_params)} common parameters across all observations"
            )

            # Now extract values for common parameters only
            param_ranges = {}
            for obs in observations:
                if "suggestion" in obs and obs["suggestion"]:
                    suggestion = obs["suggestion"]

                    # Flatten the suggestion to get all parameters (use / separator like pufferlib)
                    def extract_params(d, prefix=""):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                extract_params(v, prefix + k + "/")
                            elif isinstance(v, (int, float)):
                                param_key = prefix + k
                                if (
                                    param_key in common_params
                                ):  # Only collect common params
                                    if param_key not in param_ranges:
                                        param_ranges[param_key] = []
                                    param_ranges[param_key].append(v)

                    extract_params(suggestion)

            # Build sweep config with actual parameter ranges
            sweep_config = {
                "metric": "reward",
                "goal": "maximize",
                "method": "bayes",
            }

            # Debug: Check how pufferlib flattens vs our method
            if observations and "suggestion" in observations[0]:
                import pufferlib

                puffer_flat = dict(
                    pufferlib.unroll_nested_dict(observations[0]["suggestion"])
                )
                print(f"Pufferlib flattened keys: {list(puffer_flat.keys())}")
                print(f"Our flattened keys: {list(param_ranges.keys())}")

            print(f"Found {len(param_ranges)} parameters in observations")

            # Add parameter definitions based on observed values
            for param_name, values in param_ranges.items():
                if len(values) >= 1:  # Changed from > 1 to >= 1
                    min_val = min(values)
                    max_val = max(values)
                    mean_val = np.mean(values)

                    # If all values are the same, create a small range around them
                    if min_val == max_val:
                        if min_val > 0:
                            min_val = min_val * 0.9
                            max_val = max_val * 1.1
                        else:
                            min_val = min_val - 0.1
                            max_val = max_val + 0.1

                    # Determine distribution type based on parameter name and values
                    if param_name.endswith("learning_rate") or param_name.endswith(
                        "lr"
                    ):
                        distribution = "log_normal"
                    elif (
                        param_name.endswith("gamma")
                        or param_name.endswith("gae_lambda")
                        or param_name.endswith("clip_coef")
                    ):
                        distribution = "logit_normal"
                        # Logit requires values strictly between 0 and 1
                        min_val = max(0.001, min(min_val * 0.99, 0.999))
                        max_val = max(0.001, min(max_val * 1.01, 0.999))
                        mean_val = max(0.001, min(mean_val, 0.999))
                    elif param_name.endswith("batch_size") or param_name.endswith(
                        "minibatch_size"
                    ):
                        distribution = "uniform_pow2"
                        # Values should already be powers of 2, just ensure they're integers
                        min_val = int(min_val)
                        max_val = int(max_val)
                        mean_val = int(mean_val)
                        print(
                            f"Pow2 param {param_name}: min={min_val}, max={max_val}, mean={mean_val}"
                        )
                    elif param_name.endswith("total_timesteps"):
                        distribution = "int_uniform"
                        # Ensure integers
                        min_val = int(min_val)
                        max_val = int(max_val)
                        mean_val = int(mean_val)
                    elif param_name.endswith("ent_coef"):
                        # Entropy coefficient should use log scale
                        distribution = "log_normal"
                    else:
                        distribution = "uniform"

                    # Final bounds adjustment based on distribution
                    if distribution == "log_normal":
                        # Ensure positive values for log scale
                        min_val = max(1e-10, min_val * 0.8 if min_val > 0 else 1e-6)
                        max_val = max(min_val * 2, max_val * 1.2)
                    elif distribution == "logit_normal":
                        # Already handled above
                        pass
                    elif distribution == "uniform_pow2":
                        # Already handled above
                        pass
                    else:
                        # Regular uniform - widen bounds slightly
                        range_val = max_val - min_val
                        if range_val > 0:
                            min_val = min_val - range_val * 0.1
                            max_val = max_val + range_val * 0.1

                    param_config = {
                        "distribution": distribution,
                        "min": min_val,
                        "max": max_val,
                        "mean": mean_val,
                        "scale": "auto",
                    }

                    # Debug problematic parameters
                    if distribution == "logit_normal":
                        print(
                            f"Logit param {param_name}: min={min_val}, max={max_val}, mean={mean_val}"
                        )

                    sweep_config[param_name] = param_config

            print(f"Final sweep config parameters: {list(sweep_config.keys())}")
            protein = Protein(sweep_config)
            print(
                f"Protein expects these parameters: {list(protein.hyperparameters.spaces.keys())}"
            )

            # Add observations to protein - only those with all common parameters
            successful_observations = 0
            for i, obs in enumerate(observations[:50]):  # Limit to recent observations
                if "suggestion" in obs:
                    # Check if this observation has all common parameters (use / separator)
                    suggestion_flat = {}

                    def flatten_dict(d, prefix=""):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                flatten_dict(v, prefix + k + "/")
                            else:
                                suggestion_flat[prefix + k] = v

                    flatten_dict(obs["suggestion"])

                    # Only add if it has all common parameters
                    if all(param in suggestion_flat for param in common_params):
                        try:
                            # Debug: print what we're trying to observe
                            if i == 0:  # Only for first observation
                                print(
                                    f"First observation suggestion keys: {list(suggestion_flat.keys())}"
                                )
                                print(
                                    f"Batch size value in observation: {suggestion_flat.get('trainer/batch_size', 'NOT FOUND')}"
                                )

                            protein.observe(
                                obs["suggestion"],
                                obs.get("objective", 0),
                                obs.get("cost", 1),
                                is_failure=False,
                            )
                            successful_observations += 1
                        except Exception as e:
                            print(f"Failed to add observation {i}: {e}")
                            if i == 0:  # More detail for first failure
                                print(f"  Suggestion structure: {obs['suggestion']}")

            print(
                f"Successfully added {successful_observations} observations to Protein"
            )

            # Calculate average uncertainty over parameter space
            if len(protein.success_observations) > 3:
                # Sample some test points
                n_test = min(50, len(protein.success_observations) * 5)
                test_suggestions = protein.hyperparameters.sample(n_test)
                # Convert to float32 for GP compatibility
                test_suggestions_torch = torch.from_numpy(test_suggestions).float()

                # Get GP predictions
                with torch.no_grad():
                    _, score_var = protein.gp_score(test_suggestions_torch)
                    uncertainties = np.sqrt(score_var.numpy())

                avg_uncertainty = float(np.mean(uncertainties))
                max_uncertainty = float(np.max(uncertainties))
                min_uncertainty = float(np.min(uncertainties))

                # Find high confidence regions (low uncertainty)
                confidence_threshold = np.percentile(uncertainties, 20)
                high_confidence_indices = np.where(
                    uncertainties < confidence_threshold
                )[0]
                high_confidence_ratio = float(
                    len(high_confidence_indices) / len(uncertainties)
                )
            else:
                avg_uncertainty = 1.0
                max_uncertainty = 1.0
                min_uncertainty = 1.0
                high_confidence_ratio = 0.0

        except Exception as e:
            print(f"Warning: Could not compute GP metrics: {e}")
            import traceback

            traceback.print_exc()
            avg_uncertainty = None
            max_uncertainty = None
            min_uncertainty = None
            high_confidence_ratio = None

        # Statistical significance metrics
        baseline_score = scores[0] if scores else 0  # First run's score
        current_best = max(scores) if scores else 0  # Actual best score achieved
        best_run_index = (
            scores.index(current_best) if scores else 0
        )  # Which run achieved best
        improvement = current_best - baseline_score  # Improvement from baseline

        # Simple significance test (using bootstrap would be better)
        if len(scores) > 1:
            score_std = np.std(scores)
            if score_std > 0:
                z_score = improvement / score_std
                # Rough p-value approximation
                from scipy import stats

                p_value = float(1 - stats.norm.cdf(abs(z_score)))
                is_significant = bool(p_value < 0.05)
            else:
                p_value = 1.0
                is_significant = False
        else:
            p_value = 1.0
            is_significant = False

        # Estimate runs until convergence (more realistic)
        if len(scores) > convergence_window:
            # Look at rate of improvement over recent runs
            recent_improvements = []
            for i in range(len(running_best) - convergence_window, len(running_best)):
                if i > 0:
                    recent_improvements.append(running_best[i] - running_best[i - 1])

            avg_recent_improvement = (
                np.mean(recent_improvements) if recent_improvements else 0
            )

            # If improvements are very small and std is low, we're likely converged
            if abs(avg_recent_improvement) < 0.001 and convergence_std < 0.01:
                estimated_runs_to_convergence = 0  # Already converged
            elif abs(avg_recent_improvement) > 0:
                # Estimate based on how much room for improvement and current rate
                room_for_improvement = (
                    convergence_std * 2
                )  # Assume we can improve by 2 std devs
                estimated_runs_to_convergence = int(
                    room_for_improvement / abs(avg_recent_improvement)
                )
                estimated_runs_to_convergence = max(
                    5, min(estimated_runs_to_convergence, 100)
                )
            else:
                estimated_runs_to_convergence = 20  # Default if no clear trend
        else:
            estimated_runs_to_convergence = None

        return {
            "num_observations": len(observations),
            "running_best": running_best,
            "running_mean": running_mean,
            "running_std": running_std,
            "convergence": {
                "std": convergence_std,
                "slope": convergence_slope,
                "is_converged": bool(
                    convergence_std < 0.01 and abs(convergence_slope) < 0.001
                ),
                "estimated_runs_remaining": estimated_runs_to_convergence,
            },
            "uncertainty": {
                "average": avg_uncertainty,
                "max": max_uncertainty,
                "min": min_uncertainty,
                "high_confidence_ratio": high_confidence_ratio,
            },
            "significance": {
                "baseline_score": baseline_score,
                "current_best": current_best,
                "best_run_index": best_run_index,
                "improvement": improvement,
                "p_value": p_value,
                "is_significant": is_significant,
            },
            "timestamps": list(range(len(scores))),
        }

    except Exception as e:
        print(f"Error computing confidence metrics: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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

                    seconds_since_update = None

                    # Try multiple sources for last update time
                    # 1. First try heartbeat_at (most reliable)
                    if hasattr(run, "heartbeat_at") and run.heartbeat_at:
                        try:
                            # Parse ISO format datetime
                            heartbeat_dt = datetime.fromisoformat(
                                run.heartbeat_at.replace("Z", "+00:00")
                            )
                            # Use timezone-aware UTC datetime
                            current_dt = datetime.now(timezone.utc)
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
