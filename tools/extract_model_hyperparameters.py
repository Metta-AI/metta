#!/usr/bin/env python3
"""
Extract actual hyperparameters from initialized models in both functional and Hydra trainers.
This ensures we compare the real hyperparameters used, not just configuration files.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_functional_trainer_hyperparameters() -> Dict[str, Any]:
    """Extract hyperparameters from the functional trainer by running it briefly."""

    # Create a temporary script that imports and extracts hyperparameters
    extract_script = """
#!/usr/bin/env -S uv run
import os
import sys
import json
import torch
from pathlib import Path

# Add the metta directory to the path
sys.path.insert(0, "/Users/bullm/Documents/metta2/metta")

# Import the functional trainer components
import sys
sys.path.insert(0, "/Users/bullm/Documents/metta2/metta")
from bullm_run import (
    trainer_config, ppo_config, optimizer_config, 
    checkpoint_config, simulation_config, profiler_config,
    prioritized_replay_config, vtrace_config, kickstart_config
)

# Extract all hyperparameters
hyperparams = {
    "trainer": {
        "num_workers": trainer_config.num_workers,
        "total_timesteps": trainer_config.total_timesteps,
        "batch_size": trainer_config.batch_size,
        "minibatch_size": trainer_config.minibatch_size,
        "curriculum": trainer_config.curriculum,
        "bptt_horizon": trainer_config.bptt_horizon,
        "update_epochs": trainer_config.update_epochs,
        "forward_pass_minibatch_target_size": trainer_config.forward_pass_minibatch_target_size,
        "async_factor": trainer_config.async_factor,
        "grad_mean_variance_interval": trainer_config.grad_mean_variance_interval,
        "scale_batches_by_world_size": trainer_config.scale_batches_by_world_size,
        "cpu_offload": trainer_config.cpu_offload,
        "zero_copy": trainer_config.zero_copy,
    },
    "ppo": {
        "clip_coef": ppo_config.clip_coef,
        "ent_coef": ppo_config.ent_coef,
        "gamma": ppo_config.gamma,
        "gae_lambda": ppo_config.gae_lambda,
        "max_grad_norm": ppo_config.max_grad_norm,
    },
    "optimizer": {
        "type": optimizer_config.type,
        "learning_rate": optimizer_config.learning_rate,
        "beta1": optimizer_config.beta1,
        "beta2": optimizer_config.beta2,
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
    },
    "checkpoint": {
        "checkpoint_interval": checkpoint_config.checkpoint_interval,
        "wandb_checkpoint_interval": checkpoint_config.wandb_checkpoint_interval,
    },
    "simulation": {
        "evaluate_interval": simulation_config.evaluate_interval,
    },
    "profiler": {
        "interval_epochs": profiler_config.interval_epochs,
    },
    "prioritized_experience_replay": {
        "prio_alpha": prioritized_replay_config.prio_alpha,
        "prio_beta0": prioritized_replay_config.prio_beta0,
    },
    "vtrace": {
        "vtrace_rho_clip": vtrace_config.vtrace_rho_clip,
        "vtrace_c_clip": vtrace_config.vtrace_c_clip,
    },
    "kickstart": {
        "kickstart_steps": kickstart_config.kickstart_steps,
        "teacher_uri": kickstart_config.teacher_uri,
        "action_loss_coef": kickstart_config.action_loss_coef,
        "value_loss_coef": kickstart_config.value_loss_coef,
        "anneal_ratio": kickstart_config.anneal_ratio,
    }
}

# Print as JSON for easy parsing
print(json.dumps(hyperparams, indent=2))
"""

    # Write and run the script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(extract_script)
        script_path = f.name

    try:
        # Set environment variables to match comparison setup
        env = os.environ.copy()
        env["WANDB_DISABLED"] = "true"
        env["RUN_NAME"] = "hyperparam_extraction_functional"
        env["RUN_DIR"] = "/tmp/hyperparam_extraction_functional"

        # Run the script
        result = subprocess.run(
            ["uv", "run", "python", script_path], env=env, capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        else:
            print("Error extracting functional trainer hyperparameters:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {}

    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def extract_hydra_trainer_hyperparameters() -> Dict[str, Any]:
    """Extract hyperparameters from the Hydra trainer by running it briefly."""

    # Create a temporary script that imports and extracts hyperparameters
    extract_script = """
#!/usr/bin/env -S uv run
import os
import sys
import json
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Add the metta directory to the path
sys.path.insert(0, "/Users/bullm/Documents/metta2/metta")

# Import Hydra components
from hydra import compose, initialize_config_dir
from metta.rl.trainer_config import create_trainer_config

# Initialize Hydra with the comparison config
with initialize_config_dir(version_base=None, config_dir="/Users/bullm/Documents/metta2/metta/configs"):
    cfg = compose(config_name="train_job_comparison")
    
    # Add required fields that might be missing
    if not hasattr(cfg, 'run'):
        cfg.run = "hyperparam_extraction_hydra"
    if not hasattr(cfg, 'run_dir'):
        cfg.run_dir = "/tmp/hyperparam_extraction_hydra"
    if not hasattr(cfg, 'vectorization'):
        cfg.vectorization = "serial"
    
    # Create trainer config
    trainer_config = create_trainer_config(cfg)
    
    # Extract all hyperparameters
    hyperparams = {
        "trainer": {
            "num_workers": trainer_config.num_workers,
            "total_timesteps": trainer_config.total_timesteps,
            "batch_size": trainer_config.batch_size,
            "minibatch_size": trainer_config.minibatch_size,
            "curriculum": trainer_config.curriculum,
            "bptt_horizon": trainer_config.bptt_horizon,
            "update_epochs": trainer_config.update_epochs,
            "forward_pass_minibatch_target_size": trainer_config.forward_pass_minibatch_target_size,
            "async_factor": trainer_config.async_factor,
            "grad_mean_variance_interval": trainer_config.grad_mean_variance_interval,
            "scale_batches_by_world_size": trainer_config.scale_batches_by_world_size,
            "cpu_offload": trainer_config.cpu_offload,
            "zero_copy": trainer_config.zero_copy,
        },
        "ppo": {
            "clip_coef": trainer_config.ppo.clip_coef,
            "ent_coef": trainer_config.ppo.ent_coef,
            "gamma": trainer_config.ppo.gamma,
            "gae_lambda": trainer_config.ppo.gae_lambda,
            "max_grad_norm": trainer_config.ppo.max_grad_norm,
            "vf_clip_coef": trainer_config.ppo.vf_clip_coef,
            "vf_coef": trainer_config.ppo.vf_coef,
            "l2_reg_loss_coef": trainer_config.ppo.l2_reg_loss_coef,
            "l2_init_loss_coef": trainer_config.ppo.l2_init_loss_coef,
            "norm_adv": trainer_config.ppo.norm_adv,
            "clip_vloss": trainer_config.ppo.clip_vloss,
            "target_kl": trainer_config.ppo.target_kl,
        },
        "optimizer": {
            "type": trainer_config.optimizer.type,
            "learning_rate": trainer_config.optimizer.learning_rate,
            "beta1": trainer_config.optimizer.beta1,
            "beta2": trainer_config.optimizer.beta2,
            "eps": trainer_config.optimizer.eps,
            "weight_decay": trainer_config.optimizer.weight_decay,
        },
        "checkpoint": {
            "checkpoint_interval": trainer_config.checkpoint.checkpoint_interval,
            "wandb_checkpoint_interval": trainer_config.checkpoint.wandb_checkpoint_interval,
        },
        "simulation": {
            "evaluate_interval": trainer_config.simulation.evaluate_interval,
        },
        "profiler": {
            "interval_epochs": trainer_config.profiler.interval_epochs,
        },
        "prioritized_experience_replay": {
            "prio_alpha": trainer_config.prioritized_experience_replay.prio_alpha,
            "prio_beta0": trainer_config.prioritized_experience_replay.prio_beta0,
        },
        "vtrace": {
            "vtrace_rho_clip": trainer_config.vtrace.vtrace_rho_clip,
            "vtrace_c_clip": trainer_config.vtrace.vtrace_c_clip,
        },
        "kickstart": {
            "kickstart_steps": trainer_config.kickstart.kickstart_steps,
            "teacher_uri": trainer_config.kickstart.teacher_uri,
            "action_loss_coef": trainer_config.kickstart.action_loss_coef,
            "value_loss_coef": trainer_config.kickstart.value_loss_coef,
            "anneal_ratio": trainer_config.kickstart.anneal_ratio,
        }
    }

# Print as JSON for easy parsing
print(json.dumps(hyperparams, indent=2))
"""

    # Write and run the script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(extract_script)
        script_path = f.name

    try:
        # Set environment variables to match comparison setup
        env = os.environ.copy()
        env["WANDB_DISABLED"] = "true"

        # Run the script
        result = subprocess.run(
            ["uv", "run", "python", script_path], env=env, capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        else:
            print("Error extracting Hydra trainer hyperparameters:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {}

    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def compare_hyperparameters(func_hyperparams: Dict[str, Any], hydra_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Compare hyperparameters between functional and Hydra trainers."""

    comparison = {
        "matches": {},
        "differences": {},
        "missing_in_functional": {},
        "missing_in_hydra": {},
        "summary": {
            "total_parameters": 0,
            "matching_parameters": 0,
            "different_parameters": 0,
            "missing_parameters": 0,
        },
    }

    # Get all unique parameter paths
    all_params = set()

    def collect_params(params_dict, prefix=""):
        for key, value in params_dict.items():
            if isinstance(value, dict):
                collect_params(value, f"{prefix}.{key}" if prefix else key)
            else:
                all_params.add(f"{prefix}.{key}" if prefix else key)

    collect_params(func_hyperparams)
    collect_params(hydra_hyperparams)

    comparison["summary"]["total_parameters"] = len(all_params)

    # Compare each parameter
    for param_path in sorted(all_params):
        func_value = get_nested_value(func_hyperparams, param_path)
        hydra_value = get_nested_value(hydra_hyperparams, param_path)

        if func_value is None and hydra_value is not None:
            comparison["missing_in_functional"][param_path] = hydra_value
            comparison["summary"]["missing_parameters"] += 1
        elif func_value is not None and hydra_value is None:
            comparison["missing_in_hydra"][param_path] = func_value
            comparison["summary"]["missing_parameters"] += 1
        elif func_value == hydra_value:
            comparison["matches"][param_path] = func_value
            comparison["summary"]["matching_parameters"] += 1
        else:
            comparison["differences"][param_path] = {"functional": func_value, "hydra": hydra_value}
            comparison["summary"]["different_parameters"] += 1

    return comparison


def get_nested_value(dictionary: Dict[str, Any], path: str) -> Any:
    """Get a value from a nested dictionary using dot notation."""
    keys = path.split(".")
    current = dictionary

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current


def print_comparison_report(comparison: Dict[str, Any]):
    """Print a formatted comparison report."""

    print("=" * 80)
    print("HYPERPARAMETER COMPARISON (EXTRACTED FROM INITIALIZED MODELS)")
    print("=" * 80)

    summary = comparison["summary"]
    print(f"Total Parameters: {summary['total_parameters']}")
    print(f"Matching Parameters: {summary['matching_parameters']}")
    print(f"Different Parameters: {summary['different_parameters']}")
    print(f"Missing Parameters: {summary['missing_parameters']}")
    print()

    if comparison["differences"]:
        print("❌ DIFFERENCES FOUND:")
        print("-" * 40)
        for param, values in comparison["differences"].items():
            print(f"{param}:")
            print(f"  Functional: {values['functional']}")
            print(f"  Hydra:      {values['hydra']}")
        print()

    if comparison["missing_in_functional"]:
        print("⚠️  MISSING IN FUNCTIONAL TRAINER:")
        print("-" * 40)
        for param, value in comparison["missing_in_functional"].items():
            print(f"{param}: {value}")
        print()

    if comparison["missing_in_hydra"]:
        print("⚠️  MISSING IN HYDRA TRAINER:")
        print("-" * 40)
        for param, value in comparison["missing_in_hydra"].items():
            print(f"{param}: {value}")
        print()

    if comparison["matches"]:
        print("✅ MATCHING PARAMETERS:")
        print("-" * 40)
        for param, value in comparison["matches"].items():
            print(f"{param}: {value}")
        print()

    # Overall assessment
    match_percentage = (summary["matching_parameters"] / summary["total_parameters"]) * 100
    print(f"Overall Match Rate: {match_percentage:.1f}%")

    if summary["different_parameters"] == 0 and summary["missing_parameters"] == 0:
        print("🎉 PERFECT MATCH! All hyperparameters are identical.")
    elif summary["different_parameters"] == 0:
        print("⚠️  Some parameters are missing but all present parameters match.")
    else:
        print("❌ CRITICAL DIFFERENCES FOUND! The trainers are not using identical hyperparameters.")


def main():
    """Main function to extract and compare hyperparameters."""

    print("Extracting hyperparameters from functional trainer...")
    func_hyperparams = extract_functional_trainer_hyperparameters()

    print("Extracting hyperparameters from Hydra trainer...")
    hydra_hyperparams = extract_hydra_trainer_hyperparameters()

    if not func_hyperparams:
        print("❌ Failed to extract functional trainer hyperparameters")
        return

    if not hydra_hyperparams:
        print("❌ Failed to extract Hydra trainer hyperparameters")
        return

    print("Comparing hyperparameters...")
    comparison = compare_hyperparameters(func_hyperparams, hydra_hyperparams)

    print_comparison_report(comparison)

    # Save detailed comparison to file
    with open("hyperparameter_comparison.json", "w") as f:
        json.dump(
            {
                "functional_hyperparameters": func_hyperparams,
                "hydra_hyperparameters": hydra_hyperparams,
                "comparison": comparison,
            },
            f,
            indent=2,
        )

    print("\nDetailed comparison saved to: hyperparameter_comparison.json")


if __name__ == "__main__":
    main()
