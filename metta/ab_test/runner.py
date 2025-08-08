"""
A/B Test Runner for executing experiments.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

from .config import ABExperiment, ABTestConfig, ABVariant

logger = logging.getLogger(__name__)


class ABTestRunner:
    """Runs A/B test experiments using the existing training infrastructure."""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.experiment = config.experiment
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Track run results
        self.run_results: Dict[str, List[Dict]] = {}

    def run_experiment(self) -> Dict[str, List[Dict]]:
        """Run the complete A/B test experiment."""
        logger.info(f"Starting A/B test experiment: {self.experiment.name}")
        logger.info(f"Description: {self.experiment.description}")
        logger.info(f"Variants: {list(self.experiment.variants.keys())}")
        logger.info(f"Runs per variant: {self.experiment.runs_per_variant}")

        # Create experiment directory
        experiment_dir = self.output_dir / self.experiment.name
        experiment_dir.mkdir(exist_ok=True)

        # Save experiment configuration
        self._save_experiment_config(experiment_dir)

        # Run all variants
        for variant_name, variant in self.experiment.variants.items():
            logger.info(f"Running variant: {variant_name}")
            self.run_results[variant_name] = []

            for run_idx in range(self.experiment.runs_per_variant):
                run_name = f"{variant_name}_run_{run_idx + 1}"
                logger.info(f"Starting run {run_idx + 1}/{self.experiment.runs_per_variant}: {run_name}")

                run_result = self._run_single_run(variant, run_name, experiment_dir)
                self.run_results[variant_name].append(run_result)

                if run_result["success"]:
                    logger.info(f"Run {run_name} completed successfully")
                else:
                    logger.error(f"Run {run_name} failed: {run_result['error']}")

        # Generate summary
        self._generate_summary(experiment_dir)

        logger.info("A/B test experiment completed")
        return self.run_results

    def _run_single_run(self, variant: ABVariant, run_name: str, experiment_dir: Path) -> Dict:
        """Run a single training run for a variant."""
        run_dir = experiment_dir / run_name
        run_dir.mkdir(exist_ok=True)

        # Create configuration for this run
        config = self._create_run_config(variant, run_name)

        # Save run configuration
        config_path = run_dir / "config.yaml"
        OmegaConf.save(config, config_path)

        # Execute training
        start_time = time.time()
        success = False
        error = None
        wandb_run_id = None

        try:
            if self.config.use_skypilot:
                # Use SkyPilot for cloud execution
                cmd = self._build_skypilot_command(variant, run_name, config_path)
                logger.debug(f"Executing SkyPilot command: {' '.join(cmd)}")
            else:
                # Use local execution - pass config overrides as command-line arguments
                cmd = ["python", "tools/train.py"]

                # Add configuration overrides from the generated config
                for key, value in self._flatten_config(config).items():
                    cmd.append(f"{key}={value}")

                logger.debug(f"Executing local command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=3600 * 24,  # 24 hour timeout
            )

            if result.returncode == 0:
                success = True
                # Try to extract wandb run ID from output
                for line in result.stdout.split("\n"):
                    if "wandb run" in line.lower():
                        wandb_run_id = line.strip()
                        break
            else:
                error = f"Training failed with return code {result.returncode}"
                if result.stderr:
                    error += f"\nStderr: {result.stderr}"

        except subprocess.TimeoutExpired:
            error = "Training run timed out after 24 hours"
        except Exception as e:
            error = f"Exception during training: {str(e)}"

        duration = time.time() - start_time

        return {
            "run_name": run_name,
            "variant": variant.name,
            "success": success,
            "error": error,
            "duration": duration,
            "wandb_run_id": wandb_run_id,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
        }

    def _build_skypilot_command(self, variant: ABVariant, run_name: str, config_path: Path) -> List[str]:
        """Build SkyPilot launch command for cloud execution."""
        cmd = [
            "./devops/skypilot/launch.py",
            "train",
            f"run={run_name}",
        ]

        # Add SkyPilot-specific options
        if self.config.skypilot_gpus:
            cmd.extend(["--gpus", str(self.config.skypilot_gpus)])
        if self.config.skypilot_cpus:
            cmd.extend(["--cpus", str(self.config.skypilot_cpus)])
        if self.config.skypilot_no_spot:
            cmd.append("--no-spot")
        if self.config.skypilot_max_runtime_hours:
            cmd.extend(["--max-runtime-hours", str(self.config.skypilot_max_runtime_hours)])

        # Add configuration overrides from the variant
        for key, value in variant.overrides.items():
            cmd.append(f"{key}={value}")

        return cmd

    def _create_run_config(self, variant: ABVariant, run_name: str) -> DictConfig:
        """Create the configuration for a single run."""
        # Start with base configuration
        config_dict = self.experiment.base_config.copy()

        # Apply variant overrides
        self._apply_overrides(config_dict, variant.overrides)

        # Set run-specific configuration
        config_dict.update(
            {
                "run": run_name,
                "run_dir": str(self.output_dir / self.experiment.name / run_name),
                "wandb": {
                    "enabled": True,
                    "project": self.experiment.wandb_project,
                    "entity": self.experiment.wandb_entity,
                    "group": self.experiment.name,
                    "name": run_name,
                    "run_id": run_name,
                    "tags": [f"variant:{variant.name}"] + variant.tags,
                    "notes": f"Variant: {variant.name} - {variant.description}",
                },
            }
        )

        return OmegaConf.create(config_dict)

    def _flatten_config(self, config: DictConfig) -> Dict[str, Any]:
        """Flatten a nested configuration into key=value pairs for command line."""
        flattened = {}

        def _flatten_dict(d: Any, prefix: str = "") -> None:
            if isinstance(d, dict):
                for key, value in d.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    _flatten_dict(value, new_prefix)
            elif isinstance(d, list):
                # Handle lists by converting to string representation
                flattened[prefix] = str(d)
            else:
                # Convert value to string for command line
                flattened[prefix] = str(d)

        # Convert DictConfig to dict first
        config_dict = OmegaConf.to_container(config, resolve=True)
        _flatten_dict(config_dict)

        # Handle special Hydra keys that need + prefix
        special_keys = ["defaults"]
        for key in special_keys:
            if key in flattened:
                value = flattened.pop(key)
                flattened[f"+{key}"] = value

        # Convert underscores to dots for nested keys (e.g., trainer__total_timesteps -> trainer.total_timesteps)
        converted = {}
        for key, value in flattened.items():
            if "__" in key:
                converted[key.replace("__", ".")] = value
            else:
                converted[key] = value

        return converted

    def _apply_overrides(self, config: Dict, overrides: Dict[str, Any]) -> None:
        """Apply overrides to the configuration dictionary."""
        for key, value in overrides.items():
            # Handle both dot notation (trainer.total_timesteps) and underscore notation (trainer__total_timesteps)
            if "__" in key:
                keys = key.split("__")
            else:
                keys = key.split(".")

            current = config

            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the final value
            current[keys[-1]] = value

    def _save_experiment_config(self, experiment_dir: Path) -> None:
        """Save the experiment configuration to disk."""
        config_path = experiment_dir / "experiment_config.yaml"

        # Convert experiment to dict for saving
        config_dict = {
            "experiment": {
                "name": self.experiment.name,
                "description": self.experiment.description,
                "date": self.experiment.date,
                "runs_per_variant": self.experiment.runs_per_variant,
                "wandb_project": self.experiment.wandb_project,
                "wandb_entity": self.experiment.wandb_entity,
                "variants": {
                    name: {"description": variant.description, "overrides": variant.overrides, "tags": variant.tags}
                    for name, variant in self.experiment.variants.items()
                },
                "base_config": self.experiment.base_config,
            }
        }

        OmegaConf.save(OmegaConf.create(config_dict), config_path)
        logger.info(f"Saved experiment configuration to {config_path}")

    def _generate_summary(self, experiment_dir: Path) -> None:
        """Generate a summary of the experiment results."""
        summary_path = experiment_dir / "summary.txt"

        with open(summary_path, "w") as f:
            f.write("A/B Test Experiment Summary\n")
            f.write("==========================\n\n")
            f.write(f"Experiment: {self.experiment.name}\n")
            f.write(f"Description: {self.experiment.description}\n")
            f.write(f"Date: {self.experiment.date}\n")
            f.write(f"WandB Project: {self.experiment.wandb_project}\n\n")

            for variant_name, runs in self.run_results.items():
                successful_runs = [r for r in runs if r["success"]]
                failed_runs = [r for r in runs if not r["success"]]

                f.write(f"Variant: {variant_name}\n")
                f.write(f"  Successful runs: {len(successful_runs)}/{len(runs)}\n")
                f.write(f"  Failed runs: {len(failed_runs)}/{len(runs)}\n")

                if successful_runs:
                    avg_duration = sum(r["duration"] for r in successful_runs) / len(successful_runs)
                    f.write(f"  Average duration: {avg_duration:.2f} seconds\n")

                if failed_runs:
                    f.write("  Failed run errors:\n")
                    for run in failed_runs:
                        f.write(f"    {run['run_name']}: {run['error']}\n")

                f.write("\n")

        logger.info(f"Generated experiment summary at {summary_path}")


def run_ab_test(experiment: ABExperiment, **kwargs) -> Dict[str, List[Dict]]:
    """Convenience function to run an A/B test experiment."""
    config = ABTestConfig(experiment=experiment, **kwargs)
    runner = ABTestRunner(config)
    return runner.run_experiment()
