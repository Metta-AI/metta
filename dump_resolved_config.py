#!/usr/bin/env -S uv run --script
"""Dump fully resolved Hydra config to text file."""

import sys

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="train_job")
def dump_config(cfg: DictConfig) -> None:
    """Dump the fully resolved config to a text file."""
    # Convert to regular dict for YAML serialization
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Write as YAML format for better readability
    with open("resolved_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # Also write in the format similar to the 3rd party library
    with open("resolved_config_comparison.txt", "w") as f:
        f.write("# Fully Resolved Metta Config\n")
        f.write("# Command: ./tools/train.py run=relh.dummy.run\n\n")

        # Extract key configurations that match the 3rd party format
        f.write(f"name: {cfg.get('run', 'unknown')}\n\n")

        # Training parameters
        if "trainer" in cfg:
            trainer = cfg.trainer
            f.write("# Training parameters\n")
            f.write(f"total_timesteps: {trainer.get('total_timesteps', 'N/A')}\n")
            f.write(f"batch_size: {trainer.get('batch_size', 'N/A')}\n")
            if "optim" in trainer:
                optim = trainer.optim
                f.write(f"learning_rate: {optim.get('lr', 'N/A')}\n")
                f.write(f"adam_beta1: {optim.get('beta1', 'N/A')}\n")
                f.write(f"adam_beta2: {optim.get('beta2', 'N/A')}\n")
                f.write(f"adam_eps: {optim.get('eps', 'N/A')}\n")
            f.write(f"num_workers: {trainer.get('num_workers', 'N/A')}\n")
            f.write(f"minibatch_size: {trainer.get('minibatch_size', 'N/A')}\n")
            f.write(f"bptt_horizon: {trainer.get('bptt_horizon', 'N/A')}\n")
            f.write("\n")

        # Environment configuration
        if "env" in trainer and "mettagrid" in trainer.env:
            env_cfg = trainer.env.mettagrid
            f.write("# Environment configuration\n")
            f.write(f"num_agents: {env_cfg.get('num_agents', 'N/A')}\n")
            f.write(f"obs_width: {env_cfg.get('obs_width', 'N/A')}\n")
            f.write(f"obs_height: {env_cfg.get('obs_height', 'N/A')}\n")
            f.write(f"max_steps: {env_cfg.get('episode_length', 'N/A')}\n")
            f.write("\n")

            # Game objects
            if "objects" in env_cfg:
                f.write("# Game objects\n")
                f.write("objects:\n")
                for obj_name, obj_cfg in env_cfg.objects.items():
                    f.write(f"  {obj_name}:\n")
                    for key, value in obj_cfg.items():
                        f.write(f"    {key}: {value}\n")
                f.write("\n")

        # Write full config as YAML at the end
        f.write("\n# Full Config in YAML format:\n")
        f.write("# " + "=" * 50 + "\n")
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print("Config dumped to resolved_config.yaml and resolved_config_comparison.txt")


if __name__ == "__main__":
    # Override sys.argv to pass the correct arguments
    sys.argv = ["dump_resolved_config.py", "run=relh.dummy.run"]
    dump_config()
