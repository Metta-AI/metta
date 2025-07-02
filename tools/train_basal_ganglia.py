#!/usr/bin/env python3
"""
Training script for the Basal Ganglia model.

This script trains the basal ganglia model on the simple environment
to test the two-layer architecture with learned reward shaping.
"""

import os
import sys
import hydra
from omegaconf import DictConfig

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metta.rl.trainer import MettaTrainer
from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.common.wandb.wandb_context import WandbContext
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment


@hydra.main(version_base=None, config_path="../configs", config_name="train_basal_ganglia")
def main(cfg: DictConfig):
    """Main training function for the basal ganglia model."""

    # Setup logging and environment
    setup_mettagrid_logger()
    setup_mettagrid_environment()

    print("🚀 Starting Basal Ganglia Training")
    print(f"Configuration: {cfg}")

    # Initialize components
    policy_store = PolicyStore()
    sim_suite_config = SimulationSuiteConfig(cfg.sim)

    # Initialize wandb if configured
    wandb_run = None
    if hasattr(cfg, 'wandb') and cfg.wandb is not None:
        wandb_context = WandbContext(cfg.wandb)
        wandb_run = wandb_context.get_run()

    # Initialize trainer
    trainer = MettaTrainer(
        cfg=cfg,
        wandb_run=wandb_run,
        policy_store=policy_store,
        sim_suite_config=sim_suite_config,
        stats_client=None,
    )

    try:
        # Start training
        print("🎯 Beginning training loop...")
        trainer.train()
        print("✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        trainer.close()
        if wandb_run:
            wandb_run.finish()

    print("🏁 Basal Ganglia training script finished")


if __name__ == "__main__":
    main()
