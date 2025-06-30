#!/usr/bin/env -S uv run
"""
Meta-analysis demonstration script.

This script demonstrates the proof of principle for predicting training curves
from environment and agent configurations using VAEs.
"""

import argparse
import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.script_decorators import metta_script
from metta.meta_analysis import TrainingDataCollector, MetaAnalysisModel, MetaAnalysisTrainer, TrainingCurveDataset

try:
    import wandb
except ImportError:
    wandb = None


@hydra.main(config_path="../configs", config_name="meta_analysis_demo", version_base=None)
@metta_script
def main(cfg: DictConfig) -> int:
    """Main demonstration script."""

    logger = setup_mettagrid_logger()
    logger.info("Starting meta-analysis demonstration")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect training data from wandb
    if cfg.collect_data:
        logger.info("Collecting training data from wandb...")

        collector = TrainingDataCollector(
            wandb_entity=cfg.wandb.entity,
            wandb_project=cfg.wandb.project
        )

        # Collect training runs
        training_data = collector.collect_training_runs(
            run_filters=cfg.run_filters,
            max_runs=cfg.max_runs,
            start_date=getattr(cfg, "start_date", None),
            end_date=getattr(cfg, "end_date", None)
        )

        if not training_data:
            logger.error("No training data collected!")
            return 1

        # Save dataset
        dataset_path = output_dir / "training_dataset.csv"
        df = collector.save_dataset(training_data, str(dataset_path))

        logger.info(f"Collected {len(training_data)} training runs")
        logger.info(f"Dataset saved to {dataset_path}")

        # Print dataset statistics
        logger.info("Dataset statistics:")
        logger.info(f"  Environment features: {list(df.columns[3:8])}")  # Skip run_id, run_name, final_performance
        logger.info(f"  Agent features: {list(df.columns[8:-1])}")  # Skip training_curve
        logger.info(f"  Average final performance: {df['final_performance'].mean():.3f}")

    # Step 2: Train meta-analysis model
    if cfg.train_model:
        logger.info("Training meta-analysis model...")

        # Define features
        env_features = [
            "max_steps", "num_agents", "map_width", "map_height",
            "num_rooms", "num_altars", "num_mines", "num_generators", "num_walls"
        ]

        agent_features = [
            "learning_rate", "batch_size", "minibatch_size", "gamma",
            "gae_lambda", "clip_coef", "ent_coef", "vf_coef",
            "hidden_size", "num_layers", "cnn_channels", "cnn_kernel_size"
        ]

        # Load dataset
        dataset_path = output_dir / "training_dataset.csv"
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            return 1

        dataset = TrainingCurveDataset(str(dataset_path), env_features, agent_features)

                # Split dataset with reproducible random seed
        train_split_ratio = getattr(cfg, "train_val_split", 0.8)
        random_seed = getattr(cfg, "random_seed", 42)

        train_size = int(train_split_ratio * len(dataset))
        val_size = len(dataset) - train_size

        # Set random seed for reproducible splits
        torch.manual_seed(random_seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation samples (ratio: {train_split_ratio:.1%})")

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

        # Optionally initialize wandb
        wandb_run = None
        if getattr(cfg, "wandb_log", False) and wandb is not None:
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name="meta_analysis_training",
                config=dict(cfg),
                dir=str(output_dir),
                reinit=True,
            )

        # Create model
        model = MetaAnalysisModel(
            env_input_dim=len(env_features),
            agent_input_dim=len(agent_features),
            env_latent_dim=cfg.env_latent_dim,
            agent_latent_dim=cfg.agent_latent_dim,
            curve_length=cfg.curve_length,
            hidden_dim=cfg.hidden_dim
        )

        # Create trainer
        trainer = MetaAnalysisTrainer(
            model=model,
            device=cfg.device,
            learning_rate=cfg.learning_rate,
            beta=cfg.beta,
            curve_weight=cfg.curve_weight,
            wandb_run=wandb_run,
            tsne_interval=getattr(cfg, "tsne_interval", 10),
            tsne_sample_size=getattr(cfg, "tsne_sample_size", 128),
        )

        # Train model
        save_path = output_dir / "meta_analysis_model"
        logger.info(f"Starting training for {cfg.num_epochs} epochs...")
        logger.info(f"Batch size: {cfg.batch_size}, Learning rate: {cfg.learning_rate}")

        training_results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=cfg.num_epochs,
            save_path=str(save_path),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        logger.info("Training completed!")

        # Log final metrics
        if training_results["val_losses"]:
            final_train_loss = training_results["train_losses"][-1]["total_loss"]
            final_val_loss = training_results["val_losses"][-1]["val_loss"]
            logger.info(f"Final metrics - Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}")

        # Save training results
        import json
        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)

        logger.info(f"Training results saved to {results_path}")
        if wandb_run is not None:
            wandb_run.finish()

    # Step 3: Demonstrate predictions
    if cfg.demo_predictions:
        logger.info("Demonstrating predictions...")

        # Load trained model
        model_path = output_dir / "meta_analysis_model_final.pt"
        if not model_path.exists():
            logger.error(f"Trained model not found at {model_path}")
            return 1

        # Create model and load weights
        env_features = [
            "max_steps", "num_agents", "map_width", "map_height",
            "num_rooms", "num_altars", "num_mines", "num_generators", "num_walls"
        ]
        agent_features = [
            "learning_rate", "batch_size", "minibatch_size", "gamma",
            "gae_lambda", "clip_coef", "ent_coef", "vf_coef",
            "hidden_size", "num_layers", "cnn_channels", "cnn_kernel_size"
        ]

        model = MetaAnalysisModel(
            env_input_dim=len(env_features),
            agent_input_dim=len(agent_features),
            env_latent_dim=cfg.env_latent_dim,
            agent_latent_dim=cfg.agent_latent_dim,
            curve_length=cfg.curve_length,
            hidden_dim=cfg.hidden_dim
        )

        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        trainer = MetaAnalysisTrainer(model=model, device=cfg.device)

        # Sample from latent space
        logger.info("Sampling from learned latent space...")
        env_latent, agent_latent = trainer.sample_latent_space(num_samples=5)

        # Generate predicted curves
        predicted_curves = trainer.model.predict_curve(env_latent, agent_latent)

        logger.info("Generated predicted training curves:")
        for i, curve in enumerate(predicted_curves):
            logger.info(f"  Sample {i+1}: Final reward = {curve[-1]:.3f}")

        # Save predictions
        import numpy as np
        predictions_path = output_dir / "latent_space_predictions.npz"
        np.savez(
            predictions_path,
            env_latent=env_latent.cpu().numpy(),
            agent_latent=agent_latent.cpu().numpy(),
            predicted_curves=predicted_curves.cpu().numpy()
        )

        logger.info(f"Predictions saved to {predictions_path}")

    logger.info("Meta-analysis demonstration completed!")
    return 0


if __name__ == "__main__":
    main()
