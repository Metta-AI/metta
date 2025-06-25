#!/usr/bin/env python3
"""
Functional Training Demo for Metta

This demonstrates how to train a Metta agent using a functional approach,
without the MettaTrainer class. All training logic is exposed as a simple
while loop that calls rollout and train_ppo functions.
"""

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

# Add metta to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent))

from metta.agent.policy_store import PolicyStore
from metta.common.stopwatch import Stopwatch
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functional_trainer import rollout, train_ppo
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.vecenv import make_vecenv
from metta.util.logging import setup_mettagrid_logger

# Ensure pufferlib is available
try:
    from pufferlib import _C  # noqa: F401
except ImportError:
    raise ImportError("Failed to import C/CUDA advantage kernel. Please install pufferlib.") from None

torch.set_float32_matmul_precision("high")


class FunctionalTrainingDemo:
    """A demo class that shows how to do functional training."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = setup_mettagrid_logger("demo")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Timer for performance tracking
        self.timer = Stopwatch(self.logger)
        self.timer.start()

    def setup_environment(self):
        """Create the vectorized environment."""
        self.logger.info("Setting up environment...")

        # Create curriculum from config
        curriculum_config = "/env/mettagrid/curriculum/simple"
        env_overrides = DictConfig({"desync_episodes": True})
        curriculum = curriculum_from_config_path(curriculum_config, env_overrides)

        # Environment parameters
        num_workers = 1
        num_agents = curriculum.get_task().env_cfg().game.num_agents
        batch_size = 128 // num_agents  # Target batch size per agent
        num_envs = batch_size * 2  # async_factor = 2

        self.logger.info(f"Creating {num_envs} environments with {num_agents} agents each")

        # Create vectorized environment
        self.vecenv = make_vecenv(
            curriculum,
            "multiprocessing",
            num_envs=num_envs,
            batch_size=batch_size,
            num_workers=num_workers,
            zero_copy=True,
        )

        # Reset with seed
        seed = self.cfg.get("seed", 42)
        self.vecenv.async_reset(seed)

        # Get driver environment
        self.metta_grid_env = self.vecenv.driver_env
        assert isinstance(self.metta_grid_env, MettaGridEnv)

        return self.vecenv

    def setup_policy(self):
        """Create and initialize the policy."""
        self.logger.info("Setting up policy...")

        # Create policy store
        self.policy_store = PolicyStore(self.cfg, wandb_run=None)

        # Load or create policy
        checkpoint_dir = Path("./demo_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = TrainerCheckpoint.load(str(checkpoint_dir.parent))

        if checkpoint.policy_path:
            policy_record = self.policy_store.load_from_uri(checkpoint.policy_path)
            self.logger.info(f"Loaded existing policy from {checkpoint.policy_path}")
        else:
            policy_record = self.policy_store.create(self.metta_grid_env)
            self.logger.info("Created new policy")

        self.policy = policy_record.policy().to(self.device)

        # Activate actions
        self.policy.activate_actions(self.metta_grid_env.action_names, self.metta_grid_env.max_action_args, self.device)

        return self.policy, checkpoint

    def setup_training(self, checkpoint):
        """Set up training components."""
        self.logger.info("Setting up training components...")

        # Experience buffer
        self.experience = Experience(
            total_agents=self.vecenv.num_agents,
            batch_size=128,
            bptt_horizon=64,
            minibatch_size=64,
            max_minibatch_size=64,
            obs_space=self.vecenv.single_observation_space,
            atn_space=self.vecenv.single_action_space,
            device=self.device,
            hidden_size=getattr(self.policy, "hidden_size", 128),
            cpu_offload=False,
            num_lstm_layers=2,
            agents_per_batch=getattr(self.vecenv, "agents_per_batch", None),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=0.0003,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Load optimizer state if available
        if checkpoint.agent_step > 0:
            try:
                self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                self.logger.info("Loaded optimizer state from checkpoint")
            except Exception as e:
                self.logger.warning(f"Could not load optimizer state: {e}")

        # Loss tracker
        self.losses = Losses()

        # Kickstarter (for imitation learning, optional)
        self.kickstarter = Kickstarter(
            self.cfg, self.policy_store, self.metta_grid_env.action_names, self.metta_grid_env.max_action_args
        )

        # Training state
        self.agent_step = checkpoint.agent_step
        self.epoch = checkpoint.epoch

        return checkpoint

    def train_loop(self, num_epochs: int = 10):
        """Main functional training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs...")

        # PPO hyperparameters
        ppo_config = {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "norm_adv": True,
            "clip_vloss": True,
            "vf_clip_coef": 0.1,
            "update_epochs": 4,
            "target_kl": None,
            "l2_reg_loss_coef": 0.0,
            "l2_init_loss_coef": 0.0,
            "clip_range": 0.0,
            "prio_alpha": 0.0,
            "prio_beta0": 0.6,
            "total_timesteps": 100000,
            "batch_size": 128,
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
        }

        for epoch_idx in range(num_epochs):
            epoch_start_time = time.time()
            steps_before = self.agent_step

            # ========== ROLLOUT PHASE ==========
            self.logger.info(f"Epoch {self.epoch}: Starting rollout...")
            with self.timer("rollout"):
                self.agent_step, stats = rollout(
                    policy=self.policy,
                    vecenv=self.vecenv,
                    experience=self.experience,
                    device=self.device,
                    agent_step=self.agent_step,
                    timer=self.timer,
                )

            # Log rollout stats
            steps_collected = self.agent_step - steps_before
            self.logger.info(f"  Collected {steps_collected} steps")

            if stats and "episode/reward" in stats:
                mean_reward = np.mean(stats["episode/reward"])
                self.logger.info(f"  Mean episode reward: {mean_reward:.2f}")

            # ========== TRAINING PHASE ==========
            self.logger.info(f"Epoch {self.epoch}: Starting PPO training...")
            with self.timer("train"):
                self.epoch = train_ppo(
                    policy=self.policy,
                    optimizer=self.optimizer,
                    experience=self.experience,
                    device=self.device,
                    losses=self.losses,
                    epoch=self.epoch,
                    cfg=self.cfg,
                    lr_scheduler=None,
                    timer=self.timer,
                    kickstarter=self.kickstarter,
                    agent_step=self.agent_step,
                    **ppo_config,
                )

            # Log training stats
            loss_stats = self.losses.stats()
            self.logger.info(f"  Policy loss: {loss_stats.get('policy_loss', 0):.4f}")
            self.logger.info(f"  Value loss: {loss_stats.get('value_loss', 0):.4f}")
            self.logger.info(f"  Entropy: {loss_stats.get('entropy', 0):.4f}")

            # Calculate timing
            epoch_time = time.time() - epoch_start_time
            steps_per_sec = steps_collected / epoch_time if epoch_time > 0 else 0

            rollout_time = self.timer.get_last_elapsed("rollout")
            train_time = self.timer.get_last_elapsed("train")
            total_time = rollout_time + train_time

            train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
            rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0

            self.logger.info(
                f"Epoch {self.epoch} complete - "
                f"{steps_per_sec:.0f} steps/sec "
                f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout)"
            )

            # Save checkpoint every 5 epochs
            if self.epoch % 5 == 0:
                self.save_checkpoint()

        self.logger.info("Training complete!")

    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = Path("./demo_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        # Save policy
        name = f"model_{self.epoch:04d}.pt"
        path = str(checkpoint_dir / name)

        pr = self.policy_store.save(
            name=name, path=path, policy=self.policy, metadata={"epoch": self.epoch, "agent_step": self.agent_step}
        )

        # Save trainer checkpoint
        checkpoint = TrainerCheckpoint(
            agent_step=self.agent_step,
            epoch=self.epoch,
            total_agent_step=self.agent_step,
            optimizer_state_dict=self.optimizer.state_dict(),
            policy_path=pr.uri if pr else None,
            extra_args={},
        )
        checkpoint.save(str(checkpoint_dir.parent))

        self.logger.info(f"Saved checkpoint at epoch {self.epoch}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "vecenv"):
            self.vecenv.close()


@hydra.main(version_base=None, config_path="configs", config_name="demo")
def main(cfg: DictConfig) -> None:
    """Main entry point for the demo."""
    print("=" * 70)
    print("METTA FUNCTIONAL TRAINING DEMO")
    print("=" * 70)
    print()
    print("This demo shows how to train a Metta agent using a functional approach.")
    print("All training logic is exposed as simple function calls in a loop.")
    print()

    # Create demo instance
    demo = FunctionalTrainingDemo(cfg)

    try:
        # Setup components
        demo.setup_environment()
        policy, checkpoint = demo.setup_policy()
        demo.setup_training(checkpoint)

        # Run training
        demo.train_loop(num_epochs=10)

    finally:
        # Cleanup
        demo.cleanup()

    print()
    print("Demo complete! Check ./demo_checkpoints for saved models.")


if __name__ == "__main__":
    # If no config exists, create a minimal one
    demo_config_path = Path("configs/demo.yaml")
    if not demo_config_path.exists():
        demo_config_path.parent.mkdir(exist_ok=True)
        with open(demo_config_path, "w") as f:
            f.write("""# Demo configuration
run: demo_functional_training
device: cpu
seed: 42

agent:
  _target_: metta.agent.metta_agent.MettaAgent
  observations:
    obs_key: grid_obs
  clip_range: 0
  analyze_weights_interval: 0
  l2_init_weight_update_interval: 0
  components:
    _obs_:
      _target_: metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper
      sources: null
    obs_normalizer:
      _target_: metta.agent.lib.observation_normalizer.ObservationNormalizer
      sources:
      - name: _obs_
    cnn1:
      _target_: metta.agent.lib.nn_layer_library.Conv2d
      sources:
      - name: obs_normalizer
      nn_params:
        out_channels: 32
        kernel_size: 3
        stride: 1
    obs_flattener:
      _target_: metta.agent.lib.nn_layer_library.Flatten
      sources:
      - name: cnn1
    fc1:
      _target_: metta.agent.lib.nn_layer_library.Linear
      sources:
      - name: obs_flattener
      nn_params:
        out_features: 128
    _core_:
      _target_: metta.agent.lib.lstm.LSTM
      sources:
      - name: fc1
      output_size: 128
      nn_params:
        num_layers: 2
    critic_1:
      _target_: metta.agent.lib.nn_layer_library.Linear
      sources:
      - name: _core_
      nn_params:
        out_features: 256
    _value_:
      _target_: metta.agent.lib.nn_layer_library.Linear
      sources:
      - name: critic_1
      nn_params:
        out_features: 1
      nonlinearity: null
    actor_1:
      _target_: metta.agent.lib.nn_layer_library.Linear
      sources:
      - name: _core_
      nn_params:
        out_features: 256
    _action_embeds_:
      _target_: metta.agent.lib.action.ActionEmbedding
      sources: null
      nn_params:
        num_embeddings: 100
        embedding_dim: 16
    _action_:
      _target_: metta.agent.lib.actor.MettaActorSingleHead
      sources:
      - name: actor_1
      - name: _action_embeds_
""")

    main()
