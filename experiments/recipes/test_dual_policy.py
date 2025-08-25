"""
Simple test recipe for dual policy with the roomba v9 NPC checkpoint.

Usage:
    python tools/run.py experiments.recipes.test_dual_policy:test
"""

# The NPC checkpoint you want to test with
# Note: Wandb artifact names use entity/project/artifact:version format
# You can override this by setting an environment variable or passing it as an argument
import os

from metta.cogworks.curriculum import env_curriculum
from metta.mettagrid.config.envs import make_arena
from metta.rl.trainer_config import DualPolicyConfig, EvaluationConfig, TrainerConfig
from metta.tools.train import TrainTool

NPC_CHECKPOINT = os.environ.get(
    "NPC_CHECKPOINT",
    "wandb://metta-research/dual_policy_training/model-bullm_dual_policy_against_roomba_v9:v2",
)


def test():
    """Test dual policy with roomba v9 NPC."""

    env = make_arena(num_agents=24)

    trainer_cfg = TrainerConfig(
        curriculum=env_curriculum(env),
        total_timesteps=100_000,  # Short test run
        # Dual policy configuration
        dual_policy=DualPolicyConfig(
            enabled=True,
            training_agents_pct=0.5,  # 50% training, 50% NPC
            checkpoint_npc=NPC_CHECKPOINT,
        ),
        # Skip git check for local testing
        evaluation=EvaluationConfig(
            skip_git_check=True,
            evaluate_interval=0,  # Disable evaluation for quick test
        ),
        # Basic training settings
        batch_size=2048,
        minibatch_size=256,
        rollout_workers=4,
    )

    return TrainTool(trainer=trainer_cfg)
