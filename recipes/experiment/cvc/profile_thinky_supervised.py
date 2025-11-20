"""Quick profiling script for thinky_supervised_to_ppo.

Run on GPU machine to identify bottlenecks:
    uv run python recipes/experiment/cvc/profile_thinky_supervised.py
"""

from __future__ import annotations

import time
from typing import Optional

from metta.rl.loss.action_supervised_and_critic_profiled import ActionSupervisedAndCriticConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.loss.ppo import PPOConfig
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.training.scheduler import HyperUpdateRule, LossRunGate, SchedulerConfig
from metta.tools.train import TrainTool
from recipes.experiment.cvc import mission_variant_curriculum as mvc


def profile_supervised_rollout(
    base_missions: Optional[list[str] | str] = "extractor_hub_30",
    num_cogs: int = 4,
    supervised_steps: int = 50_000,
    run_name: Optional[str] = None,
) -> TrainTool:
    """Profiling-enabled version of thinky_supervised_to_ppo.train().

    Changes from production version:
    - Uses profiled loss implementation
    - Enables PyTorch profiler
    - Disables local evaluation
    - Shorter total timesteps for quick profiling
    """
    if run_name is None:
        run_name = f"msb_profile_{time.strftime('%Y%m%d_%H%M%S')}"

    # Minimal curriculum for profiling
    curriculum = mvc.make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=False,
        variants=None,
        exclude_variants=None,
        stats_max_cap=1.0,
    )

    # Same loss config but with profiled implementation
    loss_config = LossesConfig(
        supervisor=ActionSupervisedAndCriticConfig(
            enabled=True,
            teacher_class_path="cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
            teacher_uri=None,
            teacher_lead_prob=0.0,
            action_loss_coef=1.0,
            rollout_forward_enabled=True,
            train_forward_enabled=True,
            sample_enabled=True,
            gamma=0.977,
            gae_lambda=0.891477,
            vf_coef=0.897619,
            vf_clip_coef=0.1,
            clip_vloss=True,
        ),
        ppo=PPOConfig(
            enabled=True,
            clip_coef=0.264407,
            ent_coef=0.010000,
            gae_lambda=0.891477,
            gamma=0.977,
            vf_clip_coef=0.1,
            vf_coef=0.897619,
            norm_adv=True,
            clip_vloss=True,
        ),
    )

    # Enable PyTorch profiler
    profiler_config = TorchProfilerConfig(
        enabled=True,
        wait_steps=2,
        warmup_steps=2,
        active_steps=4,
        repeat=1,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    trainer_cfg = TrainerConfig(
        losses=loss_config,
        profiler=profiler_config,
        total_timesteps=10_000,  # Short run for profiling
    )

    scheduler = SchedulerConfig(
        run_gates=[
            LossRunGate(
                loss_instance_name="action_supervisor",
                phase="rollout",
                begin_at_step=0,
                end_at_step=supervised_steps,
            ),
            LossRunGate(
                loss_instance_name="action_supervisor",
                phase="train",
                begin_at_step=0,
                end_at_step=supervised_steps,
            ),
            LossRunGate(
                loss_instance_name="ppo",
                phase="rollout",
                begin_at_step=supervised_steps,
            ),
            LossRunGate(
                loss_instance_name="ppo",
                phase="train",
                begin_at_step=supervised_steps,
            ),
        ],
        rules=[
            HyperUpdateRule(
                loss_instance_name="action_supervisor",
                attr_path="action_loss_coef",
                mode="progress",
                style="linear",
                start_value=1.0,
                end_value=0.0,
                start_agent_step=supervised_steps - 20_000,
                end_agent_step=supervised_steps,
            ),
        ],
    )

    # Disable local evaluation for profiling
    evaluator_cfg = EvaluatorConfig(
        simulations=[],
        evaluate_local=False,
    )

    return TrainTool(
        run=run_name,
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Profiling thinky_supervised_to_ppo")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Run 10k timesteps with detailed timing logs")
    print("  2. Generate PyTorch profiler trace")
    print("  3. Print bottleneck analysis")
    print("\nProfiler output will be in:")
    print("  ./train_dir/msb_profile_*/profiler_trace_*.json")
    print("\nView with: chrome://tracing\n")
    print("=" * 60)

    tool = profile_supervised_rollout()
    result = tool.invoke({})
    sys.exit(result or 0)
