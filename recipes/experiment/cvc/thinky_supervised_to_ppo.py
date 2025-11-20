"""Mission-variant curriculum with supervised learning from Nim Thinky, then PPO.

This recipe:
1. Trains with supervised learning using Nim Thinky (fast scripted agent) as teacher for 100k steps
2. Student policy is always in the lead (student actions go to environment)
3. Teacher sees the same raw observations as student and generates reference actions
4. Critic is trained with Bellman updates during supervised phase
5. After 100k steps, switches to pure PPO reinforcement learning

The Nim Thinky agent is a fast, deterministic scripted agent that performs well on
CoGs vs Clips missions. It provides strong supervision signal for the student policy.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional, Sequence

from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from metta.rl.loss.action_supervised_and_critic import ActionSupervisedAndCriticConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.loss.ppo import PPOConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.training.scheduler import HyperUpdateRule, LossRunGate, SchedulerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from recipes.experiment import cogs_v_clips
from recipes.experiment.cvc import mission_variant_curriculum as mvc


def train(
    base_missions: Optional[list[str] | str] = None,
    num_cogs: int = 4,
    enable_detailed_slice_logging: bool = False,
    variants: Optional[Sequence[str]] = None,
    exclude_variants: Optional[Sequence[str] | str] = None,
    all_variants_per_mission: bool = False,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    supervised_steps: int = 100_000,  # 100k steps of supervised learning
    run_name: Optional[str] = None,
) -> TrainTool:
    """Create a training tool with Thinky supervised learning, then PPO.

    Args:
        base_missions: Mission names to include. Can be:
            - None: Uses FULL_CURRICULUM_MISSIONS
            - A mission set name: "eval_missions", "diagnostic_missions", "training_facility_missions", "all"
            - A comma-separated string of mission names or set names
            - A list of mission names or set names
        num_cogs: Number of agents per mission
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        variants: Optional mission variants to apply (only used when all_variants_per_mission=False)
        exclude_variants: Optional list of variant names to exclude, or comma-separated string
            (only used when all_variants_per_mission=True)
        all_variants_per_mission: If True, create separate tasks for each mission-variant combination.
            If False, apply the same variants to all missions (or no variants if variants=None).
        eval_variants: Optional mission variants to apply during evaluation
        eval_difficulty: Difficulty variant for evaluation
        supervised_steps: Number of agent steps to train with supervised learning before switching to PPO
        run_name: Optional run name. If not provided, generates one with format "msb_YYYYMMDD_vcts"

    Returns:
        A TrainTool configured with supervised learning from Thinky, then PPO
    """
    # Generate run name if not provided
    if run_name is None:
        run_name = f"msb_{time.strftime('%Y%m%d')}_vcts"

    # When all_variants_per_mission=True, we want all variants (unless exclude_variants is specified)
    resolved_exclude_variants = exclude_variants
    if all_variants_per_mission and exclude_variants is None:
        resolved_exclude_variants = []

    # Use the same curriculum as mission_variant_curriculum
    curriculum = mvc.make_curriculum(
        base_missions=base_missions,
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        variants=variants,
        exclude_variants=resolved_exclude_variants,
        stats_max_cap=0.5 if all_variants_per_mission else 1.0,
    )

    # Configure losses: both action_supervised_and_critic (as supervisor) and PPO
    loss_config = LossesConfig(
        supervisor=ActionSupervisedAndCriticConfig(
            enabled=True,
            # Teacher is the Nim Thinky scripted agent (fast and smart)
            teacher_class_path="cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
            # No teacher_uri means use the scripted agent from teacher_class_path
            teacher_uri=None,
            # Student actions always go to environment (student-led)
            teacher_lead_prob=0.0,
            # Coefficient for action supervision loss
            action_loss_coef=1.0,
            # Enable rollout forward pass (needed to get student actions)
            rollout_forward_enabled=True,
            # Enable train forward pass (needed for backward pass)
            train_forward_enabled=True,
            # Enable sequential sampling from replay buffer
            sample_enabled=True,
            # Critic training parameters (for Bellman updates)
            gamma=0.977,  # Same as PPO default
            gae_lambda=0.891477,  # Same as PPO default
            vf_coef=0.897619,  # Same as PPO default
            vf_clip_coef=0.1,
            clip_vloss=True,
        ),
        ppo=PPOConfig(
            enabled=True,
            # Standard PPO hyperparameters (will be used after supervised phase)
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

    trainer_cfg = TrainerConfig(
        losses=loss_config,
    )

    # Configure scheduler to gate which loss runs when
    # Note: "action_supervisor" is the instance name from LossesConfig._configs()
    scheduler = SchedulerConfig(
        run_gates=[
            # Phase 1: Supervised learning (0 to supervised_steps)
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
            # Phase 2: PPO only (after supervised_steps)
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
            # Optional: Fade out supervised loss coefficient near the transition
            # This provides a smoother handoff from supervised to RL
            HyperUpdateRule(
                loss_instance_name="action_supervisor",
                attr_path="action_loss_coef",
                mode="progress",
                style="linear",
                start_value=1.0,
                end_value=0.0,
                start_agent_step=supervised_steps - 20_000,  # Start fading 20k steps before transition
                end_agent_step=supervised_steps,
            ),
        ],
    )

    # Build eval suite with both standard missions and diagnostic missions
    resolved_eval_variants = cogs_v_clips._resolve_eval_variants(variants, eval_variants)

    # Standard eval missions
    standard_eval_suite = cogs_v_clips.make_eval_suite(
        num_cogs=num_cogs,
        difficulty=eval_difficulty,
        variants=resolved_eval_variants,
    )

    # Add diagnostic missions to eval suite
    diagnostic_missions = []
    for diagnostic_cls in DIAGNOSTIC_EVALS:
        diagnostic_mission = diagnostic_cls()  # type: ignore[call-arg]
        prepared_mission = cogs_v_clips._prepare_mission(
            diagnostic_mission,
            num_cogs=num_cogs,
            variant_names=list(resolved_eval_variants) if resolved_eval_variants else [],
        )
        env_cfg = prepared_mission.make_env()
        sim = SimulationConfig(
            suite="cogs_vs_clips_diagnostic",
            name=f"{diagnostic_mission.name}_{num_cogs}cogs",
            env=env_cfg,
        )
        diagnostic_missions.append(sim)

    # Combine standard and diagnostic missions
    full_eval_suite = standard_eval_suite + diagnostic_missions

    evaluator_cfg = EvaluatorConfig(
        simulations=full_eval_suite,
        evaluate_local=True,  # Enable local evaluation during training
        epoch_interval=10,  # Run local eval every 10 epochs
    )

    return TrainTool(
        run=run_name,
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
        scheduler=scheduler,
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
    num_cogs: int = 4,
    difficulty: str | None = "standard",
    subset: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> EvaluateTool:
    """Evaluate policies on CoGs vs Clips missions."""
    return EvaluateTool(
        simulations=cogs_v_clips.make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            subset=subset,
            variants=variants,
        ),
        policy_uris=policy_uris,
    )


def play(
    policy_uri: Optional[str] = None,
    mission: str = "extractor_hub_30",
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
) -> PlayTool:
    """Play a single mission with a policy."""
    env = cogs_v_clips.make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=variants,
    )
    sim = SimulationConfig(suite="cogs_vs_clips", name=f"{mission}_{num_cogs}cogs", env=env)
    return PlayTool(sim=sim, policy_uri=policy_uri)


def experiment(
    base_missions: Optional[list[str]] = None,
    run_name: Optional[str] = None,
    num_cogs: int = 4,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    variants: Optional[list[str]] = None,
    exclude_variants: Optional[list[str]] = None,
    all_variants_per_mission: bool = False,
    supervised_steps: int = 100_000,
    additional_args: Optional[list[str]] = None,
) -> None:
    """Submit a training job on AWS with 4 GPUs.

    Args:
        base_missions: Optional mission names to include. Can be:
            - None: Uses FULL_CURRICULUM_MISSIONS (if all_variants_per_mission=False)
            - A mission set name: "eval_missions", "diagnostic_missions", "training_facility_missions", "all"
            - A list of mission names or set names
        run_name: Optional run name. If not provided, generates one with timestamp.
        num_cogs: Number of agents per mission (default: 4).
        heartbeat_timeout: Heartbeat timeout in seconds (default: 3600).
        skip_git_check: Whether to skip git check (default: True).
        variants: Optional mission variants to apply (only used when all_variants_per_mission=False)
        exclude_variants: Optional list of variant names to exclude (only used when all_variants_per_mission=True)
        all_variants_per_mission: If True, create separate tasks for each mission-variant combination.
            If False, apply the same variants to all missions (or no variants if variants=None).
        supervised_steps: Number of agent steps to train with supervised learning before switching to PPO
        additional_args: Additional arguments to pass to the training command.
    """
    if run_name is None:
        run_name = f"thinky_supervised_to_ppo_{time.strftime('%Y-%m-%d_%H%M%S')}"

    cmd = [
        "./devops/skypilot/launch.py",
        "recipes.experiment.cvc.thinky_supervised_to_ppo.train",
        f"run={run_name}",
        f"num_cogs={num_cogs}",
        f"supervised_steps={supervised_steps}",
        "--gpus=4",
        f"--heartbeat-timeout={heartbeat_timeout}",
    ]

    if base_missions:
        # Pass base_missions as comma-separated string (shell-safe format)
        missions_str = ",".join(base_missions)
        cmd.append(f"base_missions={missions_str}")

    if skip_git_check:
        cmd.append("--skip-git-check")

    if all_variants_per_mission:
        cmd.append("all_variants_per_mission=True")
        if exclude_variants:
            exclude_str = ",".join(exclude_variants)
            cmd.append(f"exclude_variants={exclude_str}")
    else:
        cmd.append("all_variants_per_mission=False")
        if variants:
            variants_str = ",".join(variants)
            cmd.append(f"variants={variants_str}")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching training job: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    subprocess.run(cmd, check=True)
    print(f"✓ Successfully launched job: {run_name}")


__all__ = [
    "train",
    "evaluate",
    "play",
    "experiment",
]
