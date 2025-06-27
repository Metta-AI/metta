"""
Clean API for Metta - provides direct instantiation without Hydra.
"""

import os
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

# Direct instantiation functions


def make_agent(
    obs_space,
    action_space,
    obs_width: int,
    obs_height: int,
    feature_normalizations: Dict[int, float],
    global_features: list,
    device: torch.device,
    obs_key: str = "grid_obs",
    clip_range: float = 0,
    analyze_weights_interval: int = 300,
    l2_init_weight_update_interval: int = 0,
):
    """Create a Metta agent instance directly."""
    from metta.agent.metta_agent import MettaAgent

    # Create agent config directly
    config = {
        "_target_": "metta.agent.metta_agent.MettaAgent",
        "observations": {"obs_key": obs_key},
        "clip_range": clip_range,
        "analyze_weights_interval": analyze_weights_interval,
        "l2_init_weight_update_interval": l2_init_weight_update_interval,
        "components": {
            "_obs_": {"_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper", "sources": None},
            "obs_normalizer": {
                "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                "sources": [{"name": "_obs_"}],
            },
            "cnn1": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "obs_normalizer"}],
                "nn_params": {"out_channels": 64, "kernel_size": 5, "stride": 3},
            },
            "cnn2": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "cnn1"}],
                "nn_params": {"out_channels": 64, "kernel_size": 3, "stride": 1},
            },
            "obs_flattener": {"_target_": "metta.agent.lib.nn_layer_library.Flatten", "sources": [{"name": "cnn2"}]},
            "fc1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "obs_flattener"}],
                "nn_params": {"out_features": 128},
            },
            "encoded_obs": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "fc1"}],
                "nn_params": {"out_features": 128},
            },
            "_core_": {
                "_target_": "metta.agent.lib.lstm.LSTM",
                "sources": [{"name": "encoded_obs"}],
                "output_size": 128,
                "nn_params": {"num_layers": 2},
            },
            "core_relu": {"_target_": "metta.agent.lib.nn_layer_library.ReLU", "sources": [{"name": "_core_"}]},
            "critic_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "core_relu"}],
                "nn_params": {"out_features": 1024},
                "nonlinearity": "nn.Tanh",
                "effective_rank": True,
            },
            "_value_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "critic_1"}],
                "nn_params": {"out_features": 1},
                "nonlinearity": None,
            },
            "actor_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "core_relu"}],
                "nn_params": {"out_features": 512},
            },
            "_action_embeds_": {
                "_target_": "metta.agent.lib.action.ActionEmbedding",
                "sources": None,
                "nn_params": {"num_embeddings": 100, "embedding_dim": 16},
            },
            "_action_": {
                "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
                "sources": [{"name": "actor_1"}, {"name": "_action_embeds_"}],
            },
        },
    }

    return MettaAgent(
        obs_space=obs_space,
        obs_width=obs_width,
        obs_height=obs_height,
        action_space=action_space,
        feature_normalizations=feature_normalizations,
        global_features=global_features,
        device=device,
        **config,
    )


def make_optimizer(
    parameters,
    learning_rate: float = 0.0004573146765703167,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-12,
    weight_decay: float = 0,
    type: str = "adam",
) -> torch.optim.Optimizer:
    """Create an optimizer directly."""
    if type == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {type}")


def make_experience_buffer(
    total_agents: int,
    batch_size: int,
    bptt_horizon: int,
    minibatch_size: int,
    max_minibatch_size: int,
    obs_space,
    atn_space,
    device: torch.device,
    hidden_size: int,
    cpu_offload: bool = False,
    num_lstm_layers: int = 2,
    agents_per_batch: Optional[int] = None,
):
    """Create an experience buffer directly."""
    from metta.rl.experience import Experience

    return Experience(
        total_agents=total_agents,
        batch_size=batch_size,
        bptt_horizon=bptt_horizon,
        minibatch_size=minibatch_size,
        max_minibatch_size=max_minibatch_size,
        obs_space=obs_space,
        atn_space=atn_space,
        device=device,
        hidden_size=hidden_size,
        cpu_offload=cpu_offload,
        num_lstm_layers=num_lstm_layers,
        agents_per_batch=agents_per_batch,
    )


def make_loss_module(
    policy: torch.nn.Module,
    vf_coef: float = 0.44,
    ent_coef: float = 0.0021,
    clip_coef: float = 0.1,
    vf_clip_coef: float = 0.1,
    norm_adv: bool = True,
    clip_vloss: bool = True,
    gamma: float = 0.977,
    gae_lambda: float = 0.916,
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
    l2_reg_loss_coef: float = 0.0,
    l2_init_loss_coef: float = 0.0,
    kickstarter: Optional[Any] = None,
):
    """Create a PPO loss module directly."""
    from metta.rl.objectives import ClipPPOLoss

    return ClipPPOLoss(
        policy=policy,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        clip_coef=clip_coef,
        vf_clip_coef=vf_clip_coef,
        norm_adv=norm_adv,
        clip_vloss=clip_vloss,
        gamma=gamma,
        gae_lambda=gae_lambda,
        vtrace_rho_clip=vtrace_rho_clip,
        vtrace_c_clip=vtrace_c_clip,
        l2_reg_loss_coef=l2_reg_loss_coef,
        l2_init_loss_coef=l2_init_loss_coef,
        kickstarter=kickstarter,
    )


def make_vecenv(
    env_config: Dict[str, Any],
    num_envs: int = 16,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    device: str = "cpu",
    zero_copy: bool = True,
    vectorization: str = "serial",
):
    """Create a vectorized environment directly."""
    from metta.mettagrid.curriculum.core import SingleTaskCurriculum
    from metta.rl.vecenv import make_vecenv

    curriculum = SingleTaskCurriculum("task", DictConfig(env_config))

    return make_vecenv(
        curriculum=curriculum,
        vectorization=vectorization,
        num_envs=num_envs,
        batch_size=batch_size,
        num_workers=num_workers,
        zero_copy=zero_copy,
        is_training=True,
    )


def env(
    num_agents: int = 2,
    width: int = 15,
    height: int = 10,
    max_steps: int = 1000,
    obs_width: int = 11,
    obs_height: int = 11,
) -> Dict[str, Any]:
    """Create a default MetaGrid environment configuration."""
    return {
        "sampling": 0,
        "desync_episodes": False,
        "replay_level_prob": 0.0,
        "game": {
            "num_agents": num_agents,
            "obs_width": obs_width,
            "obs_height": obs_height,
            "num_observation_tokens": 200,
            "max_steps": max_steps,
            "inventory_item_names": [
                "ore.red",
                "battery.red",
                "heart",
                "laser",
                "armor",
            ],
            "diversity_bonus": {"enabled": False, "similarity_coef": 0.5, "diversity_coef": 0.5},
            "agent": {
                "default_item_max": 50,
                "heart_max": 255,
                "freeze_duration": 10,
                "rewards": {
                    "action_failure_penalty": 0,
                    "ore.red": 0.01,
                    "battery.red": 0.02,
                    "heart": 1,
                    "heart_max": 1000,
                },
            },
            "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
            "objects": {
                "altar": {
                    "input_battery.red": 1,
                    "output_heart": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_items": 1,
                },
                "mine_red": {
                    "output_ore.red": 1,
                    "color": 0,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 50,
                    "initial_items": 1,
                },
                "generator_red": {
                    "input_ore.red": 1,
                    "output_battery.red": 1,
                    "color": 0,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 50,
                    "initial_items": 1,
                },
                "wall": {"swappable": False},
                "block": {"swappable": True},
            },
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "attack": {"enabled": False},
                "swap": {"enabled": True},
                "change_color": {"enabled": False},
            },
            "reward_sharing": {"groups": {}},
            "map_builder": {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": width,
                "height": height,
                "border_width": 2,
                "agents": num_agents,
                "objects": {"mine_red": 2, "generator_red": 1, "altar": 1, "wall": 5, "block": 3},
            },
        },
    }


def build_runtime_config(
    run: str = "default_run",
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 0,
    vectorization: str = "serial",
) -> Dict[str, Any]:
    """Build the runtime configuration for Metta."""
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "./train_dir")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "run": run,
        "data_dir": data_dir,
        "run_dir": f"{data_dir}/{run}",
        "policy_uri": f"file://{data_dir}/{run}/checkpoints",
        "torch_deterministic": True,
        "vectorization": vectorization,
        "seed": seed,
        "device": device,
        "stats_user": os.environ.get("USER", "unknown"),
        "dist_cfg_path": None,
        "hydra": {"callbacks": {"resolver_callback": {"_target_": "metta.common.util.resolvers.ResolverRegistrar"}}},
    }


def setup_metta_environment(config: Dict[str, Any]) -> None:
    """Setup the Metta environment with the given configuration."""
    from metta.common.util.runtime_configuration import setup_mettagrid_environment

    setup_mettagrid_environment(DictConfig(config))


def get_logger(name: str):
    """Get a configured logger for Metta."""
    from metta.common.util.logging import setup_mettagrid_logger

    return setup_mettagrid_logger(name)


# High-level convenience functions


def quick_train(
    run_name: str = "default_run",
    timesteps: int = 50_000_000_000,  # Match original default
    batch_size: int = 262_144,  # Match original default
    num_agents: int = 2,
    num_workers: int = 1,
    learning_rate: float = 0.0004573146765703167,  # Match original default
    checkpoint_interval: int = 60,  # seconds, not epochs
    evaluate_interval: int = 300,  # seconds
    device: str = "cuda",
    vectorization: str = "serial",
    env_width: int = 15,
    env_height: int = 10,
    bptt_horizon: int = 64,  # Match original default
    minibatch_size: int = 16_384,  # Match original default
    update_epochs: int = 1,  # Match original default
    max_grad_norm: float = 0.5,  # Match original default
    logger=None,
) -> str:
    """Quick training function with sensible defaults matching the original trainer.

    Args:
        run_name: Name of the training run
        timesteps: Total timesteps to train
        batch_size: Batch size for training
        num_agents: Number of agents per environment
        num_workers: Number of workers
        learning_rate: Learning rate
        checkpoint_interval: How often to save checkpoints (in seconds)
        evaluate_interval: How often to evaluate (in seconds)
        device: Device to use
        vectorization: Vectorization mode
        env_width: Environment width
        env_height: Environment height
        bptt_horizon: BPTT horizon for LSTM training
        minibatch_size: Minibatch size
        update_epochs: Number of epochs to update per rollout
        max_grad_norm: Maximum gradient norm for clipping
        logger: Optional logger instance

    Returns:
        Path to the final checkpoint
    """
    import os
    import time

    import gymnasium as gym
    import numpy as np

    from metta.common.stopwatch import Stopwatch
    from metta.rl.functional_trainer import (
        compute_initial_advantages,
        perform_rollout_step,
        process_rollout_infos,
    )
    from metta.rl.losses import Losses

    if logger is None:
        logger = get_logger("quick_train")

    # Setup directories
    checkpoint_dir = f"./train_dir/{run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create environment
    env_config = env(
        num_agents=num_agents,
        width=env_width,
        height=env_height,
        max_steps=1000,
    )

    # Calculate environment count for batch size
    # Match original trainer logic
    forward_pass_minibatch_target_size = 2048  # From original config
    target_batch_size = forward_pass_minibatch_target_size // num_agents
    if target_batch_size < max(2, num_workers):
        target_batch_size = num_workers

    # Adjust batch_size to be multiple of num_workers
    env_batch_size = (target_batch_size // num_workers) * num_workers
    async_factor = 2  # From original config
    num_envs = env_batch_size * async_factor

    logger.info(f"Using {num_envs} environments with batch size {env_batch_size}")
    logger.info(f"Total agents: {num_envs * num_agents}")

    # Create vectorized environment
    vecenv = make_vecenv(
        env_config=env_config,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=env_batch_size if num_workers > 1 else None,
        device=device,
        vectorization=vectorization,
    )

    env_info = vecenv.driver_env

    # Create observation space
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env_info.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Create agent
    device_obj = torch.device(device)
    policy = make_agent(
        obs_space=obs_space,
        action_space=env_info.single_action_space,
        obs_width=env_info.obs_width,
        obs_height=env_info.obs_height,
        feature_normalizations=env_info.feature_normalizations,
        global_features=env_info.global_features,
        device=device_obj,
    )
    policy.activate_actions(env_info.action_names, env_info.max_action_args, device_obj)

    # For experience buffer, use actual batch_size parameter
    # The original trainer uses a different batch_size for experience vs environments
    total_agents = vecenv.num_agents

    # Validate batch_size is large enough
    min_batch_size = total_agents * bptt_horizon
    if batch_size < min_batch_size:
        logger.warning(
            f"Batch size {batch_size} is too small for {total_agents} agents with "
            f"bptt_horizon {bptt_horizon}. Adjusting to minimum {min_batch_size}."
        )
        batch_size = min_batch_size

    # Create experience buffer with proper minibatch calculation
    # Ensure minibatch_size divides batch_size evenly
    while batch_size % minibatch_size != 0 and minibatch_size > 1:
        minibatch_size -= 1

    experience = make_experience_buffer(
        total_agents=total_agents,
        batch_size=batch_size,
        bptt_horizon=bptt_horizon,
        minibatch_size=minibatch_size,
        max_minibatch_size=minibatch_size,
        obs_space=env_info.single_observation_space,
        atn_space=env_info.single_action_space,
        device=device_obj,
        hidden_size=policy.hidden_size,
        num_lstm_layers=policy.core_num_layers,
        agents_per_batch=getattr(vecenv, "agents_per_batch", None),
    )

    # Create optimizer and loss module
    optimizer = make_optimizer(policy.parameters(), learning_rate=learning_rate)
    loss_module = make_loss_module(policy=policy)
    losses = Losses()

    # Training setup
    timer = Stopwatch(logger)
    timer.start()

    # Track time for interval-based operations
    last_checkpoint_time = time.time()
    last_eval_time = time.time()
    start_time = time.time()

    logger.info("Starting training...")
    logger.info(f"Total timesteps: {timesteps:,}")
    logger.info(f"Batch size: {batch_size:,}")
    logger.info(f"Minibatch size: {minibatch_size:,}")
    logger.info(f"Update epochs: {update_epochs}")
    logger.info(f"BPTT horizon: {bptt_horizon}")

    vecenv.async_reset(seed=0)

    agent_step = 0
    epoch = 0
    all_rollout_stats = {}

    while agent_step < timesteps:
        steps_before = agent_step

        # Rollout
        with timer("rollout"):
            raw_infos = []
            experience.reset_for_rollout()

            while not experience.ready_for_training:
                num_steps, info, _ = perform_rollout_step(
                    policy=policy,
                    vecenv=vecenv,
                    experience=experience,
                    device=device_obj,
                    timer=timer,
                )
                agent_step += num_steps
                if info:
                    raw_infos.extend(info)

            # Process rollout stats
            rollout_stats = process_rollout_infos(raw_infos)

            # Accumulate stats
            for k, v in rollout_stats.items():
                if k not in all_rollout_stats:
                    all_rollout_stats[k] = []
                if isinstance(v, list):
                    all_rollout_stats[k].extend(v)
                else:
                    all_rollout_stats[k].append(v)

        # Train
        with timer("train"):
            losses.zero()
            experience.reset_importance_sampling_ratios()

            # Compute advantages
            advantages = compute_initial_advantages(
                experience, gamma=0.977, gae_lambda=0.916, vtrace_rho_clip=1.0, vtrace_c_clip=1.0, device=device_obj
            )

            # Update epochs
            for update_epoch in range(update_epochs):
                # Train minibatches
                for mb_idx in range(experience.num_minibatches):
                    minibatch = experience.sample_minibatch(
                        advantages=advantages,
                        prio_alpha=0.0,  # No prioritized replay by default
                        prio_beta=0.6,
                        minibatch_idx=mb_idx,
                        total_minibatches=experience.num_minibatches,
                    )

                    loss = loss_module(
                        minibatch=minibatch,
                        experience=experience,
                        losses=losses,
                        agent_step=agent_step,
                        device=device_obj,
                    )
                    losses.minibatches_processed += 1

                    optimizer.zero_grad()
                    loss.backward()

                    if (mb_idx + 1) % experience.accumulate_minibatches == 0:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                        optimizer.step()
                        if hasattr(policy, "clip_weights"):
                            policy.clip_weights()

        # Calculate and log metrics
        steps_in_epoch = agent_step - steps_before
        rollout_time = timer.get_last_elapsed("rollout")
        train_time = timer.get_last_elapsed("train")
        total_time = rollout_time + train_time
        steps_per_sec = steps_in_epoch / total_time if total_time > 0 else 0

        train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
        rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0

        # Calculate average reward from stats
        mean_reward = 0.0
        if all_rollout_stats:
            reward_keys = [k for k in all_rollout_stats.keys() if "reward" in k and "mean" in k]
            if reward_keys:
                all_rewards = []
                for k in reward_keys:
                    if isinstance(all_rollout_stats[k], list):
                        all_rewards.extend(all_rollout_stats[k])
                    else:
                        all_rewards.append(all_rollout_stats[k])
                if all_rewards:
                    mean_reward = np.mean(all_rewards)

        loss_stats = losses.stats()

        logger.info(
            f"Epoch {epoch} - Steps: {agent_step:,}/{timesteps:,} - "
            f"{steps_per_sec:.0f} sps "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout) - "
            f"Policy loss: {loss_stats['policy_loss']:.4f} - "
            f"Value loss: {loss_stats['value_loss']:.4f} - "
            f"Reward: {mean_reward:.3f}"
        )

        # Time-based checkpoint saving
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval:
            checkpoint_path = f"{checkpoint_dir}/policy_epoch_{epoch}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path} (after {current_time - start_time:.0f}s)")
            last_checkpoint_time = current_time

        # Clear accumulated stats periodically
        if epoch % 10 == 0:
            all_rollout_stats.clear()

        epoch += 1

    # Save final checkpoint
    final_checkpoint = f"{checkpoint_dir}/policy_final.pt"
    torch.save(policy.state_dict(), final_checkpoint)
    logger.info(f"Training complete! Final checkpoint: {final_checkpoint}")

    # Log timing summary
    elapsed_time = time.time() - start_time
    logger.info(f"Total training time: {elapsed_time:.1f}s")
    logger.info(f"Average SPS: {agent_step / elapsed_time:.0f}")

    vecenv.close()
    return final_checkpoint


def quick_eval(
    checkpoint_path: str,
    num_episodes: int = 10,
    num_envs: int = 32,
    num_agents: int = 2,
    device: str = "cuda",
    vectorization: str = "multiprocessing",
    env_width: int = 15,
    env_height: int = 10,
    logger=None,
) -> Dict[str, Any]:
    """Quick evaluation function.

    Args:
        checkpoint_path: Path to checkpoint to evaluate
        num_episodes: Number of episodes to run
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        device: Device to use
        vectorization: Vectorization mode
        env_width: Environment width
        env_height: Environment height
        logger: Optional logger instance

    Returns:
        Dictionary with evaluation results
    """
    import gymnasium as gym
    import numpy as np

    if logger is None:
        logger = get_logger("quick_eval")

    # Create environment
    env_config = env(
        num_agents=num_agents,
        width=env_width,
        height=env_height,
        max_steps=1000,
    )

    # Create vectorized environment
    vecenv = make_vecenv(
        env_config=env_config,
        num_envs=num_envs,
        num_workers=1,
        device=device,
        vectorization=vectorization,
    )

    env_info = vecenv.driver_env

    # Create observation space
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": env_info.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Create agent
    device_obj = torch.device(device)
    policy = make_agent(
        obs_space=obs_space,
        action_space=env_info.single_action_space,
        obs_width=env_info.obs_width,
        obs_height=env_info.obs_height,
        feature_normalizations=env_info.feature_normalizations,
        global_features=env_info.global_features,
        device=device_obj,
    )
    policy.activate_actions(env_info.action_names, env_info.max_action_args, device_obj)

    # Load checkpoint
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()

    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # Run evaluation
    rewards = []
    episode_lengths = []
    episodes_completed = 0

    # Reset and start environments
    vecenv.async_reset(seed=42)

    # Initialize hidden state
    from metta.agent.policy_state import PolicyState
    from metta.mettagrid.mettagrid_env import dtype_actions

    state = PolicyState()
    if hasattr(policy, "core_num_layers"):
        state.lstm_h = torch.zeros(policy.core_num_layers, vecenv.num_agents, policy.hidden_size, device=device_obj)
        state.lstm_c = torch.zeros(policy.core_num_layers, vecenv.num_agents, policy.hidden_size, device=device_obj)

    step_count = 0

    logger.info(f"Starting evaluation with {num_envs} environments, collecting {num_episodes} episodes")

    while episodes_completed < num_episodes:
        # Receive from environment
        o, r, d, t, info, env_id, mask = vecenv.recv()
        step_count += 1

        # Convert observations to tensors
        o = torch.as_tensor(o).to(device_obj, non_blocking=True)

        with torch.no_grad():
            actions, _, _, _, _ = policy(o, state)

        # Send actions to environment
        vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Process episode completions
        if info:
            # Debug first few steps to see info structure
            if step_count <= 5:
                logger.info(f"Info at step {step_count}: {info}")

            # Process info like in training - it might be nested
            from metta.mettagrid.util.dict_utils import unroll_nested_dict

            for idx, info_dict in enumerate(info):
                if info_dict:
                    # Unroll nested dictionary
                    flat_info = dict(unroll_nested_dict(info_dict))

                    # Check various possible keys for episode completion
                    episode_done = False
                    episode_return = None
                    episode_length = None

                    # Look for task_reward pattern (e.g., "task_reward/task/rewards.mean")
                    for key, value in flat_info.items():
                        if key.startswith("task_reward/") and key.endswith("/rewards.mean"):
                            episode_return = value
                            episode_done = True
                            logger.info(f"Found episode completion with key: {key} = {value}")
                            break

                    # Also check for episode length/steps
                    if "attributes" in flat_info and isinstance(flat_info["attributes"], dict):
                        if "steps" in flat_info["attributes"]:
                            episode_length = flat_info["attributes"]["steps"]

                    if episode_done and episode_return is not None:
                        rewards.append(float(episode_return))
                        if episode_length is not None:
                            episode_lengths.append(int(episode_length))
                        episodes_completed += 1

                        logger.info(
                            f"Episode {episodes_completed}/{num_episodes} completed: "
                            f"reward={episode_return:.2f}, length={episode_length or 'N/A'}"
                        )

                        if episodes_completed >= num_episodes:
                            logger.info(f"Collected {num_episodes} episodes after {step_count} steps")
                            break

        # Debug: check what's in info periodically
        if step_count % 5000 == 0 and info:
            logger.info(f"Sample info at step {step_count}: {info[0] if info else 'None'}")

        # Log progress every 1000 steps
        if step_count % 1000 == 0:
            logger.info(f"Evaluation step {step_count}, episodes completed: {episodes_completed}/{num_episodes}")

    vecenv.close()

    # Compute results
    results = {
        "num_episodes": len(rewards),
        "avg_reward": np.mean(rewards) if rewards else 0.0,
        "std_reward": np.std(rewards) if rewards else 0.0,
        "min_reward": np.min(rewards) if rewards else 0.0,
        "max_reward": np.max(rewards) if rewards else 0.0,
    }

    if episode_lengths:
        results["avg_episode_length"] = np.mean(episode_lengths)
        results["episode_lengths"] = episode_lengths

    return results


def quick_sim(
    run_name: str,
    policy_uri: str,
    num_episodes: int = 10,
    num_envs: int = 32,
    num_agents: int = 2,
    device: str = "cuda",
    logger=None,
) -> Dict[str, Any]:
    """Quick simulation/evaluation function using direct evaluation.

    Args:
        run_name: Name of the run
        policy_uri: URI of the policy to evaluate
        num_episodes: Number of episodes to run
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        device: Device to use
        logger: Optional logger instance

    Returns:
        Dictionary with simulation results
    """
    import os

    if logger is None:
        logger = get_logger("quick_sim")

    # Extract checkpoint path from URI
    if policy_uri.startswith("file://"):
        checkpoint_path = policy_uri[7:]
    else:
        checkpoint_path = policy_uri

    # Make sure path is absolute
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)

    logger.info(f"Evaluating policy: {checkpoint_path}")

    # Use quick_eval to run the evaluation
    results = quick_eval(
        checkpoint_path=checkpoint_path,
        num_episodes=num_episodes,
        num_envs=num_envs,
        num_agents=num_agents,
        device=device,
        vectorization="multiprocessing",
        logger=logger,
    )

    # Format results
    policy_name = os.path.basename(checkpoint_path)
    return {
        "policies": [
            {
                "name": policy_name,
                "uri": policy_uri,
                "metrics": results,
            }
        ]
    }


# Configuration factory functions (from metta/__init__.py)


def agent_config(
    obs_key: str = "grid_obs",
    clip_range: float = 0,
    analyze_weights_interval: int = 300,
    l2_init_weight_update_interval: int = 0,
) -> Dict[str, Any]:
    """Create a default Metta agent configuration dict."""
    return {
        "_target_": "metta.agent.metta_agent.MettaAgent",
        "observations": {"obs_key": obs_key},
        "clip_range": clip_range,
        "analyze_weights_interval": analyze_weights_interval,
        "l2_init_weight_update_interval": l2_init_weight_update_interval,
        "components": {
            "_obs_": {"_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper", "sources": None},
            "obs_normalizer": {
                "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                "sources": [{"name": "_obs_"}],
            },
            "cnn1": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "obs_normalizer"}],
                "nn_params": {"out_channels": 64, "kernel_size": 5, "stride": 3},
            },
            "cnn2": {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": "cnn1"}],
                "nn_params": {"out_channels": 64, "kernel_size": 3, "stride": 1},
            },
            "obs_flattener": {"_target_": "metta.agent.lib.nn_layer_library.Flatten", "sources": [{"name": "cnn2"}]},
            "fc1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "obs_flattener"}],
                "nn_params": {"out_features": 128},
            },
            "encoded_obs": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "fc1"}],
                "nn_params": {"out_features": 128},
            },
            "_core_": {
                "_target_": "metta.agent.lib.lstm.LSTM",
                "sources": [{"name": "encoded_obs"}],
                "output_size": 128,
                "nn_params": {"num_layers": 2},
            },
            "core_relu": {"_target_": "metta.agent.lib.nn_layer_library.ReLU", "sources": [{"name": "_core_"}]},
            "critic_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "core_relu"}],
                "nn_params": {"out_features": 1024},
                "nonlinearity": "nn.Tanh",
                "effective_rank": True,
            },
            "_value_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "critic_1"}],
                "nn_params": {"out_features": 1},
                "nonlinearity": None,
            },
            "actor_1": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "core_relu"}],
                "nn_params": {"out_features": 512},
            },
            "_action_embeds_": {
                "_target_": "metta.agent.lib.action.ActionEmbedding",
                "sources": None,
                "nn_params": {"num_embeddings": 100, "embedding_dim": 16},
            },
            "_action_": {
                "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
                "sources": [{"name": "actor_1"}, {"name": "_action_embeds_"}],
            },
        },
    }


def trainer_config(
    total_timesteps: int = 10_000,
    batch_size: int = 256,
    learning_rate: float = 0.0003,
    checkpoint_dir: str = "./checkpoints",
    num_workers: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """Create a default trainer configuration dict."""
    # Calculate appropriate minibatch_size based on batch_size
    minibatch_size = min(32, batch_size)
    while batch_size % minibatch_size != 0 and minibatch_size > 1:
        minibatch_size -= 1

    defaults = {
        "target": "metta.rl.trainer.MettaTrainer",
        "total_timesteps": total_timesteps,
        "clip_coef": 0.1,
        "ent_coef": 0.0021,
        "gae_lambda": 0.916,
        "gamma": 0.977,
        "learning_rate": learning_rate,
        "max_grad_norm": 0.5,
        "vf_clip_coef": 0.1,
        "vf_coef": 0.44,
        "norm_adv": True,
        "clip_vloss": True,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "bptt_horizon": 8,
        "num_workers": num_workers,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_interval": 100,
    }

    # Override defaults with any provided kwargs
    defaults.update(kwargs)
    return defaults


def sim_suite_config(
    name: str = "eval",
    num_envs: int = 32,
    num_episodes: int = 10,
    map_preview_limit: int = 32,
) -> Dict[str, Any]:
    """Create a default simulation suite configuration dict."""
    return {
        "_target_": "metta.sim.simulation_config.SimulationSuiteConfig",
        "name": name,
        "num_envs": num_envs,
        "num_episodes": num_episodes,
        "map_preview_limit": map_preview_limit,
        "suites": [],
    }


def wandb_config(
    mode: str = "disabled",
    project: str = "metta",
    entity: Optional[str] = None,
    tags: Optional[list] = None,
) -> Dict[str, Any]:
    """Create a default WandB configuration dict."""
    return {
        "mode": mode,
        "project": project,
        "entity": entity,
        "tags": tags or [],
    }
