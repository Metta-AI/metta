"""Rollout phase functions for Metta training."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


def rollout(
    vecenv: Any,
    policy: Any,
    experience: Any,
    device: torch.device,
    timer: Any,
) -> tuple[int, list]:
    """Perform a complete rollout phase.

    Returns:
        Tuple of (total_steps, raw_infos)
    """
    raw_infos = []
    experience.reset_for_rollout()
    total_steps = 0

    while not experience.ready_for_training:
        # Get observation
        o, r, d, t, info, training_env_id, mask, num_steps = get_observation(vecenv, device, timer)
        total_steps += num_steps

        # Run policy inference
        actions, selected_action_log_probs, values, lstm_state_to_store = run_policy_inference(
            policy, o, experience, training_env_id.start, device
        )

        # Store experience
        experience.store(
            obs=o,
            actions=actions,
            logprobs=selected_action_log_probs,
            rewards=r,
            dones=d,
            truncations=t,
            values=values,
            env_id=training_env_id,
            mask=mask,
            lstm_state=lstm_state_to_store,
        )

        # Send actions back to environment
        with timer("_rollout.env"):
            vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Collect info for batch processing
        if info:
            raw_infos.extend(info)

    return total_steps, raw_infos


def run_dual_policy_rollout(
    training_policy: torch.nn.Module,
    npc_policy_record: PolicyRecord,
    observations: Tensor,
    experience: Experience,
    training_env_id_start: int,
    device: torch.device,
    training_agents_pct: float,
    num_agents_per_env: int,
    num_envs: int,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
    """Run dual-policy rollout where some agents use training policy and others use NPC policy.

    Args:
        training_policy: The policy being trained
        npc_policy_record: The NPC policy record loaded from wandb URI
        observations: Observations tensor of shape (total_agents, *obs_shape)
        experience: Experience buffer
        training_env_id_start: Starting environment ID
        device: Device to run inference on
        training_agents_pct: Percentage of agents that use training policy
        num_agents_per_env: Number of agents per environment
        num_envs: Number of environments

    Returns:
        Tuple of (actions, selected_action_log_probs, values, lstm_state_to_store)
        Only includes data from training policy agents, NPC data is excluded
    """
    with torch.no_grad():
        # Calculate agent indices for training vs NPC policies
        training_agents_per_env = max(1, int(num_agents_per_env * training_agents_pct))
        npc_agents_per_env = num_agents_per_env - training_agents_per_env

        # Create index matrices for all environments
        total_agents = num_envs * num_agents_per_env
        idx_matrix = torch.arange(total_agents, device=device).reshape(num_envs, num_agents_per_env)

        # Training policy agents: first training_agents_per_env agents in each env
        training_idxs = idx_matrix[:, :training_agents_per_env].reshape(-1)
        # NPC agents: remaining agents in each env
        npc_idxs = (
            idx_matrix[:, training_agents_per_env:].reshape(-1)
            if npc_agents_per_env > 0
            else torch.tensor([], device=device, dtype=torch.long)
        )

        # Get training policy actions for training agents
        training_obs = observations[training_idxs]
        training_state = PolicyState()
        lstm_h, lstm_c = experience.get_lstm_state(training_env_id_start)
        if lstm_h is not None:
            training_state.lstm_h = lstm_h
            training_state.lstm_c = lstm_c

        training_actions, training_log_probs, _, training_values, _ = training_policy(training_obs, training_state)

        # Get NPC policy actions for NPC agents (if any)
        if len(npc_idxs) > 0:
            npc_obs = observations[npc_idxs]
            npc_policy = npc_policy_record.policy
            npc_state = PolicyState()

            # Initialize NPC policy to environment if needed
            if not hasattr(npc_policy, "_initialized_to_env"):
                # This would need to be done during setup, but for now we'll assume it's already done
                pass

            npc_actions, npc_log_probs, _, npc_values, _ = npc_policy(npc_obs, npc_state)

            # Combine actions: training agents first, then NPC agents
            # Reshape to (num_envs, agents_per_env, action_dim)
            training_actions_reshaped = training_actions.reshape(num_envs, training_agents_per_env, -1)
            npc_actions_reshaped = npc_actions.reshape(num_envs, npc_agents_per_env, -1)

            # Concatenate along agents dimension
            all_actions = torch.cat([training_actions_reshaped, npc_actions_reshaped], dim=1)
            # Flatten back to (total_agents, action_dim)
            actions = all_actions.reshape(-1, all_actions.shape[-1])
        else:
            # No NPC agents, use only training actions
            actions = training_actions

        # Store LSTM state for training policy only
        lstm_state_to_store = None
        if training_state.lstm_h is not None and training_state.lstm_c is not None:
            lstm_state_to_store = {"lstm_h": training_state.lstm_h.detach(), "lstm_c": training_state.lstm_c.detach()}

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    # Return only training policy data (actions, log_probs, values)
    # The actions tensor contains all agents' actions for environment step
    # But we only return training policy's log_probs and values for experience storage
    return actions, training_log_probs, training_values.flatten(), lstm_state_to_store


def process_dual_policy_stats(
    raw_infos: list,
    training_agents_pct: float,
    num_agents_per_env: int,
    num_envs: int,
) -> None:
    """Process dual-policy statistics from raw infos and add to info structure.

    This function extracts separate rewards and hearts for training policy agents and NPC agents,
    then adds them to the info structure under a 'dual_policy' section.
    """
    for info in raw_infos:
        if "agent" in info and "game" in info:
            # Calculate agent indices for training vs NPC policies
            training_agents_per_env = max(1, int(num_agents_per_env * training_agents_pct))
            npc_agents_per_env = num_agents_per_env - training_agents_per_env

            # Get episode rewards if available
            episode_rewards = info.get("episode_rewards", None)
            if episode_rewards is not None and len(episode_rewards) == num_envs * num_agents_per_env:
                # Separate rewards for training and NPC agents
                training_rewards = []
                npc_rewards = []

                for env_idx in range(num_envs):
                    env_start = env_idx * num_agents_per_env
                    # Training agents: first training_agents_per_env agents in each env
                    training_rewards.extend(episode_rewards[env_start : env_start + training_agents_per_env])
                    # NPC agents: remaining agents in each env
                    if npc_agents_per_env > 0:
                        npc_rewards.extend(
                            episode_rewards[env_start + training_agents_per_env : env_start + num_agents_per_env]
                        )

                # Calculate statistics
                training_reward_sum = sum(training_rewards) if training_rewards else 0.0
                training_reward_mean = training_reward_sum / len(training_rewards) if training_rewards else 0.0
                npc_reward_sum = sum(npc_rewards) if npc_rewards else 0.0
                npc_reward_mean = npc_reward_sum / len(npc_rewards) if npc_rewards else 0.0
                combined_reward_sum = training_reward_sum + npc_reward_sum
                combined_reward_mean = (
                    combined_reward_sum / (len(training_rewards) + len(npc_rewards))
                    if (training_rewards or npc_rewards)
                    else 0.0
                )

                # Add to info structure
                info["dual_policy"] = {
                    "training_reward_sum": training_reward_sum,
                    "training_reward_mean": training_reward_mean,
                    "training_reward_count": len(training_rewards),
                    "npc_reward_sum": npc_reward_sum,
                    "npc_reward_mean": npc_reward_mean,
                    "npc_reward_count": len(npc_rewards),
                    "combined_reward_sum": combined_reward_sum,
                    "combined_reward_mean": combined_reward_mean,
                    "total_agent_count": len(training_rewards) + len(npc_rewards),
                }

            # Process agent stats for hearts
            agent_stats = info.get("agent", {})
            if agent_stats and "heart" in agent_stats:
                # For hearts, we need to calculate per-agent statistics
                # Since agent stats are averaged across all agents, we need to reconstruct
                total_hearts = agent_stats["heart"] * (num_envs * num_agents_per_env)

                # Estimate hearts for training vs NPC agents based on percentage
                training_hearts = total_hearts * training_agents_pct
                npc_hearts = total_hearts * (1 - training_agents_pct)

                # Add hearts to dual_policy section
                if "dual_policy" not in info:
                    info["dual_policy"] = {}

                info["dual_policy"].update(
                    {
                        "training_hearts": training_hearts,
                        "npc_hearts": npc_hearts,
                        "combined_hearts": total_hearts,
                        "training_hearts_per_agent": training_hearts / (num_envs * training_agents_per_env)
                        if training_agents_per_env > 0
                        else 0.0,
                        "npc_hearts_per_agent": npc_hearts / (num_envs * npc_agents_per_env)
                        if npc_agents_per_env > 0
                        else 0.0,
                        "combined_hearts_per_agent": total_hearts / (num_envs * num_agents_per_env),
                    }
                )


def get_observation(
    vecenv: Any,
    device: torch.device,
    timer: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, list, slice, Tensor, int]:
    """Get observations and other data from the vectorized environment and convert to tensors.

    Returns:
        Tuple of (observations, rewards, dones, truncations, info, training_env_id, mask, num_steps)
    """
    # Receive environment data
    with timer("_rollout.env"):
        o, r, d, t, info, env_id, mask = vecenv.recv()

    training_env_id = slice(env_id[0], env_id[-1] + 1)

    mask = torch.as_tensor(mask)
    num_steps = int(mask.sum().item())

    # Convert to tensors
    o = torch.as_tensor(o).to(device, non_blocking=True)
    r = torch.as_tensor(r).to(device, non_blocking=True)
    d = torch.as_tensor(d).to(device, non_blocking=True)
    t = torch.as_tensor(t).to(device, non_blocking=True)

    return o, r, d, t, info, training_env_id, mask, num_steps


def run_policy_inference(
    policy: torch.nn.Module,
    observations: Tensor,
    experience: Experience,
    training_env_id_start: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
    """Run the policy to get actions and value estimates.

    Returns:
        Tuple of (actions, selected_action_log_probs, values, lstm_state_to_store)
    """
    with torch.no_grad():
        state = PolicyState()
        lstm_h, lstm_c = experience.get_lstm_state(training_env_id_start)
        if lstm_h is not None:
            state.lstm_h = lstm_h
            state.lstm_c = lstm_c

        actions, selected_action_log_probs, _, value, _ = policy(observations, state)

        if __debug__:
            assert_shape(selected_action_log_probs, ("BT",), "selected_action_log_probs")
            assert_shape(actions, ("BT", 2), "actions")

        lstm_state_to_store = None
        if state.lstm_h is not None and state.lstm_c is not None:
            lstm_state_to_store = {"lstm_h": state.lstm_h.detach(), "lstm_c": state.lstm_c.detach()}

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    return actions, selected_action_log_probs, value.flatten(), lstm_state_to_store


def get_lstm_config(policy: Any) -> Tuple[int, int]:
    """Extract LSTM configuration from policy."""
    hidden_size = getattr(policy, "hidden_size", 256)
    num_lstm_layers = 2  # Default value

    # Try to get actual number of LSTM layers from policy
    if hasattr(policy, "components") and "_core_" in policy.components:
        lstm_module = policy.components["_core_"]
        if hasattr(lstm_module, "_net") and hasattr(lstm_module._net, "num_layers"):
            num_lstm_layers = lstm_module._net.num_layers

    return hidden_size, num_lstm_layers
