"""Rollout functionality for training."""

from typing import Any, Tuple

import torch

from metta.mettagrid.mettagrid_env import dtype_actions
from metta.rl.util.rollout import get_observation, run_policy_inference


def rollout(
    vecenv: Any,
    policy: Any,
    experience: Any,
    device: torch.device,
    timer: Any,
) -> Tuple[int, list]:
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
