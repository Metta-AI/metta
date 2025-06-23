"""Rollout collectors for gathering experience from environments."""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from metta.agent.policy_state import PolicyState
from metta.rl.experience import Experience
from mettagrid.mettagrid_env import dtype_actions
from mettagrid.util.dict_utils import unroll_nested_dict

if TYPE_CHECKING:
    from metta.agent import BaseAgent

logger = logging.getLogger(__name__)


class RolloutCollector:
    """Collects rollouts from vectorized environments.

    This class handles the interaction between agents and environments,
    collecting experience data for training.
    """

    def __init__(
        self,
        vecenv,
        policy: "BaseAgent",
        experience_buffer: Experience,
        device: torch.device,
    ):
        """Initialize the rollout collector.

        Args:
            vecenv: Vectorized environment
            policy: Policy to collect rollouts with
            experience_buffer: Buffer to store experience
            device: Device to run computations on
        """
        self.vecenv = vecenv
        self.policy = policy
        self.experience = experience_buffer
        self.device = device
        self.agent_steps = 0

    def collect(
        self,
        num_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Collect rollouts from the environment.

        Args:
            num_steps: Number of steps to collect (if None, fills experience buffer)

        Returns:
            Tuple of (stats_dict, steps_collected)
        """
        infos = defaultdict(list)
        raw_infos = []
        steps_collected = 0

        # Reset for new rollout
        self.experience.reset_for_rollout()

        # Determine collection condition
        if num_steps is not None:
            should_continue = lambda: steps_collected < num_steps
        else:
            should_continue = lambda: not self.experience.ready_for_training

        while should_continue():
            # Receive from environment
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            # Convert to tensors
            mask_tensor = torch.as_tensor(mask)
            num_env_steps = int(mask_tensor.sum().item())
            steps_collected += num_env_steps
            self.agent_steps += num_env_steps

            o = torch.as_tensor(o).to(self.device, non_blocking=True)
            r = torch.as_tensor(r).to(self.device, non_blocking=True)
            d = torch.as_tensor(d).to(self.device, non_blocking=True)
            t = torch.as_tensor(t).to(self.device, non_blocking=True)

            # Get actions from policy
            with torch.no_grad():
                state = PolicyState()

                # Get LSTM state if available
                training_env_id = slice(env_id[0], env_id[-1] + 1)
                lstm_h, lstm_c = self.experience.get_lstm_state(training_env_id.start)
                if lstm_h is not None:
                    state.lstm_h = lstm_h
                    state.lstm_c = lstm_c

                # Forward pass
                actions, selected_action_log_probs, _, value, _ = self.policy(o, state)

                # Store LSTM state for next step
                lstm_state_to_store = None
                if state.lstm_h is not None:
                    lstm_state_to_store = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c}

                if str(self.device).startswith("cuda"):
                    torch.cuda.synchronize()

            # Flatten value tensor
            value = value.flatten()

            # Store in experience buffer
            self.experience.store(
                obs=o,
                actions=actions,
                logprobs=selected_action_log_probs,
                rewards=r,
                dones=d,
                truncations=t,
                values=value,
                env_id=training_env_id,
                mask=mask_tensor,
                lstm_state=lstm_state_to_store,
            )

            # Collect info for stats
            if info:
                raw_infos.extend(info)

            # Send actions to environment
            self.vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Process collected infos into stats
        for i in raw_infos:
            for k, v in unroll_nested_dict(i):
                infos[k].append(v)

        # Convert to final stats format
        stats = {}
        for k, v in infos.items():
            if isinstance(v, list):
                stats[k] = v
            else:
                stats[k] = [v]

        return stats, steps_collected


class AsyncRolloutCollector(RolloutCollector):
    """Asynchronous rollout collector for distributed training.

    This extends RolloutCollector with support for asynchronous
    environment stepping, useful for distributed training setups.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional setup for async collection
        self._pending_actions = None

    def collect_async(self, num_steps: Optional[int] = None) -> Tuple[Dict[str, Any], int]:
        """Collect rollouts asynchronously.

        This method allows overlapping computation and environment stepping
        for improved throughput.
        """
        # Implementation would handle async environment stepping
        # For now, fallback to sync collection
        return self.collect(num_steps)
