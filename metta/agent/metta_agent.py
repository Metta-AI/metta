import logging
from typing import Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions
from mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger("metta_agent")


class MettaAgentBuilder:
    def __init__(self, cfg):
        self._cfg = cfg

    def build_from_brain_policy(self, env) -> "MettaAgent":
        obs_space = gym.spaces.Dict(
            {
                "grid_obs": env.single_observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

        # Here's where we create a BrainPolicy.
        brain = hydra.utils.instantiate(
            self._cfg.agent,
            obs_space=obs_space,
            obs_width=env.obs_width,
            obs_height=env.obs_height,
            action_space=env.single_action_space,
            feature_normalizations=env.feature_normalizations,
            device=self._cfg.device,
            _target_="metta.agent.brain_policy.BrainPolicy",
            _recursive_=False,
        )
        return MettaAgent(brain)

    def build_from_pytorch_policy(self, path) -> "MettaAgent":
        from metta.rl.policy import load_policy

        policy = load_policy(path, self._cfg.device, puffer=self._cfg.puffer)
        return MettaAgent(policy)

    def load_from_disk(self, path) -> "MettaAgent":
        """Load a model from disk, supporting both torch.jit and state_dict formats."""
        checkpoint = torch.load(path, map_location=self._cfg.device)

        if isinstance(checkpoint, dict) and checkpoint.get("use_jit", False) and "jit_model" in checkpoint:
            # Load torch.jit model
            jit_model = checkpoint["jit_model"]
            return MettaAgent(jit_model)
        else:
            # Fall back to state_dict loading through PolicyStore
            raise NotImplementedError(
                "Loading non-jit models from disk should go through PolicyStore. "
                "Use PolicyStore._load_from_file instead."
            )


def make_policy(env: MettaGridEnv, cfg: ListConfig | DictConfig) -> "MettaAgent":
    builder = MettaAgentBuilder(cfg)
    return builder.build_from_brain_policy(env)


class DistributedMettaAgent(DistributedDataParallel):
    def __init__(self, agent, device):
        logger.info("Converting BatchNorm layers to SyncBatchNorm for distributed training...")
        agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
        super().__init__(agent, device_ids=[device], output_device=device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device: torch.device) -> None:
        return self.module.activate_actions(action_names, action_max_params, device)


class MettaAgent(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def __getattr__(self, name):
        """Pass through attributes to the underlying policy."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)

    def forward(
        self, x: torch.Tensor, state: PolicyState, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.forward(x, state, action)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        if hasattr(self.policy, "activate_actions"):
            self.policy.activate_actions(action_names, action_max_params, device)
        else:
            logger.warning(
                f"Policy of type {type(self.policy).__name__} does not have 'activate_actions' method. Skipping."
            )

    @property
    def lstm(self):
        return self.policy._net

    @property
    def total_params(self):
        return sum(p.numel() for p in self.policy.parameters())

    def forward_inference(
        self, value: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference mode - samples new actions based on the policy.

        Args:
            value: Value estimate tensor, shape (BT, 1)
            logits: Action logits tensor, shape (BT, A)

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
            - action: Sampled action, shape (BT, 2)
            - action_log_prob: Log probability of the sampled action, shape (BT,)
            - entropy: Entropy of the action distribution, shape (BT,)
            - value: Value estimate, shape (BT, 1)
            - log_probs: Log-softmax of logits, shape (BT, A)
        """
        if __debug__:
            assert_shape(value, ("BT", 1), "inference_value")
            assert_shape(logits, ("BT", "A"), "inference_logits")

        # Sample actions
        action_logit_index, action_log_prob, entropy, log_probs = sample_actions(logits)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "action_logit_index")
            assert_shape(action_log_prob, ("BT",), "action_log_prob")
            assert_shape(entropy, ("BT",), "entropy")
            assert_shape(log_probs, ("BT", "A"), "log_probs")

        # Convert logit index to action
        action = self._convert_logit_index_to_action(action_logit_index)

        if __debug__:
            assert_shape(action, ("BT", 2), "inference_output_action")

        return action, action_log_prob, entropy, value, log_probs

    def forward_training(
        self, value: torch.Tensor, logits: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training mode - evaluates the policy on provided actions.

        Args:
            value: Value estimate tensor, shape (BT, 1)
            logits: Action logits tensor, shape (BT, A)
            action: Action tensor for evaluation, shape (B, T, 2)

        Returns:
            Tuple of (action, action_log_prob, entropy, value, log_probs)
            - action: Same as input action, shape (B, T, 2)
            - action_log_prob: Log probability of the provided action, shape (BT,)
            - entropy: Entropy of the action distribution, shape (BT,)
            - value: Value estimate, shape (BT, 1)
            - log_probs: Log-softmax of logits, shape (BT, A)
        """
        if __debug__:
            assert_shape(value, ("BT", 1), "training_value")
            assert_shape(logits, ("BT", "A"), "training_logits")
            assert_shape(action, ("B", "T", 2), "training_input_action")

        B, T, A = action.shape
        flattened_action = action.view(B * T, A)
        action_logit_index = self._convert_action_to_logit_index(flattened_action)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "converted_action_logit_index")

        action_log_prob, entropy, log_probs = evaluate_actions(logits, action_logit_index)

        if __debug__:
            assert_shape(action_log_prob, ("BT",), "training_action_log_prob")
            assert_shape(entropy, ("BT",), "training_entropy")
            assert_shape(log_probs, ("BT", "A"), "training_log_probs")
            assert_shape(action, ("B", "T", 2), "training_output_action")

        return action, action_log_prob, entropy, value, log_probs

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """
        Convert (action_type, action_param) pairs to discrete action indices
        using precomputed offsets.

        Args:
            flattened_action: Tensor of shape [B*T, 2] containing (action_type, action_param) pairs

        Returns:
            action_logit_indices: Tensor of shape [B*T] containing flattened action indices
        """
        if __debug__:
            assert_shape(flattened_action, ("BT", 2), "flattened_action")

        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()

        # Use precomputed cumulative sum with vectorized indexing
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        action_logit_indices = action_type_numbers + cumulative_sum + action_params

        if __debug__:
            assert_shape(action_logit_indices, ("BT",), "action_logit_indices")

        return action_logit_indices

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """
        Convert logit indices back to action pairs using tensor indexing.

        Args:
            action_logit_index: Tensor of shape [B*T] containing flattened action indices

        Returns:
            action: Tensor of shape [B*T, 2] containing (action_type, action_param) pairs
        """
        if __debug__:
            assert_shape(action_logit_index, ("BT",), "action_logit_index")

        action = self.action_index_tensor[action_logit_index]

        if __debug__:
            assert_shape(action, ("BT", 2), "actions")

        return action

    def _apply_to_components(self, method_name, *args, **kwargs) -> list[torch.Tensor]:
        """
        Apply a method to all components, collecting and returning the results.

        Args:
            method_name: Name of the method to call on each component
            *args, **kwargs: Arguments to pass to the method

        Returns:
            list: Results from calling the method on each component

        Raises:
            AttributeError: If any component doesn't have the requested method
            TypeError: If a component's method is not callable
            AssertionError: If no components are available
        """
        assert len(self.policy.components) != 0, "No components available to apply method"

        results = []
        for name, component in self.policy.components.items():
            if not hasattr(component, method_name):
                raise AttributeError(f"Component '{name}' does not have method '{method_name}'")

            method = getattr(component, method_name)
            if not callable(method):
                raise TypeError(f"Component '{name}' has {method_name} attribute but it's not callable")

            result = method(*args, **kwargs)
            if result is not None:
                results.append(result)

        return results

    def l2_reg_loss(self) -> torch.Tensor:
        """L2 regularization loss is on by default although setting l2_norm_coeff to 0 effectively turns it off. Adjust
        it by setting l2_norm_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        component_loss_tensors = self._apply_to_components("l2_reg_loss")
        if len(component_loss_tensors) > 0:
            return torch.sum(torch.stack(component_loss_tensors))
        else:
            return torch.tensor(0.0, device=self.device)

    def l2_init_loss(self) -> torch.Tensor:
        """L2 initialization loss is on by default although setting l2_init_coeff to 0 effectively turns it off. Adjust
        it by setting l2_init_scale in your component config to a multiple of the global loss value or 0 to turn it off.
        """
        component_loss_tensors = self._apply_to_components("l2_init_loss")
        if len(component_loss_tensors) > 0:
            return torch.sum(torch.stack(component_loss_tensors))
        else:
            return torch.tensor(0.0, device=self.device)

    def update_l2_init_weight_copy(self):
        """Update interval set by l2_init_weight_update_interval. 0 means no updating."""
        self._apply_to_components("update_l2_init_weight_copy")

    def clip_weights(self):
        """Weight clipping is on by default although setting clip_range or clip_scale to 0, or a large positive value
        effectively turns it off. Adjust it by setting clip_scale in your component config to a multiple of the global
        loss value or 0 to turn it off."""
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for all components that have weights enabled for analysis.
        Returns a list of metric dictionaries, one per component. Set analyze_weights to True in the config to turn it
        on for a given component."""
        results = {}
        for name, component in self.policy.components.items():
            method_name = "compute_weight_metrics"
            if not hasattr(component, method_name):
                continue  # Skip components that don't have this method instead of raising an error

            method = getattr(component, method_name)
            assert callable(method), f"Component '{name}' has {method_name} attribute but it's not callable"

            results[name] = method(delta)

        metrics_list = [metrics for metrics in results.values() if metrics is not None]
        return metrics_list
