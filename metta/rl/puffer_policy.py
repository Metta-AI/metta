"""
PufferLib checkpoint loading and conversion utilities.

This module provides integration between PufferLib checkpoints (state_dict format)
and Metta agents. It detects checkpoint formats, preprocesses state dictionaries,
and loads them into compatible Metta agents.
"""

import logging
from typing import Any, Dict, Optional, TypeGuard

import einops
import torch
from tensordict import TensorDict
from torch import nn

from metta.agent.components.actor import ActionProbsConfig
from metta.agent.components.lstm import LSTM, LSTMConfig
from metta.agent.policy import Policy, PolicyArchitecture

logger = logging.getLogger(__name__)


def _is_puffer_state_dict(loaded_obj: Any) -> TypeGuard[Dict[str, torch.Tensor]]:
    """Return True if the object appears to be a PufferLib state_dict."""
    return isinstance(loaded_obj, dict) and bool(loaded_obj) and any(key.startswith("policy.") for key in loaded_obj)


def _preprocess_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map PufferLib-specific keys to Metta-compatible keys."""
    processed = {}

    key_mappings = {
        # Convolution layers
        "policy.conv1.weight": "conv1.weight",
        "policy.conv1.bias": "conv1.bias",
        "policy.conv2.weight": "conv2.weight",
        "policy.conv2.bias": "conv2.bias",
        # Fully connected layers
        "policy.network.0.weight": "network.0.weight",
        "policy.network.0.bias": "network.0.bias",
        "policy.network.2.weight": "network.2.weight",
        "policy.network.2.bias": "network.2.bias",
        "policy.network.5.weight": "network.5.weight",
        "policy.network.5.bias": "network.5.bias",
        # LSTM mappings (different structure in PufferLib)
        "lstm.weight_ih_l0": "lstm.net.weight_ih_l0",
        "lstm.weight_hh_l0": "lstm.net.weight_hh_l0",
        "lstm.bias_ih_l0": "lstm.net.bias_ih_l0",
        "lstm.bias_hh_l0": "lstm.net.bias_hh_l0",
        # Alternate cell mappings (duplicates in checkpoint)
        "cell.weight_ih": "lstm.net.weight_ih_l0",
        "cell.weight_hh": "lstm.net.weight_hh_l0",
        "cell.bias_ih": "lstm.net.bias_ih_l0",
        "cell.bias_hh": "lstm.net.bias_hh_l0",
        # Value head
        "policy.value.weight": "value.weight",
        "policy.value.bias": "value.bias",
        # Actor head
        "policy.actor.0.weight": "actor.0.weight",
        "policy.actor.0.bias": "actor.0.bias",
        "policy.actor.1.weight": "actor.1.weight",
        "policy.actor.1.bias": "actor.1.bias",
    }

    for src_key, dst_key in key_mappings.items():
        if src_key in state_dict:
            processed[dst_key] = state_dict[src_key]
        else:
            logger.debug(f"Missing expected key in checkpoint: {src_key}")

    logger.info(f"Preprocessed checkpoint: {len(state_dict)} -> {len(processed)} parameters")
    return processed


def _create_metta_agent(device: str | torch.device = "cpu") -> Any:
    """Instantiate a PufferLib-compatible Metta policy for checkpoint loading."""

    from mettagrid import MettaGridEnv
    from mettagrid.builder.envs import make_arena

    # Minimal environment for initialization
    env_cfg = make_arena(num_agents=60)
    temp_env = MettaGridEnv(env_cfg, render_mode="rgb_array")

    policy_cfg = PufferLibCompatibleConfig()
    policy = PufferLibCompatiblePolicy(temp_env, policy_cfg).to(device)

    temp_env.close()
    return policy


def _load_state_dict_into_agent(policy: Any, state_dict: Dict[str, torch.Tensor]) -> Any:
    """Load a state_dict into a policy, handling key and shape mismatches."""
    policy_state = policy.state_dict()
    compatible_state = {}
    shape_mismatches = []

    keys_matched = 0
    for key, value in state_dict.items():
        if key in policy_state:
            target_param = policy_state[key]
            if target_param.shape == value.shape:
                compatible_state[key] = value
                keys_matched += 1
            else:
                shape_mismatches.append(f"{key}: checkpoint {value.shape} vs policy {target_param.shape}")
                logger.debug(f"Shape mismatch for {key}: checkpoint {value.shape} vs policy {target_param.shape}")
        else:
            logger.debug(f"Skipping unmatched parameter: {key}")

    if shape_mismatches:
        logger.warning(f"Shape mismatches found for {len(shape_mismatches)} parameters")

    logger.info(f"Loaded {keys_matched}/{len(state_dict)} compatible parameters")

    if not compatible_state:
        raise RuntimeError("No compatible parameters found in checkpoint")

    try:
        policy.load_state_dict(compatible_state, strict=False)
        logger.info("Successfully loaded checkpoint into Metta policy")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    return policy


class PufferLibCheckpoint:
    """Loader for checkpoints in both Metta and PufferLib formats."""

    def load_checkpoint(self, checkpoint_data: Any, device: str | torch.device = "cpu") -> Any:
        logger.info("Loading checkpoint in PufferLib state_dict format")
        if not isinstance(checkpoint_data, dict):
            raise TypeError("Expected checkpoint_data to be a dict (state_dict format)")

        logger.debug(f"Checkpoint sample keys: {list(checkpoint_data.keys())[:10]}")
        policy = _create_metta_agent(device)
        processed_state = _preprocess_state_dict(checkpoint_data)
        return _load_state_dict_into_agent(policy, processed_state)


class PufferLibCompatibleConfig(PolicyArchitecture):
    """
    Policy configuration that exactly matches PufferLib architecture for checkpoint loading.

    Based on analysis of PufferLib checkpoint:
    - CNN: 128 channels, 24 input channels, 5x5 and 3x3 kernels
    - LSTM: 512 hidden size
    - Actor: 5 actions + 9 action args
    - Critic: Single value output
    """

    class_path: str = "metta.agent.policies.pufferlib_compatible.PufferLibCompatiblePolicy"

    lstm_config: LSTMConfig = LSTMConfig(
        in_key="encoded_obs",
        out_key="core",
        latent_size=512,  # Match PufferLib: 256 (self) + 256 (cnn) = 512 input to LSTM
        hidden_size=512,  # Match PufferLib LSTM: 512 not 128
        num_layers=1,
    )

    # Minimal action_probs_config to satisfy base class requirement
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class PufferLibCompatiblePolicy(Policy):
    """Policy that exactly matches PufferLib architecture for seamless checkpoint loading."""

    def __init__(self, env, config: Optional[PufferLibCompatibleConfig] = None):
        super().__init__()
        self.config = config or PufferLibCompatibleConfig()
        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        self.active_action_names = []
        self.num_active_actions = 100  # Default
        self.action_index_tensor = None
        self.cum_action_max_params = None

        self.out_width = env.obs_width
        self.out_height = env.obs_height

        self.num_layers = max(env.feature_normalizations.keys())

        self.conv1 = nn.Conv2d(24, 128, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1)

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
        )

        self.max_vec = [1.0] * 24
        for feature_id, norm_value in self.env.feature_normalizations.items():
            print(f"feature_id: {feature_id}, norm_value: {norm_value}")
            if feature_id < 24:
                self.max_vec[feature_id] = norm_value if norm_value > 0 else 1.0
        self.max_vec = torch.tensor(self.max_vec, dtype=torch.float32)
        self.max_vec = torch.maximum(self.max_vec, torch.ones_like(self.max_vec))
        self.max_vec = self.max_vec[None, :, None, None]

        action_nvec = self.env.single_action_space.nvec
        self.actor = nn.ModuleList([nn.Linear(512, n) for n in action_nvec])
        self.value = nn.Linear(512, 1)

        lstm_config = LSTMConfig(
            in_key="encoded_obs",
            out_key="core",
            latent_size=512,
            hidden_size=512,
            num_layers=1,
        )
        self.lstm = LSTM(lstm_config)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Converts raw observation tokens into a concatenated self + CNN feature vector."""
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")

        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range, but we need to make them long for indexing)
        x_coords = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coords = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        atr_values = observations[..., 2].float()  # Shape: [B_TT, M]

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=observations.device,
        )

        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )

        batch_idx = torch.arange(B * TT, device=observations.device).unsqueeze(-1).expand_as(atr_values)
        box_obs[
            batch_idx[valid_tokens],
            atr_indices[valid_tokens],
            x_coords[valid_tokens],
            y_coords[valid_tokens],
        ] = atr_values[valid_tokens]

        # Normalize features with epsilon for numerical stability
        max_vec_device = self.max_vec.to(box_obs.device)
        features = box_obs / (max_vec_device + 1e-8)

        # Self encoder processes aggregated per-channel features (sum across spatial dimensions)
        # Shape: [B, num_layers=24] -> [B, 256]
        self_input = features.sum(dim=(-2, -1))  # Sum across height and width dimensions
        self_features = self.self_encoder(self_input)

        # CNN processes spatial features normally
        # Shape: [B, 24, H, W] -> [B, 256]
        cnn_features = self.network(features)

        # Concatenate self and CNN features: [B, 256] + [B, 256] = [B, 512]
        result = torch.cat([self_features, cnn_features], dim=1)
        return result

    def decode_actions(self, hidden):
        # hidden = self.layer_norm(hidden)
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value

    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        observations = td["env_obs"]
        hidden = self.encode_observations(observations)

        td["encoded_obs"] = hidden

        # Pass through LSTM: [B, 512] -> [B, 512]
        self.lstm(td)
        hidden = td["core"]
        # Decode actions and value
        logits, value = self.decode_actions(hidden)

        # Convert logits to actions by sampling from categorical distributions
        actions = []
        entropies = []
        action_log_probs = []

        for logit_tensor in logits:
            action_probs = torch.softmax(logit_tensor, dim=-1)
            sampled_action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
            actions.append(sampled_action)

            # Correct entropy calculation
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
            entropies.append(entropy)

            # Get log probabilities of the selected actions
            log_probs = torch.log(action_probs + 1e-8)
            selected_log_probs = log_probs.gather(-1, sampled_action.unsqueeze(-1)).squeeze(-1)
            action_log_probs.append(selected_log_probs)

        actions_tensor = torch.stack(actions, dim=-1)
        entropies_tensor = torch.stack(entropies, dim=-1)
        log_probs_tensor = torch.stack(action_log_probs, dim=-1)

        if action is None:
            td["actions"] = actions_tensor
            td["act_log_prob"] = log_probs_tensor
            td["values"] = value.flatten()
            td["entropy"] = entropies_tensor
        else:
            td["act_log_prob"] = log_probs_tensor
            td["entropy"] = entropies_tensor
            td["full_log_probs"] = log_probs_tensor  # TODO: This is not correct (fix this later)

        return td

    @property
    def action_names(self) -> list[str]:
        """Return list of action names."""
        return getattr(self.env, "action_names", [])

    @property
    def observation_space(self):
        """Return observation space."""
        return self.env.observation_space

    def get_action_and_value(self, obs, state=None, action=None, **kwargs):
        """Get action and value prediction."""
        td = TensorDict({"env_obs": obs}, batch_size=obs.shape[:-3])
        td = self.forward(td, state=state, action=action)

        action_probs = td["action_probs"]
        values = td["values"]

        return action_probs, values, td.get("lstm_state")

    def get_value(self, obs, state=None, **kwargs):
        """Get value prediction only."""
        td = TensorDict({"env_obs": obs}, batch_size=obs.shape[:-3])
        td = self.forward(td, state=state)
        return td["values"]

    @property
    def device(self) -> torch.device:
        """Get the device this policy is on."""
        return next(self.parameters()).device

    def reset_memory(self):
        """Reset policy memory/state if any."""
        pass
