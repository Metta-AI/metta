import logging

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

logger = logging.getLogger(__name__)


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy=None, cnn_channels=128, input_size=128, hidden_size=128):
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        super().__init__(env, policy, input_size, hidden_size)

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        """
        Initialize the policy to the current environment's features and actions.

        Args:
            features: Dictionary mapping feature names to their properties:
                {
                    feature_name: {
                        "id": byte,  # The feature_id to use during this run
                        "type": "scalar" | "categorical",
                        "normalization": float (optional, only for scalar features)
                    }
                }
            action_names: List of action names
            action_max_params: List of maximum parameters for each action
            device: Device to place tensors on
            is_training: Deprecated. Training mode is now automatically detected.
        """
        self._initialize_observations(features, device, is_training)
        self.activate_actions(action_names, action_max_params, device)

    def _initialize_observations(self, features: dict[str, dict], device, is_training: bool):
        """Initialize observation features by storing the feature mapping."""
        self.active_features = features
        self.device = device

        # Create quick lookup mappings
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props
        }

        # Store original feature mapping on first initialization
        if not hasattr(self, "original_feature_mapping"):
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
            logger.info(f"Stored original feature mapping with {len(self.original_feature_mapping)} features")
        else:
            # Create remapping for subsequent initializations
            self._create_feature_remapping(features, is_training)

    def _create_feature_remapping(self, features: dict[str, dict], is_training: bool):
        """Create a remapping dictionary to translate new feature IDs to original ones."""
        UNKNOWN_FEATURE_ID = 255
        self.feature_id_remap = {}
        unknown_features = []

        for name, props in features.items():
            new_id = props["id"]
            if name in self.original_feature_mapping:
                # Remap known features to their original IDs
                original_id = self.original_feature_mapping[name]
                if new_id != original_id:
                    self.feature_id_remap[new_id] = original_id
            elif not is_training:
                # In eval mode, map unknown features to UNKNOWN_FEATURE_ID
                self.feature_id_remap[new_id] = UNKNOWN_FEATURE_ID
                unknown_features.append(name)
            else:
                # In training mode, learn new features
                self.original_feature_mapping[name] = new_id

        if self.feature_id_remap:
            logger.info(
                f"Created feature remapping: {len(self.feature_id_remap)} remapped, {len(unknown_features)} unknown"
            )
            self._apply_feature_remapping(features, UNKNOWN_FEATURE_ID)

    def _apply_feature_remapping(self, features: dict[str, dict], unknown_id: int):
        """Apply feature remapping to observation processing and update normalizations."""
        # Create remapping tensor for observation processing
        remap_tensor = torch.arange(256, dtype=torch.uint8, device=self.device)
        for new_id, original_id in self.feature_id_remap.items():
            remap_tensor[new_id] = original_id

        # Map unused feature IDs to UNKNOWN
        current_feature_ids = {props["id"] for props in features.values()}
        for feature_id in range(256):
            if feature_id not in self.feature_id_remap and feature_id not in current_feature_ids:
                remap_tensor[feature_id] = unknown_id

        # Store remap tensor as a buffer
        self.register_buffer("feature_remap_tensor", remap_tensor)

        # Update normalization factors
        self._update_normalization_factors(features)

    def _update_normalization_factors(self, features: dict[str, dict]):
        """Update normalization factors for observation processing."""
        norm_tensor = torch.ones(256, dtype=torch.float32, device=self.device)
        for name, props in features.items():
            if name in self.original_feature_mapping and "normalization" in props:
                original_id = self.original_feature_mapping[name]
                norm_tensor[original_id] = props["normalization"]

        # Store normalization tensor as a buffer
        self.register_buffer("norm_factors", norm_tensor)

    def activate_actions(self, action_names, action_max_params, device):
        """Initialize the action space."""
        assert isinstance(action_max_params, list), "action_max_params must be a list"
        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for action index conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=self.device, dtype=torch.long), dim=0
        )

        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])
        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)

        logger.info(f"Policy actions initialized with: {self.active_actions}")

    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"].to(self.device)

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Encode obs
        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # Prepare LSTM state
        lstm_h, lstm_c = state.get("lstm_h"), state.get("lstm_c")
        if lstm_h is not None and lstm_c is not None:
            lstm_h = lstm_h.to(self.device)[: self.lstm.num_layers]
            lstm_c = lstm_c.to(self.device)[: self.lstm.num_layers]
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        # Forward LSTM
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, in_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode
        logits_list, value = self.policy.decode_actions(flat_hidden)

        if action is None:
            # ---------- Inference Mode ----------
            log_probs = F.log_softmax(logits_list, dim=-1)  # [batch_size, num_actions]
            action_probs = torch.exp(log_probs)  # [batch_size, num_actions]

            actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]

            batch_indices = torch.arange(actions.shape[0], device=actions.device)
            full_log_probs = log_probs[batch_indices, actions]  # [batch_size]

            action = self._convert_logit_index_to_action(actions)

            td["actions"] = action.to(dtype=torch.int32)
            td["act_log_prob"] = full_log_probs
            td["values"] = value.flatten()
            td["full_log_probs"] = log_probs

        else:
            # ---------- Training Mode ----------
            action = action.to(self.device)
            if action.dim() == 3:  # (B, T, 2) → flatten to (BT, 2)
                B, T, A = action.shape
                action = action.view(B * T, A)

            action_log_probs = F.log_softmax(logits_list, dim=-1)

            action_probs = torch.exp(action_log_probs)
            action_logit_index = self._convert_action_to_logit_index(action)  # shape [BT]

            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            full_log_probs = action_log_probs[batch_indices, action_logit_index]

            entropy = -(action_probs * action_log_probs).sum(dim=-1)  # [batch_size]

            td["act_log_prob"] = full_log_probs.view(B, TT)
            td["entropy"] = entropy.view(B, TT)
            td["full_log_probs"] = action_log_probs.view(B, T, -1)
            td["value"] = value.view(B, TT)

        return td

    def clip_weights(self):
        """Clip weights of the actor heads to prevent large updates."""
        pass

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return cumulative_sum + action_params


class Policy(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        self.cnn1 = pufferlib.pytorch.layer_init(nn.Conv2d(in_channels=22, out_channels=64, kernel_size=5, stride=3))
        self.cnn2 = pufferlib.pytorch.layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1))

        test_input = torch.zeros(1, 22, 11, 11)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        self.fc1 = pufferlib.pytorch.layer_init(nn.Linear(self.flattened_size, 128))
        self.encoded_obs = pufferlib.pytorch.layer_init(nn.Linear(128, 128))
        self.critic_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1024))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(1024, 1), std=1.0)
        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 512))
        self.action_embeddings = nn.Embedding(100, 16)

        # Action heads - will be initialized based on action space
        action_nvec = self.action_space.nvec if hasattr(self.action_space, "nvec") else [100]

        self.actor_heads = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(512 + 16, n), std=0.01) for n in action_nvec]
        )

        max_vec = torch.tensor(
            [
                9.0,
                1.0,
                1.0,
                10.0,
                3.0,
                254.0,
                1.0,
                1.0,
                235.0,
                8.0,
                9.0,
                250.0,
                29.0,
                1.0,
                1.0,
                8.0,
                1.0,
                1.0,
                6.0,
                3.0,
                1.0,
                2.0,
            ],
            dtype=torch.float32,
        )[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        self.to(self.device)

    def network_forward(self, x):
        x = x / self.max_vec
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.encoded_obs(x)
        return x

    def encode_observations(self, observations, state=None):
        """
        Encode observations into a hidden representation.
        """

        observations = observations.to(self.device)
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1 if token_observations.dim() == 3 else token_observations.shape[1]
        if token_observations.dim() != 3:
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"
        token_observations[token_observations == 255] = 0

        coords_byte = token_observations[..., 0].to(torch.uint8)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()
        y_coord_indices = (coords_byte & 0x0F).long()
        atr_indices = token_observations[..., 1].long()
        atr_values = token_observations[..., 2].float()

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )
        batch_indices = torch.arange(B * TT, device=token_observations.device).unsqueeze(-1).expand_as(atr_values)

        valid_tokens = coords_byte != 0xFF
        valid_tokens = valid_tokens & (x_coord_indices < self.out_width) & (y_coord_indices < self.out_height)
        valid_tokens = valid_tokens & (atr_indices < self.num_layers)

        box_obs[
            batch_indices[valid_tokens],
            atr_indices[valid_tokens],
            x_coord_indices[valid_tokens],
            y_coord_indices[valid_tokens],
        ] = atr_values[valid_tokens]

        return self.network_forward(box_obs)

    def decode_actions(self, hidden):
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        actor_features = self.actor_1(hidden)

        action_embed = self.action_embeddings.weight.mean(dim=0).unsqueeze(0).expand(actor_features.shape[0], -1)
        combined_features = torch.cat([actor_features, action_embed], dim=-1)

        logits = torch.cat([head(combined_features) for head in self.actor_heads], dim=-1)  # (B, sum(A_i))

        return logits, value
