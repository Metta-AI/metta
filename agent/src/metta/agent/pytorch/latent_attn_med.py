import logging

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from torch import nn

from metta.agent.external.models.encoders import ObsLatentAttn, ObsSelfAttn
from metta.agent.external.models.tokenizers import ObsAttrEmbedFourier, ObsAttrValNorm, ObsTokenPadStrip
from tensordict import TensorDict

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

        # Prepare LSTM state
        lstm_h, lstm_c = state.get("lstm_h"), state.get("lstm_c")
        if lstm_h is not None and lstm_c is not None:
            lstm_h = lstm_h.to(self.device)[:self.lstm.num_layers]
            lstm_c = lstm_c.to(self.device)[:self.lstm.num_layers]
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        # Encode observations
        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # LSTM forward pass
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, input_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode actions and value
        logits_list, value = self.policy.decode_actions(flat_hidden)

        if action is None:
            # ---------- Inference Mode ----------
            action_log_probs = F.log_softmax(logits_list, dim=-1)
            action_probs = torch.exp(action_log_probs)

            sampled_actions = torch.multinomial(action_probs, num_samples=1).view(-1)
            batch_indices = torch.arange(sampled_actions.shape[0], device=sampled_actions.device)
            full_log_probs = action_log_probs[batch_indices, sampled_actions]

            converted_action = self._convert_logit_index_to_action(sampled_actions)

            td["actions"] = converted_action.to(dtype=torch.int32)
            td["act_log_prob"] = full_log_probs
            td["values"] = value.flatten()
            td["full_log_probs"] = action_log_probs

        else:
            # ---------- Training Mode ----------
            action = action.to(self.device)
            if action.dim() == 3:  # (B, T, A) -> (BT, A)
                B, T, A = action.shape
                action = action.view(B * T, A)

            action_log_probs = F.log_softmax(logits_list, dim=-1)
            action_probs = torch.exp(action_log_probs)

            action_logit_index = self._convert_action_to_logit_index(action)
            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            full_log_probs = action_log_probs[batch_indices, action_logit_index]

            entropy = -(action_probs * action_log_probs).sum(dim=-1)

            td["act_log_prob"] = full_log_probs.view(B, TT)
            td["entropy"] = entropy.view(B, TT)
            td["full_log_probs"] = action_log_probs.view(B, TT, -1)
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

        # Define components with explicit names and sources
        self.obs_ = ObsTokenPadStrip(
            obs_shape=(200, 3),
        )
        self.obs_norm = ObsAttrValNorm(
            feature_normalizations=[1.0] * 256,
        )
        self.obs_fourier = ObsAttrEmbedFourier(
            12,
            8,
        )

        self.obs_latent_query_attn = ObsLatentAttn(
            out_dim=32,
            _feat_dim=45,
            use_mask=True,
            num_query_tokens=10,
            query_token_dim=32,
            num_heads=8,
            num_layers=3,
        )
        self.obs_latent_self_attn = ObsSelfAttn(
            out_dim=128,
            _feat_dim=32,
            num_heads=8,
            num_layers=3,
            use_mask=False,
            use_cls_token=True,
        )

        self.critic_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1024))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(1024, 1), std=1.0)
        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 512))
        self.action_embeddings = nn.Embedding(100, 16)

        # Action heads - will be initialized based on action space
        action_nvec = self.action_space.nvec if hasattr(self.action_space, "nvec") else [100]

        self.actor_heads = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(512 + 16, n), std=0.01) for n in action_nvec]
        )

        self.to(self.device)

    def network_forward(self, x):
        x, mask, B_TT = self.obs_(x)
        x = self.obs_norm(x)
        x = self.obs_fourier(x)
        x = self.obs_latent_query_attn(x, mask, B_TT)  # output shape: [24, 10, 32]
        x = self.obs_latent_self_attn(x, mask)
        return x

    def encode_observations(self, observations, state=None):
        """
        Encode observations into a hidden representation.

        Args:
            observations: Input tensor, shape (B, TT, M, 3) or (B, M, 3)
            state: Optional state dictionary

        Returns:
            hidden: Encoded representation, shape (B * TT, hidden_size)
        """
        observations = observations.to(self.device)

        # Initialize dictionary for TensorDict
        td = {"x": observations, "state": None}

        # Safely handle LSTM state
        if state is not None and state.get("lstm_h") is not None and state.get("lstm_c") is not None:
            lstm_h = state.get("lstm_h").to(observations.device)
            lstm_c = state.get("lstm_c").to(observations.device)
            td["state"] = torch.cat([lstm_h, lstm_c], dim=0)

        if observations.dim() == 4:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")
            td["x"] = observations
        elif observations.dim() != 3:
            raise ValueError(f"Expected observations with 3 or 4 dimensions, got shape: {observations.shape}")

        return self.network_forward(td)

    def decode_actions(self, hidden):
        critic_features = F.tanh(self.critic_1(hidden))

        value = self.value_head(critic_features)

        actor_features = self.actor_1(hidden)

        action_embed = self.action_embeddings.weight.mean(dim=0).unsqueeze(0).expand(actor_features.shape[0], -1)
        combined_features = torch.cat([actor_features, action_embed], dim=-1)

        logits = torch.cat([head(combined_features) for head in self.actor_heads], dim=-1)  # (B, sum(A_i))

        return logits, value


