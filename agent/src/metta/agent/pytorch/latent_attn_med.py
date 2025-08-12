import logging

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from torch import nn

from metta.agent.lib.obs_enc import ObsLatentAttn, ObsSelfAttn
from metta.agent.lib.obs_tokenizers import ObsAttrEmbedFourier, ObsAttrValNorm, ObsTokenPadStrip

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

    def get_agent_experience_spec(self):
        """Provide experience spec for compatibility with trainer."""
        import torch
        from torchrl.data import Composite, UnboundedDiscrete

        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
        )

    def reset_memory(self):
        """Reset LSTM memory if needed."""
        # LSTM state is managed through TensorDict, no explicit reset needed
        pass

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

    def forward(self, td, action=None):
        """
        Forward pass using TensorDict interface compatible with MettaAgent.

        Args:
            td: TensorDict containing at least "env_obs"
            action: Optional action tensor for training mode

        Returns:
            TensorDict with outputs
        """
        from metta.agent.util.distribution_utils import evaluate_actions, sample_actions

        # Handle batch dimensions like MettaAgent
        if td.batch_dims > 1:
            B, TT = td.batch_size[0], td.batch_size[1]
            td = td.reshape(B * TT)
            td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))
        else:
            B = td.batch_size.numel()
            TT = 1
            td.set("bptt", torch.full((B,), 1, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=td.device, dtype=torch.long))

        # Get observations
        observations = td.get("env_obs")
        observations = observations.to(self.device)

        # Prepare state for backward compatibility
        state = {"lstm_h": td.get("lstm_h", None), "lstm_c": td.get("lstm_c", None), "hidden": td.get("hidden", None)}

        # Encode observations
        hidden = self.policy.encode_observations(observations, state)

        # Prepare LSTM state
        lstm_h = state.get("lstm_h", None)
        lstm_c = state.get("lstm_c", None)

        if lstm_h is not None and lstm_c is not None:
            lstm_h = lstm_h.to(self.device)
            lstm_c = lstm_c.to(self.device)

            expected_num_layers = self.lstm.num_layers
            lstm_h = lstm_h[:expected_num_layers, :, :]
            lstm_c = lstm_c[:expected_num_layers, :, :]
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        # LSTM forward pass
        batch_size = B * TT
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # Shape: (TT, B, input_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(batch_size, -1)

        # Store LSTM states
        td["lstm_h"] = new_lstm_h
        td["lstm_c"] = new_lstm_c

        # Decode actions and value
        logits_list, value = self.policy.decode_actions(flat_hidden)

        # For simplicity, use only the first action head for now
        # (full multi-discrete support would require action space conversion)
        primary_logits = logits_list[0]

        if action is None:
            # Inference mode - sample actions
            action_indices, act_log_prob, entropy, full_log_probs = sample_actions(primary_logits)

            # Convert to multi-dimensional action format
            if len(logits_list) >= 2:
                # Sample from second action head too
                secondary_logits = logits_list[1]
                secondary_probs = F.softmax(secondary_logits, dim=-1)
                secondary_actions = torch.multinomial(secondary_probs, num_samples=1).squeeze(-1)
                actions_tensor = torch.stack([action_indices, secondary_actions], dim=-1)
            else:
                actions_tensor = torch.stack([action_indices, torch.zeros_like(action_indices)], dim=-1)

            td["actions"] = actions_tensor
            td["act_log_prob"] = act_log_prob
            td["values"] = value.squeeze(-1) if value.dim() > 1 else value
            td["full_log_probs"] = full_log_probs

            output_td = td
        else:
            # Training mode - evaluate given actions
            # Extract action indices from the multi-dimensional action tensor
            if action.dim() == 3:  # (B, TT, 2)
                action_indices = action.reshape(batch_size, -1)[:, 0]
            else:
                action_indices = action[:, 0]

            act_log_prob, entropy, full_log_probs = evaluate_actions(primary_logits, action_indices)

            td["act_log_prob"] = act_log_prob
            td["entropy"] = entropy
            td["value"] = value
            td["full_log_probs"] = full_log_probs

            # Reshape back to (B, TT)
            output_td = td.reshape(B, TT)

        return output_td


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
        td = {"env_obs": observations, "state": None}

        # Safely handle LSTM state
        if state is not None and state.get("lstm_h") is not None and state.get("lstm_c") is not None:
            lstm_h = state.get("lstm_h").to(observations.device)
            lstm_c = state.get("lstm_c").to(observations.device)
            td["state"] = torch.cat([lstm_h, lstm_c], dim=0)

        if observations.dim() == 4:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")
            td["env_obs"] = observations
        elif observations.dim() != 3:
            raise ValueError(f"Expected observations with 3 or 4 dimensions, got shape: {observations.shape}")

        return self.network_forward(td)

    def decode_actions(self, hidden):
        critic_features = F.tanh(self.critic_1(hidden))

        value = self.value_head(critic_features)

        actor_features = self.actor_1(hidden)

        action_embed = self.action_embeddings.weight.mean(dim=0).unsqueeze(0).expand(actor_features.shape[0], -1)
        combined_features = torch.cat([actor_features, action_embed], dim=-1)
        logits = [head(combined_features) for head in self.actor_heads]

        return logits, value
