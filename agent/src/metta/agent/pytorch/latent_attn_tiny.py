import logging

import einops
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from metta.agent.modules.encoders import ObsLatentAttn
from metta.agent.modules.tokenizers import ObsAttrEmbedFourier, ObsAttrValNorm, ObsTokenPadStrip
from metta.agent.pytorch.base import LSTMWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class LatentAttnTiny(PyTorchAgentMixin, LSTMWrapper):
    def __init__(
        self,
        env,
        policy=None,
        cnn_channels=128,
        input_size=128,
        hidden_size=128,
        num_layers=2,
        **kwargs,
    ):
        """Initialize LatentAttnTiny policy with mixin support."""
        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)

        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        # Use enhanced LSTMWrapper with num_layers support
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Determine dimensions from observations
        if observations.dim() == 4:  # Training
            B = observations.shape[0]
            TT = observations.shape[1]
            # Reshape TD for training if needed
            if td.batch_dims > 1:
                td = td.reshape(B * TT)
        else:  # Inference
            B = observations.shape[0]
            TT = 1

        # Now set TensorDict fields with mixin (TD is already reshaped if needed)
        self.set_tensordict_fields(td, observations)

        # Encode observations
        hidden = self.policy.encode_observations(observations, state)

        # Use base class for proper LSTM state management (includes detachment!)
        lstm_h, lstm_c, env_id = self._manage_lstm_state(td, B, TT, observations.device)
        lstm_state = (lstm_h, lstm_c)

        # LSTM forward pass
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, input_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)

        # CRITICAL: Store with automatic detachment to prevent gradient accumulation
        self._store_lstm_state(new_lstm_h, new_lstm_c, env_id)

        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode actions and value
        logits_list, value = self.policy.decode_actions(flat_hidden)

        # Use mixin for mode-specific processing
        if action is None:
            # Mixin handles inference mode
            td = self.forward_inference(td, logits_list, value)
        else:
            # Mixin handles training mode with proper reshaping
            td = self.forward_training(td, action, logits_list, value)

        return td


class Policy(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False
        self.action_space = env.single_action_space

        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        # Define layer dimensions that are used multiple times
        self.actor_hidden_dim = 512  # Used in actor_1 and bilinear calculation
        self.action_embed_dim = 16  # Used in action_embeddings and bilinear calculation

        # Define components with explicit names and sources
        self.obs_ = ObsTokenPadStrip(
            obs_shape=(200, 3),
        )
        self.obs_norm = ObsAttrValNorm(
            feature_normalizations=[1.0] * 256,
        )
        self.obs_fourier = ObsAttrEmbedFourier(
            attr_embed_dim=10,
            num_freqs=4,
        )

        self.obs_latent_query_attn = ObsLatentAttn(
            out_dim=128,
            _feat_dim=27,
            use_mask=True,
            num_query_tokens=1,
            query_token_dim=32,
            num_heads=4,
            num_layers=2,
            qk_dim=32,
        )

        self.critic_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1024))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(1024, 1), std=1.0)
        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.actor_hidden_dim))
        self.action_embeddings = nn.Embedding(100, self.action_embed_dim)

        # Create action heads using mixin pattern
        self.actor_heads = PyTorchAgentMixin.create_action_heads(
            self, env, input_size=self.actor_hidden_dim + self.action_embed_dim
        )

    def network_forward(self, x):
        x, mask, B_TT = self.obs_(x)
        x = self.obs_norm(x)
        x = self.obs_fourier(x)
        x = self.obs_latent_query_attn(x, mask, B_TT)
        return x

    def encode_observations(self, observations, state=None):
        """Encode observations into a hidden representation."""

        # Initialize dictionary for TensorDict
        td = {"env_obs": observations, "state": None}

        # Safely handle LSTM state
        if state is not None and state.get("lstm_h") is not None and state.get("lstm_c") is not None:
            lstm_h = state.get("lstm_h")
            lstm_c = state.get("lstm_c")
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
        logits = torch.cat([head(combined_features) for head in self.actor_heads], dim=-1)  # (B, sum(A_i))

        return logits, value
