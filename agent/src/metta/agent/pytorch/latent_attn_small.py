import logging

import einops
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

import pufferlib.pytorch
from metta.agent.modules.encoders import ObsLatentAttn, ObsSelfAttn
from metta.agent.modules.tokenizers import ObsAttrEmbedFourier, ObsAttrValNorm, ObsTokenPadStrip
from metta.agent.pytorch.base import LSTMWrapper

logger = logging.getLogger(__name__)


class LatentAttnSmall(LSTMWrapper):
    def __init__(self, env, policy=None, cnn_channels=128, input_size=128, hidden_size=128, num_layers=2):
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        # Use enhanced LSTMWrapper with num_layers support
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Prepare LSTM state
        lstm_h, lstm_c = state.get("lstm_h"), state.get("lstm_c")
        if lstm_h is not None and lstm_c is not None:
            lstm_h = lstm_h[: self.lstm.num_layers]
            lstm_c = lstm_c[: self.lstm.num_layers]
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
            action = action
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
            8,
            8,
        )

        self.obs_latent_query_attn = ObsLatentAttn(
            out_dim=32,
            _feat_dim=41,
            use_mask=True,
            num_query_tokens=10,
            query_token_dim=32,
            num_heads=4,
            num_layers=1,
        )
        self.obs_latent_self_attn = ObsSelfAttn(
            out_dim=128, _feat_dim=32, num_heads=4, num_layers=2, use_mask=False, use_cls_token=True
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

    def network_forward(self, x):
        x, mask, B_TT = self.obs_(x)
        x = self.obs_norm(x)
        x = self.obs_fourier(x)
        x = self.obs_latent_query_attn(x, mask, B_TT)
        x = self.obs_latent_self_attn(x, mask)
        return x

    def encode_observations(self, observations, state=None):
        """
        Encode observations into a hidden representation.
        """

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
