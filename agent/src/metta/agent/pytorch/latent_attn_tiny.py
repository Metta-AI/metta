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

logger = logging.getLogger(__name__)


class LatentAttnTiny(LSTMWrapper):
    def __init__(
        self,
        env,
        policy=None,
        cnn_channels=128,
        input_size=128,
        hidden_size=128,
        num_layers=2,
        clip_range=0,
        analyze_weights_interval=300,
        **kwargs,
    ):
        """Initialize LatentAttnTiny policy with configuration parameters.

        Args:
            env: Environment
            policy: Optional inner policy
            cnn_channels: Number of CNN channels
            input_size: LSTM input size
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            clip_range: Weight clipping range (0 = disabled)
            analyze_weights_interval: Interval for weight analysis
            **kwargs: Additional configuration parameters (for compatibility)
        """
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        # Use enhanced LSTMWrapper with num_layers support
        super().__init__(env, policy, input_size, hidden_size, num_layers=num_layers)

        # Store configuration parameters
        self.clip_range = clip_range
        self.analyze_weights_interval = analyze_weights_interval

        if kwargs:
            logger.info(f"[DEBUG] Additional config parameters: {kwargs}")

    def clip_weights(self):
        """Clip weights to prevent large updates during training.

        This matches ComponentPolicy's weight clipping behavior.
        """
        if self.clip_range > 0:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    if hasattr(module, "weight") and module.weight is not None:
                        module.weight.data.clamp_(-self.clip_range, self.clip_range)
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.data.clamp_(-self.clip_range, self.clip_range)

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def forward(self, td: TensorDict, state=None, action=None):
        observations = td["env_obs"]

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # CRITICAL FIX: Set bptt and batch fields to match ComponentPolicy behavior
        # ComponentPolicy sets these in every forward pass, and components depend on them
        if observations.dim() == 4:  # Training: [B, T, obs_tokens, 3]
            B = observations.shape[0]
            TT = observations.shape[1]
            # Flatten batch dimension and set fields exactly like ComponentPolicy
            total_batch = B * TT
            td.set("bptt", torch.full((total_batch,), TT, device=observations.device, dtype=torch.long))
            td.set("batch", torch.full((total_batch,), B, device=observations.device, dtype=torch.long))
        else:  # Inference: [B, obs_tokens, 3]
            B = observations.shape[0]
            TT = 1
            # Set fields for inference mode
            td.set("bptt", torch.full((B,), 1, device=observations.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=observations.device, dtype=torch.long))

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
            # CRITICAL: ComponentPolicy expects the action to be flattened already during training
            # The TD should be reshaped to match the flattened batch dimension
            if action.dim() == 3:  # (B, T, A) -> (BT, A)
                batch_size_orig, time_steps, A = action.shape
                action = action.view(batch_size_orig * time_steps, A)
                # Also flatten the TD to match
                if td.batch_dims > 1:
                    td = td.reshape(td.batch_size.numel())

            action_log_probs = F.log_softmax(logits_list, dim=-1)
            action_probs = torch.exp(action_log_probs)

            action_logit_index = self._convert_action_to_logit_index(action)
            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            full_log_probs = action_log_probs[batch_indices, action_logit_index]

            entropy = -(action_probs * action_log_probs).sum(dim=-1)

            # Store in flattened TD (will be reshaped by caller if needed)
            td["act_log_prob"] = full_log_probs
            td["entropy"] = entropy
            td["full_log_probs"] = action_log_probs
            td["value"] = value

            # ComponentPolicy reshapes the TD after training forward based on td["batch"] and td["bptt"]
            # The reshaping happens in ComponentPolicy.forward() after forward_training()
            if "batch" in td.keys() and "bptt" in td.keys():
                batch_size = td["batch"][0].item()
                bptt_size = td["bptt"][0].item()
                td = td.reshape(batch_size, bptt_size)

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
