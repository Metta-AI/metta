import logging

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy=None, cnn_channels=128, input_size=512, hidden_size=512):
        if policy is None:
            policy = Policy(env, cnn_channels=cnn_channels, hidden_size=hidden_size, input_size=input_size)
        super().__init__(env, policy, input_size, hidden_size)

    def activate_actions(self, action_names, action_max_params, device):
        """
        Initialize the action space, similar to MettaAgent.activate_actions.

        Args:
            action_names: List of action names
            action_max_params: List of maximum parameters for each action head
            device: Device to place tensors on
        """
        assert isinstance(action_max_params, list), "action_max_params must be a list"
        self.device = device
        self.action_max_params = action_max_params
        self.action_names = action_names
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for action index conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=self.device, dtype=torch.long), dim=0
        )

        # Create action_index tensor for conversion
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])
        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)

        logger.info(f"Policy actions initialized with: {self.active_actions}")

    def forward(self, observations, state=None, action=None):
        """
        Forward pass through the recurrent policy.

        Args:
            observations: Input observation tensor, shape (B, TT, M, 3) or (B, M, 3)
            state: Dictionary with 'lstm_h', 'lstm_c', 'hidden' (optional)
            action: Optional action tensor for training, shape (B, T, num_action_heads)

        Returns:
            Tuple of (actions, action_log_prob, entropy, value, log_probs)
        """
        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        observations = observations.to(self.device)
        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # Prepare LSTM state
        lstm_h = state.get("lstm_h", None)
        lstm_c = state.get("lstm_c", None)
        if lstm_h is not None and lstm_c is not None:
            lstm_h = lstm_h.to(self.device)
            lstm_c = lstm_c.to(self.device)
            # Ensure LSTM state shapes match
            expected_num_layers = self.lstm.num_layers
            lstm_h = lstm_h[:expected_num_layers, :, :]
            lstm_c = lstm_c[:expected_num_layers, :, :]
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        # LSTM forward pass
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # Shape: (TT, B, input_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)  # Shape: (B * TT, hidden_size)

        # Decode actions and value
        logits_list, value = self.policy.decode_actions(flat_hidden)

        actions = []
        selected_action_log_probs = []
        entropies = []

        for _, logits in enumerate(logits_list):  # Each logits tensor: [batch_size, action_dim_i]
            action_log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, action_dim_i]
            action_probs = torch.exp(action_log_probs)  # [batch_size, action_dim_i]

            # Sample action from categorical distribution
            action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)  # [batch_size]

            # Gather log-prob of the sampled action
            batch_indices = torch.arange(action.shape[0], device=action.device)
            selected_log_prob = action_log_probs[batch_indices, action]  # [batch_size]

            # Entropy: H(π) = -∑π(a)log π(a)
            entropy = -torch.sum(action_probs * action_log_probs, dim=-1)  # [batch_size]

            actions.append(action)
            selected_action_log_probs.append(selected_log_prob)
            entropies.append(entropy)

        # Convert actions to the expected format [batch_size, 2]
        # Assuming first action is action_type and second is action_param
        if len(actions) >= 2:
            # Stack the first two actions as [action_type, action_param]
            actions_tensor = torch.stack([actions[0], actions[1]], dim=-1)  # [batch_size, 2]
        else:
            # If only one action head, pad with zeros
            actions_tensor = torch.stack([actions[0], torch.zeros_like(actions[0])], dim=-1)  # [batch_size, 2]

        # Stack log probs and entropy over action heads
        selected_action_log_probs = torch.stack(selected_action_log_probs, dim=-1)  # [batch_size, num_action_heads]
        entropy = torch.stack(entropies, dim=-1).sum(-1)  # [batch_size] — total entropy over heads

        # Return: sampled actions [B, 2], mean log-probs [B], total entropy [B], value [B, 1], logits
        return (
            torch.zeros(actions_tensor.shape).to(dtype=torch.int32),
            selected_action_log_probs.mean(dim=-1),
            entropy,
            value,  # Keep value as [B*TT, 1] which should be [B, 1] when TT=1
            logits_list,
        )


class Policy(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.out_width = 11
        self.out_height = 11
        self.num_layers = 22

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.num_layers, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, hidden_size // 2)),
            nn.ReLU(),
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
        self.register_buffer("max_vec", max_vec.to(self.device))

        action_nvec = self.action_space.nvec
        self.actor = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in action_nvec]
        )
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        self.to(self.device)

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

        features = box_obs / self.max_vec
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden):
        """
        Decode hidden representation into logits and value.

        Args:
            hidden: Hidden representation, shape (B * TT, hidden_size)

        Returns:
            logits: List of logits for each action head, [shape (B * TT, n_i)]
            value: Value estimate, shape (B * TT, 1)
        """
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value
