import logging
import einops
import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn.functional as F
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
        self.activate_actions(action_names, action_max_params, device)

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

    def forward(self, observations, state=None, action=None):
        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        observations = observations.to(self.device)
        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # prepare lstm state
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
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # Shape: (TT, B, input_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)  # Shape: (B * TT, hidden_size)

        # Decode actions and value
        logits_list, value = self.policy.decode_actions(flat_hidden)

        actions = []
        selected_action_log_probs = []
        entropies = []


        for _, logits in enumerate(logits_list):
            action_log_probs = F.log_softmax(logits, dim=-1)
            action_probs = torch.exp(action_log_probs)

            # Sample action from categorical distribution
            action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

            # Gather log-prob of the sampled action
            batch_indices = torch.arange(action.shape[0], device=action.device)
            selected_log_prob = action_log_probs[batch_indices, action]

            # Entropy
            entropy = -torch.sum(action_probs * action_log_probs, dim=-1)

            actions.append(action)
            selected_action_log_probs.append(selected_log_prob)
            entropies.append(entropy)

        # Convert actions to expected format
        if len(actions) >= 2:
            actions_tensor = torch.stack([actions[0], actions[1]], dim=-1)
        else:
            actions_tensor = torch.stack([actions[0], torch.zeros_like(actions[0])], dim=-1)

        selected_action_log_probs = torch.stack(selected_action_log_probs, dim=-1)
        entropy = torch.stack(entropies, dim=-1).sum(-1)

        return (
            torch.zeros(actions_tensor.shape).to(dtype=torch.int32),
            selected_action_log_probs.mean(dim=-1),
            entropy,
            value,
            logits_list,
        )




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

        self.cnn1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=22, out_channels=64, kernel_size=5, stride=3)
        )
        self.cnn2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        )

        test_input = torch.zeros(1, 22, 11, 11)
        with torch.no_grad():
            test_output = self.cnn2(self.cnn1(test_input))
            self.flattened_size = test_output.numel() // test_output.shape[0]

        self.flatten = nn.Flatten()

        self.fc1 = pufferlib.pytorch.layer_init(
            nn.Linear(self.flattened_size, 128)
        )
        self.encoded_obs = pufferlib.pytorch.layer_init(
            nn.Linear(128, 128)
        )
        self.critic_1 = pufferlib.pytorch.layer_init(
            nn.Linear(self.hidden_size, 1024)
        )
        self.value_head = pufferlib.pytorch.layer_init(
            nn.Linear(1024, 1), std=1.0
        )
        self.actor_1 = pufferlib.pytorch.layer_init(
            nn.Linear(self.hidden_size, 512)
        )
        self.action_embeddings = nn.Embedding(100, 16)

        # Action heads - will be initialized based on action space
        action_nvec = self.action_space.nvec if hasattr(self.action_space, 'nvec') else [100]

        self.actor_heads = nn.ModuleList([
            pufferlib.pytorch.layer_init(nn.Linear(512 + 16, n), std=0.01)
            for n in action_nvec
        ])

        max_vec = torch.tensor([
            9.0, 1.0, 1.0, 10.0, 3.0, 254.0, 1.0, 1.0, 235.0, 8.0, 9.0,
            250.0, 29.0, 1.0, 1.0, 8.0, 1.0, 1.0, 6.0, 3.0, 1.0, 2.0
        ], dtype=torch.float32)[None, :, None, None]
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


        return self.network_forward(box_obs)

    def decode_actions(self, hidden):

        critic_features = self.critic_1(hidden)

        value = self.value_head(critic_features)

        actor_features = self.actor_1(hidden)

        action_embed = self.action_embeddings.weight.mean(dim=0).unsqueeze(0).expand(
            actor_features.shape[0], -1
        )
        combined_features = torch.cat([actor_features, action_embed], dim=-1)
        logits = [head(combined_features) for head in self.actor_heads]

        return logits, value
