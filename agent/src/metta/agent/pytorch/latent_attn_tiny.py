import logging

import einops
import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn

from metta.agent.lib.obs_enc import ObsLatentAttn
from metta.agent.lib.obs_tokenizers import ObsAttrEmbedFourier, ObsAttrValNorm, ObsTokenPadStrip
from metta.agent.pytorch.pytorch_base import PytorchAgentBase

logger = logging.getLogger(__name__)


class Recurrent(PytorchAgentBase):
    def __init__(self, env, policy=None, cnn_channels=128, input_size=64, hidden_size=64):
        if policy is None:
            policy = Policy(
                env,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        super().__init__(env, policy, input_size, hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store these for compatibility with MettaAgent's activate_actions
        self.action_index_tensor = None
        self.cum_action_max_params = None


class Policy(nn.Module):
    def __init__(self, env, input_size=64, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.action_space = env.single_action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = nn.Sequential(
            ObsTokenPadStrip("all"),
            ObsAttrEmbedFourier(
                attr_embed_dim=8,  # Fixed parameter name
                num_freqs=3,
            ),
            ObsAttrValNorm(),
        )

        self.encoder = ObsLatentAttn(
            d_model=8,
            n_encoders=1,
            n_heads=2,
            d_head=16,
            d_ff=64,
            encoder_norm="pre",
            return_type="cls",
            positional_encoder="sinusoidal_indexed",
            n_latents=4,
            n_latent_heads=1,
            n_latent_blocks=2,
            latent_dim=64,
            latent_ff_mult=2,
            latent_norm="layer",
            latent_skip=False,
            output_dim=input_size,
        )

        self.critic_1 = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        self.actor_1 = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size))

        action_nvec = self.action_space.nvec
        num_actions = sum(action_nvec)

        self.action_embeddings = nn.Embedding(num_actions, hidden_size)

        self.actor_heads = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size * 2, n), std=0.01) for n in action_nvec]
        )

        self.to(self.device)

    def network_forward(self, x):
        return self.network(x)

    def encode_observations(self, observations, state=None):
        """
        Encode observations into a hidden representation.
        """
        observations = observations.to(self.device)

        # Get tokenized observations
        if observations.dim() == 3:
            tokenized_obs = observations
        else:
            tokenized_obs = einops.rearrange(observations, "b t m c -> (b t) m c")

        # Pass through tokenizer
        tokenized_obs = self.tokenizer(tokenized_obs)

        # Pass through encoder (which returns the final hidden representation for tiny model)
        hidden = self.encoder(tokenized_obs)

        return hidden

    def decode_actions(self, hidden):
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)

        actor_features = torch.tanh(self.actor_1(hidden))

        action_embed = self.action_embeddings.weight.mean(dim=0).unsqueeze(0).expand(actor_features.shape[0], -1)
        combined_features = torch.cat([actor_features, action_embed], dim=-1)
        logits = torch.cat([head(combined_features) for head in self.actor_heads], dim=-1)  # (B, sum(A_i))

        return logits, value
