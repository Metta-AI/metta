import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
from cogames.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation

logger = logging.getLogger("cogames.policies.simple_policy")


class SimplePolicyNet(torch.nn.Module):
    """Feed-forward baseline for discrete action spaces."""

    def __init__(self, env: MettaGridEnv) -> None:
        super().__init__()
        self.hidden_size = 64  # 128

        # # --- vit obs ---
        # shim_config = ObsShimTokensConfig(in_key="env_obs", out_key="box_obs", max_tokens=96)
        # self.obs_shim = shim_config.make_component(env)
        # embed_config = ObsAttrEmbedFourierConfig(in_key="box_obs", out_key="embed_obs", attr_embed_dim=8, num_freqs=3)
        # self.obs_embed = embed_config.make_component(env)
        # vit_config = ObsPerceiverLatentConfig(
        #     in_key="embed_obs",
        #     out_key="vit_obs",
        #     feat_dim=21,
        #     latent_dim=64,
        #     num_latents=12,
        #     num_heads=4,
        #     num_layers=2,
        #     pool="mean",
        # )
        # self.vit = vit_config.make_component(env)
        # # --- end vit obs ---

        # --- old obs ---
        obs_size = int(np.prod(env.single_observation_space.shape))
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_size, self.hidden_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(self.hidden_size, self.hidden_size)),
        )
        # --- end old obs ---

        self.action_nvec = tuple(env.single_action_space.nvec)

        self.action_head = torch.nn.Linear(self.hidden_size, sum(self.action_nvec))
        self.value_head = torch.nn.Linear(self.hidden_size, 1)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch_size = observations.shape[0]

        # --- old obs ---
        obs_flat = observations.view(batch_size, -1).float() / 255.0
        hidden = self.net(obs_flat)
        # --- end old obs ---

        # # --- vit obs ---
        # td = tensordict.TensorDict({"env_obs": observations}, batch_size=(batch_size,))
        # td = self.obs_shim(td)
        # td = self.obs_embed(td)
        # td = self.vit(td)
        # hidden = td["vit_obs"]
        # # --- end vit obs ---

        logits = self.action_head(hidden)
        logits_split = logits.split(self.action_nvec, dim=1)

        values = self.value_head(hidden)
        return list(logits_split), values

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        return self.forward_eval(observations, state)


class SimpleAgentPolicyImpl(AgentPolicy):
    """Per-agent policy wrapper sharing the feed-forward network."""

    def __init__(self, net: SimplePolicyNet, device: torch.device, action_nvec: tuple[int, ...]) -> None:
        self._net = net
        self._device = device
        self._action_nvec = action_nvec

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        obs_tensor = torch.tensor(obs, device=self._device).unsqueeze(0).float()

        with torch.no_grad():
            self._net.eval()
            logits, _ = self._net.forward_eval(obs_tensor)

            actions: list[int] = []
            for logit in logits:
                dist = torch.distributions.Categorical(logits=logit)
                actions.append(dist.sample().item())

            return np.array(actions, dtype=np.int32)


class SimplePolicy(TrainablePolicy):
    """Simple feedforward policy."""

    def __init__(self, env: MettaGridEnv, device: torch.device) -> None:
        super().__init__()
        self._net = SimplePolicyNet(env).to(device)
        self._device = device
        self.action_nvec = tuple(env.single_action_space.nvec)

    def network(self) -> nn.Module:
        return self._net

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return SimpleAgentPolicyImpl(self._net, self._device, self.action_nvec)

    def load_policy_data(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        self._net.load_state_dict(state_dict)
        self._net = self._net.to(self._device)

    def save_policy_data(self, checkpoint_path: str) -> None:
        torch.save(self._net.state_dict(), checkpoint_path)
