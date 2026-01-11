import logging
from typing import List, Optional

import numpy as np
import torch
from cortex.stacks import build_cortex_auto_config
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn
from torchrl.data import Composite, UnboundedDiscrete

import pufferlib.pytorch
from metta.agent.components.actor import ActionProbs, ActionProbsConfig
from metta.agent.components.cnn_encoder import CNNEncoder, CNNEncoderConfig
from metta.agent.components.cortex import CortexTD, CortexTDConfig
from metta.agent.components.obs_shim import ObsShimBox, ObsShimBoxConfig
from metta.agent.policy import Policy
from metta.agent.policy_architecture import PolicyArchitecture
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

logger = logging.getLogger(__name__)


class _OptionalLoadLinear(nn.Linear):
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        if weight_key not in state_dict:
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class FastConfig(PolicyArchitecture):
    """
    Fast uses a CNN encoder so is not flexible to changing observation features but it runs faster than the ViT encoder.

    This particular class is also set up without using PolicyAutoBuilder to demonstrate an alternative way to build a
    policy, affording more control over the process at the expense of more code. It demonstrates that we can use config
    objects (ie CortexTDConfig) for classes as layers (self.core) or attributes (ie actor_hidden_dim) for simple torch
    classes as layers (ie self.critic_1), wrap them in a TensorDictModule, and intermix."""

    class_path: str = "metta.agent.policies.fast.FastPolicy"

    _hidden_size = 128

    obs_shim_config: ObsShimBoxConfig = ObsShimBoxConfig(in_key="env_obs", out_key="obs_normalizer")
    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig(in_key="obs_normalizer", out_key="encoded_obs")
    cortex_core_config: CortexTDConfig = CortexTDConfig(
        in_key="encoded_obs",
        out_key="core",
        d_hidden=_hidden_size,
        out_features=_hidden_size,
        key_prefix="fast_cortex_state",
        stack_cfg=build_cortex_auto_config(
            d_hidden=_hidden_size,
            num_layers=1,
            pattern="L",
            post_norm=False,
        ),
    )
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class FastPolicy(Policy):
    def __init__(self, policy_env_info: "PolicyEnvInterface", config: Optional[FastConfig] = None):
        super().__init__(policy_env_info)
        self.config = config or FastConfig()
        self.is_continuous = False
        self.action_space = policy_env_info.action_space

        self.active_action_names = []
        self.num_active_actions = 100  # Default

        self.out_width = policy_env_info.obs_width
        self.out_height = policy_env_info.obs_height

        self.obs_shim = ObsShimBox(policy_env_info=policy_env_info, config=self.config.obs_shim_config)

        self.cnn_encoder = CNNEncoder(config=self.config.cnn_encoder_config, policy_env_interface=policy_env_info)

        self.core = CortexTD(config=self.config.cortex_core_config)
        core_width = int(self.config.cortex_core_config.out_features or self.config.cortex_core_config.d_hidden)

        module = pufferlib.pytorch.layer_init(nn.Linear(core_width, self.config.actor_hidden_dim), std=1.0)
        self.actor_1 = TDM(module, in_keys=["core"], out_keys=["actor_1"])

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        module = pufferlib.pytorch.layer_init(nn.Linear(core_width, self.config.critic_hidden_dim), std=np.sqrt(2))
        self.critic_1 = TDM(module, in_keys=["core"], out_keys=["critic_1"])
        self.critic_activation = nn.Tanh()
        module = pufferlib.pytorch.layer_init(nn.Linear(self.config.critic_hidden_dim, 1), std=1.0)
        self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])
        module = pufferlib.pytorch.layer_init(_OptionalLoadLinear(self.config.critic_hidden_dim, 1), std=1.0)
        self.gtd_aux = TDM(module, in_keys=["critic_1"], out_keys=["h_values"])

        # Actor branch
        self.actor_logits = TDM(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.config.actor_hidden_dim, int(self.action_space.n)),
                std=0.01,
            ),
            in_keys=["actor_1"],
            out_keys=["logits"],
        )
        self.action_probs = ActionProbs(config=self.config.action_probs_config)

    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations hurting performance
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        self.obs_shim(td)
        self.cnn_encoder(td)
        self.core(td)
        self.actor_1(td)
        td["actor_1"] = torch.relu(td["actor_1"])
        self.critic_1(td)
        td["critic_1"] = self.critic_activation(td["critic_1"])
        self.value_head(td)
        self.gtd_aux(td)
        self.actor_logits(td)
        self.action_probs(td, action)
        td["values"] = td["values"].flatten()
        td["h_values"] = td["h_values"].flatten()

        return td

    def initialize_to_environment(
        self,
        policy_env_info: PolicyEnvInterface,
        device: torch.device,
    ) -> List[str]:
        device = torch.device(device)
        self.to(device)

        log = self.obs_shim.initialize_to_environment(policy_env_info, device)
        self.action_probs.initialize_to_environment(policy_env_info, device)
        return [log]

    def reset_memory(self):
        self.core.reset_memory()

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
