import logging
import typing

import numpy as np
import tensordict
import tensordict.nn
import torch
import torchrl.data

import metta.agent.components.actor
import metta.agent.components.cnn_encoder
import metta.agent.components.lstm
import metta.agent.components.obs_shim
import metta.agent.policy
import mettagrid.policy.policy_env_interface
import pufferlib.pytorch

logger = logging.getLogger(__name__)


class FastConfig(metta.agent.policy.PolicyArchitecture):
    """
    Fast uses a CNN encoder so is not flexible to changing observation features but it runs faster than the ViT encoder.

    This particular class is also set up without using PolicyAutoBuilder to demonstrate an alternative way to build a
    policy, affording more control over the process at the expense of more code. It demonstrates that we can use config
    objects (ie LSTMConfig) for classes as layers (self.lstm) or attributes (ie actor_hidden_dim) for simple torch
    classes as layers (ie self.critic_1), wrap them in a TensorDictModule, and intermix."""

    class_path: str = "metta.agent.policies.fast.FastPolicy"

    obs_shim_config: metta.agent.components.obs_shim.ObsShimBoxConfig = (
        metta.agent.components.obs_shim.ObsShimBoxConfig(in_key="env_obs", out_key="obs_normalizer")
    )
    cnn_encoder_config: metta.agent.components.cnn_encoder.CNNEncoderConfig = (
        metta.agent.components.cnn_encoder.CNNEncoderConfig(in_key="obs_normalizer", out_key="encoded_obs")
    )
    lstm_config: metta.agent.components.lstm.LSTMConfig = metta.agent.components.lstm.LSTMConfig(
        in_key="encoded_obs", out_key="core", latent_size=128, hidden_size=128, num_layers=2
    )
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    action_probs_config: metta.agent.components.actor.ActionProbsConfig = (
        metta.agent.components.actor.ActionProbsConfig(in_key="logits")
    )


class FastPolicy(metta.agent.policy.Policy):
    def __init__(self, policy_env_info: "PolicyEnvInterface", config: typing.Optional[FastConfig] = None):
        super().__init__(policy_env_info)
        self.config = config or FastConfig()
        self.policy_env_info = policy_env_info
        self.is_continuous = False
        self.action_space = policy_env_info.action_space

        self.active_action_names = []
        self.num_active_actions = 100  # Default

        self.out_width = policy_env_info.obs_width
        self.out_height = policy_env_info.obs_height

        self.obs_shim = metta.agent.components.obs_shim.ObsShimBox(
            policy_env_info=policy_env_info, config=self.config.obs_shim_config
        )

        self.cnn_encoder = metta.agent.components.cnn_encoder.CNNEncoder(
            config=self.config.cnn_encoder_config, policy_env_interface=policy_env_info
        )

        self.lstm = metta.agent.components.lstm.LSTM(config=self.config.lstm_config)

        module = pufferlib.pytorch.layer_init(
            torch.nn.Linear(self.config.lstm_config.hidden_size, self.config.actor_hidden_dim), std=1.0
        )
        self.actor_1 = tensordict.nn.TensorDictModule(module, in_keys=["core"], out_keys=["actor_1"])

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        module = pufferlib.pytorch.layer_init(
            torch.nn.Linear(self.config.lstm_config.hidden_size, self.config.critic_hidden_dim), std=np.sqrt(2)
        )
        self.critic_1 = tensordict.nn.TensorDictModule(module, in_keys=["core"], out_keys=["critic_1"])
        self.critic_activation = torch.nn.Tanh()
        module = pufferlib.pytorch.layer_init(torch.nn.Linear(self.config.critic_hidden_dim, 1), std=1.0)
        self.value_head = tensordict.nn.TensorDictModule(module, in_keys=["critic_1"], out_keys=["values"])

        # Actor branch
        self.actor_logits = tensordict.nn.TensorDictModule(
            pufferlib.pytorch.layer_init(
                torch.nn.Linear(self.config.actor_hidden_dim, int(self.action_space.n)),
                std=0.01,
            ),
            in_keys=["actor_1"],
            out_keys=["logits"],
        )
        self.action_probs = metta.agent.components.actor.ActionProbs(config=self.config.action_probs_config)

    @torch._dynamo.disable  # Avoid graph breaks from TensorDict operations hurting performance
    def forward(self, td: tensordict.TensorDict, state=None, action: torch.Tensor = None):
        self.obs_shim(td)
        self.cnn_encoder(td)
        self.lstm(td)
        self.actor_1(td)
        td["actor_1"] = torch.relu(td["actor_1"])
        self.critic_1(td)
        td["critic_1"] = self.critic_activation(td["critic_1"])
        self.value_head(td)
        self.actor_logits(td)
        self.action_probs(td, action)
        td["values"] = td["values"].flatten()

        return td

    def initialize_to_environment(
        self,
        policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface,
        device: torch.device,
    ) -> typing.List[str]:
        device = torch.device(device)
        self.to(device)

        log = self.obs_shim.initialize_to_environment(policy_env_info, device)
        self.action_probs.initialize_to_environment(policy_env_info, device)
        return [log]

    def reset_memory(self):
        self.lstm.reset_memory()

    def get_agent_experience_spec(self) -> torchrl.data.Composite:
        return torchrl.data.Composite(
            env_obs=torchrl.data.UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=torchrl.data.UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=torchrl.data.UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
