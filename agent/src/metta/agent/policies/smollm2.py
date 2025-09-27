import logging
from typing import List, Optional

import torch
from tensordict import TensorDict

from metta.agent.components.actor import ActionProbsConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.smollm import SmolLM2BackboneConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.agent.policy_auto_builder import PolicyAutoBuilder
from metta.rl.training import EnvironmentMetaData

logger = logging.getLogger(__name__)


class SmolLM2Config(PolicyArchitecture):
    """Configuration for SmolLM2 policies built with the auto-builder."""

    class_path: str = "metta.agent.policies.smollm2.SmolLM2Policy"

    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    max_sequence_length: int = 24
    freeze_llm: bool = False

    tokens_key: str = "smollm2_tokens"
    logits_key: str = "smollm2_logits"
    hidden_key: Optional[str] = "smollm2_hidden"

    components: List[ComponentConfig] = []
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="smollm2_logits")

    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        self.components = self.build_components()
        if self.action_probs_config.in_key != self.logits_key:
            self.action_probs_config = self.action_probs_config.model_copy(update={"in_key": self.logits_key})

    def build_components(self) -> List[ComponentConfig]:
        return [
            ObsShimTokensConfig(in_key="env_obs", out_key=self.tokens_key),
            SmolLM2BackboneConfig(
                in_key=self.tokens_key,
                logits_key=self.logits_key,
                values_key="values",
                hidden_key=self.hidden_key,
                model_name=self.model_name,
                max_sequence_length=self.max_sequence_length,
                freeze_llm=self.freeze_llm,
            ),
        ]


class SmolLM2Policy(Policy):
    """Policy wrapper that delegates to PolicyAutoBuilder with SmolLM2 components."""

    def __init__(self, env_metadata: EnvironmentMetaData, config: Optional[SmolLM2Config] = None):
        super().__init__()
        self.config = config or SmolLM2Config()
        # Ensure components/action head reflect current config values
        self.config.components = self.config.build_components()
        if self.config.action_probs_config.in_key != self.config.logits_key:
            self.config.action_probs_config = self.config.action_probs_config.model_copy(
                update={"in_key": self.config.logits_key}
            )

        self.builder = PolicyAutoBuilder(env_metadata, self.config)
        self.components = self.builder.components
        self.network = self.builder.network
        self.action_probs = self.builder.action_probs

    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        return self.builder.forward(td, action)

    def initialize_to_environment(self, env_metadata: EnvironmentMetaData, device: torch.device):
        return self.builder.initialize_to_environment(env_metadata, device)

    def reset_memory(self):
        self.builder.reset_memory()

    def get_agent_experience_spec(self):
        return self.builder.get_agent_experience_spec()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def total_params(self) -> int:
        return self.builder.total_params
