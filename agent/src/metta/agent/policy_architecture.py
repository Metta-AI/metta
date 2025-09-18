"""Policy architecture configuration (separated from policy base)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from metta.agent.components.component_config import ComponentConfig
from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    from metta.rl.training.training_environment import EnvironmentMetaData


class PolicyArchitecture(Config):
    class_path: str
    components: List[ComponentConfig] = []
    action_probs_config: ComponentConfig

    def make_policy(self, env_metadata: "EnvironmentMetaData"):
        AgentClass = load_symbol(self.class_path)
        return AgentClass(env_metadata, self)
