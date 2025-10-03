"""Policy definition for SmolLLM-backed agents."""

from __future__ import annotations

from typing import List, Literal, Optional

from metta.agent.components.actor import ActionProbsConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.smollm import SmolLLMBackboneConfig
from metta.agent.policy import PolicyArchitecture


class SmolLLMConfig(PolicyArchitecture):
    """Policy configuration for SmolLLM-backed agents."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    max_sequence_length: int = 32
    freeze_llm: bool = True
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"

    tokens_key: str = "smollm_tokens"
    logits_key: str = "smollm_logits"
    values_key: str = "values"
    hidden_key: Optional[str] = None

    components: List[ComponentConfig] = []
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="smollm_logits")

    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        self.components = self.build_components()
        if self.action_probs_config.in_key != self.logits_key:
            self.action_probs_config = self.action_probs_config.model_copy(update={"in_key": self.logits_key})

    def build_components(self) -> List[ComponentConfig]:
        return [
            ObsShimTokensConfig(
                in_key="env_obs",
                out_key=self.tokens_key,
                max_tokens=self.max_sequence_length,
            ),
            SmolLLMBackboneConfig(
                in_key=self.tokens_key,
                logits_key=self.logits_key,
                values_key=self.values_key,
                hidden_key=self.hidden_key,
                model_name=self.model_name,
                max_sequence_length=self.max_sequence_length,
                freeze_llm=self.freeze_llm,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attn_implementation,
            ),
        ]
