"""Policy definition for SmolLLM-backed agents."""

import typing

import metta.agent.components.actor
import metta.agent.components.component_config
import metta.agent.components.obs_shim
import metta.agent.components.smollm
import metta.agent.policy


class SmolLLMConfig(metta.agent.policy.PolicyArchitecture):
    """Policy configuration for SmolLLM-backed agents."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    max_sequence_length: int = 32
    token_stride: int = 1
    freeze_llm: bool = True
    torch_dtype: typing.Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    attn_implementation: typing.Optional[str] = "flash_attention_2"

    tokens_key: str = "smollm_tokens"
    logits_key: str = "smollm_logits"
    values_key: str = "values"
    hidden_key: typing.Optional[str] = None

    actor_head_rank: typing.Optional[int] = None
    value_head_rank: typing.Optional[int] = None
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.Optional[typing.List[str]] = None

    components: typing.List[metta.agent.components.component_config.ComponentConfig] = []
    action_probs_config: metta.agent.components.actor.ActionProbsConfig = (
        metta.agent.components.actor.ActionProbsConfig(in_key="smollm_logits")
    )

    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        self.components = self.build_components()
        if self.action_probs_config.in_key != self.logits_key:
            self.action_probs_config = self.action_probs_config.model_copy(update={"in_key": self.logits_key})

    def build_components(self) -> typing.List[metta.agent.components.component_config.ComponentConfig]:
        return [
            metta.agent.components.obs_shim.ObsShimTokensConfig(
                in_key="env_obs",
                out_key=self.tokens_key,
                max_tokens=self.max_sequence_length,
            ),
            metta.agent.components.smollm.SmolLLMBackboneConfig(
                in_key=self.tokens_key,
                logits_key=self.logits_key,
                values_key=self.values_key,
                hidden_key=self.hidden_key,
                model_name=self.model_name,
                max_sequence_length=self.max_sequence_length,
                freeze_llm=self.freeze_llm,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attn_implementation,
                token_stride=self.token_stride,
                actor_head_rank=self.actor_head_rank,
                value_head_rank=self.value_head_rank,
                use_lora=self.use_lora,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_target_modules=self.lora_target_modules,
            ),
        ]
