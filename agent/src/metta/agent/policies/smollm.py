"""Policy definition for SmolLLM-backed agents with Cortex HF stack."""

from __future__ import annotations

from typing import List, Literal, Optional

import torch
from cortex.stacks import build_hf_stack_config, build_llama_stack_config_from_model
from transformers import AutoConfig, AutoModelForCausalLM

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture
from metta.agent.utils import resolve_torch_dtype


class SmolLLMConfig(PolicyArchitecture):
    """Cortexified SmolLLM policy that wraps HF layers in a Cortex stack."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    num_layers: Optional[int] = None
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    max_sequence_length: int = 32
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    mem_len: int = 128
    init_from_pretrained: bool = True

    tokens_key: str = "smollm_tokens"
    logits_key: str = "logits"
    values_key: str = "values"

    _token_embed_dim: int = 8
    _fourier_freqs: int = 3
    _num_latents: int = 12
    _num_heads: int = 4
    _num_layers: int = 2
    _actor_hidden: int = 256
    _critic_hidden: int = 512

    components: List[ComponentConfig] = []
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="smollm_logits")

    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        self.components = self.build_components()
        if self.action_probs_config.in_key != self.logits_key:
            self.action_probs_config = self.action_probs_config.model_copy(update={"in_key": self.logits_key})

    def _resolve_dtype(self) -> Optional[torch.dtype]:
        return resolve_torch_dtype(self.dtype)

    def build_components(self) -> List[ComponentConfig]:
        if self.init_from_pretrained:
            stack_cfg = build_hf_stack_config(
                self.model_name,
                trust_remote_code=True,
                dtype=self._resolve_dtype(),
                attn_implementation=self.attn_implementation,
                num_layers=self.num_layers,
                mem_len=int(self.mem_len),
                compile_blocks=False,
            )
        else:
            stack_cfg = _build_hf_stack_config_from_scratch(
                self.model_name,
                trust_remote_code=True,
                dtype=self._resolve_dtype(),
                attn_implementation=self.attn_implementation,
                num_layers=self.num_layers,
                mem_len=int(self.mem_len),
                compile_blocks=False,
            )
        hf_hidden = int(stack_cfg.d_hidden)

        feat_dim = self._token_embed_dim + (4 * self._fourier_freqs) + 1

        components: List[ComponentConfig] = [
            ObsShimTokensConfig(
                in_key="env_obs",
                out_key=self.tokens_key,
                max_tokens=self.max_sequence_length,
            ),
            ObsAttrEmbedFourierConfig(
                in_key=self.tokens_key,
                out_key="obs_attr_embed",
                attr_embed_dim=self._token_embed_dim,
                num_freqs=self._fourier_freqs,
            ),
            ObsPerceiverLatentConfig(
                in_key="obs_attr_embed",
                out_key="obs_latent_attn",
                feat_dim=feat_dim,
                latent_dim=hf_hidden,
                num_latents=self._num_latents,
                num_heads=self._num_heads,
                num_layers=self._num_layers,
            ),
            CortexTDConfig(
                in_key="obs_latent_attn",
                out_key="core",
                d_hidden=hf_hidden,
                out_features=hf_hidden,
                stack_cfg=stack_cfg,
                key_prefix="cortex_state",
                dtype=self.dtype,
            ),
            MLPConfig(
                in_key="core",
                out_key="actor_hidden",
                name="actor_mlp",
                in_features=hf_hidden,
                hidden_features=[self._actor_hidden],
                out_features=self._actor_hidden,
            ),
            MLPConfig(
                in_key="core",
                out_key=self.values_key,
                name="critic",
                in_features=hf_hidden,
                out_features=1,
                hidden_features=[self._critic_hidden],
            ),
            ActorHeadConfig(in_key="actor_hidden", out_key=self.logits_key, input_dim=self._actor_hidden),
        ]

        return components


def _build_hf_stack_config_from_scratch(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    num_layers: Optional[int] = None,
    mem_len: int = 0,
    compile_blocks: bool = False,
) -> object:
    """Build a CortexStackConfig matching a HF model's architecture but with random initialization."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if attn_implementation is not None and hasattr(config, "attn_implementation"):
        config.attn_implementation = attn_implementation  # type: ignore[attr-defined]

    model = AutoModelForCausalLM.from_config(config)

    if dtype is not None:
        model = model.to(dtype=dtype)

    model.eval()

    model_type = getattr(model.config, "model_type", None)
    if model_type != "llama":
        msg = f"_build_hf_stack_config_from_scratch currently supports LLaMA; got model_type={model_type}"
        raise NotImplementedError(msg)

    if num_layers is not None and num_layers > 0:
        model.model.layers = model.model.layers[:num_layers]  # type: ignore[attr-defined]

    return build_llama_stack_config_from_model(model, mem_len=mem_len, compile_blocks=compile_blocks)
