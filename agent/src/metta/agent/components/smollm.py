"""Components for SmolLM-based policies."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn
from transformers import AutoModelForCausalLM

import pufferlib.pytorch
from metta.agent.components.component_config import ComponentConfig
from metta.rl.training import EnvironmentMetaData

logger = logging.getLogger(__name__)


class SmolLM2BackboneConfig(ComponentConfig):
    """Configuration for the SmolLM2 backbone component."""

    in_key: str
    name: str = "smollm2_backbone"
    logits_key: str = "smollm2_logits"
    values_key: str = "values"
    hidden_key: Optional[str] = None

    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    max_sequence_length: int = 24
    freeze_llm: bool = False

    def make_component(self, env: EnvironmentMetaData):
        return SmolLM2Backbone(env, self)


class SmolLM2Backbone(nn.Module):
    """Token-processing backbone that wraps a pretrained SmolLM2 model."""

    def __init__(self, env: EnvironmentMetaData, config: SmolLM2BackboneConfig):
        super().__init__()
        self.config = config
        self.tokens_key = self.config.in_key
        self.logits_key = self.config.logits_key
        self.values_key = self.config.values_key
        self.hidden_key = self.config.hidden_key
        self.max_sequence_length = self.config.max_sequence_length

        logger.info("Loading SmolLM2 model: %s", self.config.model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        if self.config.freeze_llm:
            logger.info("Freezing SmolLM2 weights")
            for param in self.llm.parameters():
                param.requires_grad = False

        self.hidden_size = self.llm.config.hidden_size

        # Project 3-channel tokens into the LLM hidden dimension.
        self.token_projector = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self._initialize_projector()

        self.max_action_args = list(env.max_action_args)
        self.total_actions = sum(max_arg + 1 for max_arg in self.max_action_args)

        self.actor_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.total_actions), std=0.01)
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

    def forward(self, td: TensorDict) -> TensorDict:
        flat_td = td.reshape(td.batch_size.numel()) if td.batch_dims > 1 else td
        observations = flat_td[self.tokens_key]

        if observations.dim() == 4:  # e.g. [B, T, seq, 3]
            observations = observations.view(-1, observations.shape[-2], observations.shape[-1])

        model_dtype = next(self.llm.parameters()).dtype
        obs_float = observations.to(dtype=model_dtype)
        obs_float = self._compress_tokens(obs_float)
        token_embeddings = self.token_projector(obs_float)

        outputs = self.llm(
            inputs_embeds=token_embeddings,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        pooled_hidden = hidden_states.mean(dim=1)

        logits = self.actor_head(pooled_hidden).to(dtype=torch.float32)
        values = self.value_head(pooled_hidden).squeeze(-1).to(dtype=torch.float32)

        flat_td.set(self.logits_key, logits)
        flat_td.set(self.values_key, values)
        if self.hidden_key is not None:
            flat_td.set(self.hidden_key, pooled_hidden.to(dtype=torch.float32))

        return td

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device):
        self.to(device)

        llm_dtype = next(self.llm.parameters()).dtype
        self.token_projector = self.token_projector.to(device=device, dtype=llm_dtype)
        self.actor_head = self.actor_head.to(device=device, dtype=llm_dtype)
        self.value_head = self.value_head.to(device=device, dtype=llm_dtype)

        logger.info("SmolLM2 backbone initialized: %d actions", self.total_actions)
        return f"SmolLM2 actions: {self.total_actions}"

    def reset_memory(self):
        # Stateless component
        return None

    def _initialize_projector(self) -> None:
        for module in self.token_projector.modules():
            if isinstance(module, nn.Linear):
                pufferlib.pytorch.layer_init(module, std=1.0)

    def _compress_tokens(self, observations: torch.Tensor) -> torch.Tensor:
        """Reduce token sequence length while preserving informative tokens."""

        batch, seq_len, channels = observations.shape
        if seq_len <= self.max_sequence_length:
            return observations

        coords_byte = observations[..., 0]
        valid_mask = coords_byte != 255

        compressed = torch.full(
            (batch, self.max_sequence_length, channels),
            fill_value=255,
            device=observations.device,
            dtype=observations.dtype,
        )

        for idx in range(batch):
            valid_indices = torch.where(valid_mask[idx])[0]
            if valid_indices.numel() <= self.max_sequence_length:
                n_valid = valid_indices.numel()
                if n_valid > 0:
                    compressed[idx, :n_valid] = observations[idx, valid_indices]
            else:
                step = valid_indices.numel() / self.max_sequence_length
                selected = valid_indices[
                    (step * torch.arange(self.max_sequence_length, device=observations.device)).long()
                ]
                compressed[idx] = observations[idx, selected]

        return compressed
