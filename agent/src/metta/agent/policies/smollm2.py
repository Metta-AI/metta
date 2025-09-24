import logging
from typing import Optional

import pufferlib.pytorch
import torch
from tensordict import TensorDict
from torch import nn
from transformers import AutoModelForCausalLM

from metta.agent.components.actor import ActionProbs, ActionProbsConfig
from metta.agent.components.obs_shim import ObsShimTokens, ObsShimTokensConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.training import EnvironmentMetaData

logger = logging.getLogger(__name__)


class SmolLM2Config(PolicyArchitecture):
    """Configuration for the SmolLM2 policy architecture."""

    class_path: str = "metta.agent.policies.smollm2.SmolLM2Policy"

    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    max_sequence_length: int = 24
    freeze_llm: bool = False
    tokens_key: str = "smollm2_tokens"
    logits_key: str = "smollm2_logits"
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="smollm2_logits")


class SmolLM2Policy(Policy):
    """Policy that wraps a pretrained SmolLM2 language model for action selection."""

    def __init__(self, env: EnvironmentMetaData, config: Optional[SmolLM2Config] = None):
        super().__init__()
        self.config = config or SmolLM2Config()
        self.tokens_key = self.config.tokens_key
        self.logits_key = self.config.logits_key
        self.max_sequence_length = self.config.max_sequence_length

        # Observation processing to produce token sequences compatible with the LLM projector.
        obs_shim_config = ObsShimTokensConfig(in_key="env_obs", out_key=self.tokens_key)
        self.obs_shim = ObsShimTokens(env, config=obs_shim_config)

        # Load the pretrained language model (kept in FP32 for stability like other policies).
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

        # Action/value heads operate on the pooled hidden representation.
        self.max_action_args = list(env.max_action_args)
        self.total_actions = sum(max_arg + 1 for max_arg in self.max_action_args)

        self.actor_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.total_actions), std=0.01)
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

        self.config.action_probs_config.in_key = self.logits_key
        self.action_probs = ActionProbs(self.config.action_probs_config)

    def _initialize_projector(self) -> None:
        for module in self.token_projector.modules():
            if isinstance(module, nn.Linear):
                pufferlib.pytorch.layer_init(module, std=1.0)

    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Run a forward pass through the LLM-based policy."""

        original_shape = td.batch_size
        needs_reshape = td.batch_dims > 1
        flat_td = td.reshape(td.batch_size.numel()) if needs_reshape else td

        # Prepare observations
        self.obs_shim(flat_td)
        observations = flat_td[self.tokens_key]

        if observations.dim() == 4:  # [B, T, seq, 3]
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
        flat_td.set("values", values)

        if action is None:
            self.action_probs(flat_td)
        else:
            self.action_probs(flat_td, action)

        # Ensure values follow expected flattened layout
        flat_td["values"] = flat_td["values"].flatten()

        return flat_td.reshape(original_shape) if needs_reshape else flat_td

    def initialize_to_environment(self, env_metadata: EnvironmentMetaData, device: torch.device):
        self.to(device)
        log = self.obs_shim.initialize_to_environment(env_metadata, device)
        self.action_probs.initialize_to_environment(env_metadata, device)

        llm_dtype = next(self.llm.parameters()).dtype
        self.token_projector = self.token_projector.to(device=device, dtype=llm_dtype)
        self.actor_head = self.actor_head.to(device=device, dtype=llm_dtype)
        self.value_head = self.value_head.to(device=device, dtype=llm_dtype)

        logger.info("SmolLM2 initialized: %d actions, dtype %s", self.total_actions, llm_dtype)
        return [log]

    def reset_memory(self):
        # Stateless policy – nothing to reset.
        return None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _compress_tokens(self, observations: torch.Tensor) -> torch.Tensor:
        """Reduce token sequence length while preserving informative tokens."""

        batch, seq_len, channels = observations.shape
        if seq_len <= self.max_sequence_length:
            return observations

        coords_byte = observations[..., 0]
        valid_mask = coords_byte != 255

        compressed = torch.zeros(
            (batch, self.max_sequence_length, channels),
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
