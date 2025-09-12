import logging
from typing import Optional

import pufferlib.pytorch
import torch
from tensordict import TensorDict
from torch import nn
from transformers import AutoModelForCausalLM

from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class SmolLM2(PyTorchAgentMixin, nn.Module):
    """SmolLM2-based policy using HuggingFace's pre-trained language model."""

    def __init__(
        self,
        env,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        hidden_size: int = 576,  # SmolLM2-135M has 576 hidden size
        max_sequence_length: int = 24,  # Longer sequences for temporal flattening efficiency
        freeze_llm: bool = False,
        use_lora: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Extract mixin parameters
        mixin_params = self.extract_mixin_params(kwargs)

        # Initialize the SmolLM2 model with memory optimizations
        logger.info(f"Loading SmolLM2 model: {model_name}")

        # Try flash attention first, fallback gracefully if not available
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",  # Use flash attention for performance
                torch_dtype=torch.float16,  # Use FP16 for better memory efficiency
                low_cpu_mem_usage=True,  # Reduce CPU memory during model loading
            )
            logger.info("Loaded SmolLM2 with Flash Attention 2")
        except Exception as e:
            logger.warning(f"Flash attention not available ({e}), falling back to standard attention")
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Still use FP16 for memory efficiency
                low_cpu_mem_usage=True,  # Reduce CPU memory during model loading
            )

        # Store sequence length limit
        self.max_sequence_length = max_sequence_length

        # Token compression strategy for efficiency
        self.token_compression = "smart_sample"  # Options: "truncate", "smart_sample", "aggregate"

        # Note: Gradient checkpointing disabled for 135M model - unnecessary overhead

        # Get model configuration
        self.hidden_size = self.llm.config.hidden_size
        self.num_layers = self.llm.config.num_hidden_layers

        # Optionally freeze the LLM weights
        if freeze_llm:
            logger.info("Freezing LLM weights")
            for param in self.llm.parameters():
                param.requires_grad = False

        # Token embedding projection
        # Map from [batch, seq_len, 3] byte tokens to LLM embedding space
        self.token_projector = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.hidden_size),
        )

        # Initialize token projector with small weights
        for module in self.token_projector.modules():
            if isinstance(module, nn.Linear):
                pufferlib.pytorch.layer_init(module, std=0.01)

        # Store action space information
        self.action_space = env.single_action_space
        self.max_action_args = getattr(env, "max_action_args", [])
        self.num_action_types = len(self.max_action_args) if self.max_action_args else env.single_action_space.nvec[0]

        # Single action head for flattened multi-discrete space
        # This matches the format expected by the existing training system
        if hasattr(env, "max_action_args"):
            total_actions = sum(max_arg + 1 for max_arg in env.max_action_args)
        else:
            # Fallback for multi-discrete action space
            total_actions = sum(env.single_action_space.nvec)

        self.actor = nn.ModuleList([pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, total_actions), std=0.01)])

        # Value head
        self.value = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1)

        # Hidden state for recurrence (optional)
        self.hidden_state = None

        # Initialize mixin
        self.init_mixin(**mixin_params)

        logger.info(f"SmolLM2 agent initialized with {sum(p.numel() for p in self.parameters()):,} parameters")

    def _compress_tokens(self, observations: torch.Tensor) -> torch.Tensor:
        """Smart compression of tokens from [B, seq_len, 3] to [B, max_seq_len, 3]."""
        B, seq_len, C = observations.shape

        if seq_len <= self.max_sequence_length:
            return observations

        if self.token_compression == "truncate":
            # Simple truncation - take first N tokens
            return observations[:, : self.max_sequence_length, :]

        elif self.token_compression == "smart_sample":
            # Intelligent sampling based on token importance
            # Prioritize non-empty tokens (those not set to 0xFF)

            # Find valid tokens (not 0xFF which indicates empty/padding)
            coords_byte = observations[..., 0]
            valid_mask = coords_byte != 0xFF  # [B, seq_len]

            compressed_obs = torch.zeros(
                (B, self.max_sequence_length, C), device=observations.device, dtype=observations.dtype
            )

            for batch_idx in range(B):
                valid_indices = torch.where(valid_mask[batch_idx])[0]

                if len(valid_indices) <= self.max_sequence_length:
                    # If we have fewer valid tokens than max, use all valid + padding
                    n_valid = len(valid_indices)
                    compressed_obs[batch_idx, :n_valid, :] = observations[batch_idx, valid_indices, :]
                    # Rest remains zeros (padding)
                else:
                    # Smart sampling: take evenly spaced samples from valid tokens
                    step = len(valid_indices) / self.max_sequence_length
                    selected_indices = [valid_indices[int(i * step)] for i in range(self.max_sequence_length)]
                    compressed_obs[batch_idx, :, :] = observations[batch_idx, selected_indices, :]

            return compressed_obs

        elif self.token_compression == "aggregate":
            # Aggregate tokens by spatial regions
            # Group tokens by spatial proximity and aggregate their values
            # This is more complex but preserves spatial relationships

            # For now, fall back to smart sampling
            return self._compress_tokens_smart_sample(observations)

        else:
            # Default to truncation
            return observations[:, : self.max_sequence_length, :]

    def forward(self, td: TensorDict, state: Optional[dict] = None, action=None) -> TensorDict:
        """Forward pass: process observations through SmolLM2 and predict actions/values."""

        observations = td["env_obs"]

        # BPTT Temporal Flattening: Do this BEFORE TD reshaping for maximum efficiency
        if observations.dim() == 4:  # Training with BPTT: [B, TT, seq_len, 3]
            B, TT, seq_len, channels = observations.shape

            # Key optimization: Flatten temporal dimension into sequence dimension
            # This creates [B, TT*seq_len, 3] - longer sequences but fewer batch items
            observations_temporal = observations.view(B, TT * seq_len, channels)

            # Convert to model dtype, normalize, and apply smart compression to temporal sequences
            # Use token_projector dtype which is synchronized with LLM dtype in initialize_to_environment
            model_dtype = next(self.token_projector.parameters()).dtype
            obs_float = observations_temporal.to(dtype=model_dtype) / 255.0
            obs_float = self._compress_tokens(obs_float)  # [B, max_seq_len, 3]

            # Now handle TD reshaping if needed - but with fewer, longer sequences
            if td.batch_dims > 1:
                # Instead of B*TT short sequences, we want B long sequences
                # But TD expects flattened batch, so we need to adjust
                td = td.reshape(B * TT)
                # Repeat compressed sequences to match TD's expected batch size
                obs_float = obs_float.repeat_interleave(TT, dim=0)  # [B*TT, max_seq_len, 3]
            # If not reshaping TD, keep [B, max_seq_len, 3] - optimal for LLM

        else:  # Inference: [B, seq_len, 3]
            B = observations.shape[0]
            TT = 1
            # Convert to model dtype (synchronized with LLM dtype in initialize_to_environment)
            model_dtype = next(self.token_projector.parameters()).dtype
            obs_float = observations.to(dtype=model_dtype) / 255.0
            obs_float = self._compress_tokens(obs_float)

        # Use mixin to set critical TensorDict fields (after our processing)
        self.set_tensordict_fields(td, observations)

        # Project compressed tokens to LLM embedding space
        # Note: token_projector dtype is synchronized with LLM in initialize_to_environment()
        token_embeddings = self.token_projector(obs_float)  # [batch_size, max_seq_len, hidden_size]

        # Process through LLM - autocast handled automatically by training loop
        outputs = self.llm(
            inputs_embeds=token_embeddings,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use the last hidden state from the hidden_states tuple
        # CausalLMOutputWithPast always has hidden_states when output_hidden_states=True
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # [batch_size, max_seq_len, hidden_size]
        else:
            # This should not happen with output_hidden_states=True, but fallback to logits processing
            raise RuntimeError("SmolLM2 output missing hidden_states - check output_hidden_states=True parameter")

        # Pool over sequence dimension (mean pooling)
        # Note: actor/value head dtypes are synchronized with LLM in initialize_to_environment()
        pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        # Decode actions and value using single flattened head
        logits_list = [head(pooled_hidden) for head in self.actor]
        value = self.value(pooled_hidden)

        # Convert logits list to single tensor for mixin compatibility
        logits = logits_list[0] if len(logits_list) == 1 else torch.cat(logits_list, dim=-1)

        # Use mixin for mode-specific processing
        if action is None:
            # Mixin handles inference mode
            td = self.forward_inference(td, logits, value)
        else:
            # Mixin handles training mode with proper reshaping
            td = self.forward_training(td, action, logits, value)

        return td

    def initialize_to_environment(self, full_action_names: list[str], device: torch.device):
        """Initialize the agent to the current environment."""
        self.device = device
        self.to(device)

        # CRITICAL: Ensure all custom layers match LLM dtype after device placement
        # The LLM loads with fp16, but custom layers default to fp32
        llm_dtype = next(self.llm.parameters()).dtype

        # Convert token projector to match LLM dtype
        self.token_projector = self.token_projector.to(dtype=llm_dtype)
        logger.info(f"Converted token_projector to {llm_dtype}")

        # Convert actor heads to match LLM dtype
        for i, actor_head in enumerate(self.actor):
            self.actor[i] = actor_head.to(dtype=llm_dtype)
        logger.info(f"Converted {len(self.actor)} actor heads to {llm_dtype}")

        # Convert value head to match LLM dtype
        self.value = self.value.to(dtype=llm_dtype)
        logger.info(f"Converted value head to {llm_dtype}")

        # Store action names for debugging
        self.full_action_names = full_action_names

        logger.info(
            f"SmolLM2 initialized to environment with {len(full_action_names)} actions, "
            f"all components using {llm_dtype}"
        )

    def _apply_feature_remapping(self, remap_tensor: torch.Tensor):
        """Apply feature remapping for agent portability."""
        # This is primarily handled at the observation level before feeding to LLM
        pass

    def update_normalization_factors(self, features: dict[str, dict], original_mapping: dict[str, int] | None):
        """Update normalization factors after feature remapping."""
        # SmolLM2 uses its own normalization in token_projector
        pass

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for monitoring using mixin's standard approach."""
        # Use the mixin's implementation for consistency
        return super().compute_weight_metrics(delta)

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss for regularization using mixin's standard approach."""
        # Use the mixin's implementation for consistency across all agents
        return super().l2_init_loss()
