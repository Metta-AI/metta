import logging
from typing import Optional

import pufferlib.pytorch
import torch
import torch.nn.functional as F
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
        freeze_llm: bool = False,
        use_lora: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Extract mixin parameters
        mixin_params = self.extract_mixin_params(kwargs)

        # Initialize the SmolLM2 model
        logger.info(f"Loading SmolLM2 model: {model_name}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

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

        # Action and value heads
        self.action_space = env.single_action_space

        # Calculate total flattened action space
        if hasattr(env, "max_action_args"):
            total_actions = sum(max_arg + 1 for max_arg in env.max_action_args)
        else:
            # Fallback for multi-discrete action space
            total_actions = sum(env.single_action_space.nvec)

        self.actor = nn.ModuleList([pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, total_actions), std=0.01)])
        self.value = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1)

        # Hidden state for recurrence (optional)
        self.hidden_state = None

        # Initialize mixin
        self.init_mixin(**mixin_params)

        logger.info(f"SmolLM2 agent initialized with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, td: TensorDict, state: Optional[dict] = None, action=None) -> TensorDict:
        """Forward pass: process observations through SmolLM2 and predict actions/values."""

        observations = td["env_obs"]

        # Set critical TensorDict fields
        B, TT = self.set_tensordict_fields(td, observations)

        # Handle BPTT reshaping if needed
        if td.batch_dims > 1:
            total_batch = B * TT
            td = td.reshape(total_batch)

        # Process observations
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # Reshape observations for processing: [B*TT, seq_len, 3]
        if observations.dim() == 4:
            observations = observations.view(B * TT, observations.shape[2], 3)
        elif observations.dim() == 3:
            observations = observations.view(B, observations.shape[1], 3)

        # Convert byte tokens to float and normalize
        obs_float = observations.float() / 255.0

        # Project tokens to LLM embedding space
        token_embeddings = self.token_projector(obs_float)  # [B*TT, seq_len, hidden_size]

        # Process through LLM
        with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for stability
            outputs = self.llm(
                inputs_embeds=token_embeddings,
                output_hidden_states=True,
                return_dict=True,
            )

        # Use the last hidden state, pooled across sequence dimension
        hidden_states = outputs.hidden_states[-1]  # [B*TT, seq_len, hidden_size]

        # Pool over sequence dimension (mean pooling)
        pooled_hidden = hidden_states.mean(dim=1)  # [B*TT, hidden_size]

        # Decode actions and value
        logits_list = [head(pooled_hidden) for head in self.actor]
        value = self.value(pooled_hidden)

        # Sample actions
        actions, log_probs, entropies, full_log_probs = self._sample_actions(logits_list)

        # Convert actions to tensor format expected by environment
        if len(actions) >= 2:
            actions_tensor = torch.stack([actions[0], actions[1]], dim=-1)
        else:
            # For single-head action space, duplicate or pad
            if len(actions) == 1:
                # Get action type and parameter from flattened action
                action_indices = actions[0]
                actions_tensor = self._convert_logit_index_to_action(action_indices)
            else:
                actions_tensor = torch.zeros((B * TT, 2), dtype=torch.int32, device=observations.device)

        actions_tensor = actions_tensor.to(dtype=torch.int32)

        # Update TensorDict
        if action is None:
            # Inference mode
            td["actions"] = actions_tensor
            td["act_log_prob"] = log_probs.mean(dim=-1) if log_probs.dim() > 1 else log_probs
            td["values"] = value.flatten()
            td["full_log_probs"] = full_log_probs
        else:
            # Training mode
            td["act_log_prob"] = log_probs.mean(dim=-1) if log_probs.dim() > 1 else log_probs
            td["entropy"] = entropies.sum(dim=-1) if entropies.dim() > 1 else entropies
            td["value"] = value.flatten()
            td["full_log_probs"] = full_log_probs
            td = td.reshape(B, TT)

        return td

    def _sample_actions(self, logits_list: list[torch.Tensor]):
        """Samples discrete actions from logits and computes log-probs and entropy."""
        actions, selected_log_probs, entropies, full_log_probs = [], [], [], []

        # Handle both single and multi-head action spaces
        if len(logits_list) == 1:
            # Single head for flattened action space
            logits = logits_list[0]
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            action = torch.multinomial(probs, 1).squeeze(-1)
            batch_idx = torch.arange(action.shape[0], device=action.device)

            selected_log_prob = log_probs[batch_idx, action]
            entropy = -(probs * log_probs).sum(dim=-1)

            return [action], selected_log_prob, entropy, log_probs
        else:
            # Multi-head action space
            max_actions = max(logits.shape[1] for logits in logits_list)

            for logits in logits_list:
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()

                action = torch.multinomial(probs, 1).squeeze(-1)
                batch_idx = torch.arange(action.shape[0], device=action.device)

                selected_log_prob = log_probs[batch_idx, action]
                entropy = -(probs * log_probs).sum(dim=-1)

                actions.append(action)
                selected_log_probs.append(selected_log_prob)
                entropies.append(entropy)

                # Pad log_probs to max size
                pad_width = max_actions - log_probs.shape[1]
                full_log_probs.append(F.pad(log_probs, (0, pad_width), value=float("-inf")))

            return (
                actions,
                torch.stack(selected_log_probs, dim=-1),
                torch.stack(entropies, dim=-1),
                torch.stack(full_log_probs, dim=-1),
            )

    def initialize_to_environment(self, full_action_names: list[str], device: torch.device):
        """Initialize the agent to the current environment."""
        self.device = device
        self.to(device)

        # Store action names for debugging
        self.full_action_names = full_action_names

        logger.info(f"SmolLM2 initialized to environment with {len(full_action_names)} actions")

    def _apply_feature_remapping(self, remap_tensor: torch.Tensor):
        """Apply feature remapping for agent portability."""
        # This is primarily handled at the observation level before feeding to LLM
        pass

    def update_normalization_factors(self, features: dict[str, dict], original_mapping: dict[str, int] | None):
        """Update normalization factors after feature remapping."""
        # SmolLM2 uses its own normalization in token_projector
        pass

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Compute weight metrics for monitoring."""
        metrics = []

        # Metrics for token projector
        for i, layer in enumerate(self.token_projector.modules()):
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data
                metrics.append(
                    {
                        "name": f"token_projector.{i}",
                        "mean": weight.mean().item(),
                        "std": weight.std().item(),
                        "max": weight.max().item(),
                        "min": weight.min().item(),
                    }
                )

        # Metrics for action heads
        for i, head in enumerate(self.actor):
            weight = head.weight.data
            metrics.append(
                {
                    "name": f"actor.{i}",
                    "mean": weight.mean().item(),
                    "std": weight.std().item(),
                    "max": weight.max().item(),
                    "min": weight.min().item(),
                }
            )

        # Metrics for value head
        weight = self.value.weight.data
        metrics.append(
            {
                "name": "value",
                "mean": weight.mean().item(),
                "std": weight.std().item(),
                "max": weight.max().item(),
                "min": weight.min().item(),
            }
        )

        return metrics

    def l2_init_loss(self) -> torch.Tensor:
        """Calculate L2 initialization loss for regularization."""
        loss = torch.tensor(0.0, device=self.device)

        # Only regularize the heads, not the pre-trained LLM
        for module in [*self.token_projector.modules(), *self.actor, self.value]:
            if hasattr(module, "_initial_weight") and hasattr(module, "weight"):
                loss += ((module.weight - module._initial_weight) ** 2).sum()
            if hasattr(module, "_initial_bias") and hasattr(module, "bias") and module.bias is not None:
                loss += ((module.bias - module._initial_bias) ** 2).sum()

        return loss * 0.001  # Small regularization factor

    def _store_initial_weights(self):
        """Store initial weights for L2 regularization."""
        for module in [*self.token_projector.modules(), *self.actor, self.value]:
            if isinstance(module, nn.Linear):
                module._initial_weight = module.weight.data.clone()
                if module.bias is not None:
                    module._initial_bias = module.bias.data.clone()

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete logit indices to (action_type, action_param) pairs."""
        if hasattr(self, "action_index_tensor"):
            return self.action_index_tensor[logit_indices]
        else:
            # Fallback: assume simple action space
            # This would need proper implementation based on action space structure
            return torch.stack([logit_indices, torch.zeros_like(logit_indices)], dim=-1)
