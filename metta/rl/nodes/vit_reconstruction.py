from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourier
from metta.agent.policy import Policy
from metta.rl.nodes.base import NodeBase, NodeConfig
from metta.rl.nodes.registry import NodeSpec
from metta.rl.training import ComponentContext

if TYPE_CHECKING:
    from metta.rl.trainer_config import TrainerConfig
    from metta.rl.training import TrainingEnvironment


class ViTReconstructionLossConfig(NodeConfig):
    # Weights for the two parts of the loss
    id_loss_coef: float = Field(default=0.08, ge=0)
    val_loss_coef: float = Field(default=0.4, ge=0)

    # Architecture parameters (optional, derived from policy/env if not provided)
    num_attribute_classes: Optional[int] = None
    attr_embed_dim: Optional[int] = None
    fourier_freqs: Optional[int] = None
    latent_dim: Optional[int] = None

    decoder_embed_dim: int = Field(default=128)
    decoder_num_heads: int = Field(default=4)

    def create(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        env: "TrainingEnvironment",
        device: torch.device,
        instance_name: str,
    ) -> "ViTReconstructionLoss":
        return ViTReconstructionLoss(policy, trainer_cfg, env, device, instance_name, self)


class ViTReconstructionDecoder(nn.Module):
    def __init__(
        self,
        cfg: ViTReconstructionLossConfig,
        num_attribute_classes: int,
        latent_dim: int,
        attr_embed_dim: int,
        fourier_freqs: int,
        device: torch.device,
    ):
        super().__init__()
        self.cfg = cfg
        self._num_attribute_classes = num_attribute_classes
        self._latent_dim = latent_dim

        # --- Fourier Feature Precomputation ---
        self._num_freqs = fourier_freqs
        self._query_dim = 4 * self._num_freqs

        all_coords = torch.arange(256, dtype=torch.uint8, device=device)
        x_indices = (all_coords & 0x0F).float()
        y_indices = (all_coords >> 4).float()

        mu = 11.0
        x_norm = x_indices / (mu - 1.0) * 2.0 - 1.0
        y_norm = y_indices / (mu - 1.0) * 2.0 - 1.0

        freqs = 2.0 ** torch.arange(self._num_freqs, device=device)

        x_scaled = x_norm.unsqueeze(-1) * freqs.unsqueeze(0)
        y_scaled = y_norm.unsqueeze(-1) * freqs.unsqueeze(0)

        fourier_feats = torch.cat(
            [torch.cos(x_scaled), torch.sin(x_scaled), torch.cos(y_scaled), torch.sin(y_scaled)], dim=-1
        )

        self.register_buffer("fourier_table", fourier_feats)

        # --- Decoder Architecture ---
        self.query_proj = nn.Linear(self._query_dim, self.cfg.decoder_embed_dim)
        self.key_proj = nn.Linear(self._latent_dim, self.cfg.decoder_embed_dim)
        self.value_proj = nn.Linear(self._latent_dim, self.cfg.decoder_embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.decoder_embed_dim, num_heads=self.cfg.decoder_num_heads, batch_first=True
        )

        self.id_head = nn.Linear(self.cfg.decoder_embed_dim, self._num_attribute_classes)
        self.val_head = nn.Linear(self.cfg.decoder_embed_dim, self._num_attribute_classes)

    def forward(
        self, obs_shim_tokens: Tensor, obs_latent_attn: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # ... implementation of forward pass ...
        # returning pred_logits, pred_values, target_ids, target_values, valid_mask

        # 2. Prepare Queries
        coords_byte = obs_shim_tokens[..., 0].long()
        queries_raw = self.fourier_table[coords_byte]
        queries = self.query_proj(queries_raw)

        # 3. Prepare Keys/Values
        keys = self.key_proj(obs_latent_attn)
        values = self.value_proj(obs_latent_attn)

        # 4. Decoder
        attn_output, _ = self.cross_attn(query=queries, key=keys, value=values)

        # 5. Prediction Heads
        pred_logits = self.id_head(attn_output)
        pred_values = self.val_head(attn_output)

        # 6. Target Generation
        input_coords = obs_shim_tokens[..., 0].long()
        input_attrs = obs_shim_tokens[..., 1].long()
        input_vals = obs_shim_tokens[..., 2].float()

        valid_mask = input_coords != 255

        # Safeguard against out-of-bounds attribute IDs (e.g. unknown features mapped to 255)
        # We mask them out so they don't crash one_hot, and they are already excluded by valid_mask
        # if they are padding. If they are valid tokens with OOB attrs, we treat them as class 0
        # but ensure they don't contribute to the loss or targets by updating valid_mask.

        # Update valid_mask to strictly exclude OOB attributes
        is_oob = input_attrs >= self._num_attribute_classes
        valid_mask = valid_mask & (~is_oob)

        # Safe attrs for one_hot - map OOB to 0 (ignored due to match_matrix masking)
        safe_attrs = input_attrs.masked_fill(is_oob, 0)

        match_matrix = input_coords.unsqueeze(2) == input_coords.unsqueeze(1)
        match_matrix = match_matrix & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)

        attrs_one_hot = F.one_hot(safe_attrs, num_classes=self._num_attribute_classes).float()

        target_ids = torch.bmm(match_matrix.float(), attrs_one_hot)
        target_ids = torch.clamp(target_ids, max=1.0)

        weighted_attrs = attrs_one_hot * input_vals.unsqueeze(-1)
        target_values = torch.bmm(match_matrix.float(), weighted_attrs)

        return pred_logits, pred_values, target_ids, target_values, valid_mask


class ViTReconstructionLoss(NodeBase):
    """
    Reconstruction loss for ViT architectures.
    Reconstructs the input sparse observations from the latent representation.
    """

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: "TrainerConfig",
        env: "TrainingEnvironment",
        device: torch.device,
        instance_name: str,
        cfg: ViTReconstructionLossConfig,
    ) -> None:
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)
        self.cfg: ViTReconstructionLossConfig = cfg  # type: ignore
        self.decoder = None

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"obs_shim_tokens", "obs_latent_attn"}

    def _init_decoder(self, latent_dim: int, context: ComponentContext) -> None:
        # 1. Derive num_attribute_classes from environment
        if self.cfg.num_attribute_classes is not None:
            num_attribute_classes = self.cfg.num_attribute_classes
        else:
            # Derive from policy_env_info
            obs_features = self.env.policy_env_info.obs_features
            if not obs_features:
                # Fallback default if no features defined (unlikely)
                num_attribute_classes = 256
            else:
                max_id = max(f.id for f in obs_features)
                num_attribute_classes = max_id + 1

        # 2. Derive architecture params from policy
        attr_embed_dim = self.cfg.attr_embed_dim
        fourier_freqs = self.cfg.fourier_freqs

        if attr_embed_dim is None or fourier_freqs is None:
            # Inspect policy for ObsAttrEmbedFourier
            found_config = None
            for module in self.policy.modules():
                if isinstance(module, ObsAttrEmbedFourier):
                    found_config = module.config
                    break

            if found_config:
                if attr_embed_dim is None:
                    attr_embed_dim = found_config.attr_embed_dim
                if fourier_freqs is None:
                    fourier_freqs = found_config.num_freqs
            else:
                # Fallback defaults if not found
                if attr_embed_dim is None:
                    attr_embed_dim = 12  # Default from config
                if fourier_freqs is None:
                    fourier_freqs = 6  # Default from config

        self.decoder = ViTReconstructionDecoder(
            self.cfg,
            num_attribute_classes=num_attribute_classes,
            latent_dim=latent_dim,
            attr_embed_dim=attr_embed_dim,
            fourier_freqs=fourier_freqs,
            device=self.device,
        ).to(self.device)

        # Register new parameters with the optimizer since they were created after optimizer init
        if context.optimizer is not None:
            context.optimizer.add_param_group({"params": self.decoder.parameters()})

        # Handle distributed training: wrap decoder in DDP if needed
        # The policy is already DDP wrapped by Trainer, but this new module is not.
        if context.distributed.is_distributed:
            # Note: DDP wrapper broadcasts parameters from rank 0 to others on init
            self.decoder = DDP(
                self.decoder,
                device_ids=[context.distributed.config.local_rank],
                output_device=context.distributed.config.local_rank,
                # broadcast_buffers is True by default, which is correct for syncing buffers
            )

        # Attach to policy to ensure parameters are accessible if needed
        # Unwrapping policy if it's wrapped (e.g. DDP)
        # Use a unique name
        (
            self.policy.module if hasattr(self.policy, "module") else self.policy
        ).vit_reconstruction_decoder = self.decoder

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        policy_td = shared_loss_data.get("policy_td")
        if policy_td is None:
            return self._zero(), shared_loss_data, False

        obs_shim_tokens = policy_td.get("obs_shim_tokens")
        obs_latent_attn = policy_td.get("obs_latent_attn")

        if obs_shim_tokens is None or obs_latent_attn is None:
            return self._zero(), shared_loss_data, False

        # Flatten batch dimensions if needed (B, T) -> (B*T)
        if obs_shim_tokens.dim() == 4:
            obs_shim_tokens = obs_shim_tokens.flatten(0, 1)

        if obs_latent_attn.dim() == 3:
            obs_latent_attn = obs_latent_attn.flatten(0, 1)

        if obs_latent_attn.dim() == 2:
            obs_latent_attn = obs_latent_attn.unsqueeze(1)

        # Lazy initialization of decoder
        if self.decoder is None:
            self._init_decoder(latent_dim=obs_latent_attn.shape[-1], context=context)

        # Run decoder
        pred_logits, pred_values, target_ids, target_values, valid_mask = self.decoder(obs_shim_tokens, obs_latent_attn)

        # 7. Compute Losses

        # Part A: Attribute ID NodeBase (BCE)
        loss_id = F.binary_cross_entropy_with_logits(pred_logits, target_ids, reduction="none")

        # Use the derived num_attribute_classes for normalization
        # Handle both wrapped and unwrapped decoder
        num_classes = (self.decoder.module if isinstance(self.decoder, DDP) else self.decoder)._num_attribute_classes
        mask_expanded = valid_mask.unsqueeze(-1)
        loss_id = (loss_id * mask_expanded).sum() / (mask_expanded.sum() * num_classes + 1e-6)

        # Part B: Attribute Value NodeBase (Masked MSE)
        sq_error = (pred_values - target_values) ** 2
        val_loss_mask = mask_expanded * target_ids
        loss_val = (sq_error * val_loss_mask).sum() / (val_loss_mask.sum() + 1e-6)

        # Total NodeBase
        total_loss = self.cfg.id_loss_coef * loss_id + self.cfg.val_loss_coef * loss_val

        # Logging
        self.loss_tracker["vit_recon_loss"].append(total_loss.item())
        self.loss_tracker["vit_id_loss"].append(loss_id.item())
        self.loss_tracker["vit_val_loss"].append(loss_val.item())

        return total_loss, shared_loss_data, False


NODE_SPECS = [
    NodeSpec(
        key="vit_reconstruction",
        config_cls=ViTReconstructionLossConfig,
        default_enabled=False,
        has_rollout=False,
        has_train=True,
    )
]
