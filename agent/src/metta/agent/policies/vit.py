import types
from typing import List, Optional

import torch
import torch.nn as nn
from cortex.stacks import build_cortex_auto_config
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.components.diversity_injection import DiversityInjectionConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import Policy, PolicyArchitecture
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol


def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
    """Forward pass for the ViT policy with dynamics heads."""
    self.network()(td)
    self.action_probs(td, action)

    if "values" in td.keys():
        td["values"] = td["values"].flatten()

    # Dynamics/Muesli predictions - only if modules were created
    if hasattr(self, "returns_pred") and self.returns_pred is not None:
        td["pred_input"] = torch.cat([td["core"], td["logits"]], dim=-1)
        self.returns_pred(td)
        self.reward_pred(td)

        # K-step dynamics unrolling for Muesli
        if action is not None and self.unroll_steps > 0:
            _compute_unrolled_predictions(self, td, action)

    return td


def _compute_unrolled_predictions(self, td: TensorDict, actions: torch.Tensor) -> None:
    """Compute K-step unrolled predictions and write to TensorDict.

    Output shapes are (B*T, K, ...) to match TensorDict batch dimension.
    Only the first T_eff positions have valid predictions; rest are zero-padded.
    """
    K = self.unroll_steps
    hidden = td["core"]  # (B*T, H)

    B = int(td["batch"][0].item())
    T = int(td["bptt"][0].item())
    BT = B * T

    if T <= K:
        return  # Not enough timesteps

    T_eff = T - K

    # Reshape to (B, T, ...) for temporal indexing
    hidden_bt = hidden.view(B, T, -1)
    actions_bt = actions.view(B, T)
    current_h = hidden_bt[:, :T_eff]  # (B, T_eff, H)

    unrolled_logits_list: list[torch.Tensor] = []
    unrolled_rewards_list: list[torch.Tensor] = []
    unrolled_returns_list: list[torch.Tensor] = []

    for k in range(K):
        step_actions = actions_bt[:, k : k + T_eff]

        # Dynamics: (h_k, a_k) -> (h_{k+1}, r_k)
        next_h, r_pred_k = _dynamics_step(self, current_h, step_actions)

        # Prediction: h_{k+1} -> (pi_{k+1}, v_{k+1})
        p_logits_k, v_pred_k = _prediction_step(self, next_h)

        unrolled_logits_list.append(p_logits_k)  # (B, T_eff, A)
        unrolled_rewards_list.append(r_pred_k.squeeze(-1))  # (B, T_eff)
        unrolled_returns_list.append(v_pred_k.squeeze(-1))  # (B, T_eff)
        current_h = next_h

    # Stack along K dimension: (B, T_eff, K, ...)
    stacked_logits = torch.stack(unrolled_logits_list, dim=2)  # (B, T_eff, K, A)
    stacked_rewards = torch.stack(unrolled_rewards_list, dim=2)  # (B, T_eff, K)
    stacked_returns = torch.stack(unrolled_returns_list, dim=2)  # (B, T_eff, K)

    # Pad T_eff to T so we can reshape to (B*T, K, ...)
    A = stacked_logits.shape[-1]
    padded_logits = torch.zeros(B, T, K, A, device=hidden.device, dtype=hidden.dtype)
    padded_rewards = torch.zeros(B, T, K, device=hidden.device, dtype=hidden.dtype)
    padded_returns = torch.zeros(B, T, K, device=hidden.device, dtype=hidden.dtype)

    padded_logits[:, :T_eff] = stacked_logits
    padded_rewards[:, :T_eff] = stacked_rewards
    padded_returns[:, :T_eff] = stacked_returns

    # Reshape to (B*T, K, ...) to match TensorDict batch dimension
    td["unrolled_logits"] = padded_logits.view(BT, K, A)
    td["unrolled_rewards"] = padded_rewards.view(BT, K)
    td["unrolled_returns"] = padded_returns.view(BT, K)


def _dynamics_step(self, hidden: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamics function: (h_t, a_t) -> (h_{t+1}, r_t)"""
    if action.dim() > 1 and action.shape[-1] == 1:
        action = action.squeeze(-1)

    # One-hot encode actions
    if action.dtype in (torch.long, torch.int32, torch.int64):
        action_emb = torch.nn.functional.one_hot(action.long(), num_classes=self.num_actions).float()
    else:
        action_emb = action

    dyn_input = torch.cat([hidden, action_emb], dim=-1)
    output = self.dynamics_model(dyn_input)

    next_hidden = output[..., :-1]
    reward = output[..., -1:]

    return next_hidden, reward


def _prediction_step(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prediction function: h_t -> (pi_logits_t, v_t)"""
    td = TensorDict({"core": hidden}, batch_size=hidden.shape[0], device=hidden.device)

    self.components["actor_mlp"](td)
    self.components["actor_head"](td)
    logits = td["logits"]

    pred_input = torch.cat([hidden, logits], dim=-1)
    td["pred_input"] = pred_input
    self.returns_pred(td)
    returns = td["returns_pred"]

    return logits, returns


class ViTDefaultConfig(PolicyArchitecture):
    """Speed-optimized ViT variant with lighter token embeddings and attention stack.

    The trunk uses Axon blocks (post-up experts with residual connections) for efficient
    feature processing. Configure trunk depth, layer normalization, and hidden dimension
    scaling independently.
    """

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 64
    _actor_hidden = 256

    # Whether training passes cached pre-state to the Cortex core
    pass_state_during_training: bool = False
    _critic_hidden = 512

    # Trunk configuration
    # Number of Axon layers in the trunk (default: 16 for large model)
    core_resnet_layers: int = 1
    # Enable layer normalization after each trunk layer
    core_use_layer_norm: bool = True

    # Diversity injection - auto-expands exploration when gradients vanish
    # Enable with losses.diversity.enabled=True losses.diversity.diversity_coef=0.01
    use_diversity_injection: bool = False

    # Dynamics/Muesli unrolling - number of steps to unroll for model-based prediction
    # Set to > 0 to enable dynamics modules for Muesli loss
    unroll_steps: int = 0

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="obs_latent_attn",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        CortexTDConfig(
            in_key="obs_latent_attn",
            out_key="core",
            d_hidden=_latent_dim,
            out_features=_latent_dim,
            key_prefix="vit_cortex_state",
            stack_cfg=build_cortex_auto_config(
                d_hidden=_latent_dim,
                num_layers=1,  # Default to 1, can be overridden
                pattern="A",  # Axon blocks provide residual-like connections
                post_norm=True,
            ),
            pass_state_during_training=pass_state_during_training,
        ),
        MLPConfig(
            in_key="core",
            out_key="actor_hidden",
            name="actor_mlp",
            in_features=_latent_dim,
            hidden_features=[_actor_hidden],
            out_features=_actor_hidden,
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_latent_dim,
            out_features=1,
            hidden_features=[_critic_hidden],
        ),
        ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=_actor_hidden, name="actor_head"),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, policy_env_info: PolicyEnvInterface) -> Policy:
        # Apply trunk configuration dynamically to the Cortex component
        cortex = next(c for c in self.components if isinstance(c, CortexTDConfig))

        # Rebuild stack config with current parameters
        cortex.stack_cfg = build_cortex_auto_config(
            d_hidden=int(self._latent_dim),
            num_layers=self.core_resnet_layers,
            pattern="A",
            post_norm=self.core_use_layer_norm,
        )

        # Conditionally add diversity injection after Cortex
        if self.use_diversity_injection:
            # Find Cortex index and insert diversity injection after it
            cortex_idx = next(i for i, c in enumerate(self.components) if isinstance(c, CortexTDConfig))
            # Check if already inserted
            if not any(isinstance(c, DiversityInjectionConfig) for c in self.components):
                self.components.insert(
                    cortex_idx + 1,
                    DiversityInjectionConfig(
                        in_key="core",
                        out_key="core",  # in-place replacement
                        name="diversity_injection",
                    ),
                )

        AgentClass = load_symbol(self.class_path)
        if not isinstance(AgentClass, type):
            raise TypeError(f"Loaded symbol {self.class_path} is not a class")

        policy = AgentClass(policy_env_info, self)
        policy.num_actions = policy_env_info.action_space.n
        policy.unroll_steps = self.unroll_steps

        # Only create dynamics modules if unroll_steps > 0
        if self.unroll_steps > 0:
            latent_dim = int(self._latent_dim)
            num_actions = int(policy.num_actions)

            # Dynamics Model: (Hidden + Action) -> (Hidden + Reward)
            dyn_input_dim = int(latent_dim + num_actions)
            dyn_output_dim = int(latent_dim + 1)

            dynamics_net = nn.Sequential(
                nn.Linear(dyn_input_dim, 256),
                nn.SiLU(),
                nn.Linear(256, dyn_output_dim),
            )
            policy.dynamics_model = dynamics_net

            # Returns/Reward Prediction Heads (for Muesli)
            # Input: Core + Logits
            pred_input_dim = int(latent_dim + num_actions)

            returns_module = nn.Linear(pred_input_dim, 1)
            reward_module = nn.Linear(pred_input_dim, 1)

            policy.returns_pred = TDM(returns_module, in_keys=["pred_input"], out_keys=["returns_pred"])
            policy.reward_pred = TDM(reward_module, in_keys=["pred_input"], out_keys=["reward_pred"])

        # Attach methods
        policy.forward = types.MethodType(forward, policy)

        return policy
