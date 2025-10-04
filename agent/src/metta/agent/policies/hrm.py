from typing import List

from metta.agent.components.actor import ActionProbsConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.hrm import (
    HRMActorConfig,
    HRMCriticConfig,
    HRMObsEncodingConfig,
    HRMReasoningConfig,
)
from metta.agent.policy import PolicyArchitecture
from metta.agent.policy_auto_builder import PolicyAutoBuilder


class HRMPolicy(PolicyAutoBuilder):
    """Component-based Hierarchical Reasoning Model (HRM) policy implementation."""

    def __init__(self, env_metadata):
        _embed_dim = 256
        _action_embed_dim = 64
        _num_actions = 100

        config = PolicyArchitecture(
            class_path="metta.agent.policy_auto_builder.PolicyAutoBuilder",
            components=[
                # Observation encoding
                HRMObsEncodingConfig(
                    in_key="env_obs",
                    out_key="hrm_obs_encoded",
                    embed_dim=_embed_dim,
                    out_width=11,
                    out_height=11,
                ),
                # Hierarchical reasoning
                HRMReasoningConfig(
                    in_key="hrm_obs_encoded",
                    out_key="hrm_reasoning",
                    embed_dim=_embed_dim,
                    num_layers=6,
                    num_heads=8,
                ),
                # Critic (value function)
                HRMCriticConfig(
                    in_key="hrm_reasoning",
                    out_key="values",
                    embed_dim=_embed_dim,
                ),
                # Actor components
                HRMActorConfig(
                    in_key="hrm_reasoning",
                    out_key="logits",
                    embed_dim=_embed_dim,
                    num_actions=_num_actions,
                ),
            ],
            action_probs_config=ActionProbsConfig(in_key="logits"),
        )

        super().__init__(env_metadata, config)


class HRMPolicyConfig(PolicyArchitecture):
    """Component-based Hierarchical Reasoning Model (HRM) policy configuration."""

    class_path: str = "metta.agent.policies.hrm.HRMPolicy"

    _embed_dim = 256
    _action_embed_dim = 64
    _num_actions = 100

    components: List[ComponentConfig] = [
        # Observation encoding
        HRMObsEncodingConfig(
            in_key="env_obs",
            out_key="hrm_obs_encoded",
            embed_dim=_embed_dim,
            out_width=11,
            out_height=11,
        ),
        # Hierarchical reasoning
        HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="hrm_reasoning",
            embed_dim=_embed_dim,
            num_layers=6,
            num_heads=8,
        ),
        # Critic (value function)
        HRMCriticConfig(
            in_key="hrm_reasoning",
            out_key="values",
            embed_dim=_embed_dim,
        ),
        # Actor components
        HRMActorConfig(
            in_key="hrm_reasoning",
            out_key="logits",
            embed_dim=_embed_dim,
            num_actions=_num_actions,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class HRMCompactConfig(PolicyArchitecture):
    """Compact version of HRM with smaller dimensions for faster training."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 128
    _action_embed_dim = 32
    _num_actions = 100

    components: List[ComponentConfig] = [
        # Observation encoding
        HRMObsEncodingConfig(
            in_key="env_obs",
            out_key="hrm_obs_encoded",
            embed_dim=_embed_dim,
            out_width=11,
            out_height=11,
        ),
        # Hierarchical reasoning (fewer layers and heads)
        HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="hrm_reasoning",
            embed_dim=_embed_dim,
            num_layers=3,
            num_heads=4,
        ),
        # Critic (value function)
        HRMCriticConfig(
            in_key="hrm_reasoning",
            out_key="values",
            embed_dim=_embed_dim,
        ),
        # Actor components
        HRMActorConfig(
            in_key="hrm_reasoning",
            out_key="logits",
            embed_dim=_embed_dim,
            num_actions=_num_actions,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class HRMLargeConfig(PolicyArchitecture):
    """Large version of HRM with more capacity for complex reasoning."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _embed_dim = 512
    _action_embed_dim = 128
    _num_actions = 100

    components: List[ComponentConfig] = [
        # Observation encoding
        HRMObsEncodingConfig(
            in_key="env_obs",
            out_key="hrm_obs_encoded",
            embed_dim=_embed_dim,
            out_width=11,
            out_height=11,
        ),
        # Hierarchical reasoning (more layers and heads)
        HRMReasoningConfig(
            in_key="hrm_obs_encoded",
            out_key="hrm_reasoning",
            embed_dim=_embed_dim,
            num_layers=12,
            num_heads=16,
        ),
        # Critic (value function)
        HRMCriticConfig(
            in_key="hrm_reasoning",
            out_key="values",
            embed_dim=_embed_dim,
        ),
        # Actor components
        HRMActorConfig(
            in_key="hrm_reasoning",
            out_key="logits",
            embed_dim=_embed_dim,
            num_actions=_num_actions,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


# Default HRM configuration
HRMDefaultConfig = HRMPolicyConfig
