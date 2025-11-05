"""Policy helper utilities, including LSTM state adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch

from mettagrid.policy.policy import MultiAgentPolicy as Policy
from mettagrid.policy.policy import TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import ActionsConfig

_POLICY_CLASS_SHORTHAND: dict[str, str] = {
    "random": "mettagrid.policy.random.RandomMultiAgentPolicy",
    "noop": "mettagrid.policy.noop.NoopPolicy",
    "stateless": "mettagrid.policy.stateless.StatelessPolicy",
    "token": "mettagrid.policy.token.TokenPolicy",
    "lstm": "mettagrid.policy.lstm.LSTMPolicy",
    "scripted_baseline": "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
    "scripted_unclipping": "cogames.policy.scripted_agent.unclipping_agent.UnclippingPolicy",
    # Backwards compatibility aliases
    "baseline": "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
    "simple_baseline": "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
    "unclipping": "cogames.policy.scripted_agent.unclipping_agent.UnclippingPolicy",
}


def initialize_or_load_policy(
    policy_class_path: str,
    policy_data_path: Optional[str],
    actions: "ActionsConfig",
    device: "torch.device | None" = None,
    env: Optional[Any] = None,
    policy_env_info: Optional[PolicyEnvInterface] = None,
) -> Policy:
    """Initialize a policy from its class path and optionally load weights.

    Args:
        policy_class_path: Full class path to the policy
        policy_data_path: Optional path to policy checkpoint
        actions: Actions configuration from the environment
        device: Optional device to use for policy
        env: Optional environment instance (required for some policies like LSTM)
        policy_env_info: Optional PolicyEnvInterface (created from env if not provided)

    Returns:
        Initialized policy instance
    """
    import inspect

    policy_class = load_symbol(policy_class_path)

    # Create policy_env_info if not provided but env is provided
    if policy_env_info is None and env is not None:
        # Try to create policy_env_info from env
        if hasattr(env, "env_cfg"):
            from mettagrid.config.mettagrid_config import MettaGridConfig

            if isinstance(env.env_cfg, MettaGridConfig):
                policy_env_info = PolicyEnvInterface.from_mg_cfg(env.env_cfg)
        elif hasattr(env, "driver_env") and hasattr(env.driver_env, "env_cfg"):
            from mettagrid.config.mettagrid_config import MettaGridConfig

            if isinstance(env.driver_env.env_cfg, MettaGridConfig):
                policy_env_info = PolicyEnvInterface.from_mg_cfg(env.driver_env.env_cfg)

    # Create policy_env_info from actions if still not provided (backward compatibility)
    if policy_env_info is None and actions is not None:
        # Try to create a minimal PolicyEnvInterface from actions
        # This is a fallback for backward compatibility
        from mettagrid.config.mettagrid_config import MettaGridConfig

        mg_cfg = MettaGridConfig()
        mg_cfg.game.actions = actions
        policy_env_info = PolicyEnvInterface.from_mg_cfg(mg_cfg)

    # Check if policy requires obs_shape parameter (like LSTMPolicy)
    sig = inspect.signature(policy_class.__init__)
    params = list(sig.parameters.keys())
    # Remove 'self' from params
    params = [p for p in params if p != "self"]

    # Check for special cases
    first_param = params[0] if params else None
    has_policy_env_info_param = "policy_env_info" in params
    has_obs_shape_param = "obs_shape" in params
    has_actions_param = "actions" in params or "actions_cfg" in params

    # Handle policies that take no arguments (like scripted agents)
    if not params:
        policy = policy_class()  # type: ignore[misc]
    elif has_obs_shape_param:
        # LSTMPolicy-style: requires (actions_cfg, obs_shape, device, policy_env_info)
        if device is None:
            device = torch.device("cpu")

        # Extract obs_shape and actions_cfg from policy_env_info or env
        if policy_env_info is not None:
            obs_shape = policy_env_info.observation_space.shape
            actions_cfg = policy_env_info.actions
        elif env is not None:
            obs_shape = env.single_observation_space.shape
            actions_cfg = actions if actions is not None else MettaGridConfig().game.actions
        else:
            raise TypeError(f"{policy_class_path} requires either policy_env_info or env to determine obs_shape")

        if policy_env_info is None:
            raise TypeError(f"{policy_class_path} requires policy_env_info parameter")

        policy = policy_class(actions_cfg, obs_shape, device, policy_env_info)  # type: ignore[misc]
    elif first_param == "policy_env_info":
        # Policy subclasses that take policy_env_info as first parameter
        # (like FastPolicy, PufferPolicy, TransformerPolicy)
        if policy_env_info is None:
            raise TypeError(f"{policy_class_path} requires policy_env_info parameter")
        # Check if there's a config parameter
        if "config" in params:
            policy = policy_class(policy_env_info)  # type: ignore[misc]
        else:
            policy = policy_class(policy_env_info)  # type: ignore[misc]
    elif has_policy_env_info_param:
        # Policy base class: requires (policy_env_info)
        if policy_env_info is None:
            raise TypeError(f"{policy_class_path} requires policy_env_info parameter")
        policy = policy_class(policy_env_info)  # type: ignore[misc]
    elif has_actions_param:
        # Legacy policy: requires actions (but this shouldn't happen for TrainablePolicy)
        if actions is None:
            raise TypeError(f"{policy_class_path} requires actions parameter")
        policy = policy_class(actions)  # type: ignore[misc]
    else:
        # Try policy_env_info first, then fall back to actions
        if policy_env_info is not None:
            try:
                policy = policy_class(policy_env_info)  # type: ignore[misc]
            except TypeError:
                if actions is not None:
                    policy = policy_class(actions)  # type: ignore[misc]
                else:
                    msg = f"{policy_class_path} requires either policy_env_info or actions parameter"
                    raise TypeError(msg) from None
        elif actions is not None:
            policy = policy_class(actions)  # type: ignore[misc]
        else:
            raise TypeError(f"{policy_class_path} requires either policy_env_info or actions parameter")

    if policy_data_path:
        if not isinstance(policy, TrainablePolicy):
            raise TypeError("Policy data provided, but the selected policy does not support loading checkpoints.")

        policy.load_policy_data(policy_data_path)
    return policy


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path.

    Args:
        policy: Either a shorthand like "random", "stateless", "token", "lstm" or a full class path.

    Returns:
        Full class path to the policy.
    """
    full_path = _POLICY_CLASS_SHORTHAND.get(policy, policy)

    # Will raise an error if invalid
    _ = load_symbol(full_path)
    return full_path


def get_policy_class_shorthand(policy: str) -> Optional[str]:
    return {v: k for k, v in _POLICY_CLASS_SHORTHAND.items()}.get(policy)


_NOT_CHECKPOINT_PATTERNS = (
    r"trainer_state\.pt",  # trainer state file
    r"model_\d{6}\.pt",  # matches model_000001.pt etc
)


def find_policy_checkpoints(checkpoints_path: Path, env_name: Optional[str] = None) -> list[Path]:
    checkpoints = []
    if env_name:
        # Try to find the final checkpoint
        # PufferLib saves checkpoints in data_dir/env_name/
        checkpoint_dir = checkpoints_path / env_name
        if checkpoint_dir.exists():
            checkpoints = checkpoint_dir.glob("*.pt")

    # Fallback: also check directly in checkpoints_path
    if not checkpoints and checkpoints_path.exists():
        checkpoints = checkpoints_path.glob("*.pt")
    return [
        p
        for p in sorted(checkpoints, key=lambda c: c.stat().st_mtime)
        if not any(re.fullmatch(pattern, p.name) for pattern in _NOT_CHECKPOINT_PATTERNS)
    ]


def resolve_policy_data_path(
    policy_data_path: Optional[str],
) -> Optional[str]:
    """Resolve a checkpoint path if provided.

    If the supplied path does not exist locally and AWS policy storage is configured,
    this will attempt to download the checkpoint into the requested location.
    """

    if policy_data_path is None:
        return None

    path = Path(policy_data_path).expanduser()
    if path.is_file():
        return str(path)

    if path.is_dir():
        checkpoints = find_policy_checkpoints(path)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
        return str(checkpoints[-1])

    if path.exists():  # Non-pt extension but present
        return str(path)

    raise FileNotFoundError(f"Checkpoint path not found: {path}")


LSTMStateTuple = Tuple[torch.Tensor, torch.Tensor]
LSTMStateDict = Dict[str, torch.Tensor]


def _canonical_component(component: torch.Tensor, expected_layers: Optional[int]) -> torch.Tensor:
    """Return a ``(layers, batch, hidden)`` tensor, adding axes as needed."""
    if component.dim() > 3:
        msg = f"Expected tensor with <=3 dims, got {component.dim()}"
        raise ValueError(msg)

    while component.dim() < 3:
        component = component.unsqueeze(0)

    if expected_layers is not None:
        if component.shape[0] != expected_layers and component.shape[1] == expected_layers:
            component = component.transpose(0, 1)
        if component.shape[0] != expected_layers:
            msg = f"Hidden state has unexpected layer dimension. Expected {expected_layers}, got {component.shape[0]}."
            raise ValueError(msg)

    return component.contiguous()


@dataclass
class LSTMState:
    """Canonical representation of an LSTM hidden/cell state."""

    hidden: torch.Tensor
    cell: torch.Tensor

    @classmethod
    def from_tuple(
        cls,
        state: Optional[LSTMStateTuple],
        expected_layers: Optional[int],
    ) -> Optional["LSTMState"]:
        if state is None:
            return None
        hidden, cell = state
        return cls(
            _canonical_component(hidden, expected_layers),
            _canonical_component(cell, expected_layers),
        )

    @classmethod
    def from_dict(
        cls,
        state: LSTMStateDict,
        expected_layers: Optional[int],
    ) -> Optional["LSTMState"]:
        if not state:
            return None
        hidden = state.get("lstm_h")
        cell = state.get("lstm_c")
        if hidden is None or cell is None:
            return None
        return cls(
            _canonical_component(hidden, expected_layers),
            _canonical_component(cell, expected_layers),
        )

    @classmethod
    def from_any(
        cls,
        state: Optional[Union["LSTMState", LSTMStateTuple, LSTMStateDict]],
        expected_layers: Optional[int],
    ) -> Optional["LSTMState"]:
        if state is None:
            return None
        if isinstance(state, LSTMState):
            return state
        if isinstance(state, dict):
            return cls.from_dict(state, expected_layers)
        if isinstance(state, tuple):
            return cls.from_tuple(state, expected_layers)
        msg = f"Unsupported LSTM state container type: {type(state)!r}"
        raise TypeError(msg)

    def to_tuple(self) -> LSTMStateTuple:
        return self.hidden, self.cell

    def write_dict(self, target: LSTMStateDict) -> None:
        """Populate ``target`` with tensors in batch-major form."""
        target.clear()
        target["lstm_h"] = self.hidden.transpose(0, 1).contiguous().detach()
        target["lstm_c"] = self.cell.transpose(0, 1).contiguous().detach()

    def detach(self) -> "LSTMState":
        return LSTMState(self.hidden.detach(), self.cell.detach())
