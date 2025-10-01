"""CoGames-specific MettaGrid environment helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from mettagrid import MettaGridConfig, MettaGridEnv, dtype_actions


class HierarchicalActionMettaGridEnv(MettaGridEnv):
    """Expose verb-specific action arguments as independent MultiDiscrete bins.

    The underlying MettaGrid C++ environment expects actions shaped like
    ``(num_agents, 2)`` where ``[..., 0]`` selects the verb and ``[..., 1]``
    encodes that verb's argument. Each verb supports a different argument
    range, so sampling both dimensions independently from the raw
    ``MultiDiscrete([num_verbs, max_arg])`` space regularly yields invalid
    pairs.

    This wrapper expands the action space to:

    ``MultiDiscrete([num_verbs] + [max_arg_i + 1 for i in verbs])``

    Policies (and PuffeRL) can then sample each dimension independently. When
    stepping the environment we select the argument corresponding to the chosen
    verb and forward the compact action to the base implementation.
    """

    def __init__(
        self,
        env_cfg: MettaGridConfig,
        render_mode: Optional[str] = None,
        *,
        buf: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # ``MettaGridEnv`` does not accept shared-memory buffers during
        # construction; these are wired up later via ``set_buffers``. We accept
        # ``buf`` here so callers (e.g., PuffeRL) can keep passing it, but we
        # intentionally ignore it for the super() call.
        self._hierarchical_action_space: Optional[spaces.MultiDiscrete] = None
        super().__init__(env_cfg=env_cfg, render_mode=render_mode, **kwargs)

        base_space = super().single_action_space
        self._base_action_dims = int(base_space.nvec[0])
        self._max_action_args = np.asarray(super().max_action_args, dtype=np.int64)
        self._arg_dims = (self._max_action_args + 1).tolist()

        nvec = np.array([self._base_action_dims] + self._arg_dims, dtype=np.int64)
        self._hierarchical_action_space = spaces.MultiDiscrete(nvec)

        # Ensure Gym-style attribute access works (e.g. env.action_space.contains)
        self.action_space = self._hierarchical_action_space

    # ---------------------------------------------------------------------
    # Action space overrides
    # ---------------------------------------------------------------------
    @property  # type: ignore[override]
    def single_action_space(self) -> spaces.MultiDiscrete:
        if self._hierarchical_action_space is None:
            return MettaGridEnv.single_action_space.__get__(self)  # type: ignore[misc]
        return self._hierarchical_action_space

    # ------------------------------------------------------------------
    # Core step conversion
    # ------------------------------------------------------------------
    def _convert_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        expected_dims = self._hierarchical_action_space.nvec.size
        if actions.shape[1] != expected_dims:
            raise ValueError(
                f"Expected actions with {expected_dims} dimensions, got shape {actions.shape}."
            )

        num_agents = actions.shape[0]
        verb_indices = actions[:, 0].astype(np.int64, copy=True)
        np.clip(verb_indices, 0, self._base_action_dims - 1, out=verb_indices)

        arg_offsets = 1 + verb_indices
        chosen_args = actions[np.arange(num_agents), arg_offsets].astype(np.int64, copy=True)

        # Guard against rogue values (e.g., truncated buffers)
        valid_arg_limits = self._max_action_args[verb_indices]
        np.clip(chosen_args, 0, valid_arg_limits, out=chosen_args)

        base_actions = np.stack([verb_indices, chosen_args], axis=-1).astype(dtype_actions, copy=False)
        return base_actions

    def step(self, actions: np.ndarray):  # type: ignore[override]
        converted = self._convert_actions(actions)
        return super().step(converted)

    def project_actions(self, actions: np.ndarray) -> np.ndarray:
        """Expose the base (verb, argument) encoding for logging or tooling."""

        return self._convert_actions(actions)


def make_hierarchical_env(
    env_cfg: MettaGridConfig,
    *,
    render_mode: Optional[str] = None,
    buf: Optional[Any] = None,
) -> HierarchicalActionMettaGridEnv:
    """Helper factory mirroring MettaGridEnv constructor signature."""

    return HierarchicalActionMettaGridEnv(env_cfg=env_cfg, render_mode=render_mode, buf=buf)
