"""Vectorized environment helpers shared across Tribal tooling."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pufferlib.pufferlib import set_buffers


class TribalEnvFactory:
    """Picklable factory for vectorized Tribal Village environments."""

    def __init__(self, base_config: dict[str, Any]):
        self._base_config = dict(base_config)

    def clone_cfg(self) -> dict[str, Any]:
        return dict(self._base_config)

    def __call__(
        self,
        cfg: Optional[dict[str, Any]] = None,
        buf: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Any:
        from tribal_village_env.environment import TribalVillageEnv

        merged_cfg = dict(self._base_config)
        if cfg is not None:
            merged_cfg.update(cfg)
        if seed is not None and "seed" not in merged_cfg:
            merged_cfg["seed"] = seed

        env = TribalVillageEnv(config=merged_cfg)
        set_buffers(env, buf)
        return env


class FlattenVecEnv:
    """Adapter to present contiguous agents_per_batch to the trainer."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.driver_env = getattr(inner, "driver_env", None)
        for attr in (
            "single_observation_space",
            "single_action_space",
            "action_space",
            "observation_space",
            "atn_batch_shape",
        ):
            setattr(self, attr, getattr(inner, attr, None))

        self.agents_per_batch = getattr(inner, "agents_per_batch", getattr(inner, "num_agents", 1))
        self.num_agents = self.agents_per_batch
        self.num_envs = getattr(inner, "num_envs", getattr(inner, "num_environments", None))

    def async_reset(self, seed: int = 0) -> None:
        self.inner.async_reset(seed)

    def reset(self, seed: int = 0):
        self.async_reset(seed)
        return self.recv()

    def send(self, actions):
        actions_arr = np.asarray(actions)
        self.inner.send(actions_arr)

    def recv(self):
        result = self.inner.recv()
        if len(result) == 8:
            o, r, d, t, ta, infos, env_ids, masks = result
        else:
            o, r, d, t, infos, env_ids, masks = result
            ta = None

        o = np.asarray(o, copy=False).reshape(self.agents_per_batch, *self.single_observation_space.shape)
        r = np.asarray(r, copy=False).reshape(self.agents_per_batch)
        d = np.asarray(d, copy=False).reshape(self.agents_per_batch)
        t = np.asarray(t, copy=False).reshape(self.agents_per_batch)
        mask = (
            np.asarray(masks, copy=False).reshape(self.agents_per_batch)
            if masks is not None
            else np.ones(self.agents_per_batch, dtype=bool)
        )
        env_ids = (
            np.asarray(env_ids, copy=False).reshape(self.agents_per_batch)
            if env_ids is not None
            else np.arange(self.agents_per_batch, dtype=np.int32)
        )
        infos = infos if isinstance(infos, list) else []
        return o, r, d, t, ta, infos, env_ids, mask

    def close(self):
        if hasattr(self.inner, "close"):
            self.inner.close()


__all__ = ["FlattenVecEnv", "TribalEnvFactory"]
