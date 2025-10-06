from __future__ import annotations

# ruff: noqa: E402

"""Metta TensorDict adapter for Cortex stacks (stateful).

Two‑cache design:
- `rollout_cache` advances only during rollout (TT == 1).
- `train_cache` is a frozen snapshot of `rollout_cache`, taken at the start
  of the next rollout following a training phase. It is used to initialize
  training segments and is never mutated during training. We do not store
  per‑timestep hidden state in replay; instead we persist only `env_id` so
  training can index `train_cache` deterministically.

This avoids clearing memory on trainer hooks and keeps repeated samples (PER)
consistent without extra metadata.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from cortex.stacks import CortexStack
from einops import rearrange
from tensordict import TensorDict

FlatKey = str
LeafPath = Tuple[str, str, str]  # (block_key, cell_key, leaf_key)


def _as_reset_mask(
    dones: Optional[torch.Tensor],
    truncateds: Optional[torch.Tensor],
    *,
    B: int,
    TT: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if dones is None and truncateds is None:
        return None
    if dones is None:
        dones = torch.zeros_like(truncateds)
    if truncateds is None:
        truncateds = torch.zeros_like(dones)
    if dones.numel() == 0 and truncateds.numel() == 0:
        return None
    resets_bool = (dones.bool() | truncateds.bool()).to(device=device)
    try:
        return resets_bool.view(B) if TT == 1 else resets_bool.view(B, TT)
    except RuntimeError:
        return resets_bool.reshape(-1)[:B] if TT == 1 else resets_bool.reshape(B, TT)


@dataclass
class MettaTDAdapter:
    """Stateful adapter integrating a CortexStack with Metta TensorDicts.

    Args:
        stack: CortexStack instance.
        in_key: Input tensor key in TensorDict (expects [B*TT, H]).
        out_key: Output tensor key to write ([B*TT, H_out]).
        d_hidden: External hidden size of the stack (for validation and proj).
        out_features: Optional output projection size; if None or == d_hidden,
            no projection is applied.
        key_prefix: Prefix for flat state keys written to TensorDict for replay.
    """

    stack: CortexStack
    in_key: str
    out_key: str
    d_hidden: int
    out_features: Optional[int] = None
    key_prefix: str = "cortex_state"

    def __post_init__(self) -> None:
        # Discover state schema for per-env caches
        template = self.stack.init_state(batch=1, device=torch.device("cpu"), dtype=torch.float32)
        entries: List[Tuple[FlatKey, LeafPath]] = []
        for block_key in template.keys():
            block_state = template.get(block_key)
            if not isinstance(block_state, TensorDict):
                continue
            for cell_key in block_state.keys():
                cell_state = block_state.get(cell_key)
                if not isinstance(cell_state, TensorDict):
                    continue
                for leaf_key in cell_state.keys():
                    tensor = cell_state.get(leaf_key)
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    fkey = self._flat_key(block_key, cell_key, leaf_key)
                    entries.append((fkey, (block_key, cell_key, leaf_key)))
        self._flat_entries = entries

        # Output projection (optional)
        self._out_proj: nn.Module
        if self.out_features is None or int(self.out_features) == int(self.d_hidden):
            self._out_proj = nn.Identity()
        else:
            self._out_proj = nn.Linear(int(self.d_hidden), int(self.out_features))

        # Per-env caches: env_id -> TensorDict(batch=[1])
        self._rollout_cache: Dict[int, TensorDict] = {}
        self._train_cache: Dict[int, TensorDict] = {}

        # Phase state machine flags
        self._in_training: bool = False
        self._snapshot_done: bool = False

    # ----------------------------- Public API -----------------------------
    @torch._dynamo.disable
    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.in_key]
        if "bptt" not in td.keys():
            raise KeyError("TensorDict is missing required 'bptt' metadata")
        TT = int(td["bptt"][0].item())
        if TT <= 0:
            raise ValueError("bptt entries must be positive")

        total_batch = x.shape[0]
        B, rem = divmod(total_batch, TT)
        if rem != 0:
            raise ValueError("input batch must be divisible by bptt")
        if "batch" in td.keys():
            B = int(td["batch"][0].item())

        device = x.device
        dtype = x.dtype

        # Env IDs for per-env cache
        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is not None:
            env_ids = training_env_ids.reshape(-1)[:B]
        else:
            env_ids = torch.arange(B, device=device)
        env_id_list = [int(env_ids[i].item()) for i in range(B)]

        # Resets
        resets = _as_reset_mask(td.get("dones", None), td.get("truncateds", None), B=B, TT=TT, device=device)

        # Update phase flags
        if TT > 1:
            self._in_training = True
            self._snapshot_done = False
        # Take snapshot at first rollout call after training
        if TT == 1 and self._in_training and not self._snapshot_done:
            self._snapshot_train_cache()
            self._in_training = False
            self._snapshot_done = True
        # Seed train_cache on first ever rollout if empty
        if TT == 1 and not self._train_cache:
            self._snapshot_train_cache()

        if TT == 1:
            # ----------------------------- Rollout path -----------------------------
            state_prev = self._gather_state_batch(
                env_id_list, cache=self._rollout_cache, B=B, device=device, dtype=dtype
            )
            if resets is not None:
                state_prev = self.stack.reset_state(state_prev, resets)

            # Expose env_id for replay (so training can index train_cache)
            td.set("env_id", env_ids.detach())

            # Step
            x_step = x.view(B, -1)
            y, state_next = self.stack.step(x_step, state_prev, resets=resets)
            self._scatter_state_batch(state_next, env_id_list, cache=self._rollout_cache)

            y = self._out_proj(y)
            td.set(self.out_key, y.reshape(B * TT, -1))
            return td

        # ------------------------------ Training path ------------------------------
        # Use env_id from replay to select initial states from frozen train_cache
        if "env_id" in td.keys():
            env_ids_train = td.get("env_id").to(device=device)
            env_ids_train = env_ids_train.view(B, TT)[:, 0]
            env_id_list_train = [int(env_ids_train[i].item()) for i in range(B)]
        else:
            # Fallback to local 0..B-1 indexing (not ideal, but keeps training running)
            env_id_list_train = list(range(B))
        state0 = self._gather_state_batch(env_id_list_train, cache=self._train_cache, B=B, device=device, dtype=dtype)

        x_seq = rearrange(x, "(b t) h -> b t h", b=B, t=TT)
        y_seq, _ = self.stack(x_seq, state0, resets=resets)
        y_seq = self._out_proj(y_seq)
        td.set(self.out_key, rearrange(y_seq, "b t h -> (b t) h"))
        return td

    def reset_memory(self) -> None:
        # No-op by design; trainer hooks call reset_memory() at rollout/train starts.
        # Per-step resets are handled inside forward via masks, and caches persist.
        return

    def get_memory(self) -> Dict[str, Dict[int, TensorDict]]:
        return {"rollout": self._rollout_cache, "train": self._train_cache}

    def set_memory(self, memory: Dict[str, Dict[int, TensorDict]]) -> None:
        self._rollout_cache = dict(memory.get("rollout", {}))
        self._train_cache = dict(memory.get("train", {}))

    def experience_keys(self) -> Dict[FlatKey, torch.Size]:
        """Replay keys required by the adapter.

        We only need `env_id` so the frozen training cache can be indexed
        deterministically; hidden states are not stored per timestep.
        """

        return {"env_id": torch.Size([])}

    # --------------------------- Internal helpers ---------------------------
    def _gather_state_batch(
        self,
        env_ids: List[int],
        *,
        cache: Dict[int, TensorDict],
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TensorDict:
        batch_state = self.stack.init_state(batch=B, device=device, dtype=dtype)
        for i, env_id in enumerate(env_ids):
            env_state = cache.get(env_id)
            if env_state is None:
                env_state = self.stack.init_state(batch=1, device=device, dtype=dtype)
            # Copy leaves into batch_state at row i
            for _fkey, (bkey, ckey, lkey) in self._flat_entries:
                leaf_src = env_state.get(bkey).get(ckey).get(lkey)  # [1, ...]
                leaf_dst = batch_state.get(bkey).get(ckey).get(lkey)  # [B, ...]
                leaf_dst[i].copy_(leaf_src[0])
        return batch_state

    def _scatter_state_batch(self, state: TensorDict, env_ids: List[int], *, cache: Dict[int, TensorDict]) -> None:
        B = len(env_ids)
        for i in range(B):
            env_id = env_ids[i]
            # Build a per-env state TD with batch=[1]
            env_td = TensorDict({}, batch_size=[1], device=state.device)
            for _fkey, (bkey, ckey, lkey) in self._flat_entries:
                leaf_src = state.get(bkey).get(ckey).get(lkey)  # [B, ...]
                leaf_one = leaf_src[i].unsqueeze(0)  # [1, ...]
                # Create nested structure lazily
                if bkey not in env_td.keys():
                    env_td[bkey] = TensorDict({}, batch_size=[1], device=state.device)
                if ckey not in env_td[bkey].keys():
                    env_td[bkey][ckey] = TensorDict({}, batch_size=[1], device=state.device)
                env_td[bkey][ckey][lkey] = leaf_one.detach()
            cache[env_id] = env_td

    def _snapshot_train_cache(self) -> None:
        # Deep clone rollout cache into train cache (detach tensors)
        new_cache: Dict[int, TensorDict] = {}
        for env_id, td in self._rollout_cache.items():
            env_td = TensorDict({}, batch_size=td.batch_size, device=td.device)
            for _fkey, (bkey, ckey, lkey) in self._flat_entries:
                leaf = td.get(bkey).get(ckey).get(lkey)
                # Lazily build nested structure
                if bkey not in env_td.keys():
                    env_td[bkey] = TensorDict({}, batch_size=td.batch_size, device=td.device)
                if ckey not in env_td[bkey].keys():
                    env_td[bkey][ckey] = TensorDict({}, batch_size=td.batch_size, device=td.device)
                env_td[bkey][ckey][lkey] = leaf.clone().detach()
            new_cache[env_id] = env_td
        self._train_cache = new_cache

    def _flat_key(self, block_key: str, cell_key: str, leaf_key: str) -> FlatKey:
        return f"{self.key_prefix}__{block_key}__{cell_key}__{leaf_key}"


__all__ = ["MettaTDAdapter"]
