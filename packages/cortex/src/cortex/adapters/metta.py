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


class MettaTDAdapter(nn.Module):
    """Stateful adapter integrating a CortexStack with Metta TensorDicts.

    This is an ``nn.Module`` so its parameters (the wrapped ``CortexStack`` and the
    optional projection) are registered and optimized as part of the policy.

    Args:
        stack: CortexStack instance.
        in_key: Input tensor key in TensorDict (expects ``[B*TT, H]``).
        out_key: Output tensor key to write (``[B*TT, H_out]``).
        d_hidden: External hidden size of the stack (for validation and proj).
        out_features: Optional output projection size; if ``None`` or equal to
          ``d_hidden``, no projection is applied.
        key_prefix: Prefix for flat state keys written to TensorDict for replay.
    """

    def __init__(
        self,
        *,
        stack: CortexStack,
        in_key: str,
        out_key: str,
        d_hidden: int,
        out_features: Optional[int] = None,
        key_prefix: str = "cortex_state",
    ) -> None:
        super().__init__()
        # Constructor args
        self.stack: CortexStack = stack
        self.in_key: str = in_key
        self.out_key: str = out_key
        self.d_hidden: int = int(d_hidden)
        self.out_features: Optional[int] = out_features
        self.key_prefix: str = key_prefix

        # Discover state schema for per-env caches
        self._flat_entries: List[Tuple[FlatKey, LeafPath]] = self._discover_state_entries()

        # Output projection (optional)
        if self.out_features is None or int(self.out_features) == int(self.d_hidden):
            self._out_proj = nn.Identity()
        else:
            self._out_proj = nn.Linear(int(self.d_hidden), int(self.out_features))

        # Dense per-leaf caches: LeafPath -> Tensor[N_slots, *leaf_shape]
        # These enable vectorized gather/scatter via index_select / index_copy_.
        self._leaf_shapes: Dict[LeafPath, Tuple[int, ...]] = self._discover_leaf_shapes()
        self._rollout_store: Dict[LeafPath, torch.Tensor] = {}
        self._train_store: Dict[LeafPath, torch.Tensor] = {}
        self._store_dtype: torch.dtype = torch.bfloat16

        # Compact id→slot mappings (contiguous slots 0..N-1)
        self._rollout_id2slot: Dict[int, int] = {}
        self._rollout_next_slot: int = 0
        self._train_id2slot: Dict[int, int] = {}

        # Phase state machine flags
        self._in_training: bool = False
        self._snapshot_done: bool = False

        # Persistent rollout batched state (for TT==1 fast path)
        self._rollout_current_state: Optional[TensorDict] = None
        self._rollout_current_env_ids: Optional[torch.Tensor] = None  # Long[Br]

    def _discover_state_entries(self) -> List[Tuple[FlatKey, LeafPath]]:
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
        return entries

    def _discover_leaf_shapes(self) -> Dict[LeafPath, Tuple[int, ...]]:
        template = self.stack.init_state(batch=1, device=torch.device("cpu"), dtype=torch.float32)
        shapes: Dict[LeafPath, Tuple[int, ...]] = {}
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
                    if isinstance(tensor, torch.Tensor):
                        shapes[(block_key, cell_key, leaf_key)] = tuple(tensor.shape[1:])
        return shapes

    # ----------------------------- Public API -----------------------------
    @torch._dynamo.disable
    def forward(self, td: TensorDict) -> TensorDict:
        # Backward-compatible lazy init (handles old checkpoints)
        if not hasattr(self, "_store_dtype"):
            self._store_dtype = torch.bfloat16
        if not hasattr(self, "_rollout_id2slot"):
            self._rollout_id2slot = {}
        if not hasattr(self, "_rollout_next_slot"):
            self._rollout_next_slot = 0
        if not hasattr(self, "_train_id2slot"):
            self._train_id2slot = {}

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

        # No debug/timing; only raise on errors.

        # Env IDs for per-env cache
        if "training_env_ids" not in td.keys():
            raise KeyError("TensorDict is missing required 'training_env_ids' key")

        env_ids = td["training_env_ids"].squeeze(-1).reshape(-1)[:B]
        env_ids_long = env_ids.to(device=device, dtype=torch.long)

        # Resets
        resets = _as_reset_mask(td.get("dones", None), td.get("truncateds", None), B=B, TT=TT, device=device)

        # Update phase flags
        if TT > 1:
            self._in_training = True
            self._snapshot_done = False
            # If this is the first training and train_store is empty, snapshot now
            # (after first rollout has completed and populated rollout_store)
            if not self._train_store:
                self._flush_rollout_current_to_store()
                self._snapshot_train_store()
            # Free persistent rollout batch during training
            self._rollout_current_state = None
            self._rollout_current_env_ids = None
        # Take snapshot at first rollout call after training
        if TT == 1 and self._in_training and not self._snapshot_done:
            # Flush any live rollout state to the dense store, then snapshot
            self._flush_rollout_current_to_store()
            self._snapshot_train_store()
            self._in_training = False
            self._snapshot_done = True
            # Free the persistent rollout batch after snapshot
            self._rollout_current_state = None
            self._rollout_current_env_ids = None

        if TT == 1:
            # ----------------------------- Rollout path -----------------------------
            # Maintain persistent batched state for current env slice
            state_prev = self._ensure_rollout_current_state(env_ids_long, B=B, device=device, dtype=dtype)

            # Step
            x_step = x.view(B, -1)
            y, state_next = self.stack.step(x_step, state_prev, resets=resets)
            # Update persistent rollout state in-place; no per-step scatter
            self._rollout_current_state = state_next
            y = self._out_proj(y)
            td.set(self.out_key, y.reshape(B * TT, -1))
            return td
        else:
            # ------------------------------ Training path ------------------------------
            # Use training_env_ids from replay to select initial states from frozen train_cache
            env_ids_train = td["training_env_ids"].squeeze(-1).to(device=device)
            env_ids_train = env_ids_train.view(B, TT)[:, 0]
            env_ids_train_long = env_ids_train.to(dtype=torch.long)

            # Map env ids to compact slot ids (unknown → -1) and gather by slots
            train_slots = self._map_ids_to_slots(env_ids_train_long, self._train_id2slot, create_missing=False)
            state0 = self._gather_state_by_slots(train_slots, store=self._train_store, B=B, device=device, dtype=dtype)

            x_seq = rearrange(x, "(b t) h -> b t h", b=B, t=TT)
            y_seq, _ = self.stack(x_seq, state0, resets=resets)
            y_seq = self._out_proj(y_seq)
            td.set(self.out_key, rearrange(y_seq, "b t h -> (b t) h"))
            return td

    def reset_memory(self) -> None:
        # No-op by design; trainer hooks call reset_memory() at rollout/train starts.
        # Per-step resets are handled inside forward via masks, and caches persist.
        return

    def get_memory(self) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        """Serialize dense stores for checkpointing.

        Returns a nested mapping with per-leaf tensors; current live rollout
        batch is not serialized (it will be re-gathered on next call).
        """

        def pack(store: Dict[LeafPath, torch.Tensor]) -> Dict[str, torch.Tensor]:
            packed: Dict[str, torch.Tensor] = {}
            for (b_key, c_key, leaf_key), t in store.items():
                packed[f"{b_key}|{c_key}|{leaf_key}"] = t
            return packed

        return {
            "rollout_store": {"data": pack(self._rollout_store)},
            "train_store": {"data": pack(self._train_store)},
        }

    def set_memory(self, memory) -> None:  # type: ignore[override]
        """Restore dense stores if provided; raises on malformed input.

        Accepts the structure emitted by ``get_memory``. If the provided
        object does not match the expected schema, a RuntimeError is raised.
        """
        try:

            def unpack(blob: Dict[str, torch.Tensor]) -> Dict[LeafPath, torch.Tensor]:
                store: Dict[LeafPath, torch.Tensor] = {}
                for k, t in blob.items():
                    b_key, c_key, leaf_key = k.split("|")
                    store[(b_key, c_key, leaf_key)] = t
                return store

            rollout_blob = memory.get("rollout_store", {}).get("data", {})
            train_blob = memory.get("train_store", {}).get("data", {})
            if rollout_blob or train_blob:
                self._rollout_store = unpack(rollout_blob)
                self._train_store = unpack(train_blob)
                # Ensure dtype policy is respected after restore
                # (actual dtype will be enforced on first gather/scatter)
                if not hasattr(self, "_store_dtype"):
                    self._store_dtype = torch.bfloat16
            else:
                raise RuntimeError("MettaTDAdapter.set_memory: missing 'rollout_store'/'train_store' data blobs")
        except Exception as e:
            raise RuntimeError(f"MettaTDAdapter.set_memory: malformed memory structure: {e}") from e

    def experience_keys(self) -> Dict[FlatKey, torch.Size]:
        """Replay keys required by the adapter.

        We only need `training_env_ids` so the frozen training cache can be indexed
        deterministically; hidden states are not stored per timestep.
        """

        return {"training_env_ids": torch.Size([1])}

    # --------------------------- Internal helpers ---------------------------
    # --------------------------- Dense store helpers ---------------------------
    def _ensure_store_capacity(
        self,
        store: Dict[LeafPath, torch.Tensor],
        min_slots: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        cur_cap = 0
        if store:
            any_leaf = next(iter(store.values()))
            cur_cap = int(any_leaf.shape[0])
            ok = all(t.device == device and t.dtype == dtype and t.shape[0] >= min_slots for t in store.values())
            if ok and cur_cap >= min_slots:
                return
        new_cap = max(min_slots, max(1, cur_cap * 2))
        # Expand or create per-leaf storage
        for leaf_path, leaf_shape in self._leaf_shapes.items():
            if leaf_path not in store:
                store[leaf_path] = torch.zeros((new_cap, *leaf_shape), device=device, dtype=dtype)
            else:
                old = store[leaf_path]
                if old.shape[0] < new_cap or old.device != device or old.dtype != dtype:
                    new = torch.zeros((new_cap, *leaf_shape), device=device, dtype=dtype)
                    new[: old.shape[0]].copy_(old.to(device=device, dtype=dtype))
                    store[leaf_path] = new
        # capacity tracked per-store via leaf shapes

    def _gather_state_by_slots(
        self,
        slot_ids: torch.Tensor,
        *,
        store: Dict[LeafPath, torch.Tensor],
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TensorDict:
        # Pick an appropriate storage dtype for this device (bf16 if supported, else f32)
        if device.type == "cuda":
            bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            self._store_dtype = torch.bfloat16 if bf16_ok else torch.float32
        else:
            # CPU supports bf16 tensors; keep bf16 for memory unless caller prefers otherwise
            self._store_dtype = torch.bfloat16

        # Ensure capacity for any large slot id; sanitize negatives if any (-1)
        if slot_ids.numel() == 0:
            return self.stack.init_state(batch=B, device=device, dtype=dtype)

        if slot_ids.dim() != 1:
            slot_ids = slot_ids.reshape(-1)
        valid_mask = slot_ids >= 0
        slot_ids_clamped = slot_ids.clamp_min(0)
        max_slot = int(slot_ids_clamped.max().item()) + 1 if slot_ids_clamped.numel() > 0 else 0
        self._ensure_store_capacity(store, max_slot, device=device, dtype=self._store_dtype)

        batch_state = self.stack.init_state(batch=B, device=device, dtype=dtype)
        for _fkey, (bkey, ckey, lkey) in self._flat_entries:
            src = store[(bkey, ckey, lkey)]  # [N, ...]
            cap = int(src.shape[0])
            if max_slot > cap:
                raise RuntimeError(
                    f"[MettaTDAdapter] slot out of bounds: need<{max_slot} got cap={cap} for leaf {(bkey, ckey, lkey)}"
                )
            gathered = src.index_select(0, slot_ids_clamped)  # [B, ...] (bf16)
            if not bool(valid_mask.all()):
                gathered[~valid_mask] = 0
            gathered = gathered.to(dtype=dtype)
            # Assign into nested TD
            block_td = batch_state.get(bkey)
            cell_td = block_td.get(ckey)
            cell_td.set(lkey, gathered)
        return batch_state

    def _scatter_state_by_slots(
        self, state: TensorDict, slot_ids: torch.Tensor, *, store: Dict[LeafPath, torch.Tensor]
    ) -> None:
        B = int(slot_ids.numel())
        if B == 0:
            return
        # Set store dtype for this device
        leaf_device = None
        for _fk, (b_key, c_key, leaf_key) in self._flat_entries:
            t = state.get(b_key).get(c_key).get(leaf_key)
            if isinstance(t, torch.Tensor):
                leaf_device = t.device
                break
        if leaf_device is None:
            leaf_device = state.device if hasattr(state, "device") else torch.device("cpu")
        if leaf_device.type == "cuda":
            bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            self._store_dtype = torch.bfloat16 if bf16_ok else torch.float32
        else:
            self._store_dtype = torch.bfloat16
        if slot_ids.dim() != 1:
            slot_ids = slot_ids.reshape(-1)
        # Ensure capacity (derive device/dtype from leaves to avoid TensorDict.device mismatches)
        max_slot = int(slot_ids.max().item()) + 1
        # Determine device from first available leaf tensor
        leaf_device = None
        for _fk, (b_key, c_key, leaf_key) in self._flat_entries:
            t = state.get(b_key).get(c_key).get(leaf_key)
            if isinstance(t, torch.Tensor):
                leaf_device = t.device
                break
        if leaf_device is None:
            leaf_device = state.device if hasattr(state, "device") else torch.device("cpu")

        self._ensure_store_capacity(store, max_slot, device=leaf_device, dtype=self._store_dtype)
        for _fkey, (bkey, ckey, lkey) in self._flat_entries:
            src = state.get(bkey).get(ckey).get(lkey)  # [B, ...]
            dest = store[(bkey, ckey, lkey)]
            dest.index_copy_(0, slot_ids, src.to(dtype=dest.dtype).detach())

    def _snapshot_train_store(self) -> None:
        # Build a compact snapshot of rollout_store (only assigned slots) for training
        if not self._rollout_id2slot:
            # Nothing assigned yet; create an empty train store
            self._train_store = {leaf: tensor.clone().detach() for leaf, tensor in self._rollout_store.items()}
            self._train_id2slot = {}
            return
        items = sorted(self._rollout_id2slot.items(), key=lambda kv: kv[1])  # sort by rollout slot
        env_ids_sorted = [eid for eid, _ in items]
        old_slots = [slot for _, slot in items]
        new_slots = list(range(len(old_slots)))
        self._train_id2slot = {eid: ns for eid, ns in zip(env_ids_sorted, new_slots, strict=False)}
        # Reindex per-leaf tensors to compact [M, ...]
        self._train_store = {}
        for leaf_path, src in self._rollout_store.items():
            device = src.device
            dtype = src.dtype
            index = torch.tensor(old_slots, device=device, dtype=torch.long)
            compact = src.index_select(0, index)
            self._train_store[leaf_path] = compact.clone().detach().to(dtype=dtype)

    def _infer_dtype(self, state: TensorDict) -> torch.dtype:
        # Infer a reasonable dtype from first leaf
        for _fkey, (bkey, ckey, lkey) in self._flat_entries:
            t = state.get(bkey).get(ckey).get(lkey)
            if isinstance(t, torch.Tensor):
                return t.dtype
        return torch.float32

    def _infer_device(self, state: TensorDict) -> torch.device:
        # Prefer a leaf device; fallback to TensorDict.device
        for _fkey, (bkey, ckey, lkey) in self._flat_entries:
            t = state.get(bkey).get(ckey).get(lkey)
            if isinstance(t, torch.Tensor):
                return t.device
        return state.device if hasattr(state, "device") else torch.device("cpu")

    def _flush_rollout_current_to_store(self) -> None:
        # If we have a live rollout batched state, scatter it into the rollout store
        if self._rollout_current_state is not None and self._rollout_current_env_ids is not None:
            slots = self._map_ids_to_slots(self._rollout_current_env_ids, self._rollout_id2slot, create_missing=True)
            self._scatter_state_by_slots(self._rollout_current_state, slots, store=self._rollout_store)
            # Do not clear current state; it can remain valid until env slice changes

    def _ensure_rollout_current_state(
        self,
        env_ids_long: torch.Tensor,
        *,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TensorDict:
        # If current env slice matches, reuse live state
        if (
            self._rollout_current_state is not None
            and self._rollout_current_env_ids is not None
            and self._rollout_current_env_ids.numel() == env_ids_long.numel()
            and torch.equal(self._rollout_current_env_ids, env_ids_long)
        ):
            return self._rollout_current_state

        # Slice changed: flush previous live state to store, gather new batch
        self._flush_rollout_current_to_store()
        slots = self._map_ids_to_slots(env_ids_long, self._rollout_id2slot, create_missing=False)
        state_prev = self._gather_state_by_slots(slots, store=self._rollout_store, B=B, device=device, dtype=dtype)
        self._rollout_current_state = state_prev
        self._rollout_current_env_ids = env_ids_long.detach().clone()
        return state_prev

    def _map_ids_to_slots(
        self,
        ids_long: torch.Tensor,
        mapping: Dict[int, int],
        *,
        create_missing: bool,
    ) -> torch.Tensor:
        ids_list = ids_long.detach().view(-1).tolist()
        out: List[int] = []
        if create_missing:
            if mapping is self._rollout_id2slot:
                next_slot = self._rollout_next_slot
            else:
                next_slot = (max(mapping.values()) + 1) if mapping else 0
            for eid in ids_list:
                e = int(eid)
                slot = mapping.get(e)
                if slot is None:
                    slot = next_slot
                    mapping[e] = slot
                    next_slot += 1
                out.append(slot)
            if mapping is self._rollout_id2slot:
                self._rollout_next_slot = next_slot
        else:
            for eid in ids_list:
                out.append(mapping.get(int(eid), -1))
        return torch.tensor(out, device=ids_long.device, dtype=torch.long)

    def _flat_key(self, block_key: str, cell_key: str, leaf_key: str) -> FlatKey:
        return f"{self.key_prefix}__{block_key}__{cell_key}__{leaf_key}"


__all__ = ["MettaTDAdapter"]
