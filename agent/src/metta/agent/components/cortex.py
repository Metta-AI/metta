from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from cortex.config import CortexStackConfig
from cortex.factory import build_cortex
from cortex.stacks import CortexStack
from einops import rearrange
from pydantic import ConfigDict
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig

logger = logging.getLogger(__name__)

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
    if truncateds is None:
        if dones is None:
            return None
        truncateds = torch.zeros_like(dones)
    if dones is None:
        dones = torch.zeros_like(truncateds)
    if dones.numel() == 0 and truncateds.numel() == 0:
        return None
    resets_bool = (dones.bool() | truncateds.bool()).to(device=device)
    try:
        return resets_bool.view(B) if TT == 1 else resets_bool.view(B, TT)
    except RuntimeError:
        return resets_bool.reshape(-1)[:B] if TT == 1 else resets_bool.reshape(B, TT)


class CortexTDConfig(ComponentConfig):
    """Config for integrating a Cortex stack as a Metta component.

    This component manages state caching internally and exposes a TensorDict API
    compatible with ``PolicyAutoBuilder``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    in_key: str
    out_key: str
    name: str = "cortex"

    d_hidden: int = 128
    out_features: Optional[int] = None

    # JSON‑serializable config for building the Cortex stack.
    stack_cfg: CortexStackConfig

    key_prefix: str = "cortex_state"
    # Cache storage dtype for CortexTD: 'fp32' (default) or 'bf16'
    store_dtype: str = "fp32"

    def make_component(self, env: Any = None) -> nn.Module:
        return CortexTD(config=self)


class CortexTD(nn.Module):
    """Stateful component integrating a ``CortexStack`` with Metta TensorDicts."""

    def __init__(self, config: CortexTDConfig) -> None:
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key

        # Build the stack from config
        scfg: CortexStackConfig = config.stack_cfg
        stack = build_cortex(scfg)
        # Optional sanity check: d_hidden should match the stack's external size
        try:
            stack_hidden = int(stack.cfg.d_hidden)  # type: ignore[attr-defined]
            if stack_hidden != int(config.d_hidden):
                raise ValueError(
                    f"CortexTDConfig.d_hidden ({config.d_hidden}) does not match stack.cfg.d_hidden ({stack_hidden})."
                )
        except Exception:
            # If cfg is not present, skip the check
            pass

        self.stack: CortexStack = stack
        self.d_hidden: int = int(config.d_hidden)
        self.out_features: Optional[int] = config.out_features
        self.key_prefix: str = config.key_prefix

        store_dtype = config.store_dtype
        if store_dtype not in {"bf16", "fp32"}:
            raise ValueError("store_dtype must be 'bf16' or 'fp32'")
        self._store_dtype: torch.dtype = torch.bfloat16 if store_dtype == "bf16" else torch.float32

        self._flat_entries: List[Tuple[FlatKey, LeafPath]] = self._discover_state_entries()

        if self.out_features is None or int(self.out_features) == int(self.d_hidden):
            self._out_proj: nn.Module = nn.Identity()
        else:
            self._out_proj = nn.Linear(int(self.d_hidden), int(self.out_features))

        self._leaf_shapes: Dict[LeafPath, Tuple[int, ...]] = self._discover_leaf_shapes()
        self._rollout_store: Dict[LeafPath, torch.Tensor] = {}
        self._train_store: Dict[LeafPath, torch.Tensor] = {}

        self._rollout_id2slot: Dict[int, int] = {}
        self._rollout_next_slot: int = 0
        self._train_id2slot: Dict[int, int] = {}

        self._in_training: bool = False
        self._snapshot_done: bool = False

        self._rollout_current_state: Optional[TensorDict] = None
        self._rollout_current_env_ids: Optional[torch.Tensor] = None  # Long[Br]

        # Prime: materialize any lazily-created leaves (e.g., AxonLayer groups)
        # by running a single zero-input step on CPU. This yields a more complete
        # state template than stack.init_state() alone, so future gather/scatter
        # operations include all leaves from the start.
        self._prime_state_template()

        # Eval-time logging is done at DEBUG level each time keys are inferred

    def _prime_state_template(self) -> None:
        """Prime by running a zero‑step init and recording any lazy leaves.

        Delegates the zero‑step logic to ``_zero_step_init_state`` to avoid dupes.
        """
        s1 = self._zero_step_init_state(batch=1, device=torch.device("cpu"), dtype=torch.float32)
        self._maybe_register_new_leaves(s1)

    def _zero_step_init_state(self, *, batch: int, device: torch.device, dtype: torch.dtype) -> TensorDict:
        """Return initial state via one zero-input step from None.

        Ensures lazy leaves are materialized and shapes/devices match runtime.
        """
        x0 = torch.zeros(batch, int(self.d_hidden), device=device, dtype=dtype)
        with torch.no_grad():
            _y, s1 = self.stack.step(x0, None)
        return s1

    def _maybe_register_new_leaves(self, state: TensorDict) -> None:
        """Register lazily-created leaves so gather/scatter includes them.

        AxonLayer-managed substates under groups like 'mlstm' or 'mlstm_qkv'
        are often created on first use. Extend our flat index and shape map
        when we observe new leaves at runtime, and ensure store capacity.
        """
        existing = {leaf_path for _, leaf_path in self._flat_entries}
        for bkey in state.keys():
            btd = state.get(bkey)
            if not isinstance(btd, TensorDict):
                continue
            for ckey in btd.keys():
                ctd = btd.get(ckey)
                if not isinstance(ctd, TensorDict):
                    continue
                for lkey in ctd.keys():
                    t = ctd.get(lkey)
                    if not isinstance(t, torch.Tensor):
                        continue
                    path = (bkey, ckey, lkey)
                    if path in existing:
                        continue
                    # Register new leaf and ensure store tensors exist
                    fkey = self._flat_key(*path)
                    self._flat_entries.append((fkey, path))
                    self._leaf_shapes[path] = tuple(t.shape[1:])

                    # Ensure both stores have tensors with current capacity
                    def _cap(store: Dict[LeafPath, torch.Tensor]) -> int:
                        return next(iter(store.values())).shape[0] if store else 1

                    dev = t.device
                    self._ensure_store_capacity(
                        self._rollout_store, _cap(self._rollout_store), device=dev, dtype=self._store_dtype
                    )
                    self._ensure_store_capacity(
                        self._train_store, _cap(self._train_store), device=dev, dtype=self._store_dtype
                    )

    @torch._dynamo.disable
    def forward(self, td: TensorDict) -> TensorDict:  # type: ignore[override]
        x = td[self.in_key]

        device = x.device
        dtype = x.dtype

        TT = int(td["bptt"][0].item())

        B = int(td["batch"][0].item())

        if TT <= 0 or B <= 0:
            raise ValueError("'bptt' and 'batch' must be positive integers")

        if x.shape[0] != B * TT:
            raise ValueError(f"input length {x.shape[0]} must equal batch*bptt ({B}*{TT})")

        # training_env_ids: default to a simple range for evaluation
        if "training_env_ids" not in td.keys():
            td.set("training_env_ids", torch.arange(B, device=device, dtype=torch.long).view(B, 1))
            logger.debug(
                "[CortexTD] Missing 'training_env_ids'; defaulting to arange(B) with B=%d.",
                B,
            )

        env_ids = td["training_env_ids"].squeeze(-1).reshape(-1)[:B]
        env_ids_long = env_ids.to(device=device, dtype=torch.long)

        resets = _as_reset_mask(td.get("dones", None), td.get("truncateds", None), B=B, TT=TT, device=device)

        if TT > 1:
            # Entering training: refresh train store from rollout on every switch
            if not self._in_training:
                self._flush_rollout_current_to_store()
                self._snapshot_train_store()
                self._in_training = True
            # While training, clear rollout cache to avoid stale carryover
            self._rollout_current_state = None
            self._rollout_current_env_ids = None
        else:  # TT == 1
            # Leaving training: mark flag only; rollout cache continues across steps
            if self._in_training:
                self._in_training = False

        if TT == 1:
            state_prev = self._ensure_rollout_current_state(env_ids_long, B=B, device=device, dtype=dtype)
            x_step = x.view(B, -1)
            y, state_next = self.stack.step(x_step, state_prev, resets=resets)
            self._maybe_register_new_leaves(state_next)
            self._rollout_current_state = state_next
            y = self._out_proj(y)
            td.set(self.out_key, y.reshape(B * TT, -1))
            return td
        else:
            env_ids_train = td["training_env_ids"].squeeze(-1).to(device=device)
            env_ids_train = env_ids_train.view(B, TT)[:, 0]
            env_ids_train_long = env_ids_train.to(dtype=torch.long)

            train_slots = self._map_ids_to_slots(env_ids_train_long, self._train_id2slot, create_missing=False)
            state0 = self._gather_state_by_slots(train_slots, store=self._train_store, B=B, device=device, dtype=dtype)

            x_seq = rearrange(x, "(b t) h -> b t h", b=B, t=TT)
            y_seq, _ = self.stack(x_seq, state0, resets=resets)
            y_seq = self._out_proj(y_seq)
            td.set(self.out_key, rearrange(y_seq, "b t h -> (b t) h"))
            return td

    def experience_keys(self) -> Dict[FlatKey, torch.Size]:
        """Replay keys required by the component."""

        return {"training_env_ids": torch.Size([1])}

    def reset_memory(self) -> None:
        return

    def get_memory(self) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        """Serialize dense stores for checkpointing."""

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
        """Restore dense stores from the structure emitted by ``get_memory``."""
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
                if not hasattr(self, "_store_dtype"):
                    self._store_dtype = torch.float32
            else:
                raise RuntimeError("CortexTD.set_memory: missing 'rollout_store'/'train_store' data blobs")
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError(f"CortexTD.set_memory: malformed memory structure: {e}") from e

    def get_agent_experience_spec(self) -> Composite:
        # Advertise minimal keys (training_env_ids) for replay; hidden state is not stored.
        spec_dict: Dict[str, UnboundedDiscrete] = {}
        for key, shape in self.experience_keys().items():
            dtype = torch.long if key == "training_env_ids" else torch.float32
            spec_dict[key] = UnboundedDiscrete(shape=torch.Size(shape), dtype=dtype)
        return Composite(spec_dict)

    def _discover_state_entries(self) -> List[Tuple[FlatKey, LeafPath]]:
        template = self._zero_step_init_state(batch=1, device=torch.device("cpu"), dtype=torch.float32)
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
        template = self._zero_step_init_state(batch=1, device=torch.device("cpu"), dtype=torch.float32)
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
        for leaf_path, leaf_shape in self._leaf_shapes.items():
            if leaf_path not in store:
                store[leaf_path] = torch.zeros((new_cap, *leaf_shape), device=device, dtype=dtype)
            else:
                old = store[leaf_path]
                if old.shape[0] < new_cap or old.device != device or old.dtype != dtype:
                    new = torch.zeros((new_cap, *leaf_shape), device=device, dtype=dtype)
                    new[: old.shape[0]].copy_(old.to(device=device, dtype=dtype))
                    store[leaf_path] = new

    def _gather_state_by_slots(
        self,
        slot_ids: torch.Tensor,
        *,
        store: Dict[LeafPath, torch.Tensor],
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TensorDict:
        """Gather a batch state from a slot-indexed store without running a forward pass.

        - For empty requests, returns a zero state built from the known leaf shapes.
        - For non-empty, builds the nested TensorDict structure and fills leaves by
          index_select from the store, zeroing rows where slot == -1.
        """
        # Fast path: no slots requested -> return all zeros
        if slot_ids.numel() == 0:
            out = TensorDict({}, batch_size=[B], device=device)
            # Build zero tensors per leaf
            for _fkey, (bkey, ckey, lkey) in self._flat_entries:
                shape = self._leaf_shapes[(bkey, ckey, lkey)]
                zero = torch.zeros((B, *shape), device=device, dtype=dtype)
                btd = out.get(bkey) if bkey in out.keys() else TensorDict({}, batch_size=[B])
                ctd = (
                    btd.get(ckey)
                    if (isinstance(btd, TensorDict) and ckey in btd.keys())
                    else TensorDict({}, batch_size=[B])
                )
                ctd.set(lkey, zero)
                btd[ckey] = ctd
                out[bkey] = btd
            return out

        if slot_ids.dim() != 1:
            slot_ids = slot_ids.reshape(-1)
        valid_mask = slot_ids >= 0
        slot_ids_clamped = slot_ids.clamp_min(0)
        max_slot = int(slot_ids_clamped.max().item()) + 1 if slot_ids_clamped.numel() > 0 else 0
        self._ensure_store_capacity(store, max_slot, device=device, dtype=self._store_dtype)

        batch_state = TensorDict({}, batch_size=[B], device=device)
        for _fkey, (bkey, ckey, lkey) in self._flat_entries:
            src = store[(bkey, ckey, lkey)]  # [N, ...]
            cap = int(src.shape[0])
            if max_slot > cap:
                raise RuntimeError(
                    f"[CortexTD] slot out of bounds: need<{max_slot} got cap={cap} for leaf {(bkey, ckey, lkey)}"
                )
            gathered = src.index_select(0, slot_ids_clamped)
            if not bool(valid_mask.all()):
                gathered[~valid_mask] = 0
            gathered = gathered.to(dtype=dtype, device=device)
            btd = batch_state.get(bkey) if bkey in batch_state.keys() else TensorDict({}, batch_size=[B])
            ctd = (
                btd.get(ckey)
                if (isinstance(btd, TensorDict) and ckey in btd.keys())
                else TensorDict({}, batch_size=[B])
            )
            ctd.set(lkey, gathered)
            btd[ckey] = ctd
            batch_state[bkey] = btd
        return batch_state

    def _scatter_state_by_slots(
        self, state: TensorDict, slot_ids: torch.Tensor, *, store: Dict[LeafPath, torch.Tensor]
    ) -> None:
        B = int(slot_ids.numel())
        if B == 0:
            return
        if slot_ids.dim() != 1:
            slot_ids = slot_ids.reshape(-1)
        max_slot = int(slot_ids.max().item()) + 1
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
        if not self._rollout_id2slot:
            self._train_store = {leaf: tensor.clone().detach() for leaf, tensor in self._rollout_store.items()}
            self._train_id2slot = {}
            return
        items = sorted(self._rollout_id2slot.items(), key=lambda kv: kv[1])  # sort by rollout slot
        env_ids_sorted = [eid for eid, _ in items]
        old_slots = [slot for _, slot in items]
        new_slots = list(range(len(old_slots)))
        self._train_id2slot = {eid: ns for eid, ns in zip(env_ids_sorted, new_slots, strict=False)}
        self._train_store = {}
        for leaf_path, src in self._rollout_store.items():
            device = src.device
            dtype = src.dtype
            index = torch.tensor(old_slots, device=device, dtype=torch.long)
            compact = src.index_select(0, index)
            self._train_store[leaf_path] = compact.clone().detach().to(dtype=dtype)

    def _infer_dtype(self, state: TensorDict) -> torch.dtype:
        for _fkey, (bkey, ckey, lkey) in self._flat_entries:
            t = state.get(bkey).get(ckey).get(lkey)
            if isinstance(t, torch.Tensor):
                return t.dtype
        return torch.float32

    def _infer_device(self, state: TensorDict) -> torch.device:
        for _fkey, (bkey, ckey, lkey) in self._flat_entries:
            t = state.get(bkey).get(ckey).get(lkey)
            if isinstance(t, torch.Tensor):
                return t.device
        return state.device if hasattr(state, "device") else torch.device("cpu")

    def _flush_rollout_current_to_store(self) -> None:
        if self._rollout_current_state is not None and self._rollout_current_env_ids is not None:
            slots = self._map_ids_to_slots(self._rollout_current_env_ids, self._rollout_id2slot, create_missing=True)
            self._scatter_state_by_slots(self._rollout_current_state, slots, store=self._rollout_store)

    def _ensure_rollout_current_state(
        self,
        env_ids_long: torch.Tensor,
        *,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TensorDict:
        if (
            self._rollout_current_state is not None
            and self._rollout_current_env_ids is not None
            and self._rollout_current_env_ids.numel() == env_ids_long.numel()
            and torch.equal(self._rollout_current_env_ids, env_ids_long)
        ):
            return self._rollout_current_state

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


__all__ = ["CortexTDConfig", "CortexTD"]
