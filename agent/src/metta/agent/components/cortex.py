from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import optree
import torch
import torch.nn as nn
from cortex.config import CortexStackConfig
from cortex.factory import build_cortex
from cortex.stacks import CortexStack
from einops import rearrange
from pydantic import ConfigDict, model_validator
from tensordict import NonTensorData, TensorDict, TensorDictBase
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig
from metta.agent.utils import resolve_torch_dtype

logger = logging.getLogger(__name__)

FlatKey = str


def _td_flatten(td: TensorDictBase) -> Tuple[Iterable[Any], tuple]:
    keys = tuple(td.keys())
    children = [td.get(k) for k in keys]
    meta = (
        keys,
        tuple(td.batch_size) if not isinstance(td.batch_size, tuple) else td.batch_size,
        getattr(td, "device", None),
        td.__class__,
    )
    return children, meta


def _td_unflatten(meta: tuple, children: Iterable[Any]) -> TensorDict:
    keys, _meta_batch_size, _meta_device, td_type = meta
    children_list = list(children)
    data = {k: c for k, c in zip(keys, children_list, strict=False)}
    inferred_bs: Optional[Tuple[int, ...]] = None
    for c in children_list:
        if isinstance(c, TensorDictBase):
            inferred_bs = tuple(c.batch_size)  # type: ignore[assignment]
            break
        if isinstance(c, torch.Tensor):
            inferred_bs = (int(c.shape[0]),)
            break
    if inferred_bs is None:
        inferred_bs = tuple(_meta_batch_size) if isinstance(_meta_batch_size, tuple) else _meta_batch_size
    return td_type(data, batch_size=inferred_bs)


_REGISTERED_TD_NODE = globals().get("_REGISTERED_TD_NODE", False)
if not _REGISTERED_TD_NODE:
    optree.register_pytree_node(TensorDictBase, _td_flatten, _td_unflatten, namespace="torch")
    _REGISTERED_TD_NODE = True


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
    return resets_bool.view(B) if TT == 1 else resets_bool.view(B, TT)


def _metadata_to_int(value: Any, *, name: str) -> int:
    if isinstance(value, NonTensorData):
        return int(value.data)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return int(value.item())
        return int(value.reshape(-1)[0].item())
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} metadata: {value}") from exc


class CortexTDConfig(ComponentConfig):
    """Configuration for Cortex stack integration with TensorDict state caching."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    in_key: str
    out_key: str
    name: str = "cortex"

    d_hidden: int = 128
    out_features: Optional[int] = None
    output_nonlinearity: str = "silu"

    stack_cfg: CortexStackConfig

    key_prefix: str = "cortex_state"
    dtype: str = "float32"
    compute_dtype: Optional[str] = None
    store_dtype: Optional[str] = None  # Back-compat alias for dtype

    pass_state_during_training: bool = True

    def make_component(self, env: Any = None) -> nn.Module:
        return CortexTD(config=self)

    @model_validator(mode="after")
    def _apply_store_dtype_alias(self) -> "CortexTDConfig":
        """Allow older checkpoints to specify store_dtype instead of dtype."""
        if self.store_dtype is None:
            return self
        if self.dtype not in ("float32", self.store_dtype):
            raise ValueError(
                "CortexTDConfig found both dtype and store_dtype with conflicting values; "
                f"dtype={self.dtype}, store_dtype={self.store_dtype}"
            )
        self.dtype = self.store_dtype
        return self


class CortexTD(nn.Module):
    """Stateful Cortex stack component with TensorDict integration."""

    def __init__(self, config: CortexTDConfig) -> None:
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key

        self._storage_dtype: torch.dtype = resolve_torch_dtype(config.dtype)
        compute_override = getattr(config, "compute_dtype", None)
        self._compute_dtype: Optional[torch.dtype] = (
            resolve_torch_dtype(compute_override) if compute_override is not None else None
        )

        scfg: CortexStackConfig = config.stack_cfg
        stack = build_cortex(scfg)
        stack_hidden = int(stack.cfg.d_hidden)  # type: ignore[attr-defined]
        if stack_hidden != int(config.d_hidden):
            raise ValueError(
                f"CortexTDConfig.d_hidden ({config.d_hidden}) does not match stack.cfg.d_hidden ({stack_hidden})."
            )

        if self._storage_dtype is not None and self._storage_dtype is not torch.float32:
            stack = stack.to(dtype=self._storage_dtype)

        self.stack: CortexStack = stack
        self.d_hidden: int = int(config.d_hidden)
        self.out_features: Optional[int] = config.out_features
        self.key_prefix: str = config.key_prefix

        self._state_treedef: Optional[Any] = None
        self._leaf_shapes: List[Tuple[int, ...]] = []

        layers: List[nn.Module] = []
        if self.out_features is not None and self.out_features != self.d_hidden:
            layers.append(nn.Linear(self.d_hidden, self.out_features))
            layers.append(self._make_activation(self.config.output_nonlinearity))
        self._out: nn.Module = nn.Sequential(*layers) if layers else nn.Identity()
        if self._storage_dtype is not None and self._storage_dtype is not torch.float32:
            self._out = self._out.to(dtype=self._storage_dtype)

        self._rollout_store_leaves: List[torch.Tensor] = []
        self._row_store_leaves: List[torch.Tensor] = []

        self._rollout_id2slot: Dict[int, int] = {}
        self._rollout_next_slot: int = 0

        self._rollout_current_state: Optional[TensorDict] = None
        self._rollout_current_env_ids: Optional[torch.Tensor] = None

    def initialize_to_environment(self, _policy_env_info: Any, device: torch.device) -> Optional[str]:
        return None

    def _init_template_if_needed(self, *, B: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._state_treedef is None:
            s1 = self._zero_step_init_state(batch=B, device=device, dtype=dtype)
            self._adopt_template_from_state(s1)

    def _zero_step_init_state(self, *, batch: int, device: torch.device, dtype: torch.dtype) -> TensorDict:
        """Create initial state with zero-input forward pass to materialize all leaves."""
        x0 = torch.zeros(batch, int(self.d_hidden), device=device, dtype=dtype)
        with torch.no_grad():
            _y, s1 = self.stack.step(x0, None)
        return s1

    def _adopt_template_from_state(self, state: TensorDictBase) -> None:
        """Adopt treedef and shapes from a representative state."""
        leaves, treedef = optree.tree_flatten(state, namespace="torch")
        self._state_treedef = treedef
        self._leaf_shapes = []
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                self._leaf_shapes.append(tuple(leaf.shape[1:]))
        self._rollout_store_leaves = []
        self._row_store_leaves = []

    def _maybe_refresh_template(self, state: TensorDictBase) -> None:
        """Refresh template if current state's structure diverges."""
        leaves, _ = optree.tree_flatten(state, namespace="torch")
        if self._state_treedef is None or len(leaves) != len(self._leaf_shapes):
            self._adopt_template_from_state(state)

    def _resolve_compute_dtype(self, caller_dtype: torch.dtype) -> torch.dtype:
        if self._compute_dtype is not None:
            return self._compute_dtype
        return self._storage_dtype if hasattr(self, "_storage_dtype") else caller_dtype

    def _cast_state_dtype(self, state: TensorDictBase, dtype: torch.dtype) -> TensorDict:
        leaves, treedef = optree.tree_flatten(state, namespace="torch")
        casted: List[Any] = []
        for leaf in leaves:
            casted.append(leaf.to(dtype) if isinstance(leaf, torch.Tensor) else leaf)
        return optree.tree_unflatten(treedef, casted)

    @torch._dynamo.disable
    def forward(self, td: TensorDict) -> TensorDict:  # type: ignore[override]
        x = td[self.in_key]

        device = x.device
        storage_dtype = self._storage_dtype
        compute_dtype = self._resolve_compute_dtype(x.dtype)

        TT = _metadata_to_int(td.get("bptt", 1), name="bptt")
        B = _metadata_to_int(td.get("batch", x.shape[0]), name="batch")

        self._init_template_if_needed(B=B, device=device, dtype=storage_dtype)

        if TT <= 0 or B <= 0:
            raise ValueError("'bptt' and 'batch' must be positive integers")
        if x.shape[0] != B * TT:
            raise ValueError(f"input length {x.shape[0]} must equal batch*bptt ({B}*{TT})")

        resets = _as_reset_mask(td.get("dones", None), None, B=B, TT=TT, device=device)

        if TT == 1:
            if "training_env_ids" not in td.keys():
                td.set("training_env_ids", torch.arange(B, device=device, dtype=torch.long).view(B, 1))
                logger.debug(
                    "[CortexTD] Missing 'training_env_ids'; defaulting to arange(B) with B=%d.",
                    B,
                )
            env_ids_2d = td["training_env_ids"].to(device=device, dtype=torch.long)
            assert env_ids_2d.dim() == 2 and env_ids_2d.shape[1] == 1, "training_env_ids must be [B,1]"
            env_ids_long = env_ids_2d.view(-1)

            state_prev = self._ensure_rollout_current_state(env_ids_long, B=B, device=device, dtype=storage_dtype)
            if isinstance(state_prev, TensorDict) and compute_dtype != storage_dtype:
                state_prev = self._cast_state_dtype(state_prev, compute_dtype)

            if "row_id" not in td.keys() or "t_in_row" not in td.keys():
                logger.debug(
                    "[CortexTD] Missing 'row_id' or 't_in_row' during evaluation (TT==1); skipping row_store caching."
                )
            else:
                row_id_flat = td["row_id"].to(device=device, dtype=torch.long).view(-1)
                t_in_row_flat = td["t_in_row"].to(device=device, dtype=torch.long).view(-1)
                mask_start = t_in_row_flat == 0
                if bool(mask_start.any()):
                    idx = torch.nonzero(mask_start, as_tuple=False).reshape(-1)
                    state_sel = self._select_state_rows(state_prev, idx)
                    if isinstance(state_sel, TensorDict) and compute_dtype != storage_dtype:
                        state_sel = self._cast_state_dtype(state_sel, storage_dtype)
                    row_ids_sel = row_id_flat[idx]
                    self._scatter_state_by_slots_list(state_sel, row_ids_sel, store=self._row_store_leaves)

            x_step = x.view(B, -1) if x.dtype is compute_dtype else x.view(B, -1).to(compute_dtype)
            y, state_next = self.stack.step(x_step, state_prev, resets=resets)
            self._rollout_current_state = (
                self._cast_state_dtype(state_next, storage_dtype) if isinstance(state_next, TensorDict) else state_next
            )
            y = self._out(y)
            y = y if y.dtype is x.dtype else y.to(x.dtype)
            td.set(self.out_key, y.reshape(B * TT, -1))
            return td

        if self.config.pass_state_during_training:
            if "row_id" not in td.keys():
                raise KeyError("CortexTD training path (TT>1) requires 'row_id' when pass_state_during_training=True")
            row_tensor = td["row_id"].to(device=device, dtype=torch.long)
            if row_tensor.numel() != B * TT:
                raise ValueError("row_id must contain exactly B*TT elements")
            row_ids = row_tensor.view(B, TT)[:, 0]
            state0 = self._gather_state_by_slots_list(
                row_ids, store=self._row_store_leaves, B=B, device=device, dtype=storage_dtype
            )
            if isinstance(state0, TensorDict) and compute_dtype != storage_dtype:
                state0 = self._cast_state_dtype(state0, compute_dtype)
        else:
            state0 = None
        x_seq = rearrange(x, "(b t) h -> b t h", b=B, t=TT)
        if x_seq.dtype is not compute_dtype:
            x_seq = x_seq.to(compute_dtype)

        y_seq, _ = self.stack(x_seq, state0)
        y_seq = self._out(y_seq)
        if y_seq.dtype is not x.dtype:
            y_seq = y_seq.to(x.dtype)
        td.set(self.out_key, rearrange(y_seq, "b t h -> (b t) h"))
        return td

    def experience_keys(self) -> Dict[FlatKey, torch.Size]:
        """Replay keys required by the component."""
        return {
            "training_env_ids": torch.Size([1]),
            "row_id": torch.Size([]),
            "t_in_row": torch.Size([]),
            "dones": torch.Size([]),
            "truncateds": torch.Size([]),
        }

    def reset_memory(self) -> None:
        """Preserve rollout hidden state and row pre-states across rollouts."""
        return

    def get_memory(self) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        """Serialize dense stores for checkpointing."""

        return {
            "rollout_store": {"leaves": list(self._rollout_store_leaves)},
            "row_store": {"leaves": list(self._row_store_leaves)},
        }

    def set_memory(self, memory) -> None:  # type: ignore[override]
        """Restore dense stores from the structure emitted by ``get_memory``."""

        rollout_leaves = memory.get("rollout_store", {}).get("leaves", [])
        row_leaves = memory.get("row_store", {}).get("leaves", [])
        self._rollout_store_leaves = list(rollout_leaves)
        self._row_store_leaves = list(row_leaves)

    def get_agent_experience_spec(self) -> Composite:
        spec_dict: Dict[str, UnboundedDiscrete] = {}
        for key, shape in self.experience_keys().items():
            if key in ("training_env_ids", "row_id", "t_in_row"):
                dtype = torch.long
            else:
                dtype = torch.float32
            spec_dict[key] = UnboundedDiscrete(shape=torch.Size(shape), dtype=dtype)
        return Composite(spec_dict)

    def _ensure_store_capacity_list(
        self,
        store: List[torch.Tensor],
        min_slots: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        cur_cap = int(store[0].shape[0]) if store else 0
        new_cap = cur_cap
        if cur_cap < min_slots:
            new_cap = max(min_slots, max(1, cur_cap * 2))
        if not store:
            for shape in self._leaf_shapes:
                store.append(torch.zeros((max(1, new_cap), *shape), device=device, dtype=dtype))
            return
        if new_cap != cur_cap:
            for i, old in enumerate(store):
                new = torch.zeros((new_cap, *old.shape[1:]), device=device, dtype=dtype)
                new[: old.shape[0]].copy_(old.to(device=device, dtype=dtype))
                store[i] = new
        else:
            for i, old in enumerate(store):
                if old.device != device or old.dtype != dtype:
                    store[i] = old.to(device=device, dtype=dtype)

    def _gather_state_by_slots_list(
        self,
        slot_ids: torch.Tensor,
        *,
        store: List[torch.Tensor],
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TensorDict:
        if self._state_treedef is None:
            return TensorDict({}, batch_size=[B], device=device)

        if slot_ids.numel() == 0:
            leaves: List[torch.Tensor] = []
            for shape in self._leaf_shapes:
                leaves.append(torch.zeros((B, *shape), device=device, dtype=dtype))
            return optree.tree_unflatten(self._state_treedef, leaves)

        if slot_ids.dim() != 1:
            slot_ids = slot_ids.reshape(-1)
        valid_mask = slot_ids >= 0
        slot_ids_clamped = slot_ids.clamp_min(0)
        max_slot = int(slot_ids_clamped.max().item()) + 1 if slot_ids_clamped.numel() > 0 else 0
        self._ensure_store_capacity_list(store, max_slot, device=device, dtype=dtype)

        gathered_leaves: List[torch.Tensor] = []
        for src in store:
            cap = int(src.shape[0])
            if max_slot > cap:
                raise RuntimeError(f"[CortexTD] slot out of bounds: need<{max_slot} cap={cap}")
            gathered = src.index_select(0, slot_ids_clamped)
            gathered[~valid_mask] = 0
            gathered_leaves.append(gathered.to(dtype=dtype, device=device))
        return optree.tree_unflatten(self._state_treedef, gathered_leaves)

    def _scatter_state_by_slots_list(
        self, state: TensorDictBase, slot_ids: torch.Tensor, *, store: List[torch.Tensor]
    ) -> None:
        if slot_ids.numel() == 0:
            return
        if slot_ids.dim() != 1:
            slot_ids = slot_ids.reshape(-1)
        max_slot = int(slot_ids.max().item()) + 1
        leaves, _ = optree.tree_flatten(state, namespace="torch")
        assert all(isinstance(leaf_item, torch.Tensor) for leaf_item in leaves), "Cortex state leaves must be Tensors"
        leaf_device = leaves[0].device if leaves else torch.device("cpu")
        storage_dtype = self._storage_dtype if hasattr(self, "_storage_dtype") else leaves[0].dtype
        self._ensure_store_capacity_list(store, max_slot, device=leaf_device, dtype=storage_dtype)
        if not store:
            return
        for leaf, dest in zip(leaves, store, strict=False):
            dest.index_copy_(0, slot_ids, leaf.to(dtype=dest.dtype).detach())

    def _select_state_rows(self, state: TensorDictBase, idx: torch.Tensor) -> TensorDict:
        """Return state with each tensor leaf indexed by given rows."""
        if idx.dim() != 1:
            idx = idx.reshape(-1)
        leaves, _ = optree.tree_flatten(state, namespace="torch")
        sel_leaves: List[torch.Tensor] = [leaf.index_select(0, idx) for leaf in leaves]
        return optree.tree_unflatten(self._state_treedef, sel_leaves)

    def _flush_rollout_current_to_store(self) -> None:
        if self._rollout_current_state is not None and self._rollout_current_env_ids is not None:
            slots = self._map_ids_to_slots(self._rollout_current_env_ids, self._rollout_id2slot, create_missing=True)
            self._scatter_state_by_slots_list(self._rollout_current_state, slots, store=self._rollout_store_leaves)

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
        state_prev = self._gather_state_by_slots_list(
            slots, store=self._rollout_store_leaves, B=B, device=device, dtype=dtype
        )
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

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        n = name.lower()
        if n in ("silu", "swish"):
            return nn.SiLU()
        if n == "relu":
            return nn.ReLU()
        if n == "tanh":
            return nn.Tanh()
        if n in ("linear", "identity"):
            return nn.Identity()
        raise ValueError(f"Unsupported output_nonlinearity '{name}'. Allowed: silu/swish, relu, tanh, linear/identity.")


__all__ = ["CortexTDConfig", "CortexTD"]
