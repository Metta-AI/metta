"""Rollout phase functions for Metta training."""

import hashlib
import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import ModuleDict

from metta.common.profiling.stopwatch import Stopwatch

logger = logging.getLogger(__name__)


PufferlibVecEnv = Any


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    # Dicts/tuples handled by caller
    return np.asarray(x)


def _fingerprint(arr) -> str:
    a = _to_numpy(arr)
    h = hashlib.sha256(a.tobytes()).hexdigest()
    return h[:16]


def debug_dump_obs(obs, env_id=None, mask=None, step: int = 0, *, sample_first_valid: bool = True, max_print: int = 1):
    """
    obs: dict or ndarray/tensor batch from vecenv.recv()
    env_id, mask: from vecenv.recv(); used to pick a representative row
    step: rollout step index (for logging)
    """
    # pick a representative index in the batch
    idx = 0
    if mask is not None:
        m = _to_numpy(mask).reshape(-1).astype(bool)
        if m.any():
            idx = int(np.flatnonzero(m)[0]) if sample_first_valid else 0

    def _summ(a):
        a = _to_numpy(a)
        return dict(
            shape=tuple(a.shape),
            dtype=str(a.dtype),
            min=float(a.min()) if a.size else None,
            max=float(a.max()) if a.size else None,
            mean=float(a.mean()) if a.size else None,
            fp=_fingerprint(a),
        )

    print(
        f"\n[obs] step={step} batch_size="
        f"{next(iter(obs.values())).shape[0] if isinstance(obs, dict) else _to_numpy(obs).shape[0] if _to_numpy(obs).ndim > 0 else 1} "
        f"idx={idx} env_id={None if env_id is None else _to_numpy(env_id).reshape(-1)[idx]}"
    )

    if isinstance(obs, dict):
        # summarize each key, plus show a small slice from one sample
        for k, v in obs.items():
            v_np = _to_numpy(v)
            print(f"  key={k}: {_summ(v_np)}")
            # show a tiny preview from the chosen sample
            if v_np.ndim >= 2:
                sample = v_np[idx]
                # for 3D obs like (H,W,F), show center tile’s feature vector if plausible
                if sample.ndim == 3:
                    H, W = sample.shape[:2]
                    center_feat = sample[H // 2, W // 2]
                    print(
                        f"    sample[{k}] center tile shape={center_feat.shape} preview={center_feat[: min(8, center_feat.size)]}"
                    )
                else:
                    print(f"    sample[{k}] shape={sample.shape} preview={sample.reshape(-1)[:8]}")
            else:
                print(f"    sample[{k}] scalar={v_np}")
    else:
        v_np = _to_numpy(obs)
        print(f"  array: {_summ(v_np)}")
        sample = v_np[idx]
        if sample.ndim == 3:
            H, W = sample.shape[:2]
            center_feat = sample[H // 2, W // 2]
            print(f"    sample center tile shape={center_feat.shape} preview={center_feat[: min(8, center_feat.size)]}")
        else:
            print(f"    sample shape={sample.shape} preview={sample.reshape(-1)[:100]}")


from tensordict import TensorDict


def run_component(comp, x, *, src_key=None, device=None, dtype=torch.float32):
    # discover what key the component expects and what key it will write
    srcs = getattr(comp, "_sources", [{"name": "obs"}])
    src_key = src_key or srcs[0]["name"]
    out_key = getattr(comp, "_name", "out")

    # move to the right device
    if device is None:
        device = next((p.device for p in comp.buffers()), torch.device("cpu"))

    # wrap into a TensorDict with the correct batch size
    xt = torch.as_tensor(x, dtype=dtype, device=device)
    td = TensorDict({src_key: xt}, batch_size=xt.shape[:1], device=device)

    # run the component
    td = comp(td)

    return td[out_key], td  # return result (and the full TD if you want to chain)


def get_observation(
    vecenv: PufferlibVecEnv,
    device: torch.device,
    timer: Stopwatch,
    components: ModuleDict,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list, slice, Tensor, int]:
    """Get observations from vectorized environment and convert to tensors."""
    with timer("_rollout.env"):
        o, r, d, t, info, env_id, mask = vecenv.recv()

    # print every step for the first few, then periodically
    debug_dump_obs(o, env_id=env_id, mask=mask)

    training_env_id = slice(env_id[0], env_id[-1] + 1)

    mask = torch.as_tensor(mask)
    num_steps = int(mask.sum().item())

    # Convert to tensors
    o = torch.as_tensor(o).to(device, non_blocking=True)
    r = torch.as_tensor(r).to(device, non_blocking=True)
    d = torch.as_tensor(d).to(device, non_blocking=True)
    t = torch.as_tensor(t).to(device, non_blocking=True)

    return o, r, d, t, info, training_env_id, mask, num_steps


def send_observation(
    vecenv: PufferlibVecEnv,
    actions: Tensor,
    dtype_actions: np.dtype,
    timer: Stopwatch,
) -> None:
    """Send actions back to the vectorized environment."""
    with timer("_rollout.env"):
        vecenv.send(actions.cpu().numpy().astype(dtype_actions))
