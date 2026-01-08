import os
from typing import Optional

from torch import Tensor
from torch.library import custom_op
from torch.utils.cpp_extension import load

_mod_path = os.path.dirname(__file__)
_ext: Optional[object] = None


def _load_ext() -> object:
    global _ext
    if _ext is not None:
        return _ext
    sources = [
        os.path.join(_mod_path, "rtu_seq_allin_binding.cpp"),
        os.path.join(_mod_path, "rtu_seq_allin_kernels.cu"),
    ]
    _ext = load(
        name="rtu_seq_allin",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        # Use torch's default build cache under ~/.cache/torch_extensions
        # (avoid creating per-arch folders inside the repo).
        build_directory=None,
        verbose=False,
    )
    return _ext


@custom_op("cortex::rtu_seq_allin_forward", mutates_args=())
def _rtu_seq_allin_forward(
    x: Tensor,
    nu_log: Tensor,
    theta_log: Tensor,
    w1: Tensor,
    w2: Tensor,
    hc1_init: Tensor,
    hc2_init: Tensor,
    E_nu_c1_in: Tensor,
    E_nu_c2_in: Tensor,
    E_th_c1_in: Tensor,
    E_th_c2_in: Tensor,
    E_w1_c1_in: Tensor,
    E_w1_c2_in: Tensor,
    E_w2_c1_in: Tensor,
    E_w2_c2_in: Tensor,
    resets_u8: Tensor,
    act_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    ext = _load_ext()
    return ext.forward_allin(
        x,
        nu_log,
        theta_log,
        w1,
        w2,
        hc1_init,
        hc2_init,
        E_nu_c1_in,
        E_nu_c2_in,
        E_th_c1_in,
        E_th_c2_in,
        E_w1_c1_in,
        E_w1_c2_in,
        E_w2_c1_in,
        E_w2_c2_in,
        resets_u8,
        act_id,
    )


@_rtu_seq_allin_forward.register_fake
def _(
    x: Tensor,
    nu_log: Tensor,
    theta_log: Tensor,
    w1: Tensor,
    w2: Tensor,
    hc1_init: Tensor,
    hc2_init: Tensor,
    E_nu_c1_in: Tensor,
    E_nu_c2_in: Tensor,
    E_th_c1_in: Tensor,
    E_th_c2_in: Tensor,
    E_w1_c1_in: Tensor,
    E_w1_c2_in: Tensor,
    E_w2_c1_in: Tensor,
    E_w2_c2_in: Tensor,
    resets_u8: Tensor,
    act_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, T, H = x.shape
    y_btd_2h = x.new_empty((B, T, 2 * H))
    pre1_bth = x.new_empty((B, T, H))
    pre2_bth = x.new_empty((B, T, H))
    final_hc1_bh = x.new_empty((B, H))
    final_hc2_bh = x.new_empty((B, H))
    E_nu_c1_out = x.new_empty((B, H))
    E_nu_c2_out = x.new_empty((B, H))
    E_th_c1_out = x.new_empty((B, H))
    E_th_c2_out = x.new_empty((B, H))
    E_w1_c1_out = x.new_empty((B, H))
    E_w1_c2_out = x.new_empty((B, H))
    E_w2_c1_out = x.new_empty((B, H))
    E_w2_c2_out = x.new_empty((B, H))
    return (
        y_btd_2h,
        pre1_bth,
        pre2_bth,
        final_hc1_bh,
        final_hc2_bh,
        E_nu_c1_out,
        E_nu_c2_out,
        E_th_c1_out,
        E_th_c2_out,
        E_w1_c1_out,
        E_w1_c2_out,
        E_w2_c1_out,
        E_w2_c2_out,
    )


@custom_op("cortex::rtu_seq_allin_backward", mutates_args=())
def _rtu_seq_allin_backward(
    grad_y: Tensor,
    x: Tensor,
    nu_log: Tensor,
    theta_log: Tensor,
    w1: Tensor,
    w2: Tensor,
    pre1: Tensor,
    pre2: Tensor,
    hc1_init: Tensor,
    hc2_init: Tensor,
    resets_u8: Tensor,
    E_nu_c1_in: Tensor,
    E_nu_c2_in: Tensor,
    E_th_c1_in: Tensor,
    E_th_c2_in: Tensor,
    E_w1_c1_in: Tensor,
    E_w1_c2_in: Tensor,
    E_w2_c1_in: Tensor,
    E_w2_c2_in: Tensor,
    act_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    ext = _load_ext()
    return ext.backward_allin(
        grad_y,
        x,
        nu_log,
        theta_log,
        w1,
        w2,
        pre1,
        pre2,
        hc1_init,
        hc2_init,
        resets_u8,
        E_nu_c1_in,
        E_nu_c2_in,
        E_th_c1_in,
        E_th_c2_in,
        E_w1_c1_in,
        E_w1_c2_in,
        E_w2_c1_in,
        E_w2_c2_in,
        act_id,
    )


@_rtu_seq_allin_backward.register_fake
def _(
    grad_y: Tensor,
    x: Tensor,
    nu_log: Tensor,
    theta_log: Tensor,
    w1: Tensor,
    w2: Tensor,
    pre1: Tensor,
    pre2: Tensor,
    hc1_init: Tensor,
    hc2_init: Tensor,
    resets_u8: Tensor,
    E_nu_c1_in: Tensor,
    E_nu_c2_in: Tensor,
    E_th_c1_in: Tensor,
    E_th_c2_in: Tensor,
    E_w1_c1_in: Tensor,
    E_w1_c2_in: Tensor,
    E_w2_c1_in: Tensor,
    E_w2_c2_in: Tensor,
    act_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, T, H = x.shape
    grad_x_btd = x.new_empty((B, T, H))
    grad_nu_log_h = nu_log.new_empty(nu_log.shape)
    grad_theta_log_h = theta_log.new_empty(theta_log.shape)
    grad_w1_h = w1.new_empty(w1.shape)
    grad_w2_h = w2.new_empty(w2.shape)
    grad_hc1_init_bh = hc1_init.new_empty(hc1_init.shape)
    grad_hc2_init_bh = hc2_init.new_empty(hc2_init.shape)
    return (
        grad_x_btd,
        grad_nu_log_h,
        grad_theta_log_h,
        grad_w1_h,
        grad_w2_h,
        grad_hc1_init_bh,
        grad_hc2_init_bh,
    )


def forward_allin(
    x: Tensor,
    nu_log: Tensor,
    theta_log: Tensor,
    w1: Tensor,
    w2: Tensor,
    hc1_init: Tensor,
    hc2_init: Tensor,
    E_nu_c1_in: Tensor,
    E_nu_c2_in: Tensor,
    E_th_c1_in: Tensor,
    E_th_c2_in: Tensor,
    E_w1_c1_in: Tensor,
    E_w1_c2_in: Tensor,
    E_w2_c1_in: Tensor,
    E_w2_c2_in: Tensor,
    resets_u8: Tensor,
    act_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _rtu_seq_allin_forward(
        x,
        nu_log,
        theta_log,
        w1,
        w2,
        hc1_init,
        hc2_init,
        E_nu_c1_in,
        E_nu_c2_in,
        E_th_c1_in,
        E_th_c2_in,
        E_w1_c1_in,
        E_w1_c2_in,
        E_w2_c1_in,
        E_w2_c2_in,
        resets_u8,
        act_id,
    )


def backward_allin(
    grad_y: Tensor,
    x: Tensor,
    nu_log: Tensor,
    theta_log: Tensor,
    w1: Tensor,
    w2: Tensor,
    pre1: Tensor,
    pre2: Tensor,
    hc1_init: Tensor,
    hc2_init: Tensor,
    resets_u8: Tensor,
    E_nu_c1_in: Tensor,
    E_nu_c2_in: Tensor,
    E_th_c1_in: Tensor,
    E_th_c2_in: Tensor,
    E_w1_c1_in: Tensor,
    E_w1_c2_in: Tensor,
    E_w2_c1_in: Tensor,
    E_w2_c2_in: Tensor,
    act_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _rtu_seq_allin_backward(
        grad_y,
        x,
        nu_log,
        theta_log,
        w1,
        w2,
        pre1,
        pre2,
        hc1_init,
        hc2_init,
        resets_u8,
        E_nu_c1_in,
        E_nu_c2_in,
        E_th_c1_in,
        E_th_c2_in,
        E_w1_c1_in,
        E_w1_c2_in,
        E_w2_c1_in,
        E_w2_c2_in,
        act_id,
    )
