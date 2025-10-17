from typing import Optional

import torch
from torch.amp import custom_bwd, custom_fwd

from cortex.kernels.triton.slstm.triton_fused.slstm_bw import (
    backward_sequence as slstm_backward_sequence,
)
from cortex.kernels.triton.slstm.triton_fused.slstm_fw import (
    forward_sequence as slstm_forward_sequence,
)


def _rnn_fwbw_generator(autocast_kernel_dtype: torch.dtype) -> torch.autograd.Function:
    class _rnn_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        def forward(
            ctx,
            states_initial: torch.Tensor,  # (NS, B, NH, D)
            Wx: torch.Tensor,  # (B, T, NGI, NH, D)
            R: torch.Tensor,  # (NGR, NH, Dout, Din)
            b: torch.Tensor,  # (NGI, NH, D)
            resets: Optional[torch.Tensor] = None,  # (B, T) reset mask
            backward_recurrent_clip_val: float | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            true_batch_size = Wx.size(0)
            (all_states, last_state), all_gates = slstm_forward_sequence(
                states_initial=states_initial,
                Wx=Wx,
                R=R,
                b=b,
                resets=resets,
                output_gates_and_states_initial=True,
            )
            ctx.save_for_backward(all_states, all_gates, R)
            ctx.backward_recurrent_clip_val = backward_recurrent_clip_val
            ctx.resets = resets.detach() if resets is not None else None
            if last_state.ndim == 4:
                last_state_out = last_state[:, :true_batch_size, ...]
            elif last_state.ndim == 5:
                last_state_out = last_state[:, :, :true_batch_size, ...]
            else:
                raise ValueError(f"Invalid last_state shape: {last_state.shape}")
            return all_states[1:, :, :true_batch_size, ...], last_state_out

        @staticmethod
        @custom_bwd(device_type="cuda")
        def backward(
            ctx,
            delta_states_all_outside: torch.Tensor,  # (T, NS, B, NH, D)
            delta_states_last_outside: torch.Tensor,  # (NS, B, NH, D)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
            true_batch_size = delta_states_all_outside.size(2)
            all_states, all_gates, R = ctx.saved_tensors
            resets: Optional[torch.Tensor] = ctx.resets
            backward_recurrent_clip_val = ctx.backward_recurrent_clip_val

            delta_states_initial, delta_Wx, delta_R, delta_b = slstm_backward_sequence(
                delta_states_all_outside=delta_states_all_outside,
                delta_states_last_outside=delta_states_last_outside,
                R=R,
                states_all=all_states,
                gates_all=all_gates,
                backward_recurrent_clip_val=backward_recurrent_clip_val,
                true_B=true_batch_size,
                resets=resets,
            )
            return delta_states_initial, delta_Wx, delta_R, delta_b, None, None

    return _rnn_fwbw


slstm_tr_fp32_fwbw = _rnn_fwbw_generator(torch.float32)
slstm_tr_fp16_fwbw = _rnn_fwbw_generator(torch.float16)
slstm_tr_bf16_fwbw = _rnn_fwbw_generator(torch.bfloat16)

slstm_pt_registry = {
    "float32": slstm_tr_fp32_fwbw,
    "float16": slstm_tr_fp16_fwbw,
    "bfloat16": slstm_tr_bf16_fwbw,
}


def slstm_tr_fwbw(
    states_initial: torch.Tensor,  # (NS, B, NH, D)
    Wx: torch.Tensor,  # (B, T, NGI, NH, D)
    R: torch.Tensor,  # (NGR, NH, Dout, Din)
    b: torch.Tensor,  # (NGI, NH, D)
    resets: Optional[torch.Tensor] = None,  # (B, T) reset mask
    backward_recurrent_clip_val: float | None = None,
    autocast_kernel_dtype: str = "float32",
) -> tuple[torch.Tensor, torch.Tensor]:
    slstm_func = slstm_pt_registry[autocast_kernel_dtype]

    all_states, last_state = slstm_func.apply(
        states_initial,
        Wx,
        R,
        b,
        resets,
        backward_recurrent_clip_val,
    )
    return all_states, last_state
