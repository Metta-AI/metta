import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from metta.agent.components.component_config import ComponentConfig


class LSTMResetConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "lstm_reset"
    latent_size: int = 128
    hidden_size: int = 128
    num_layers: int = 2

    def make_component(self, env=None):
        return LSTMReset(config=self)


class LSTMReset(nn.Module):
    """
    LSTM layer that resets cell states when the episode is done or truncated in both inference and training. The file
    lstm.py only resets state in inference but runs much faster because it doesn't need to unroll the LSTM state as in
    the class above, LstmTrainStep.

    The layer leaves the choice of whether to use burn-in in the trainer loop to get to a stable LSTM state to the
    trainer.

    It also gets the correct cell state at the start of a segment during training by reading from the replay buffer.
    This has the limitation that the state gets more and more stale with more and more minibatches. However, it's a
    limitation imposed by prioritized experience replay - it doesn't let us pull from segments in temporal order.
    """

    def __init__(self, config: LSTMResetConfig):
        super().__init__()
        self.config = config
        self.latent_size = self.config.latent_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key
        self.net = nn.LSTM(self.latent_size, self.hidden_size, self.num_layers)
        self._in_training = False

        for name, param in self.net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # torch's default is uniform

        self.register_buffer("lstm_h", torch.empty(self.num_layers, 0, self.hidden_size))
        self.register_buffer("lstm_c", torch.empty(self.num_layers, 0, self.hidden_size))

    def __setstate__(self, state):
        """Ensure LSTM hidden states are properly initialized after loading from checkpoint."""
        self.__dict__.update(state)
        # Reset hidden states when loading from checkpoint to avoid batch size mismatch
        if not hasattr(self, "lstm_h"):
            self.lstm_h = torch.empty(self.num_layers, 0, self.hidden_size)
        if not hasattr(self, "lstm_c"):
            self.lstm_c = torch.empty(self.num_layers, 0, self.hidden_size)

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def forward(self, td: TensorDict):
        latent = td[self.config.in_key]  # → (batch * TT, hidden_size)

        B = td.batch_size.numel()
        if td["bptt"][0] != 1:
            TT = td["bptt"][0]
            self._in_training = True
        else:
            TT = 1
        B = B // TT

        if self._in_training and TT == 1:
            # we're at a transition from training to rollout, so we need to reset the memory
            self._in_training = False
            # self._reset_memory() # av we shouldn't need this

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            training_env_ids = torch.arange(B, device=latent.device)
        else:
            training_env_ids = training_env_ids.reshape(B * TT)  # av why reshape this? should already be B*TT

        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is not None and truncateds is not None:
            reset_mask = (dones.bool() | truncateds.bool()).view(1, -1, 1)
        else:
            # we're in eval
            reset_mask = torch.zeros(1, B * TT, 1, dtype=torch.bool, device=latent.device)

        if TT == 1:
            self.max_num_envs = training_env_ids.max() + 1
            if self.max_num_envs > self.lstm_h.size(1):
                num_allocated_envs = self.max_num_envs - self.lstm_h.size(1)
                # we haven't allocated states for these envs (ie the very first epoch or rollout)
                h_0 = torch.zeros(self.num_layers, num_allocated_envs, self.hidden_size, device=latent.device)
                c_0 = torch.zeros(self.num_layers, num_allocated_envs, self.hidden_size, device=latent.device)
                device = self.lstm_h.device
                self.lstm_h = torch.cat([self.lstm_h, h_0.detach()], dim=1).to(device)
                self.lstm_c = torch.cat([self.lstm_c, c_0.detach()], dim=1).to(device)

            h_0 = self.lstm_h[:, training_env_ids]
            c_0 = self.lstm_c[:, training_env_ids]
            h_0 = h_0.masked_fill(reset_mask, 0)
            c_0 = c_0.masked_fill(reset_mask, 0)
            td["lstm_h"] = h_0.permute(1, 0, 2).detach()
            td["lstm_c"] = c_0.permute(1, 0, 2).detach()

        latent = rearrange(latent, "(b t) h -> t b h", b=B, t=TT)
        if TT != 1:
            h_0 = td["lstm_h"]
            c_0 = td["lstm_c"]
            h_0 = rearrange(h_0, "(b t) x y -> b t x y", b=B, t=TT)[:, 0].permute(1, 0, 2)
            c_0 = rearrange(c_0, "(b t) x y -> b t x y", b=B, t=TT)[:, 0].permute(1, 0, 2)

            hidden = self._forward_train(latent, h_0, c_0, reset_mask)

        else:
            hidden, (h_n, c_n) = self.net(latent, (h_0, c_0))
            self.lstm_h[:, training_env_ids] = h_n.detach()
            self.lstm_c[:, training_env_ids] = c_n.detach()

        hidden = rearrange(hidden, "t b h -> (b t) h")

        td[self.out_key] = hidden

        return td

    def _forward_train(
        self,
        latent: torch.Tensor,
        h_0: torch.Tensor,
        c_0: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized training forward pass with per-time-step resets."""
        T, B, _ = latent.shape
        device = latent.device

        reset_mask = reset_mask.view(1, B, T, 1)
        reset_flags = reset_mask[0, :, :, 0].transpose(0, 1)  # [T, B]

        first_step_reset = reset_flags[0]
        if first_step_reset.any():
            mask = first_step_reset.view(1, B, 1)
            h_0 = h_0.masked_fill(mask, 0)
            c_0 = c_0.masked_fill(mask, 0)

        segment_start = reset_flags.clone()
        segment_start[0] = True

        time_idx = torch.arange(T, device=device).view(T, 1).expand(T, B)
        start_idx = torch.where(segment_start, time_idx, torch.zeros_like(time_idx))
        last_start_idx = torch.cummax(start_idx, dim=0)[0]
        segment_pos = time_idx - last_start_idx

        segment_ids_local = torch.cumsum(segment_start.long(), dim=0) - 1
        max_segments_per_env = int(segment_ids_local.max().item()) + 1
        stride = max_segments_per_env if max_segments_per_env > 0 else 1

        env_offsets = torch.arange(B, device=device) * stride
        global_segment_ids = segment_ids_local + env_offsets.unsqueeze(0)
        flat_segment_ids = global_segment_ids.reshape(-1)

        unique_segments, inverse_indices = torch.unique(flat_segment_ids, return_inverse=True)
        num_segments = unique_segments.numel()

        segment_lengths = torch.bincount(inverse_indices, minlength=num_segments)
        max_len = int(segment_pos.max().item()) + 1

        latent_flat = latent.reshape(-1, latent.size(-1))
        position_flat = segment_pos.reshape(-1)

        segments_padded = torch.zeros(
            max_len,
            num_segments,
            latent.size(-1),
            device=device,
            dtype=latent.dtype,
        )
        segments_padded[position_flat, inverse_indices] = latent_flat

        env_ids_per_segment = unique_segments // stride
        local_segment_ids = unique_segments % stride
        first_segment_mask = local_segment_ids == 0

        initial_h = torch.zeros(
            self.num_layers,
            num_segments,
            self.hidden_size,
            device=device,
            dtype=latent.dtype,
        )
        initial_c = torch.zeros_like(initial_h)

        if first_segment_mask.any():
            selected_envs = env_ids_per_segment[first_segment_mask]
            initial_h[:, first_segment_mask] = h_0[:, selected_envs]
            initial_c[:, first_segment_mask] = c_0[:, selected_envs]

        packed = pack_padded_sequence(
            segments_padded,
            segment_lengths.to(torch.long).cpu(),
            enforce_sorted=False,
        )

        packed_output, _ = self.net(packed, (initial_h, initial_c))
        padded_output, _ = pad_packed_sequence(packed_output)

        hidden_flat = padded_output[position_flat, inverse_indices]
        hidden = hidden_flat.view(T, B, self.hidden_size)
        return hidden

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            {
                "lstm_h": UnboundedDiscrete(
                    shape=torch.Size([self.num_layers, self.hidden_size]),
                    dtype=torch.float32,
                ),
                "lstm_c": UnboundedDiscrete(
                    shape=torch.Size([self.num_layers, self.hidden_size]),
                    dtype=torch.float32,
                ),
                "dones": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
                "truncateds": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            }
        )

    def get_memory(self):
        return self.lstm_h, self.lstm_c

    def set_memory(self, memory):
        """Cannot be called at the Policy level - use policy.<path_to_this_layer>.set_memory()"""
        self.lstm_h, self.lstm_c = memory[0], memory[1]

    def reset_memory(self):
        pass

    def _reset_memory(self):
        self.lstm_h.fill_(0)
        self.lstm_c.fill_(0)
