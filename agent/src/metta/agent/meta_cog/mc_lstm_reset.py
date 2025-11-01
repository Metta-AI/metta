from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig
from metta.agent.meta_cog.mc import MetaCogAction


class LstmTrainStep(nn.Module):
    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        self.lstm = lstm

    def forward(
        self,
        latent: torch.Tensor,
        h_t: torch.Tensor,
        c_t: torch.Tensor,
        reset_mask: torch.Tensor,
        clear_mask_bt: Optional[torch.Tensor] = None,  # [B, TT] or None
        noise_std_bt: Optional[torch.Tensor] = None,  # [B, TT] or None (0.0 for no noise)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = []
        for t in range(latent.size(0)):
            latent_t = latent[t].unsqueeze(0)
            reset_mask_t = reset_mask[0, :, t, :]  # [B, 1]

            # Apply per-step clear requests. `cm` comes from scheduled mc actions during training
            # and is shaped [B, 1] for the current unrolled timestep; we OR it into the reset mask.
            if clear_mask_bt is not None:
                cm = clear_mask_bt[:, t].view(-1, 1).to(torch.bool)
                reset_mask_t = reset_mask_t | cm

            # Zero states where reset/clear requested
            h_t = h_t.masked_fill(reset_mask_t, 0)
            c_t = c_t.masked_fill(reset_mask_t, 0)

            # Apply per-step Gaussian noise to states prior to LSTM step. `std_t` is a [B] vector
            # of standard deviations (0.0 means no noise for that env at this step). We broadcast
            # to [1, B, 1] so the noise scales per-env across all layers/hidden dims.
            if noise_std_bt is not None:
                std_t = noise_std_bt[:, t].to(torch.float32)  # [B]
                if torch.any(std_t != 0):
                    std_broadcast = std_t.view(1, -1, 1)
                    noise = torch.randn_like(h_t) * std_broadcast
                    h_t = h_t + noise
                    c_t = c_t + noise
            hidden_t, (h_t, c_t) = self.lstm(latent_t, (h_t, c_t))  # one time step
            outputs.append(hidden_t)

        hidden = torch.cat(outputs, dim=0)
        return hidden, (h_t, c_t)


class MCLSTMResetConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "mc_lstm_reset"
    latent_size: int = 128
    hidden_size: int = 128
    num_layers: int = 2

    def make_component(self, env=None):
        return MCLSTMReset(config=self)


class MCLSTMReset(nn.Module):
    """
    LSTM layer that supports MetaCog (MC) internal actions to manipulate its hidden state per
    environment. In rollout (TT==1), actions apply immediately to persistent buffers; in training
    (TT>1), actions are scheduled per row and applied at each unrolled timestep.

    Summary
    -------
    - Maintains per-environment LSTM buffers `lstm_h` and `lstm_c`.
    - Resets state on done/truncated via a binary mask.
    - Exposes mc actions:
      - `lstm_clear`: zero-out state for specified envs.
      - `lstm_noise_1`/`lstm_noise_2`: add Gaussian noise to state for specified envs.
    - Training: mc actions mark per-row pending instructions (`pending_clear_rows`,
      `pending_noise_code`). During unrolling, these are reshaped to [B, TT] and applied per
      timestep in `LstmTrainStep`.
    - Rollout: mc actions mutate `lstm_h/lstm_c` directly.

    Design details
    --------------
    - Pending instruction buffers are registered with `persistent=False`; they are reconstructed
      at init and grown per-batch, avoiding stale state in checkpoints. They are also cleared
      after use in each training forward pass.
    - No-op safety: If no mc actions are called, pending buffers remain zero/False and have no
      effect. Likewise, in rollout, if no action is called, buffers are untouched.
    - Per-step noise: Noise stds are derived from `pending_noise_code` and mapped to
      {0.0, 0.1, 0.5}. We compute `noise = randn_like(state) * std_broadcast`, broadcasting the
      per-env std across layers and hidden dimension. Zeros produce no noise cost.
    - Clear precedence: Clear is OR-ed with the environment reset mask and applied prior to the
      LSTM step, ensuring immediate zeroing at the chosen timestep.

    Constraints
    -----------
    - `training_env_ids` must be shaped [B*TT]; this class uses it to index persistent buffers in
      rollout and to derive pending buffer lengths in training.
    - Buffers must be large enough to index target envs. This class grows `lstm_h/c` lazily in
      rollout and grows pending buffers per batch in training.
    - Numeric stability: Noise is small (0.1/0.5) and added prior to the step to avoid exploding
      variance through the LSTM dynamics.

    Extension patterns
    ------------------
    - Add new mc actions by creating `MetaCogAction` members and attaching a scheduling function
      that writes into new pending buffers (training) or mutates persistent buffers directly
      (rollout). Prefer vectorized indexing over loops.
    - If additional per-step signals are needed, extend `LstmTrainStep.forward` with another
      optional tensor argument shaped [B, TT] and apply it in the loop similarly to clear/noise.
    """

    def __init__(self, config: MCLSTMResetConfig):
        super().__init__()
        self.config = config
        self.latent_size = self.config.latent_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key
        self.net = nn.LSTM(self.latent_size, self.hidden_size, self.num_layers)
        self.lstm_train_step = LstmTrainStep(self.net)
        self._in_training = False

        # define internal actions
        self.noise_1 = MetaCogAction("lstm_noise_1")
        self.noise_1.attach_apply_method(self.inject_noise_1)
        self.noise_2 = MetaCogAction("lstm_noise_2")
        self.noise_2.attach_apply_method(self.inject_noise_2)
        self.clear_action = MetaCogAction("lstm_clear")
        self.clear_action.attach_apply_method(self.clear)
        # MetaCogActions get their .initialize() called by policy auto builder when it is built

        for name, param in self.net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # torch's default is uniform

        self.register_buffer("lstm_h", torch.empty(self.num_layers, 0, self.hidden_size))
        self.register_buffer("lstm_c", torch.empty(self.num_layers, 0, self.hidden_size))

        # Per-row pending mc actions (training only). Non-persistent per batch.
        self.register_buffer("pending_noise_code", torch.zeros(0, dtype=torch.int8), persistent=False)
        self.register_buffer("pending_clear_rows", torch.zeros(0, dtype=torch.bool), persistent=False)

    @torch.no_grad()
    def _ensure_pending_capacity(self, required_rows: int, device: torch.device) -> None:
        if required_rows <= 0:
            return
        current = self.pending_noise_code.size(0)
        if required_rows <= current:
            return
        new_noise = torch.zeros(required_rows, dtype=torch.int8, device=device)
        if current > 0:
            new_noise[:current].copy_(self.pending_noise_code)
        self.pending_noise_code.resize_(required_rows)
        self.pending_noise_code.copy_(new_noise)

        new_clear = torch.zeros(required_rows, dtype=torch.bool, device=device)
        if self.pending_clear_rows.numel() > 0:
            new_clear[: self.pending_clear_rows.size(0)].copy_(self.pending_clear_rows)
        self.pending_clear_rows.resize_(required_rows)
        self.pending_clear_rows.copy_(new_clear)

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
            reset_mask = torch.zeros(1, B, 1, dtype=torch.bool, device=latent.device)

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

            # Build per-step masks from pending rows (length B*TT)
            rows = B * TT
            self._ensure_pending_capacity(rows, latent.device)
            noise_code = self.pending_noise_code[:rows]
            clear_rows = self.pending_clear_rows[:rows]

            noise_code_bt = rearrange(noise_code, "(b t) -> b t", b=B, t=TT).to(torch.int8)
            clear_mask_bt = rearrange(clear_rows, "(b t) -> b t", b=B, t=TT).to(torch.bool)

            # Map codes to stds per step
            noise_std_bt = torch.zeros(B, TT, dtype=torch.float32, device=latent.device)
            if (noise_code_bt == 1).any():
                noise_std_bt = torch.where(noise_code_bt == 1, torch.tensor(0.1, device=latent.device), noise_std_bt)
            if (noise_code_bt == 2).any():
                noise_std_bt = torch.where(noise_code_bt == 2, torch.tensor(0.5, device=latent.device), noise_std_bt)

            hidden, (h_n, c_n) = self._forward_train_step(latent, h_0, c_0, reset_mask, clear_mask_bt, noise_std_bt)

            # Clear pending per-batch instructions after use
            self.pending_noise_code.zero_()
            self.pending_clear_rows.zero_()

        else:
            hidden, (h_n, c_n) = self.net(latent, (h_0, c_0))
            self.lstm_h[:, training_env_ids] = h_n.detach()
            self.lstm_c[:, training_env_ids] = c_n.detach()

        hidden = rearrange(hidden, "t b h -> (b t) h")

        td[self.out_key] = hidden

        return td

    def _forward_train_step(self, latent, h_t, c_t, reset_mask, clear_mask_bt=None, noise_std_bt=None):
        """Run the LSTM training step with per-step clear/noise controls."""
        reset_mask = reset_mask.view(1, latent.size(1), -1, 1)  # Shape: [1, B, TT, 1]
        return self.lstm_train_step(latent, h_t, c_t, reset_mask, clear_mask_bt, noise_std_bt)

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

    def inject_noise_1(self, env_ids):
        if env_ids.numel() == 0:
            return
        if self._in_training:
            if env_ids.dtype != torch.long:
                env_ids = env_ids.long()
            self._ensure_pending_capacity(int(env_ids.max().item()) + 1, self.lstm_h.device)
            self.pending_noise_code[env_ids] = 1
        else:
            self.inject_noise(env_ids, noise_std=0.1, noise_mean=0.0, apply_to="both", generator=None)

    def inject_noise_2(self, env_ids):
        if env_ids.numel() == 0:
            return
        if self._in_training:
            if env_ids.dtype != torch.long:
                env_ids = env_ids.long()
            self._ensure_pending_capacity(int(env_ids.max().item()) + 1, self.lstm_h.device)
            self.pending_noise_code[env_ids] = 2
        else:
            self.inject_noise(env_ids, noise_std=0.5, noise_mean=0.0, apply_to="both", generator=None)

    def clear(self, env_ids):
        if env_ids.numel() == 0:
            return
        if self._in_training:
            if env_ids.dtype != torch.long:
                env_ids = env_ids.long()
            self._ensure_pending_capacity(int(env_ids.max().item()) + 1, self.lstm_h.device)
            self.pending_clear_rows[env_ids] = True
        else:
            self.lstm_h[:, env_ids].fill_(0)
            self.lstm_c[:, env_ids].fill_(0)

    def inject_noise(
        self,
        env_ids: torch.Tensor,
        noise_std: float,
        noise_mean: float = 0.0,
        apply_to: str = "both",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Inject Gaussian noise into LSTM states for the specified environments.

        Parameters
        ----------
        env_ids: torch.Tensor
            1D LongTensor of environment indices to perturb.
        noise_std: float
            Standard deviation of the Gaussian noise.
        noise_mean: float
            Mean of the Gaussian noise. Default is 0.0.
        apply_to: str
            Which states to perturb: "h", "c", or "both". Default is "both".
        generator: Optional[torch.Generator]
            Optional random generator for reproducibility.
        """
        if apply_to not in ("h", "c", "both"):
            raise ValueError("apply_to must be one of {'h', 'c', 'both'}")

        if env_ids.numel() == 0:
            return

        if env_ids.dtype != torch.long:
            env_ids = env_ids.long()

        device = self.lstm_h.device
        env_ids = env_ids.to(device)

        # Ensure buffers are large enough to index into provided env ids
        max_env_id = int(env_ids.max().item())
        if max_env_id >= self.lstm_h.size(1):
            return

        k = env_ids.numel()
        noise_shape = (self.num_layers, k, self.hidden_size)
        noise = torch.normal(
            mean=noise_mean,
            std=noise_std,
            size=noise_shape,
            device=device,
            generator=generator,
        )

        if apply_to in ("h", "both"):
            self.lstm_h[:, env_ids] = (self.lstm_h[:, env_ids] + noise).detach()
        if apply_to in ("c", "both"):
            self.lstm_c[:, env_ids] = (self.lstm_c[:, env_ids] + noise).detach()
