from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Only run when self.reset_in_training is true ie you're asking to unroll the LSTM in time."""
        # latent is (TT, B, input_size)
        # h_0 is (num_layers, B, input_size)
        # c_0 is (num_layers, B, input_size)
        # reset_mask is (1, B, TT, 1)

        outputs = []
        for t in range(latent.size(0)):
            latent_t = latent[t].unsqueeze(0)
            reset_mask_t = reset_mask[0, :, t, :]
            h_t = h_t.masked_fill(reset_mask_t, 0)
            c_t = c_t.masked_fill(reset_mask_t, 0)
            hidden_t, (h_t, c_t) = self.lstm(latent_t, (h_t, c_t))  # one time step
            outputs.append(hidden_t)

        hidden = torch.cat(outputs, dim=0)
        return hidden, (h_t, c_t)


class LSTM(LayerBase):
    """
    LSTM layer that handles tensor reshaping and state management automatically.

    This class wraps a PyTorch LSTM with proper tensor shape handling, making it easier
    to integrate LSTMs into neural network policies without dealing with complex tensor
    manipulations. It handles reshaping inputs/outputs, manages hidden states, and ensures
    consistent tensor dimensions throughout the forward pass.

    The layer processes tensors of shape [B, TT, ...] or [B, ...], where:
    - B is the batch size
    - TT is an optional time dimension

    It reshapes inputs appropriately for the LSTM, processes them through the network,
    and reshapes outputs back to the expected format, while also managing the LSTM's
    hidden state.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.hidden_size = self._nn_params["hidden_size"]
        self.num_layers = self._nn_params["num_layers"]
        self.reset_in_training = cfg.get("reset_in_training", False)
        self.lstm_train_step = None

        self.lstm_h = torch.empty(self.num_layers, 0, self.hidden_size)
        self.lstm_c = torch.empty(self.num_layers, 0, self.hidden_size)
        self.train_input_lstm_h = torch.empty(self.num_layers, 0, self.hidden_size)
        self.train_input_lstm_c = torch.empty(self.num_layers, 0, self.hidden_size)
        # self.train_output_lstm_h = torch.empty(self.num_layers, 0, self.hidden_size)
        # self.train_output_lstm_c = torch.empty(self.num_layers, 0, self.hidden_size)

    def __setstate__(self, state):
        """Ensure LSTM hidden states are properly initialized after loading from checkpoint."""
        self.__dict__.update(state)
        # AV NOTE: would it be better if I set these dicts to registered buffers?
        # Reset hidden states when loading from checkpoint to avoid batch size mismatch
        self.reset_memory()

    def on_rollout_start(self):
        # self.reset_memory()
        pass

    def on_train_phase_start(self):
        pass

    def on_mb_start(self):
        pass

    def on_eval_start(self):
        # Note, this will affect the agent when you go back to training (you just wiped its memory)
        self.reset_memory()

    def has_memory(self):
        return True

    def get_memory(self):
        return self.lstm_h, self.lstm_c

    def set_memory(self, memory):
        """Cannot be called at the MettaAgent level - use policy.component[this_layer_name].set_memory()"""
        self.lstm_h, self.lstm_c = memory[0], memory[1]

    def reset_memory(self):
        device = next(self.lstm.parameters()).device
        self.lstm_h = torch.empty(self.num_layers, 0, self.hidden_size, device=device)
        self.lstm_c = torch.empty(self.num_layers, 0, self.hidden_size, device=device)
        self.train_input_lstm_h = torch.empty(self.num_layers, 0, self.hidden_size, device=device)
        self.train_input_lstm_c = torch.empty(self.num_layers, 0, self.hidden_size, device=device)

    def setup(self, source_components):
        """Setup the layer and create the network."""
        super().setup(source_components)
        self._net = self._make_net()

    def _make_net(self):
        # Get hidden_size from _nn_params
        hidden_size = self._nn_params.get("hidden_size", self.hidden_size)
        self._out_tensor_shape = [hidden_size]

        self.lstm = nn.LSTM(self._in_tensor_shapes[0][0], **self._nn_params)
        if self.reset_in_training:
            # self.lstm_train_step = torch.jit.script(LstmTrainStep(self.lstm))
            self.lstm_train_step = LstmTrainStep(self.lstm)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # torch's default is uniform

        # make registered buffers?
        self.lstm_h = torch.empty(self.num_layers, 0, self.hidden_size)
        self.lstm_c = torch.empty(self.num_layers, 0, self.hidden_size)
        self.train_input_lstm_h = torch.empty(self.num_layers, 0, self.hidden_size)
        self.train_input_lstm_c = torch.empty(self.num_layers, 0, self.hidden_size)
        self.iter = -1
        self.burn_in = 0
        self.max_num_envs = 0
        self.training_TT = 8

        return None

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def _forward(self, td: TensorDict):
        latent = td[self._sources[0]["name"]]  # → (batch * TT, hidden_size)

        TT = td["bptt"][0]
        B = td["batch"][0]
        segment_ids = td.get("segment_ids", None)
        training_env_ids = td.get("training_env_ids", None)

        if segment_ids is None:
            # then we are in eval or similar. we just need indices for storing memory for the batch
            segment_ids = torch.arange(B, device=latent.device)
        if training_env_ids is None:
            training_env_ids = torch.arange(B, device=latent.device)
        else:
            training_env_ids = training_env_ids.reshape(B * TT)  # av why reshape this? should already be B*TT

        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is not None and truncateds is not None:
            if dones.max() > 0:
                breakpoint()
            reset_mask = (dones.bool() | truncateds.bool()).view(1, -1, 1)  # av check if this is right in TT !=1
            if TT != 1:
                breakpoint()
        else:
            # we're in eval
            reset_mask = torch.ones(1, B, 1, device=latent.device)

        if TT == 1:
            self.iter += 1
            # if segment_ids.max() >= self.lstm_h.size(1):
            self.max_num_envs = training_env_ids.max() + 1
            if self.max_num_envs > self.lstm_h.size(1):
                num_allocated_envs = self.max_num_envs - self.lstm_h.size(1)
                # we haven't allocated states for these envs (ie the very first epoch or rollout)
                h_0 = torch.zeros(self.num_layers, num_allocated_envs, self.hidden_size, device=latent.device)
                c_0 = torch.zeros(self.num_layers, num_allocated_envs, self.hidden_size, device=latent.device)
                self.lstm_h = torch.cat([self.lstm_h, h_0.detach()], dim=1)
                self.lstm_c = torch.cat([self.lstm_c, c_0.detach()], dim=1)
                print(f"Allocated {num_allocated_envs} environments on iteration {self.iter}.")

            h_0 = self.lstm_h[:, training_env_ids]
            c_0 = self.lstm_c[:, training_env_ids]
            td["lstm_h"] = h_0.permute(1, 0, 2).detach()
            td["lstm_c"] = c_0.permute(1, 0, 2).detach()

            td["check_seg_id"] = segment_ids
            td["check_env_id"] = training_env_ids

            if self.burn_in < 2 * self.training_TT:
                self.burn_in += 1
            elif self.iter % self.training_TT == 0:
                pass
                # self.train_input_lstm_h = torch.cat([self.train_input_lstm_h, h_0.detach()], dim=1)
                # self.train_input_lstm_c = torch.cat([self.train_input_lstm_c, c_0.detach()], dim=1)

        latent = rearrange(latent, "(b t) h -> t b h", b=B, t=TT)
        if self.reset_in_training and TT != 1:
            self.iter = -1
            self.burn_in = 0

            segment_ids = segment_ids.reshape(B, TT)[:, 0]
            training_env_ids = training_env_ids.reshape(B, TT)[:, 0]
            check_seg_id = td["check_seg_id"].reshape(B, TT)[:, 0]
            check_env_id = td["check_env_id"].reshape(B, TT)[:, 0]

            if check_seg_id != segment_ids or check_env_id != training_env_ids:
                breakpoint()

            h_0 = td["lstm_h"]
            c_0 = td["lstm_c"]
            h_0 = rearrange(h_0, "(b t) x y -> b t x y", b=B, t=TT)[:, 0].permute(1, 0, 2)
            c_0 = rearrange(c_0, "(b t) x y -> b t x y", b=B, t=TT)[:, 0].permute(1, 0, 2)

            # h_0 = self.train_input_lstm_h[:, segment_ids]
            # c_0 = self.train_input_lstm_c[:, segment_ids]
            hidden, (h_n, c_n) = self._forward_train_step(latent, h_0, c_0, reset_mask)

        else:
            hidden, (h_n, c_n) = self.lstm(latent, (h_0, c_0))
            self.lstm_h[:, training_env_ids] = h_n.detach()
            self.lstm_c[:, training_env_ids] = c_n.detach()

        hidden = rearrange(hidden, "t b h -> (b t) h")

        td[self._name] = hidden

        return td

    def _forward_train_step(self, latent, h_t, c_t, reset_mask):
        """Run the JIT-scripted LSTM training step."""
        reset_mask = reset_mask.view(1, latent.size(1), -1, 1)  # Shape: [1, B, TT, 1]
        return self.lstm_train_step(latent, h_t, c_t, reset_mask)
