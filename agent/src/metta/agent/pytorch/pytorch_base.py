"""Base class for PyTorch agents with common functionality."""

import logging

import pufferlib.models
import torch
import torch.nn.functional as F
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class PytorchAgentBase(pufferlib.models.LSTMWrapper):
    """Base class for PyTorch agents that provides common initialization functionality."""

    def forward(self, td: TensorDict, state=None, action=None):
        """
        Forward pass using TensorDict interface compatible with MettaAgent.

        This base implementation expects child classes to have:
        - self.policy.encode_observations(observations, state)
        - self.policy.decode_actions(hidden)
        - self.action_index_tensor (set by MettaAgent.activate_actions)
        - self.cum_action_max_params (set by MettaAgent.activate_actions)
        """
        observations = td["env_obs"].to(self.device)

        if state is None:
            state = {"lstm_h": None, "lstm_c": None, "hidden": None}

        # Encode observations
        hidden = self.policy.encode_observations(observations, state)

        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        # Prepare LSTM state
        lstm_h, lstm_c = state.get("lstm_h"), state.get("lstm_c")
        if lstm_h is not None and lstm_c is not None:
            lstm_h = lstm_h.to(self.device)[: self.lstm.num_layers]
            lstm_c = lstm_c.to(self.device)[: self.lstm.num_layers]
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        # Forward LSTM
        hidden = hidden.view(B, TT, -1).transpose(0, 1)  # (TT, B, in_size)
        lstm_output, (new_lstm_h, new_lstm_c) = self.lstm(hidden, lstm_state)
        flat_hidden = lstm_output.transpose(0, 1).reshape(B * TT, -1)

        # Decode actions and value
        logits_list, value = self.policy.decode_actions(flat_hidden)

        if action is None:
            # ---------- Inference Mode ----------
            log_probs = F.log_softmax(logits_list, dim=-1)
            action_probs = torch.exp(log_probs)

            actions = torch.multinomial(action_probs, num_samples=1).view(-1)
            batch_indices = torch.arange(actions.shape[0], device=actions.device)
            full_log_probs = log_probs[batch_indices, actions]

            action = self._convert_logit_index_to_action(actions)

            td["actions"] = action.to(dtype=torch.int32)
            td["act_log_prob"] = full_log_probs
            td["values"] = value.flatten()
            td["full_log_probs"] = log_probs

        else:
            # ---------- Training Mode ----------
            action = action.to(self.device)
            if action.dim() == 3:  # (B, T, 2) → flatten to (BT, 2)
                B, T, A = action.shape
                action = action.view(B * T, A)

            action_log_probs = F.log_softmax(logits_list, dim=-1)
            action_probs = torch.exp(action_log_probs)
            action_logit_index = self._convert_action_to_logit_index(action)

            batch_indices = torch.arange(action_logit_index.shape[0], device=action_logit_index.device)
            full_log_probs = action_log_probs[batch_indices, action_logit_index]

            entropy = -(action_probs * action_log_probs).sum(dim=-1)

            td["act_log_prob"] = full_log_probs.view(B, TT)
            td["entropy"] = entropy.view(B, TT)
            td["full_log_probs"] = action_log_probs.view(B, T, -1)
            td["value"] = value.view(B, TT)

        return td

    def clip_weights(self):
        """Clip weights of the actor heads to prevent large updates."""
        pass

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return cumulative_sum + action_params
