from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

from metta.agent.util.debug import assert_shape
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions


class ComponentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.components = None
        self.clip_range = 0.0
        self.cum_action_max_params = None
        self.action_index_tensor = None

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        """Forward pass of the ComponentPolicy - matches original MettaAgent forward() logic."""

        if self.components is None:
            raise ValueError("No components found. Ensure components are added in YAML.")

        # Handle BPTT reshaping like the original
        td.bptt = 1
        td.batch = td.batch_size.numel()
        if td.batch_dims > 1:
            B = td.batch_size[0]
            TT = td.batch_size[1]
            td = td.reshape(td.batch_size.numel())  # flatten to BT
            td.bptt = TT
            td.batch = B

        # Run value head (also runs core network if present)
        self.components["_value_"](td)

        # Run action head (reuses core network output)
        self.components["_action_"](td)

        # Select forward pass type
        if action is None:
            output_td = self.forward_inference(td)
        else:
            output_td = self.forward_training(td, action)
            output_td = output_td.reshape(td.batch, td.bptt)

        return output_td

    def forward_inference(self, td: TensorDict) -> TensorDict:
        """Inference mode - sample actions and store them in td."""
        value = td["_value_"]
        logits = td["_action_"]

        if __debug__:
            assert_shape(value, ("BT", 1), "inference_value")
            assert_shape(logits, ("BT", "A"), "inference_logits")

        action_logit_index, action_log_prob, _, full_log_probs = sample_actions(logits)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "action_logit_index")
            assert_shape(action_log_prob, ("BT",), "action_log_prob")

        action = self._convert_logit_index_to_action(action_logit_index)

        if __debug__:
            assert_shape(action, ("BT", 2), "inference_action")

        td["actions"] = action
        td["act_log_prob"] = action_log_prob
        td["values"] = value.flatten()
        td["full_log_probs"] = full_log_probs

        return td

    def forward_training(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        """Training mode - evaluate provided actions."""
        value = td["_value_"]
        logits = td["_action_"]

        if __debug__:
            assert_shape(value, ("BT", 1), "training_value")
            assert_shape(logits, ("BT", "A"), "training_logits")
            assert_shape(action, ("B", "T", 2), "training_input_action")

        B, T, A = action.shape
        flattened_action = action.view(B * T, A)
        action_logit_index = self._convert_action_to_logit_index(flattened_action)

        if __debug__:
            assert_shape(action_logit_index, ("BT",), "converted_action_logit_index")

        action_log_prob, entropy, full_log_probs = evaluate_actions(logits, action_logit_index)

        if __debug__:
            assert_shape(action_log_prob, ("BT",), "training_action_log_prob")
            assert_shape(entropy, ("BT",), "training_entropy")
            assert_shape(full_log_probs, ("BT", "A"), "training_log_probs")

        td["act_log_prob"] = action_log_prob
        td["entropy"] = entropy
        td["value"] = value
        td["full_log_probs"] = full_log_probs

        return td

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Convert (action_type, action_param) pairs to discrete indices."""
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return action_type_numbers + cumulative_sum + action_params

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        """Convert logit indices back to action pairs."""
        return self.action_index_tensor[action_logit_index]

    def clip_weights(self):
        """Apply weight clipping if enabled."""
        if self.clip_range > 0:
            self._apply_to_components("clip_weights")
