import torch
from einops import rearrange
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


def adapter_for_commit_hash_1af886(x: torch.Tensor) -> torch.Tensor:
    """
    Backwards compatibility adapter for observation format changes in commit 1af886ae.

    This commit consolidated inventory features from separate agent:inv:* and inv:* slots
    into shared inv:* slots, reducing the feature dimension from 34 to 26.

    This adapter converts the new 26-feature format back to the old 34-feature format
    by duplicating inventory features into both agent-specific and shared slots.

    Args:
        x: Input tensor with shape [..., 11, 11, 26] (new format)

    Returns:
        Tensor with shape [..., 11, 11, 34] (old format)
    """
    if x.shape[-1] != 26:
        # Not the format we need to adapt
        return x

    # Map from new consolidated format (26 features) to old format (34 features)
    # New format inventory indices
    inv_ore_red_idx = 6  # "inv:ore.red"
    inv_ore_blue_idx = 7  # "inv:ore.blue"
    inv_ore_green_idx = 8  # "inv:ore.green"
    inv_battery_idx = 9  # "inv:battery"
    inv_heart_idx = 10  # "inv:heart"
    inv_armor_idx = 11  # "inv:armor"
    inv_laser_idx = 12  # "inv:laser"
    inv_blueprint_idx = 13  # "inv:blueprint"

    # Create the expanded tensor for old format
    batch_dims = x.shape[:-1]
    expanded_shape = batch_dims + (34,)
    expanded_x = torch.zeros(expanded_shape, dtype=x.dtype, device=x.device)

    # Copy features to old format positions:

    # Features 0-5: agent, agent:group, hp, agent:frozen, agent:orientation, agent:color
    expanded_x[..., 0:6] = x[..., 0:6]

    # Features 6-13: Copy inventory to agent:inv:* slots (old format)
    expanded_x[..., 6] = x[..., inv_ore_red_idx]  # agent:inv:ore.red
    expanded_x[..., 7] = x[..., inv_ore_blue_idx]  # agent:inv:ore.blue
    expanded_x[..., 8] = x[..., inv_ore_green_idx]  # agent:inv:ore.green
    expanded_x[..., 9] = x[..., inv_battery_idx]  # agent:inv:battery
    expanded_x[..., 10] = x[..., inv_heart_idx]  # agent:inv:heart
    expanded_x[..., 11] = x[..., inv_armor_idx]  # agent:inv:armor
    expanded_x[..., 12] = x[..., inv_laser_idx]  # agent:inv:laser
    expanded_x[..., 13] = x[..., inv_blueprint_idx]  # agent:inv:blueprint

    # Features 14-18: wall, swappable, mine, color, converting
    expanded_x[..., 14:19] = x[..., 14:19]

    # Features 19-26: Copy inventory to shared inv:* slots (old format)
    expanded_x[..., 19:27] = x[..., 6:14]  # inv:ore.red through inv:blueprint

    # Features 27-33: generator, altar, armory, lasery, lab, factory, temple
    expanded_x[..., 27:34] = x[..., 19:26]


class ObsShaper(LayerBase):
    """
    This class does the following:
    1) permutes input observations from [B, H, W, C] or [B, TT, H, W, C] to [..., C, H, W]
    2) inspects tensor shapes, ensuring that input observations match expectations from the environment
    3) inserts batch size, TT, and B * TT into the tensor dict for certain other layers in the network to use
       if they need reshaping.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent is instantiated
    and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, obs_shape, num_objects, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)  # make sure no Omegaconf types are used in forward passes
        self._out_tensor_shape = [self._obs_shape[2], self._obs_shape[0], self._obs_shape[1]]
        self._output_size = num_objects

    def _forward(self, td: TensorDict):
        x = td["x"]

        # Apply backwards compatibility adapter
        x = adapter_for_commit_hash_1af886(x)

        x_shape, space_shape = x.shape, self._obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if tuple(x_shape[-space_n:]) != tuple(space_shape):
            expected_shape = f"[B(, T), {', '.join(str(dim) for dim in space_shape)}]"
            actual_shape = f"{list(x_shape)}"
            raise ValueError(
                f"Shape mismatch error:\n"
                f"x.shape: {x.shape}\n"
                f"self._obs_shape: {self._obs_shape}\n"
                f"Expected tensor with shape {expected_shape}\n"
                f"Got tensor with shape {actual_shape}\n"
                f"The last {space_n} dimensions should match {tuple(space_shape)}"
            )

        # Validate overall tensor dimensionality with improved error message
        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError(
                f"Invalid input tensor dimensionality:\n"
                f"Expected tensor with {space_n + 1} or {space_n + 2} dimensions\n"
                f"Got tensor with {x_n} dimensions: {list(x_shape)}\n"
                f"Expected format: [batch_size(, time_steps), {', '.join(str(dim) for dim in space_shape)}]"
            )

        x = x.reshape(B * TT, *space_shape)
        x = x.float()

        x = self._permute(x)

        td["_TT_"] = TT
        td["_batch_size_"] = B
        td["_BxTT_"] = B * TT
        td[self._name] = x
        return td

    def _permute(self, x):
        if x.device.type == "mps":
            x = self._mps_permute(x)
        else:
            x = rearrange(x, "b h w c -> b c h w")
        return x

    def _mps_permute(self, x):
        """For compatibility with MPS, it throws an error on .permute()"""
        bs, h, w, c = x.shape
        x = x.contiguous().view(bs, h * w, c)
        x = x.transpose(1, 2)
        x = x.contiguous().view(bs, c, h, w)
        return x
