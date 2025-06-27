"""Feature ID remapper for observation tokens."""

import torch
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class ObsFeatureRemapper(LayerBase):
    """
    Remaps feature IDs in observation tokens to match the original training mapping.

    When a trained model is loaded into a new environment, feature IDs might be different.
    For example, "mineral" might be ID 5 in the new environment but was ID 3 during training.
    This layer translates new feature IDs to the original ones the model expects.

    The remapping happens on the second byte of each observation token (the feature_id field).
    """

    def __init__(self, **cfg):
        super().__init__(**cfg)
        # Initialize with identity mapping (no remapping)
        self.register_buffer("feature_id_remap", torch.arange(256, dtype=torch.uint8))
        self._remapping_active = False

    def _initialize(self):
        """Initialize the layer - called during setup."""
        # No learnable parameters, just remapping
        if self._sources is not None and len(self._sources) > 0:
            self._out_tensor_shape = self._in_tensor_shapes[0].copy()
        else:
            # Default shape for token observations
            self._out_tensor_shape = [0, 3]  # Variable length, 3 channels

    def update_remapping(self, feature_id_remap: torch.Tensor):
        """
        Update the remapping table.

        Args:
            feature_id_remap: A 256-element tensor where index is new_id and value is original_id
        """
        self.feature_id_remap = feature_id_remap.to(self.feature_id_remap.device)
        # Check if remapping is actually needed (not identity)
        identity = torch.arange(256, dtype=torch.uint8, device=self.feature_id_remap.device)
        self._remapping_active = not torch.equal(self.feature_id_remap, identity)

    def _forward(self, td: TensorDict) -> TensorDict:
        observations = td[self._sources[0]["name"]]

        if self._remapping_active:
            # Clone to avoid modifying the original
            observations = observations.clone()

            # Extract feature IDs (second byte of each token)
            feature_ids = observations[..., 1].long()

            # Remap feature IDs
            remapped_ids = self.feature_id_remap[feature_ids]

            # Update the observations with remapped IDs
            observations[..., 1] = remapped_ids

        td[self._name] = observations
        return td
