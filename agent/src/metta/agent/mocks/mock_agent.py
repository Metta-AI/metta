import torch
from tensordict import TensorDict

from metta.agent.policy import Policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class MockAgent(Policy):
    """
    An agent that always does nothing. Used for tests and to run play without requiring a policy.

    This mock agent supports feature remapping for testing purposes while maintaining
    minimal functionality for simulation runs.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface | None = None) -> None:
        if policy_env_info is None:
            from mettagrid.config.mettagrid_config import MettaGridConfig

            policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
        super().__init__(policy_env_info)

        # Initialize required attributes that MettaAgent expects
        self.components_with_memory = []
        self.components = torch.nn.ModuleDict()  # Use ModuleDict for proper nn.Module handling
        self._device: torch.device = torch.device("cpu")
        self.policy = None  # MockAgent doesn't have a separate policy

        # Initialize feature remapping attributes
        self.original_feature_mapping = None
        self.feature_id_remap = {}

    def _apply_feature_remapping(self, features: dict[str, dict], unknown_id: int):
        """Apply feature remapping to observation components."""
        # Build complete remapping tensor
        remap_tensor = torch.arange(256, dtype=torch.uint8, device=self.device)

        # Apply explicit remappings
        for old_id, new_id in self.feature_id_remap.items():
            remap_tensor[old_id] = new_id

        # Map original IDs not in current env to unknown
        if self.original_feature_mapping:
            current_ids = {props["id"] for props in features.values()}
            for original_id in self.original_feature_mapping.values():
                if original_id not in current_ids and original_id < 256:
                    remap_tensor[original_id] = unknown_id

        # Apply to observation components
        for name, component in self.components.items():
            if name.startswith("_obs_") and hasattr(component, "update_feature_remapping"):
                component.update_feature_remapping(remap_tensor)

    def get_original_feature_mapping(self) -> dict[str, int] | None:
        """Get the original feature mapping for persistence."""
        return self.original_feature_mapping.copy() if self.original_feature_mapping else None

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        """
        Mock forward pass - always returns "do nothing" actions.

        This is a minimal implementation that satisfies the simulation's requirements:
        - Takes a TensorDict with "env_obs"
        - Adds "actions" key with shape [num_agents] representing discrete action ids
        - Returns the modified TensorDict

        Args:
            td: TensorDict containing at least "env_obs"
            action: Optional action tensor (ignored in mock - only used in training)

        Returns:
            TensorDict with mock actions added
        """
        # Get batch size from env_obs if it exists, otherwise default to 1
        if "env_obs" in td:
            env_obs = td["env_obs"]
            num_agents = env_obs.shape[0]
        else:
            # Some tests might not provide env_obs, default to batch size 1
            num_agents = td.batch_size[0] if td.batch_size else 1

        # Create "do nothing" actions (flattened index 0)
        actions = torch.zeros((num_agents,), dtype=torch.long)

        # Add required outputs to the TensorDict
        # The simulation expects at least the "actions" key
        td["actions"] = actions

        # These are optional but might be expected by some code paths:
        # td["act_log_prob"] = torch.zeros((num_agents,))
        # td["values"] = torch.zeros((num_agents,))

        return td

    def initialize_to_environment(
        self,
        policy_env_info: "PolicyEnvInterface",
        device: torch.device,
        *,
        is_training: bool | None = None,
    ) -> None:
        """Initialize the agent to work with a specific environment."""

        self._device = torch.device(device)

        if is_training is None:
            is_training = self.training
        self.training = is_training

        # Action configuration
        self.action_names = list([action.name for action in policy_env_info.actions.actions()])

        features = policy_env_info.obs_features
        self.feature_id_to_name = {feat.id: feat.name for feat in features}
        self.feature_normalizations = {feat.id: feat.normalization for feat in features}

        if self.original_feature_mapping is None:
            self.original_feature_mapping = {name: feat.id for name, feat in features.items()}
            return

        UNKNOWN_FEATURE_ID = 255
        self.feature_id_remap = {}

        for feat in features:
            new_id = feat.id
            if feat.name in self.original_feature_mapping:
                original_id = self.original_feature_mapping[feat.name]
                if new_id != original_id:
                    self.feature_id_remap[new_id] = original_id
            elif not self.training:
                self.feature_id_remap[new_id] = UNKNOWN_FEATURE_ID

    def reset_memory(self) -> None:
        """Mock implementation - no memory to reset."""
        pass

    def get_memory(self) -> dict:
        """Mock implementation - returns empty memory dict."""
        return {}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def total_params(self) -> int:
        return sum(param.numel() for param in self.parameters())
