import torch
from tensordict import TensorDict


class MockAgent(torch.nn.Module):
    """
    A mock agent that always does nothing. Used for tests and to run play without requiring a policy.

    This mock agent supports feature remapping for testing purposes while maintaining
    minimal functionality for simulation runs. It mimics the interface of MettaAgent but
    without the complexity of environment setup and policy management.
    """

    def __init__(self):
        super().__init__()
        # Initialize required attributes that tests expect
        self.components_with_memory = []
        self.components = torch.nn.ModuleDict()  # Use ModuleDict for proper nn.Module handling
        self.device = "cpu"
        self.training = True  # PyTorch's training mode flag

        # Feature remapping attributes
        self.original_feature_mapping = None
        self.feature_id_remap = {}
        self.active_features = None
        self.feature_id_to_name = {}
        self.feature_normalizations = {}

        # Action attributes
        self.action_names = None
        self.action_max_params = None
        self.active_actions = []
        self.cum_action_max_params = None
        self.action_index_tensor = None

    def forward(self, td: TensorDict, state=None, action: torch.Tensor | None = None) -> TensorDict:
        """
        Mock forward pass - always returns "do nothing" actions.

        This is a minimal implementation that satisfies the simulation's requirements:
        - Takes a TensorDict with "env_obs"
        - Adds "actions" key with shape [num_agents, 2] for [action_type, action_param]
        - Returns the modified TensorDict

        Args:
            td: TensorDict containing at least "env_obs"
            state: Optional state (ignored in mock)
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

        # Create "do nothing" actions (action_type=0, action_param=0)
        # These are the minimal valid actions that won't cause errors
        actions = torch.zeros((num_agents, 2), dtype=torch.long)

        # Add required outputs to the TensorDict
        # The simulation expects at least the "actions" key
        td["actions"] = actions

        # These are optional but might be expected by some code paths:
        # td["act_log_prob"] = torch.zeros((num_agents,))
        # td["values"] = torch.zeros((num_agents,))

        return td

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        """
        Initialize the agent to work with a specific environment.

        For MockAgent, this sets up feature remapping support while maintaining
        minimal functionality.

        Note: is_training parameter is deprecated and ignored.
        """
        # Store action configuration
        self.activate_actions(action_names, action_max_params, device)

        # Initialize observations to support feature remapping
        self.activate_observations(features, device)

    def activate_actions(self, action_names: list[str], action_max_params: list[int], device):
        """Store action configuration for testing."""
        self.action_names = action_names
        self.action_max_params = action_max_params
        self.device = device
        self.active_actions = list(zip(action_names, action_max_params, strict=False))

        # Precompute cumulative sums for efficient action conversion
        self.cum_action_max_params = torch.cumsum(
            torch.tensor([0] + action_max_params, device=self.device, dtype=torch.long), dim=0
        )

        # Create action index tensor for conversions
        action_index = []
        for action_type_idx, max_param in enumerate(action_max_params):
            for j in range(max_param + 1):
                action_index.append([action_type_idx, j])

        self.action_index_tensor = torch.tensor(action_index, device=self.device, dtype=torch.int32)

    def activate_observations(self, features: dict[str, dict], device):
        """Activate observation features by storing the feature mapping."""
        self.active_features = features
        self.device = device

        # Create quick lookup mappings
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props
        }

        # Store original feature mapping on first initialization
        if self.original_feature_mapping is None:
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
        else:
            # Create remapping for subsequent initializations
            self._create_feature_remapping(features)

    def _create_feature_remapping(self, features: dict[str, dict]):
        """Create a remapping dictionary to translate new feature IDs to original ones."""
        UNKNOWN_FEATURE_ID = 255
        self.feature_id_remap = {}
        unknown_features = []

        for name, props in features.items():
            new_id = props["id"]
            if name in self.original_feature_mapping:
                # Remap known features to their original IDs
                original_id = self.original_feature_mapping[name]
                if new_id != original_id:
                    self.feature_id_remap[new_id] = original_id
            elif not self.training:
                # In eval mode, map unknown features to UNKNOWN_FEATURE_ID
                self.feature_id_remap[new_id] = UNKNOWN_FEATURE_ID
                unknown_features.append(name)
            else:
                # In training mode, learn new features
                self.original_feature_mapping[name] = new_id

        if self.feature_id_remap:
            self._apply_feature_remapping(features, UNKNOWN_FEATURE_ID)

    def _apply_feature_remapping(self, features, unknown_feature_id):
        """
        Apply feature remapping to observation components.

        This is called by _create_feature_remapping to update any observation
        components with the new feature ID mapping.
        """
        if "_obs_" not in self.components:
            return

        # Create remap table (identity mapping by default)
        remap_table = torch.arange(256, dtype=torch.uint8, device=self.device)

        # Apply remappings
        for current_id, target_id in self.feature_id_remap.items():
            remap_table[current_id] = target_id

        # Map original IDs not in current environment to UNKNOWN
        if not self.training:
            original_ids = set(self.original_feature_mapping.values())
            current_ids = {props["id"] for props in features.values()}
            for original_id in original_ids:
                if original_id not in current_ids and original_id < 256:
                    remap_table[original_id] = unknown_feature_id

        # Update the observation component
        if hasattr(self.components["_obs_"], "update_feature_remapping"):
            self.components["_obs_"].update_feature_remapping(remap_table)

    def get_original_feature_mapping(self) -> dict[str, int] | None:
        """Get the original feature mapping for saving in metadata."""
        return self.original_feature_mapping

    def restore_original_feature_mapping(self, mapping: dict[str, int]) -> None:
        """Restore the original feature mapping from metadata."""
        # Make a copy to avoid shared state between agents
        self.original_feature_mapping = mapping.copy()

    def reset_memory(self):
        """Mock implementation - no memory to reset."""
        pass

    def get_memory(self):
        """Mock implementation - returns empty memory dict."""
        return {}
