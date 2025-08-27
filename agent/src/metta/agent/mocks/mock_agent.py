import torch
from tensordict import TensorDict

from metta.agent.metta_agent import MettaAgent


class MockAgent(MettaAgent):
    """
    An agent that always does nothing. Used for tests and to run play without requiring a policy.

    This mock agent supports feature remapping for testing purposes while maintaining
    minimal functionality for simulation runs.
    """

    def __init__(self):
        # Don't call parent __init__ as it requires many parameters we don't have
        # Instead, manually initialize as nn.Module and set required attributes
        torch.nn.Module.__init__(self)

        # Initialize required attributes that MettaAgent expects
        self.components_with_memory = []
        self.components = torch.nn.ModuleDict()  # Use ModuleDict for proper nn.Module handling
        self.device = "cpu"
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
        - Adds "actions" key with shape [num_agents, 2] for [action_type, action_param]
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
        is_training: bool = None,
    ):
        """Initialize the agent to work with a specific environment.

        One-stop shop for setting up agents to interact with environments.
        Handles both new agents and agents loaded from disk with existing feature mappings.

        Auto-detects training vs simulation context:
        - Training context (gradients enabled): Learn new features, remap known features
        - Simulation context (gradients disabled): Remap known features, map unknown to 255
        """
        self.device = device

        # Auto-detect training context if not explicitly provided
        if is_training is None:
            # Use the module's training state (set by .train()/.eval())
            # Training context: self.training=True → learn new features
            # Simulation context: self.training=False → map unknown to 255
            is_training = self.training

        self.training = is_training

        # Store action configuration
        self.action_names = action_names
        self.action_max_params = action_max_params

        # Build feature mappings
        self.feature_id_to_name = {props["id"]: name for name, props in features.items()}
        self.feature_normalizations = {
            props["id"]: props.get("normalization", 1.0) for props in features.values() if "normalization" in props
        }

        # Handle feature remapping for backward compatibility
        if self.original_feature_mapping is None:
            # First initialization - store the mapping
            self.original_feature_mapping = {name: props["id"] for name, props in features.items()}
        else:
            # Re-initialization - create remapping
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
                # Apply the remapping to any observation components
                self._apply_feature_remapping(features, UNKNOWN_FEATURE_ID)

    def reset_memory(self):
        """Mock implementation - no memory to reset."""
        pass

    def get_memory(self):
        """Mock implementation - returns empty memory dict."""
        return {}
