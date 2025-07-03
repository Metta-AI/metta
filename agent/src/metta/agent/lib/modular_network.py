"""Modular network for Metta architecture.

Author: Axel
Created: 2024-03-19

"""

from typing import Dict, List, Set

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_module import MettaModule


class ModularNetwork(MettaModule):
    """A network composed of MettaModules that can be dynamically assembled.

    This class provides a way to build networks by connecting MettaModules together.
    Each module's output keys are mapped to input keys of subsequent modules through
    the network's forward pass.

    The way this works is that each node corresponds to a MettaModule, and the network
    keeps track of the mapping between output keys and the node that produces them.
    The network then uses this mapping to route the output tensors to the input keys of
    the next node in the network.

    For example, if we have:
    - Node A with out_keys=["x", "y"]
    - Node B with in_keys=["x", "z"]
    The network will automatically route Node A's "x" output to Node B's "x" input.

    Attributes:
        nodes: Dictionary mapping node identifiers to their MettaModules
        out_key_to_node: Maps output keys to node identifiers for routing
    """

    def __init__(self):
        super().__init__(in_keys=[], out_keys=[])
        self.nodes: nn.ModuleDict = nn.ModuleDict()
        self.all_in_keys: Set[str] = set()
        self.all_out_keys: Set[str] = set()
        self.out_key_to_node: Dict[str, str] = {}
        self._computed_components: Set[str] = set()

    def add_component(self, node_id: str, module: MettaModule) -> None:
        """Add a component to the network.

        Args:
            node_id: Unique identifier for the network node
            module: MettaModule instance to add to this node

        Raises:
            ValueError: If node_id is already used or module is not a MettaModule
        """
        if node_id in self.nodes:
            raise ValueError(f"Node identifier '{node_id}' is already used")

        self.nodes[node_id] = module
        # Update routing map for this module's output keys
        for out_key in module.out_keys:
            self.out_key_to_node[out_key] = node_id

        self._update_network_keys()

        if __debug__:
            self._validate_network()

    def compute_component(self, node_id: str, td: TensorDict) -> TensorDict:
        """Compute the output of a component.
        Caution: This function does not clear the computation cache.
        If you want to clear the cache, you need to call _computed_components.clear() first.

        Args:
            node_id: Identifier of the component to compute
            td: Input TensorDict containing initial tensors

        Returns:
            The updated TensorDict with this component's outputs
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node identifier '{node_id}' not found")

        if node_id in self._computed_components:
            return td

        module = self.nodes[node_id]  # type: ignore[index]

        # Recursively compute inputs
        for in_key in module.in_keys:  # type: ignore[attr-defined]
            if in_key in self.out_key_to_node:
                higher_level_component = self.out_key_to_node[in_key]
                self.compute_component(higher_level_component, td)

        # Compute this component's outputs
        td = module(td)
        self._computed_components.add(node_id)

        return td

    @property
    def network_in_keys(self) -> List[str]:
        """Returns in-keys unmatched by out_keys"""
        return list(self.all_in_keys - self.all_out_keys)

    @property
    def network_out_keys(self) -> List[str]:
        """Returns out-keys unmatched by in_keys"""
        return list(self.all_out_keys - self.all_in_keys)

    def _update_network_keys(self) -> None:
        """Update the network's in_keys and out_keys based on the current components."""
        self.all_in_keys = set()
        self.all_out_keys = set()
        for module in self.nodes.values():  # type: ignore[attr-defined]
            self.all_in_keys.update(module.in_keys)  # type: ignore[attr-defined]
            self.all_out_keys.update(module.out_keys)  # type: ignore[attr-defined, union-attr]

        self.in_keys = self.network_in_keys
        self.out_keys = self.network_out_keys

    def _compute(self, td: TensorDict) -> Dict[str, torch.Tensor]:
        """Compute outputs by processing through all components.

        Args:
            td: Input TensorDict containing initial tensors

        Returns:
            Dictionary mapping output keys to their tensors
        """
        # Clear computation cache for new forward pass
        self._computed_components.clear()

        # Compute all output keys
        for out_key in self.out_keys:
            self.compute_component(self.out_key_to_node[out_key], td)

        # Return only the requested output keys
        return {key: td[key] for key in self.out_keys}

    def _validate_network(self) -> None:
        """Validate the network."""
        # Check for cycles in the network
        for node_id, module in self.nodes.items():
            for in_key in module.in_keys:
                if in_key in self.out_key_to_node:
                    higher_level_component = self.out_key_to_node[in_key]
                    if higher_level_component == node_id:
                        raise ValueError(f"Cycle detected in network: {node_id} -> {higher_level_component}")

        # Check that all out_keys are unique
        if len(self.out_keys) != len(set(self.out_keys)):
            raise ValueError("Out-keys are not unique")
