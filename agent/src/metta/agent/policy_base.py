from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from metta.agent.policy_state import PolicyState

import torch
import torch.nn as nn

class PolicyBase(nn.Module, ABC):
    """Base class for all policies with standardized interface."""

    def __init__(self, name: str = "BasePolicy"):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(
        self,
        agent: "MettaAgent",
        obs: Dict[str, torch.Tensor],
        state: Optional[PolicyState] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Execute policy forward pass."""
        pass
