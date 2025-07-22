"""Multi-agent training API for Metta."""

from .reward import Competition, Collaboration
from .evolution import Stable, Evolving
from .diversity import WeightDiversity, BehavioralDiversity
from .trainer import MultiAgentTrainer

__all__ = [
    "Competition",
    "Collaboration", 
    "Stable",
    "Evolving",
    "WeightDiversity",
    "BehavioralDiversity",
    "MultiAgentTrainer",
]