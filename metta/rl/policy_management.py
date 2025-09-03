import logging

import torch

from metta.agent.metta_agent import DistributedMettaAgent, PolicyAgent

logger = logging.getLogger(__name__)


def wrap_agent_distributed(agent: PolicyAgent, device: torch.device) -> PolicyAgent:
    if torch.distributed.is_initialized():
        return DistributedMettaAgent(agent, device)
    return agent
