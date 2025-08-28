"""Minimal PolicyStore replacement for simulation tools."""

import logging
from pathlib import Path
import torch
from metta.rl.simple_checkpoint_manager import SimpleCheckpointManager
from metta.sim.policy_wrapper import PolicyWrapper

logger = logging.getLogger(__name__)


class SimplePolicyStore:
    """Minimal PolicyStore replacement that works with SimpleCheckpointManager."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    @classmethod
    def create(cls, device: str = "cpu", **kwargs):
        """Create a SimplePolicyStore - ignores all legacy parameters."""
        return cls(device=device)
    
    def policy_record(self, policy_uri: str | None = None) -> PolicyWrapper | None:
        """Load policy and return wrapped in PolicyWrapper."""
        if not policy_uri:
            return None
            
        return self.policy_record_or_mock(policy_uri, "unknown")
    
    def policy_record_or_mock(self, policy_uri: str | None, run_name: str) -> PolicyWrapper:
        """Load policy or return mock if not found."""
        if policy_uri and policy_uri.startswith("file://"):
            # Handle file:// URIs - assume they point to checkpoint directories
            checkpoint_dir = policy_uri.replace("file://", "")
            if Path(checkpoint_dir).is_dir():
                try:
                    # Try to load from SimpleCheckpointManager
                    parent_dir = str(Path(checkpoint_dir).parent) 
                    run_name_actual = Path(checkpoint_dir).parent.name
                    checkpoint_manager = SimpleCheckpointManager(
                        run_dir=parent_dir, run_name=run_name_actual
                    )
                    
                    agent = checkpoint_manager.load_agent()
                    if agent:
                        logger.info(f"Loaded agent from SimpleCheckpointManager: {policy_uri}")
                        return PolicyWrapper(agent, uri=policy_uri, run_name=run_name_actual)
                except Exception as e:
                    logger.warning(f"Failed to load policy from {policy_uri}: {e}")
        
        # Return mock if we can't load the real policy
        logger.info(f"Creating mock policy for {policy_uri}")
        from metta.agent.mocks.mock_agent import MockAgent
        mock_agent = MockAgent()
        return PolicyWrapper(mock_agent, uri=policy_uri or "mock://", run_name=run_name)