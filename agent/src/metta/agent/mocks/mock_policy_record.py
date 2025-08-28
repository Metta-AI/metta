"""Mock PolicyRecord implementation for testing SimpleCheckpointManager migration."""

from metta.agent.mocks.mock_agent import MockAgent
from metta.sim.policy_wrapper import PolicyRecord


class MockPolicyRecord(PolicyRecord):
    """Mock PolicyRecord for testing that doesn't require real checkpoint files."""
    
    def __init__(self, uri: str = "mock://test", epoch: int = 0, score: float = 0.5, agent_step: int = 0, policy=None):
        if policy is None:
            policy = MockAgent()
        
        super().__init__(
            policy=policy, 
            uri=uri, 
            run_name="mock_run", 
            epoch=epoch
        )
        
        # Override metadata with test values
        self.metadata.score = score
        self.metadata.agent_step = agent_step