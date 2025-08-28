"""Policy wrapper and compatibility layer for SimpleCheckpointManager migration."""

class PolicyWrapper:
    """Wrapper that provides compatibility with PolicyRecord interface using SimpleCheckpointManager."""
    
    def __init__(self, policy, uri: str = None, run_name: str = None, epoch: int = None):
        self.policy = policy
        self.uri = uri or "direct://loaded"
        self.run_name = run_name or "direct_load"
        
        # Create metadata compatible with PolicyRecord interface
        self.metadata = type('PolicyMetadata', (), {
            'epoch': epoch if epoch is not None else getattr(policy, 'epoch', 0),
            'score': getattr(policy, 'score', 0.0),
            'agent_step': getattr(policy, 'agent_step', 0),
        })()


# Alias for backward compatibility with eval system
PolicyRecord = PolicyWrapper