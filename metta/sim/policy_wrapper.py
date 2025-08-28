"""Simple policy wrapper for simulation compatibility during SimpleCheckpointManager migration."""

class PolicyWrapper:
    """Simple wrapper that provides .policy attribute for simulation compatibility."""
    
    def __init__(self, policy, uri: str = None, run_name: str = None):
        self.policy = policy
        self.uri = uri or "direct://loaded"
        self.run_name = run_name or "direct_load"
        
        # Create minimal metadata for compatibility
        self.metadata = type('PolicyMetadata', (), {
            'epoch': getattr(policy, 'epoch', 0),
            'score': getattr(policy, 'score', 0.0),
        })()