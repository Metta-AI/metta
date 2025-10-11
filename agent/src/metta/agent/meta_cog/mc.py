import torch


class MetaCogAction:
    def __init__(self, name):
        self.name = name  # name must be unique to others in the policy
        self.action_fn = None

    def initialize(self, index):
        self.action_index = index

    def attach_apply_method(self, fn):
        """Attach a callable that takes env_ids tensor."""
        self.action_fn = fn
        return self

    def __call__(self, env_ids: torch.Tensor):
        if self.action_fn is None:
            raise RuntimeError(f"MetaCogAction '{self.name}' called before action_fn was attached")
        return self.action_fn(env_ids)
