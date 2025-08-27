import torch.nn as nn


class MemoryManager:
    def __init__(self, policy: nn.Module):
        self.policy = policy

    def get_states(self):
        return self.policy.get_states()

    def reset_states(self):
        self.policy.reset_states()

    def get_memory(self):
        return self.policy.get_memory()

    def reset_memory(self):
        self.policy.reset_memory()
