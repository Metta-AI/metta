from tensordict import TensorDict

class Algorithm:
    def __init__(self):
        pass

    def make_experience_buffers(self, experience: TensorDict):
        pass

    def on_pre_step(self, experience: TensorDict, state: TensorDict):
        pass

    def on_post_step(self, experience: TensorDict, state: TensorDict):
        pass

    def store_experience(self, experience: TensorDict, state: TensorDict):
        pass
