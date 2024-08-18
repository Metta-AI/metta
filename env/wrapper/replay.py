import torch
import os

class Replay():
    def __init__(self, filename):
        self.filename = filename
        self.data = {
            "steps": [],
            "global_obs": [],
        }

    def record_step(self, actions, obs, rewards, infos, global_obs):
        self.data["global_obs"].append(global_obs)

    def close(self):
        print(f"Saving replay to {self.filename}")
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save(self.data, self.filename)

    @staticmethod
    def load(filename):
        data = torch.load(filename)
        r = Replay(filename)
        r.data = data
        return r
