import numpy as np

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

from .ppo import PPO

class RewardClip():
    def __init__(self):
       pass

    def on_post_step(self, experience, state):
        # Clip rewards
        if self.trainer_cfg.clip_reward:
            r = torch.clamp(state["env"]["rewards"], -1, 1)
