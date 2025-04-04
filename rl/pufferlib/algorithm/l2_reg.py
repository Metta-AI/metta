import numpy as np

import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch


class L2Reg():
    def __init__(self, l2_reg_loss_coef, l2_init_loss_coef):
        self.l2_reg_loss_coef = l2_reg_loss_coef
        self.l2_init_loss_coef = l2_init_loss_coef

    def compute_loss(self, losses):
        if self.l2_reg_loss_coef > 0:
            losses['l2_reg'] = self.l2_reg_loss_coef * self.policy.l2_reg_loss().to(self.device)

        if self.l2_init_loss_coef > 0:
            losses['l2_init'] = self.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)


