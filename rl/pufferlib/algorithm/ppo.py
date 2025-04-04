import fast_gae
import numpy as np

from tensordict import TensorDict
import torch

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

from .algorithm import Algorithm
from ..experience import Experience

class PPO(Algorithm):
    def __init__(self, gamma, gae_lambda, clip_vloss, vf_clip_coef, vf_coef, ent_coef):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_vloss = clip_vloss
        self.vf_clip_coef = vf_clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def make_experience_buffers(self, experience: TensorDict):
        batch_size = experience["env"].batch_size
        experience["ppo"] = TensorDict({
            "values": torch.zeros(batch_size),
            "advantages": torch.zeros(batch_size),
            "returns": torch.zeros(batch_size),
            "logprobs": torch.zeros(batch_size),
            "entropy": torch.zeros(batch_size),
        }, batch_size=batch_size)

    def store_experience(self, experience: TensorDict, state: TensorDict):
        # Mask learner and Ensure indices do not exceed batch size
        mask = state["env", "mask"]
        env_id = state["env", "env_id"]
        batch_size = experience["env"].batch_size

        indices = np.where(mask)[0]
        num_indices = indices.size
        end = experience["ptr"] + num_indices
        dst = slice(experience["ptr"], end)

        # Zero-copy indexing for contiguous env_id
        if num_indices == mask.size and isinstance(env_id, slice):
            gpu_inds = cpu_inds = slice(0, min(batch_size - experience["ptr"], num_indices))
        else:
            cpu_inds = indices[:batch_size - experience["ptr"]]
            gpu_inds = torch.as_tensor(cpu_inds).to(experience["env", "obs"].device, non_blocking=True)

        if experience["env", "obs"].device.type == 'cuda':
            experience["env", "obs"][dst] = state["env", "obs"][gpu_inds]
        else:
            experience["env", "obs"][dst] = state["env", "cpu_obs"][cpu_inds]


        self.actions[dst] = action[gpu_inds]
        self.logprobs[dst] = logprob[gpu_inds]
        self.rewards[dst] = reward[cpu_inds].to(self.rewards.device) # ???
        self.dones[dst] = done[cpu_inds].to(self.dones.device) # ???
        self._store_value(experience, state, dst)

        if isinstance(env_id, slice):
            self.sort_keys[dst, 1] = np.arange(env_id.start, env_id.stop, dtype=np.int32)
        else:
            self.sort_keys[dst, 1] = env_id[cpu_inds]

        self.sort_keys[dst, 2] = self.step
        self.ptr = end
        self.step += 1


    def _store_value(self, experience, dst, value):
        experience[dst] = value

    def prepare_experience(self, experience, state, idxs):
        values_np = experience.values[idxs].to('cpu', non_blocking=True).numpy()
        dones_np = experience.dones[idxs].to('cpu', non_blocking=True).numpy()
        rewards_np = experience.rewards[idxs].to('cpu', non_blocking=True).numpy()
        torch.cuda.synchronize()
        advantages_np = fast_gae.compute_gae(dones_np, values_np, rewards_np, self.gamma, self.gae_lambda)
        advantages = torch.as_tensor(advantages_np).to(self.device, non_blocking=True)
        self.b_advantages = advantages.reshape(
            self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size)

        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)[b_idxs].contiguous()
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)[b_idxs]
        self.b_dones = self.dones.to(self.device, non_blocking=True)[b_idxs]
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_values = self.values.to(self.device, non_blocking=True)[b_flat]
        self.returns = advantages + self.values # Check sorting of values here
        self.b_returns = self.b_advantages + self.b_values # Check sorting of values here


    def update_state(self, experience, state):
        state["values"] = experience["b_values"]

    def compute_losses(self, losses):
        self.compute_policy_loss(losses)
        self.compute_entropy_loss(losses)
        self.compute_value_loss(losses)
        losses['entropy'] = -entropy.mean()

    def compute_policy_loss(self, losses):
        logratio = newlogprob - log_probs.reshape(-1)
        ratio = logratio.exp()

        # TODO: Only do this if we are KL clipping? Saves 1-2% compute
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

        adv = adv.reshape(-1)
        if self.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Policy loss
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        losses['policy'] = pg_loss


    def compute_value_loss(self, losses):
        newvalue = newvalue.flatten()
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - ret) ** 2
            v_clipped = val + torch.clamp(
                newvalue - val,
                -self.vf_clip_coef,
                self.vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - ret) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

        losses['value'] = v_loss


