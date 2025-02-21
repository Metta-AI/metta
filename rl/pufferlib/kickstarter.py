import torch
import numpy as np
from typing import List, Tuple

from trainer import PufferTrainer, Experience
from agent.policy_store import PolicyStore

class Kickstarter(PufferTrainer):
    def __init__(self, *args, **kwargs):
        self.activated = self.trainer_cfg.kickstart
        if not self.activated:
            return
        
        self._load_policies()
        self._make_experience_buffer()

    def _load_policies(self):
        self.teachers = []
        teachers_cfg = list(self.cfg.trainer.teachers)
        for teacher_cfg in teachers_cfg:
            policy_record = self.policy_store.policy(teacher_cfg['policy_uri'])
            policy = policy_record.policy
            policy.action_coef = teacher_cfg['action_coef']
            policy.value_coef = teacher_cfg['value_coef']
            if self.trainer_cfg.compile:
                policy = torch.compile(policy, mode=self.trainer_cfg.compile_mode)
            self.teachers.append(policy)

    def _make_experience_buffer(self):
        obs_shape = self.vecenv.single_observation_space.shape
        obs_dtype = self.vecenv.single_observation_space.dtype
        atn_shape = self.vecenv.single_action_space.shape
        atn_dtype = self.vecenv.single_action_space.dtype
        total_agents = self.vecenv.num_agents

        self.teacher_experiences = []
        for teacher in self.teachers:
            lstm = teacher.lstm
            self.teacher_experiences.append(Experience(self.trainer_cfg.batch_size, self.trainer_cfg.bptt_horizon,
                self.trainer_cfg.minibatch_size, self.policy.hidden_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
                self.trainer_cfg.cpu_offload, self.device, lstm, total_agents))
            
    def reset_experiences(self):
        if not self.activated:
            return
        
        for teacher_experience in self.teacher_experiences:
            teacher_experience.ptr = 0
            teacher_experience.step = 0
        
    def forward(self, o_device, env_id):
        if not self.activated:
            return
        
        for i, teacher_experience in enumerate(self.teacher_experiences):
            if teacher_experience.lstm_h is not None:
                h = teacher_experience.lstm_h[:, env_id]
                c = teacher_experience.lstm_c[:, env_id]
            else:
                h = None
                c = None
            
            # need to support teacher e3b
            _, logprob, _, value, (h, c), next_e3b, intrinsic_reward = self.teachers[i](o_device, (h, c), e3b=None)

            value = value.flatten()
            actions = actions.cpu().numpy()
            mask = torch.as_tensor(mask)# * policy.mask)
            o = o if self.trainer_cfg.cpu_offload else o_device
            teacher_experience.store(o, value, _, logprob, _, _, env_id, _)
            # does omitting all this prevent sorting from working??

    def sort_experiences(self) -> List[np.ndarray]:
        if not self.activated:
            return []
        idxs = []
        for teacher_experience in self.teacher_experiences:
            idxs.append(teacher_experience.sort_training_data())
        return idxs

    def loss(self, mb, student_logprobs, student_values) -> Tuple[torch.Tensor, torch.Tensor]:
        ks_value_loss = torch.tensor(0.0, device=self.device)
        ks_action_loss = torch.tensor(0.0, device=self.device)

        if not self.activated:
            return ks_action_loss, ks_value_loss

        for i, teacher_experience in enumerate(self.teacher_experiences):
            teacher_logprobs = teacher_experience.logprobs_np[mb]
            teacher_values = teacher_experience.values_np[mb]

            # fix cross entropy loss formula
            ks_action_loss -= (teacher_logprobs * student_logprobs).sum(dim=-1).mean() * self.teachers[i].action_coef
            ks_value_loss += (teacher_values - student_values) ** 2 * self.teachers[i].value_coef
            
        return ks_action_loss, ks_value_loss