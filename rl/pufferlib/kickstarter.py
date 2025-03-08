import pufferlib
import torch
from typing import Tuple

class Kickstarter:
    def __init__(self, cfg, policy_store, single_action_space):
        self.device = cfg.device
        self.enabled = cfg.trainer.kickstart
        if not self.enabled:
            return

        self.teacher_cfgs = cfg.trainer.teachers
        self.compile = cfg.trainer.compile
        self.compile_mode = cfg.trainer.compile_mode
        self.policy_store = policy_store
        self.kickstart_steps = cfg.trainer.kickstart_steps_pct * cfg.trainer.total_timesteps

        if isinstance(single_action_space, pufferlib.spaces.Discrete):
            self.multi_discrete = False
        else:
            self.multi_discrete = True
        
        self._load_policies()

    def _load_policies(self):
        self.teachers = []
        for teacher_cfg in self.teacher_cfgs:
            policy_record = self.policy_store.policy(teacher_cfg['policy_uri'])
            policy = policy_record.policy()
            policy.action_coef = teacher_cfg['action_coef']
            policy.value_coef = teacher_cfg['value_coef']
            if self.compile:
                policy = torch.compile(policy, mode=self.compile_mode)
            self.teachers.append(policy)

    def loss(self, agent_step, student_normalized_logits, student_value, o, teacher_lstm_state=None):
        ks_value_loss = torch.tensor(0.0, device=self.device)
        ks_action_loss = torch.tensor(0.0, device=self.device)

        if not self.enabled or agent_step > self.kickstart_steps:
            return ks_action_loss, ks_value_loss, teacher_lstm_state
        
        if teacher_lstm_state is None:
            teacher_lstm_state = [None for _ in range(len(self.teachers))]

        for i, teacher in enumerate(self.teachers):
            teacher_value, teacher_normalized_logits, teacher_lstm_state[i] = self._forward(teacher, o, teacher_lstm_state[i])

            if self.multi_discrete: 
                ks_action_loss -= (teacher_normalized_logits[0].exp() * student_normalized_logits[0]).sum(dim=-1).mean() * teacher.action_coef
                ks_action_loss -= (teacher_normalized_logits[1].exp() * student_normalized_logits[1]).sum(dim=-1).mean() * teacher.action_coef
            else:
                ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean() * teacher.action_coef
                
            ks_value_loss += ((teacher_value.squeeze() - student_value) ** 2).mean() * teacher.value_coef
     
        return ks_action_loss, ks_value_loss, teacher_lstm_state
    
    def _forward(self, teacher, o, teacher_lstm_state):
        # teacher e3b?
        _, _, _, teacher_value, teacher_lstm_state, next_e3b, intrinsic_reward, teacher_normalized_logits = teacher(o, teacher_lstm_state, e3b=None)

        return teacher_value, teacher_normalized_logits, teacher_lstm_state