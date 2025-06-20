from typing import List

import torch

from metta.agent.policy_state import PolicyState


class Kickstarter:
    def __init__(self, cfg, policy_store, action_names, action_max_params):
        """
        Initializes the Kickstarter module.

        Kickstarting is a technique to initialize a student policy with the knowledge of one or more teacher policies.
        This is done by adding a loss term that encourages the student's output (action logits and value) to match the
        teacher's.

        The kickstarting loss is annealed over a number of steps (`kickstart_steps`).
        The `anneal_ratio` parameter controls what fraction of the `kickstart_steps` are used for annealing.
        The annealing is symmetric: it ramps up at the beginning and ramps down at the end, using a cosine schedule
        for both. For example, if `anneal_ratio` is 0.2, the loss coefficient will ramp up from 0 to 1.0 over the
        first 10% of `kickstart_steps`, stay at 1.0 for the next 80%, then anneal from 1.0 down to 0 over the last
        10%.
        """
        self.device = cfg.device
        self.teacher_cfgs = cfg.trainer.kickstart.additional_teachers
        self.anneal_ratio = cfg.trainer.kickstart.anneal_ratio
        assert 0 <= self.anneal_ratio <= 1, "Anneal_ratio must be between 0 and 1."

        self.teacher_uri = cfg.trainer.kickstart.teacher_uri
        if self.teacher_uri is not None:
            if self.teacher_cfgs is None:
                self.teacher_cfgs = []
            self.teacher_cfgs.append(
                {
                    "teacher_uri": self.teacher_uri,
                    "action_loss_coef": cfg.trainer.kickstart.action_loss_coef,
                    "value_loss_coef": cfg.trainer.kickstart.value_loss_coef,
                }
            )

        self.enabled = True
        if self.teacher_cfgs is None:
            self.enabled = False
            return

        self.compile = cfg.trainer.compile
        self.compile_mode = cfg.trainer.compile_mode
        self.policy_store = policy_store
        self.kickstart_steps = cfg.trainer.kickstart.kickstart_steps
        self.action_names = action_names
        self.action_max_params = action_max_params
        self.anneal_factor = 1.0

        self._load_policies()

    def _load_policies(self):
        self.teachers = []
        for teacher_cfg in self.teacher_cfgs:
            policy_record = self.policy_store.policy(teacher_cfg["teacher_uri"])
            policy = policy_record.policy()
            policy.action_loss_coef = teacher_cfg["action_loss_coef"]
            policy.value_loss_coef = teacher_cfg["value_loss_coef"]
            policy.activate_actions(self.action_names, self.action_max_params, self.device)
            if self.compile:
                policy = torch.compile(policy, mode=self.compile_mode)
            self.teachers.append(policy)

    def loss(self, agent_step, student_normalized_logits, student_value, o, teacher_lstm_state: List[PolicyState]):
        ks_value_loss = torch.tensor(0.0, device=self.device)
        ks_action_loss = torch.tensor(0.0, device=self.device)

        if not self.enabled or agent_step > self.kickstart_steps:
            return ks_action_loss, ks_value_loss

        if self.anneal_ratio > 0:
            anneal_ramp_steps = self.kickstart_steps * self.anneal_ratio / 2.0
            ramp_up_end_step = anneal_ramp_steps
            ramp_down_start_step = self.kickstart_steps - anneal_ramp_steps

            if agent_step < ramp_up_end_step:
                # Ramp up
                progress = agent_step / ramp_up_end_step
                self.anneal_factor = 0.5 * (1 - torch.cos(torch.tensor(torch.pi * progress, device=self.device)))
            elif agent_step > ramp_down_start_step:
                # Ramp down
                progress = (agent_step - ramp_down_start_step) / anneal_ramp_steps
                self.anneal_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress, device=self.device)))
            else:
                # Plateau
                self.anneal_factor = 1.0
        else:
            self.anneal_factor = 1.0

        if len(teacher_lstm_state) == 0:
            teacher_lstm_state = [PolicyState() for _ in range(len(self.teachers))]

        for i, teacher in enumerate(self.teachers):
            teacher_value, teacher_normalized_logits = self._forward(teacher, o, teacher_lstm_state[i])
            ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
            ks_action_loss *= teacher.action_loss_coef * self.anneal_factor

            ks_value_loss += (
                ((teacher_value.squeeze() - student_value) ** 2).mean() * teacher.value_loss_coef * self.anneal_factor
            )

        return ks_action_loss, ks_value_loss

    def _forward(self, teacher, o, teacher_lstm_state: PolicyState):
        _, _, _, value, norm_logits = teacher(o, teacher_lstm_state)
        return value, norm_logits
