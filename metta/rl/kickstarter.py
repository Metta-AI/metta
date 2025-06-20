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
        Annealing begins at `(1 - anneal_ratio) * kickstart_steps` and goes down to 0 at `kickstart_steps` using a
        cosine schedule. For example, if `anneal_ratio` is 0.2, the loss coefficient is 1.0 for the first 80% of
        `kickstart_steps`, then anneals to 0 over the last 20%.
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
        self.start_anneal_step = (1 - self.anneal_ratio) * self.kickstart_steps
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

        if agent_step >= self.start_anneal_step and self.anneal_ratio > 0:
            progress = (agent_step - self.start_anneal_step) / (self.kickstart_steps * self.anneal_ratio)
            # Cosine annealing from 1 to 0
            self.anneal_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress, device=self.device)))

        if len(teacher_lstm_state) == 0:
            teacher_lstm_state = [PolicyState() for _ in range(len(self.teachers))]

        for i, teacher in enumerate(self.teachers):
            teacher_value, teacher_normalized_logits, teacher_lstm_state[i] = self._forward(
                teacher, o, teacher_lstm_state[i]
            )
            ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
            ks_action_loss *= teacher.action_loss_coef * self.anneal_factor

            ks_value_loss += (
                ((teacher_value.squeeze() - student_value) ** 2).mean() * teacher.value_loss_coef * self.anneal_factor
            )

        return ks_action_loss, ks_value_loss

    def _forward(self, teacher, o, teacher_lstm_state: PolicyState):
        _, _, _, value, norm_logits = teacher(o, teacher_lstm_state)
        return value, norm_logits
