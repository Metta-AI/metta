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
        The annealing is linear and only at the end. For example, if `anneal_ratio` is 0.2, the loss coefficient will
        be 1.0 for the first 80% of `kickstart_steps`, then anneal linearly from 1.0 down to 0 over the last 20%.
        """
        self.device = cfg.device
        self.teacher_cfgs = cfg.trainer.kickstart.additional_teachers
        kickstart_cfg = cfg.trainer.kickstart
        # self.anneal_ratio = cfg.trainer.kickstart.anneal_ratio
        # assert 0 <= self.anneal_ratio <= 1, "Anneal_ratio must be between 0 and 1."
        # New annealing parameters
        self.warmup_steps = kickstart_cfg.kickstart_warmup_steps
        self.anneal_duration = kickstart_cfg.kickstart_anneal_duration
        self.start_multiplier = kickstart_cfg.kickstart_anneal_start_multiplier
        self.end_multiplier = kickstart_cfg.kickstart_anneal_end_multiplier
        self.total_kickstart_steps = self.warmup_steps + self.anneal_duration
        self.log_start_multiplier = torch.log(torch.tensor(self.start_multiplier, device=self.device))
        self.log_end_multiplier = torch.log(torch.tensor(self.end_multiplier, device=self.device))


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
        # self.kickstart_steps = cfg.trainer.kickstart.kickstart_steps
        self.action_names = action_names
        self.action_max_params = action_max_params
        # self.anneal_factor = 1.0

        # if self.anneal_ratio > 0:
        #     self.anneal_duration = self.kickstart_steps * self.anneal_ratio
        #     self.ramp_down_start_step = self.kickstart_steps - self.anneal_duration
        # else:
        #     self.anneal_duration = 0
        #     self.ramp_down_start_step = self.kickstart_steps

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

    def loss(
            self,
            agent_step,
            student_normalized_logits,
            student_value,
            o,
            teacher_lstm_state: List[PolicyState],
            ppo_loss_avg: float
    ):
        ks_value_loss = torch.tensor(0.0, device=self.device)
        ks_action_loss = torch.tensor(0.0, device=self.device)

        if not self.enabled or agent_step > self.total_kickstart_steps or ppo_loss_avg <= 1e-9:
            return ks_action_loss, ks_value_loss

        # Determine multiplier
        if agent_step < self.warmup_steps:
            multiplier = self.start_multiplier
        else:
            progress = (agent_step - self.warmup_steps) / self.anneal_duration
            progress = min(progress, 1.0)  # clamp progress
            current_log_multiplier = (
                self.log_start_multiplier + progress * (self.log_end_multiplier - self.log_start_multiplier)
            )
            multiplier = torch.exp(current_log_multiplier)

        raw_ks_action_loss = torch.tensor(0.0, device=self.device)
        raw_ks_value_loss = torch.tensor(0.0, device=self.device)

        if len(teacher_lstm_state) == 0:
            teacher_lstm_state = [PolicyState() for _ in range(len(self.teachers))]

        action_loss_coef_sum = 0
        value_loss_coef_sum = 0

        for i, teacher in enumerate(self.teachers):
            teacher_value, teacher_normalized_logits = self._forward(teacher, o, teacher_lstm_state[i])

            action_loss_coef_sum += teacher.action_loss_coef
            value_loss_coef_sum += teacher.value_loss_coef

            # Calculate raw losses (sum over all teachers)
            raw_ks_action_loss -= (
                (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
                * teacher.action_loss_coef
            )
            raw_ks_value_loss += (
                (teacher_value.squeeze() - student_value) ** 2
            ).mean() * teacher.value_loss_coef

        # Normalize by the sum of coefficients to correctly handle multiple teachers
        if action_loss_coef_sum > 0:
            raw_ks_action_loss /= action_loss_coef_sum
        if value_loss_coef_sum > 0:
            raw_ks_value_loss /= value_loss_coef_sum

        # Scale the raw losses to match the target based on PPO loss average
        raw_total_ks_loss = raw_ks_action_loss + raw_ks_value_loss
        target_total_ks_loss = ppo_loss_avg * multiplier

        # Add epsilon to prevent division by zero
        scale_factor = target_total_ks_loss / (raw_total_ks_loss + 1e-9)

        final_ks_action_loss = raw_ks_action_loss * scale_factor
        final_ks_value_loss = raw_ks_value_loss * scale_factor

        return final_ks_action_loss, final_ks_value_loss

    def _forward(self, teacher, o, teacher_lstm_state: PolicyState):
        _, _, _, value, norm_logits = teacher(o, teacher_lstm_state)
        return value, norm_logits
