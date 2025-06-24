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

        The kickstarting loss is annealed over a number of steps.
        """
        self.device = cfg.device
        self.teacher_cfgs = cfg.trainer.kickstart.additional_teachers
        kickstart_cfg = cfg.trainer.kickstart

        # Annealing parameters
        self.warmup_steps = kickstart_cfg.kickstart_warmup_steps
        self.warmup_multiplier = kickstart_cfg.kickstart_warmup_multiplier
        self.anneal_1_steps = kickstart_cfg.kickstart_anneal_1_steps
        self.anneal_2_steps = kickstart_cfg.kickstart_anneal_2_steps
        self.anneal_1_end_multiplier = kickstart_cfg.kickstart_anneal_1_end_multiplier
        self.anneal_2_end_multiplier = kickstart_cfg.kickstart_anneal_2_end_multiplier
        self.ppo_loss_avg_duration_steps = kickstart_cfg.ppo_loss_avg_duration_steps

        self.anneal_1_start_step = self.warmup_steps
        self.anneal_2_start_step = self.anneal_1_start_step + self.anneal_1_steps
        self.total_kickstart_steps = self.anneal_2_start_step + self.anneal_2_steps

        self.bptt_horizon = cfg.trainer.bptt_horizon

        # State for kickstarter loss averaging
        self.ks_loss_history = []
        self.ks_loss_avg = 0.0
        self.initial_ks_to_ppo_ratio = None

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
        self.action_names = action_names
        self.action_max_params = action_max_params

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
        ppo_loss_avg: float,
    ):
        ks_value_loss = torch.tensor(0.0, device=self.device)
        ks_action_loss = torch.tensor(0.0, device=self.device)

        if not self.enabled or agent_step > self.total_kickstart_steps or ppo_loss_avg <= 1e-9:
            return ks_action_loss, ks_value_loss

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
            raw_ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(
                dim=-1
            ).mean() * teacher.action_loss_coef
            raw_ks_value_loss += ((teacher_value.squeeze() - student_value) ** 2).mean() * teacher.value_loss_coef

        # Normalize by the sum of coefficients to correctly handle multiple teachers
        if action_loss_coef_sum > 0:
            raw_ks_action_loss /= action_loss_coef_sum
        if value_loss_coef_sum > 0:
            raw_ks_value_loss /= value_loss_coef_sum

        raw_total_ks_loss = raw_ks_action_loss + raw_ks_value_loss

        # Update kickstarter loss average during warmup
        if agent_step < self.warmup_steps:
            minibatch_steps = o.shape[0] * self.bptt_horizon
            self.ks_loss_history.append((raw_total_ks_loss.item(), minibatch_steps))
            total_steps = sum(steps for _, steps in self.ks_loss_history)
            while total_steps > self.ppo_loss_avg_duration_steps and len(self.ks_loss_history) > 1:
                _, steps_to_remove = self.ks_loss_history.pop(0)
                total_steps -= steps_to_remove

            if total_steps > 0:
                self.ks_loss_avg = sum(loss * steps for loss, steps in self.ks_loss_history) / total_steps

        # Determine multiplier
        if agent_step < self.warmup_steps:
            multiplier = self.warmup_multiplier
            scale_factor = 1
        else:
            # First time in annealing phase, calculate the starting ratio.
            if self.initial_ks_to_ppo_ratio is None:
                if self.ks_loss_avg > 1e-9 and ppo_loss_avg > 1e-9:
                    self.initial_ks_to_ppo_ratio = self.ks_loss_avg / ppo_loss_avg
                else:
                    self.initial_ks_to_ppo_ratio = self.warmup_multiplier  # Fallback

            # Annealing phase 1
            if agent_step < self.anneal_2_start_step:
                progress = (agent_step - self.warmup_steps) / self.anneal_1_steps
                progress = min(progress, 1.0)
                multiplier = self.initial_ks_to_ppo_ratio + progress * (
                    self.anneal_1_end_multiplier - self.initial_ks_to_ppo_ratio
                )
            # Annealing phase 2
            else:
                progress = (agent_step - self.anneal_2_start_step) / self.anneal_2_steps
                progress = min(progress, 1.0)
                multiplier = self.anneal_1_end_multiplier + progress * (
                    self.anneal_2_end_multiplier - self.anneal_1_end_multiplier
                )

            target_total_ks_loss = ppo_loss_avg * multiplier

            # Add epsilon to prevent division by zero
            scale_factor = target_total_ks_loss / (raw_total_ks_loss + 1e-9)

        final_ks_action_loss = raw_ks_action_loss * scale_factor
        final_ks_value_loss = raw_ks_value_loss * scale_factor

        return final_ks_action_loss, final_ks_value_loss

    def _forward(self, teacher, o, teacher_lstm_state: PolicyState):
        _, _, _, value, norm_logits = teacher(o, teacher_lstm_state)
        return value, norm_logits
