import math
from typing import Callable, List

import torch

from metta.agent.policy_state import PolicyState


class Kickstarter:
    def __init__(self, cfg, policy_store, action_names, action_max_params):
        """
        Initializes the Kickstarter module.

        Kickstarting is a technique to initialize a student policy with the knowledge of one or more teacher policies.
        This is done by adding a loss term that encourages the student's output (action logits and value) to match the
        teacher's.

        The loss coefficients (`action_loss_coef`, `value_loss_coef`) can be defined as a fixed float or as a string
        representing a function of the agent step `t`. For example: "max(0, 1.0 - t / 1e6)". This provides a flexible
        way to schedule the kickstarting influence over time.
        """
        self.device = cfg.device
        self.teacher_cfgs = cfg.trainer.kickstart.additional_teachers

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

        self._compile_schedules()
        self._load_policies()

    def _compile_schedules(self):
        """
        Pre-compiles any schedule strings into callable functions for efficiency.
        This uses a safe version of `eval` that only allows access to the `math` module and basic built-ins.
        """
        self.schedule_fns: dict[str, Callable[[float], float]] = {}
        safe_env = {
            "__builtins__": {"max": max, "min": min, "pow": pow, "abs": abs, "float": float},
            "math": math,
        }
        for teacher_cfg in self.teacher_cfgs:
            for coef_name in ["action_loss_coef", "value_loss_coef"]:
                coef_val = teacher_cfg.get(coef_name)
                if isinstance(coef_val, str):
                    if coef_val not in self.schedule_fns:
                        try:
                            self.schedule_fns[coef_val] = eval(f"lambda t: float({coef_val})", safe_env)
                        except Exception as e:
                            raise ValueError(f"Invalid schedule string for {coef_name}: '{coef_val}'") from e

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
        importance_sampling_ratio: torch.Tensor,
        clip_coef: float,
    ):
        ks_value_loss = torch.tensor(0.0, device=self.device)
        ks_action_loss = torch.tensor(0.0, device=self.device)

        if not self.enabled or agent_step > self.kickstart_steps:
            return ks_action_loss, ks_value_loss

        if len(teacher_lstm_state) == 0:
            teacher_lstm_state = [PolicyState() for _ in range(len(self.teachers))]

        # Clamp the importance sampling ratio from below, which is the PPO-style
        # simplification for a loss/cost (equivalent to negative advantage).
        clipped_ratio = torch.clamp(importance_sampling_ratio, 1.0 - clip_coef, None)

        for i, teacher in enumerate(self.teachers):
            with torch.no_grad():
                teacher_value, teacher_normalized_logits = self._forward(teacher, o, teacher_lstm_state[i])

            # The distillation cost is the negative log likelihood of the teacher's actions
            # under the student's policy. This is equivalent to the cross-entropy term used to
            # minimize KL(teacher || student), and is always a positive value.
            # The cost is calculated per-item in the batch.
            distillation_cost = -torch.sum(teacher_normalized_logits.exp() * student_normalized_logits, dim=-1)

            # The final distillation loss is the mean of the clipped, ratio-weighted cost.
            distillation_loss = (clipped_ratio.flatten() * distillation_cost).mean()

            action_coef = self._get_coef(teacher.action_loss_coef, agent_step)
            value_coef = self._get_coef(teacher.value_loss_coef, agent_step)

            ks_action_loss += distillation_loss * action_coef

            # Value distillation loss is unchanged.
            ks_value_loss += ((teacher_value.squeeze() - student_value) ** 2).mean() * value_coef

        return ks_action_loss, ks_value_loss

    def _get_coef(self, coef_val: float | str, agent_step: int) -> float:
        """Retrieves the coefficient, evaluating it if it's a schedule string."""
        if isinstance(coef_val, str):
            return self.schedule_fns[coef_val](agent_step)
        return coef_val

    def _forward(self, teacher, o, teacher_lstm_state: PolicyState):
        _, _, _, value, norm_logits = teacher(o, teacher_lstm_state)
        return value, norm_logits
