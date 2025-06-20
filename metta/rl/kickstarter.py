from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyStore


class Kickstarter:
    def __init__(
        self,
        cfg: DictConfig,
        policy_store: PolicyStore,
        action_names: list[str],
        action_max_params: list[int],
    ) -> None:
        self.device: torch.device = cfg.device
        self.teacher_cfgs: list[dict[str, Any]] = cfg.trainer.kickstart.additional_teachers or []

        self.teacher_uri: str | None = cfg.trainer.kickstart.teacher_uri
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

        self.enabled: bool = True
        if self.teacher_cfgs is None:
            self.enabled = False
            return

        self.compile: bool = cfg.trainer.compile
        self.compile_mode: str = cfg.trainer.compile_mode
        self.policy_store = policy_store
        self.kickstart_steps: int = cfg.trainer.kickstart.kickstart_steps
        self.action_names: list[str] = action_names
        self.action_max_params: list[int] = action_max_params

        self._load_policies()

    def _load_policies(self) -> None:
        self.teachers: list[nn.Module] = []
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
        agent_step: int,
        student_normalized_logits: Tensor,
        student_value: Tensor,
        o: Tensor,  # Observation tensor
        teacher_policy_state: list[PolicyState],
    ) -> tuple[Tensor, Tensor]:
        ks_value_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        ks_action_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if not self.enabled or agent_step > self.kickstart_steps:
            return ks_action_loss, ks_value_loss

        if len(teacher_policy_state) == 0:
            teacher_policy_state = [PolicyState() for _ in range(len(self.teachers))]

        for i, teacher in enumerate(self.teachers):
            # teacher_policy_state will be updated as a side effect
            _, _, _, teacher_value, teacher_normalized_logits = teacher(teacher, o, teacher_policy_state[i])

            ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
            ks_action_loss *= teacher.action_loss_coef

            ks_value_loss += ((teacher_value.squeeze() - student_value) ** 2).mean() * teacher.value_loss_coef

        return ks_action_loss, ks_value_loss
