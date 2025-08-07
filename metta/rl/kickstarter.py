import torch
from tensordict import TensorDict
from torch import Tensor, nn

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.mettagrid import MettaGridEnv
from metta.rl.kickstarter_config import KickstartConfig, KickstartTeacherConfig


class KickstartTeacher:
    def __init__(self, policy: nn.Module, action_loss_coef: float, value_loss_coef: float):
        self.policy = policy
        self.action_loss_coef = action_loss_coef
        self.value_loss_coef = value_loss_coef

    def __call__(self, td: TensorDict) -> TensorDict:
        return self.policy(td)


class Kickstarter:
    def __init__(
        self, cfg: KickstartConfig, device: torch.device, policy_store: PolicyStore, metta_grid_env: MettaGridEnv
    ):
        """
        Kickstarting is a technique to initialize a student policy with the knowledge of one or more teacher policies.
        This is done by adding a loss term that encourages the student's output (action logits and value) to match the
        teacher's.

        The kickstarting loss is annealed over a number of steps (`kickstart_steps`).
        The `anneal_ratio` parameter controls what fraction of the `kickstart_steps` are used for annealing.
        The annealing is linear and only at the end. For example, if `anneal_ratio` is 0.2, the loss coefficient will
        be 1.0 for the first 80% of `kickstart_steps`, then anneal linearly from 1.0 down to 0 over the last 20%.

        Linear annealing is used because cosine's rapid dropping phase can come when the policy transition is unstable
        although this hunch hasn't been tested yet.
        """
        self.device = device
        self.metta_grid_env = metta_grid_env
        self.teacher_cfgs = cfg.additional_teachers
        self.anneal_ratio = cfg.anneal_ratio
        assert 0 <= self.anneal_ratio <= 1, "Anneal_ratio must be between 0 and 1."

        self.teacher_uri = cfg.teacher_uri
        if self.teacher_uri is not None:
            if self.teacher_cfgs is None:
                self.teacher_cfgs = []
            self.teacher_cfgs.append(
                KickstartTeacherConfig(
                    teacher_uri=self.teacher_uri,
                    action_loss_coef=cfg.action_loss_coef,
                    value_loss_coef=cfg.value_loss_coef,
                )
            )

        self.enabled: bool = True
        if self.teacher_cfgs is None:
            self.enabled = False
            return

        self.policy_store: PolicyStore = policy_store
        self.kickstart_steps: int = cfg.kickstart_steps
        self.anneal_factor = 1.0

        if self.anneal_ratio > 0:
            self.anneal_duration = self.kickstart_steps * self.anneal_ratio
            self.ramp_down_start_step = self.kickstart_steps - self.anneal_duration
        else:
            self.anneal_duration = 0
            self.ramp_down_start_step = self.kickstart_steps

        self._load_policies()

    def _load_policies(self) -> None:
        self.teachers: list[KickstartTeacher] = []
        for teacher_cfg in self.teacher_cfgs or []:
            policy_record = self.policy_store.policy_record(teacher_cfg.teacher_uri)
            policy: PolicyAgent = policy_record.policy
            policy.action_loss_coef = teacher_cfg.action_loss_coef
            policy.value_loss_coef = teacher_cfg.value_loss_coef
            # Support both new and old initialization methods
            if hasattr(policy, "initialize_to_environment"):
                # Note: We don't have features here, so we pass None
                # The policy should handle this gracefully
                features = self.metta_grid_env.get_observation_features()
                policy.initialize_to_environment(
                    features, self.metta_grid_env.action_names, self.metta_grid_env.max_action_args, self.device
                )
            teacher = KickstartTeacher(
                policy=policy,
                action_loss_coef=teacher_cfg.action_loss_coef,
                value_loss_coef=teacher_cfg.value_loss_coef,
            )
            self.teachers.append(teacher)

    def loss(
        self,
        agent_step: int,
        student_normalized_logits: Tensor,
        student_value: Tensor,
        td: TensorDict,  # Observation tensor
    ) -> tuple[Tensor, Tensor]:
        ks_value_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        ks_action_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if not self.enabled or agent_step > self.kickstart_steps:
            return ks_action_loss, ks_value_loss

        if self.anneal_ratio > 0 and agent_step > self.ramp_down_start_step:
            # Ramp down
            progress = (agent_step - self.ramp_down_start_step) / self.anneal_duration
            self.anneal_factor = 1.0 - progress

        for _, teacher in enumerate(self.teachers):
            teacher_value, teacher_normalized_logits = self._forward(teacher, td)
            ks_action_loss -= (teacher_normalized_logits.exp() * student_normalized_logits).sum(dim=-1).mean()
            ks_action_loss *= teacher.action_loss_coef * self.anneal_factor

            ks_value_loss += (
                ((teacher_value.squeeze() - student_value) ** 2).mean() * teacher.value_loss_coef * self.anneal_factor
            )

        return ks_action_loss, ks_value_loss

    def _forward(self, teacher, td):
        td = teacher(td)
        return td["value"], td["full_log_probs"]
