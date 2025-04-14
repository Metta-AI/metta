import torch

class Kickstarter:
    def __init__(self, cfg, policy_store, single_action_space):
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
        self.spaces = len(single_action_space.nvec)

        self._load_policies()

    def _load_policies(self):
        self.teachers = []
        for teacher_cfg in self.teacher_cfgs:
            policy_record = self.policy_store.policy(teacher_cfg["teacher_uri"])
            policy = policy_record.policy()
            policy.action_loss_coef = teacher_cfg["action_loss_coef"]
            policy.value_loss_coef = teacher_cfg["value_loss_coef"]
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
            teacher_value, teacher_normalized_logits, teacher_lstm_state[i] = self._forward(
                teacher, o, teacher_lstm_state[i]
            )

            for i in range(self.spaces):
                ks_action_loss -= (teacher_normalized_logits[i].exp() * student_normalized_logits[i]).sum(dim=-1).mean()

            ks_action_loss *= teacher.action_loss_coef

            ks_value_loss += ((teacher_value.squeeze() - student_value) ** 2).mean() * teacher.value_loss_coef

        return ks_action_loss, ks_value_loss, teacher_lstm_state

    def _forward(self, teacher, o, teacher_lstm_state):
        # teacher e3b?
        _, _, _, teacher_value, teacher_lstm_state, next_e3b, intrinsic_reward, teacher_normalized_logits = teacher(
            o, teacher_lstm_state, e3b=None
        )

        return teacher_value, teacher_normalized_logits, teacher_lstm_state
