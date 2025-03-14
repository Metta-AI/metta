import torch

class Kickstarter:
    def __init__(self, cfg, policy_store, vecenv):
        self.device = cfg.device

        self.teacher_cfgs = cfg.trainer.kickstart.teachers
        # self.enabled = True
        
        self.enabled = cfg.trainer.kickstart_enabled
        if not self.enabled:
            return

        if self.teacher_cfgs is None:
            self.enabled = False
            return

        self.compile = cfg.trainer.compile
        self.compile_mode = cfg.trainer.compile_mode
        self.policy_store = policy_store
        self.kickstart_steps = cfg.trainer.kickstart.kickstart_steps
        self.spaces = len(vecenv.single_action_space.nvec)
        self.action_names = vecenv.driver_env.action_names()
        
        self._load_policies()

    def _load_policies(self):
        self.teachers = []
        for teacher_cfg in self.teacher_cfgs:
            policy_record = self.policy_store.policy(teacher_cfg['policy_uri'])        
            policy = policy_record.policy()
            policy.action_loss_coef = teacher_cfg['action_loss_coef']
            policy.value_loss_coef = teacher_cfg['value_loss_coef']

            if policy_record.metadata["action_names"] != self.action_names:
                raise ValueError(
                    "Action names do not match between policy and environment: "
                    f"{policy_record.metadata['action_names']} != {self.action_names}")

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

            for i in range(self.spaces):
                ks_action_loss -= (teacher_normalized_logits[i].exp() * student_normalized_logits[i]).sum(dim=-1).mean()
                
            ks_action_loss *= teacher.action_loss_coef
                
            ks_value_loss += ((teacher_value.squeeze() - student_value) ** 2).mean() * teacher.value_loss_coef
     
        return ks_action_loss, ks_value_loss, teacher_lstm_state
    
    def _forward(self, teacher, o, teacher_lstm_state):
        # teacher e3b?
        _, _, _, teacher_value, teacher_lstm_state, next_e3b, intrinsic_reward, teacher_normalized_logits = teacher(o, teacher_lstm_state, e3b=None)

        return teacher_value, teacher_normalized_logits, teacher_lstm_state
    

    def _create_action_mapping(self, teacher_action_names):
        """Create mapping between teacher and environment action spaces."""
        mapping = {}
        # For each action space
        for space_idx in range(self.spaces):
            # Find indices of actions in teacher that exist in environment
            teacher_to_env_idx = []
            for i, action in enumerate(teacher_action_names):
                if action in self.action_names:
                    # Map to the correct position in environment actions
                    env_idx = self.action_names.index(action)
                    teacher_to_env_idx.append((i, env_idx))
            
            # Find actions in environment that don't exist in teacher
            missing_actions = []
            for i, action in enumerate(self.action_names):
                if action not in teacher_action_names:
                    missing_actions.append(i)
            
            mapping[space_idx] = {
                'teacher_to_env': teacher_to_env_idx,
                'missing_in_teacher': missing_actions
            }