import os
import torch
import logging
import warnings

logger = logging.getLogger("trainer")

class TrainerCheckpoint:
    def __init__(
        self,
        agent_steps: int = 0,
        epoch: int = 0,
        optimizer_state_dict: dict = None,
        policy_path: str = None,
        **kwargs
    ):
        self.agent_steps = agent_steps
        self.epoch = epoch
        self.optimizer_state_dict = optimizer_state_dict
        self.policy_path = policy_path
        self.extra_args = kwargs

    def save(self, run_dir: str):
        state = {
            'optimizer_state_dict': self.optimizer_state_dict,
            'agent_step': self.agent_steps,
            'epoch': self.epoch,
            'model_path': self.policy_path,
        }
        state_path = os.path.join(run_dir, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.rename(state_path + '.tmp', state_path)
        logger.info(f"Saved model to {self.policy_path}")

    @staticmethod
    def load(run_dir: str) -> 'TrainerCheckpoint':
        trainer_path = os.path.join(run_dir, 'trainer_state.pt')
        logger.info(f"Loading trainer state from {trainer_path}")

        if not os.path.exists(trainer_path):
            logger.info('No trainer state found. Assuming new run')
            return TrainerCheckpoint(0, 0, None, None)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return TrainerCheckpoint(**torch.load(trainer_path))