import logging
import os
import warnings
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("TrainerCheckpoint")


class TrainerCheckpoint:
    def __init__(
        self,
        agent_step: int = 0,
        epoch: int = 0,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        policy_path: Optional[str] = None,
    ):
        self.agent_step = agent_step
        self.epoch = epoch
        self.optimizer_state_dict = optimizer_state_dict
        self.policy_path = policy_path

    def save(self, run_dir: str) -> None:
        state = {
            "optimizer_state_dict": self.optimizer_state_dict,
            "agent_step": self.agent_step,
            "epoch": self.epoch,
            "policy_path": self.policy_path,
        }

        state_path = os.path.join(run_dir, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)
        logger.info(f"Saved trainer state to {state_path}")

    @staticmethod
    def load(run_dir: str) -> "TrainerCheckpoint":
        trainer_path = os.path.join(run_dir, "trainer_state.pt")
        logger.info(f"Loading trainer state from {trainer_path}")
        if not os.path.exists(trainer_path):
            logger.info("No trainer state found. Assuming new run")
            return TrainerCheckpoint(0, 0, None, None)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            checkpoint_data = torch.load(trainer_path, weights_only=False)

        # Extract only the expected fields
        return TrainerCheckpoint(
            agent_step=checkpoint_data.get("agent_step", 0),
            epoch=checkpoint_data.get("epoch", 0),
            optimizer_state_dict=checkpoint_data.get("optimizer_state_dict"),
            policy_path=checkpoint_data.get("policy_path"),
        )
