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
        total_agent_step: Optional[int] = None,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        policy_path: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ):
        self.agent_step = agent_step
        self.epoch = epoch
        self.total_agent_step = total_agent_step or agent_step
        self.optimizer_state_dict = optimizer_state_dict
        self.policy_path = policy_path
        self.extra_args = extra_args or {}

    def save(self, run_dir: str) -> None:
        state = {
            "optimizer_state_dict": self.optimizer_state_dict,
            "agent_step": self.agent_step,
            "epoch": self.epoch,
            "total_agent_step": self.total_agent_step,
            "policy_path": self.policy_path,
            **self.extra_args,  # Include extra args in the saved state
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
            return TrainerCheckpoint(0, 0, None, None, None, None)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            checkpoint_data = torch.load(trainer_path, weights_only=False)

        # Extract known fields
        return TrainerCheckpoint(
            agent_step=checkpoint_data.get("agent_step", 0),
            epoch=checkpoint_data.get("epoch", 0),
            total_agent_step=checkpoint_data.get("total_agent_step"),
            optimizer_state_dict=checkpoint_data.get("optimizer_state_dict"),
            policy_path=checkpoint_data.get("policy_path"),
            extra_args={
                k: v
                for k, v in checkpoint_data.items()
                if k not in ["agent_step", "epoch", "total_agent_step", "optimizer_state_dict", "policy_path"]
            },
        )
