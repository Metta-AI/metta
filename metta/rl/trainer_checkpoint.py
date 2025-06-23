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
        stopwatch_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.agent_step = agent_step
        self.epoch = epoch
        self.optimizer_state_dict = optimizer_state_dict
        self.policy_path = policy_path
        self.stopwatch_state = stopwatch_state
        self.extra_args = kwargs

    def save(self, run_dir: str) -> None:
        state = {
            "optimizer_state_dict": self.optimizer_state_dict,
            "agent_step": self.agent_step,
            "epoch": self.epoch,
            "policy_path": self.policy_path,
            "stopwatch_state": self.stopwatch_state,
            **self.extra_args,
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
            return TrainerCheckpoint(0, 0, None, None, None)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            state = torch.load(trainer_path, weights_only=False)

            # handle backward compatibility
            if "stopwatch_state" not in state:
                state["stopwatch_state"] = None

            return TrainerCheckpoint(**state)
