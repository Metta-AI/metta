import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger("TrainerCheckpoint")


class TrainerCheckpoint:
    def __init__(
        self,
        agent_step: int = 0,
        epoch: int = 0,
        optimizer_state_dict: Optional[dict[str, Any]] = None,
        policy_path: Optional[str] = None,
        stopwatch_state: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self.agent_step = agent_step
        self.epoch = epoch
        self.optimizer_state_dict = optimizer_state_dict
        self.policy_path = policy_path
        self.stopwatch_state = stopwatch_state
        self.extra_args = kwargs

    def save(self, run_dir: str, filename: str = "trainer_state.pt") -> None:
        state = {
            "optimizer_state_dict": self.optimizer_state_dict,
            "agent_step": self.agent_step,
            "epoch": self.epoch,
            "policy_path": self.policy_path,
            "stopwatch_state": self.stopwatch_state,
            **self.extra_args,
        }

        checkpoint_path = Path(run_dir) / filename

        # Write to a temporary file first to avoid leaving a partially written checkpoint
        # if the program crashes or is interrupted. We then atomically replace the final
        # state file using os.replace(), which ensures that either the old file remains,
        # or the new file is fully written — never a corrupted intermediate.
        with tempfile.NamedTemporaryFile(dir=run_dir, delete=False, suffix=".pt") as tmp_file:
            torch.save(state, tmp_file.name)
            tmp_path = tmp_file.name

        try:
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            # Clean up temp file if replace fails to avoid cluttering disk
            os.unlink(tmp_path)
            raise

        logger.info(f"[TrainerCheckpoint] Saved trainer state to {checkpoint_path}")

    @staticmethod
    def load(run_dir: str, filename: str = "trainer_state.pt") -> Optional["TrainerCheckpoint"]:
        checkpoint_path = Path(run_dir) / filename

        if not checkpoint_path.exists():
            logger.info(f"[TrainerCheckpoint] No trainer state found at {checkpoint_path}. Assuming a new run!")
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            state = torch.load(checkpoint_path, weights_only=False)

            # handle backward compatibility
            if "stopwatch_state" not in state:
                state["stopwatch_state"] = None

            return TrainerCheckpoint(**state)
