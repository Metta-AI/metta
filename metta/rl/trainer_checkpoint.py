import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import torch

from metta.common.util.fs import atomic_write

logger = logging.getLogger("TrainerCheckpoint")


class TrainerCheckpoint:
    def __init__(
        self,
        agent_step: int = 0,
        epoch: int = 0,
        optimizer_state_dict: Optional[dict[str, Any]] = None,
        policy_path: Optional[str] = None,
        stopwatch_state: Optional[dict[str, Any]] = None,
        extra_args: Optional[dict[str, Any]] = None,
    ):
        self.agent_step = agent_step
        self.epoch = epoch
        self.optimizer_state_dict = optimizer_state_dict
        self.policy_path = policy_path
        self.stopwatch_state = stopwatch_state
        self.extra_args = extra_args or {}

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
        atomic_write(lambda path: torch.save(state, path), checkpoint_path, suffix=".pt")

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

            required_keys = {
                "agent_step",
                "epoch",
                "optimizer_state_dict",
                "policy_path",
            }

            # Validate all required keys are present
            for k in required_keys:
                if k not in state:
                    raise ValueError(f"Checkpoint is missing required key: {k}")

            additional_known_keys_map = {"stopwatch_state": None}

            for k, v in additional_known_keys_map.items():
                if k not in state:
                    state[k] = v

            # Handle unexpected keys
            state_keys = set(state.keys())
            all_known_keys = required_keys | set(additional_known_keys_map.keys())
            unexpected_keys = state_keys - all_known_keys
            if unexpected_keys:
                logger.warning(
                    f"Loaded checkpoint contains unexpected keys: {unexpected_keys}. "
                    f"These will be stored in extra_args."
                )

            constructor_kwargs = {key: state[key] for key in all_known_keys}
            if unexpected_keys:
                constructor_kwargs["extra_args"] = {key: state[key] for key in unexpected_keys}

            return TrainerCheckpoint(**constructor_kwargs)
