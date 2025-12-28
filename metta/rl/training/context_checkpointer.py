"""Trainer state checkpoint management component."""

import logging
from typing import Any, Dict, Optional

import torch

from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import ComponentContext, DistributedHelper, TrainerComponent

logger = logging.getLogger(__name__)


class ContextCheckpointer(TrainerComponent):
    """Persist and restore optimizer/timing state alongside policy checkpoints."""

    trainer_attr = "trainer_checkpointer"

    def __init__(
        self,
        *,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
    ) -> None:
        super().__init__(epoch_interval=1)
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._last_synced_policy_epoch: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        target_path = self._checkpoint_manager.checkpoint_dir
        target_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Trainer checkpoints will be written to %s", target_path)
        self._last_synced_policy_epoch = context.latest_saved_policy_epoch

    # ------------------------------------------------------------------
    # Public API used by Trainer
    # ------------------------------------------------------------------
    def restore(self, context: ComponentContext) -> None:
        """Load trainer state if checkpoints exist and broadcast to all ranks."""
        payload: Optional[Dict[str, Any]] = None

        if self._distributed.is_master():
            raw = self._checkpoint_manager.load_trainer_state()
            if raw:
                logger.info(
                    "Restoring trainer state from epoch=%s agent_step=%s", raw.get("epoch"), raw.get("agent_step")
                )
                payload = {
                    "agent_step": raw.get("agent_step", 0),
                    "epoch": raw.get("epoch", 0),
                    "avg_reward": raw.get("avg_reward"),
                    "optimizer": raw.get("optimizer", {}),
                    "stopwatch_state": raw.get("stopwatch_state"),
                    "curriculum_state": raw.get("curriculum_state"),
                    "loss_states": raw.get("loss_states", {}),
                }

        payload = self._distributed.broadcast_from_master(payload)
        if payload is None:
            return

        restored_epoch = payload["epoch"]
        context.agent_step = payload["agent_step"]
        context.epoch = restored_epoch
        context.latest_saved_policy_epoch = restored_epoch
        self._last_synced_policy_epoch = context.latest_saved_policy_epoch

        total_agents = int(context.experience.total_agents)
        device = context.experience.device
        default_avg_reward = context.config.advantage.reward_centering.initial_reward_mean
        avg_reward = payload.get("avg_reward", default_avg_reward)
        if avg_reward is None:
            avg_reward = default_avg_reward
        avg_reward = torch.as_tensor(avg_reward).to(device=device, dtype=torch.float32)
        context.state.avg_reward = torch.broadcast_to(avg_reward, (total_agents,)).clone()

        optimizer_state = payload.get("optimizer")
        context.state.optimizer_state = optimizer_state
        if optimizer_state:
            context.optimizer.load_state_dict(optimizer_state)
            # Drop reference to the restored state to avoid retaining GPU buffers
            context.state.optimizer_state = None

        stopwatch_state = payload.get("stopwatch_state")
        context.state.stopwatch_state = stopwatch_state
        wall_time_baseline = 0.0
        if stopwatch_state:
            context.stopwatch.load_state(stopwatch_state, resume_running=True)
            wall_time_baseline = context.stopwatch.get_elapsed()

        curriculum_state = payload.get("curriculum_state")
        context.state.curriculum_state = curriculum_state
        if curriculum_state and context.curriculum is not None:
            context.curriculum.load_state(curriculum_state)
            logger.info("Successfully restored curriculum state")

        loss_states = payload.get("loss_states") or {}
        context.state.loss_states = loss_states
        losses = getattr(context, "losses", None)
        if losses:
            for name, loss in losses.items():
                stored = loss_states.get(name)
                if stored is None:
                    continue
                loss.load_state_dict(stored, strict=False)
        context.state.loss_states = {}

        context.timing_baseline = {
            "agent_step": context.agent_step,
            "wall_time": wall_time_baseline,
        }

    # ------------------------------------------------------------------
    # Callback entry-points
    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if not self._distributed.should_checkpoint():
            return

        policy_epoch = self.context.latest_saved_policy_epoch
        if policy_epoch != self._last_synced_policy_epoch:
            self._save_state()

    def on_training_complete(self) -> None:  # type: ignore[override]
        if not self._distributed.should_checkpoint():
            return

        self._save_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        context = self.context
        current_epoch = context.epoch
        agent_step = context.agent_step

        context.state.stopwatch_state = context.stopwatch.save_state()

        context.state.optimizer_state = context.optimizer.state_dict()
        losses = getattr(context, "losses", None)
        if losses:
            context.state.loss_states = {name: loss.state_dict() for name, loss in losses.items()}
        else:
            context.state.loss_states = {}

        # Capture curriculum state
        if context.curriculum is not None:
            context.state.curriculum_state = context.curriculum.get_state()
        else:
            context.state.curriculum_state = None

        self._checkpoint_manager.save_trainer_state(
            context.optimizer,
            current_epoch,
            agent_step,
            avg_reward=context.state.avg_reward,
            stopwatch_state=context.state.stopwatch_state,
            curriculum_state=context.state.curriculum_state,
            loss_states=context.state.loss_states,
        )

        self._last_synced_policy_epoch = self.context.latest_saved_policy_epoch

        # Release references so we do not pin large GPU tensors between checkpoints
        context.state.optimizer_state = None
        context.state.loss_states = {}
