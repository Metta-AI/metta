"""Policy upload management component for wandb and other destinations."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import wandb

from metta.common.wandb.wandb_context import WandbRun
from metta.mettagrid.config import Config
from metta.mettagrid.util.file import local_copy
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.component import TrainerComponent
from metta.rl.training.distributed_helper import DistributedHelper

logger = logging.getLogger(__name__)


class PolicyUploaderConfig(Config):
    """Configuration for policy uploading."""

    epoch_interval: int = 1000
    """How often to upload policy to wandb (in epochs)."""


class PolicyUploader(TrainerComponent):
    """Manages uploading policies to wandb and other destinations."""

    def __init__(
        self,
        *,
        config: PolicyUploaderConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
        policy_checkpointer,
        wandb_run: Optional[WandbRun] = None,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._policy_checkpointer = policy_checkpointer
        self._wandb_run = wandb_run

    def update_wandb_run(self, wandb_run: Optional[WandbRun]) -> None:
        self._wandb_run = wandb_run

    # ------------------------------------------------------------------
    # Callback entry-points
    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if not self._should_upload(epoch):
            return

        trainer = self._trainer
        if trainer is None:
            return

        checkpoint_uri = self._policy_checkpointer.get_latest_policy_uri()
        if not checkpoint_uri:
            logger.debug("PolicyUploader: no checkpoint available for epoch %s", epoch)
            return

        metadata = {
            "epoch": epoch,
            "agent_step": trainer.agent_step,
        }
        metadata.update(self._evaluation_metadata(trainer))

        self._upload(checkpoint_uri, epoch, metadata)

    def on_training_complete(self) -> None:  # type: ignore[override]
        trainer = self._trainer
        if trainer is None:
            return

        checkpoint_uri = self._policy_checkpointer.get_latest_policy_uri()
        if not checkpoint_uri:
            logger.debug("PolicyUploader: no checkpoint available for final upload")
            return

        metadata = {
            "epoch": trainer.epoch,
            "agent_step": trainer.agent_step,
            "final": True,
        }
        metadata.update(self._evaluation_metadata(trainer))

        self._upload(checkpoint_uri, trainer.epoch, metadata, force=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _should_upload(self, epoch: int) -> bool:
        if not self._distributed.should_checkpoint():
            return False
        if self._wandb_run is None:
            return False
        if epoch % self._config.epoch_interval != 0:
            return False
        return True

    def _evaluation_metadata(self, trainer) -> dict[str, Any]:
        evaluator = getattr(trainer, "evaluator", None)
        if evaluator is None:
            return {}
        try:
            scores = evaluator.get_latest_scores()
        except AttributeError:
            return {}
        if not scores or not (scores.category_scores or scores.simulation_scores):
            return {}
        return {
            "score": scores.avg_simulation_score,
            "avg_reward": scores.avg_category_score,
        }

    def _upload(
        self,
        checkpoint_uri: str,
        epoch: int,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        force: bool = False,
    ) -> Optional[str]:
        if not self._distributed.should_checkpoint():
            return None
        if self._wandb_run is None:
            return None
        if not force and epoch % self._config.epoch_interval != 0:
            return None

        artifact_name = f"policy-{epoch}"

        with self._materialize_checkpoint(checkpoint_uri) as local_path:
            if local_path is None:
                return None

            artifact = wandb.Artifact(name=artifact_name, type="model")
            if metadata:
                artifact.metadata.update(metadata)
            artifact.add_file(str(local_path))
            logger.info("Uploading policy checkpoint artifact %s", artifact_name)
            logged_artifact = self._wandb_run.log_artifact(artifact)
            return getattr(logged_artifact, "id", None)

    @contextmanager
    def _materialize_checkpoint(self, checkpoint_uri: str):
        """Yield a local file path for the given checkpoint URI."""
        normalized_uri = CheckpointManager.normalize_uri(checkpoint_uri)
        parsed = urlparse(normalized_uri)

        if parsed.scheme == "file":
            local_path = Path(parsed.path)
            if not local_path.exists():
                logger.warning("PolicyUploader: checkpoint path %s does not exist", local_path)
                yield None
            else:
                yield local_path
            return

        if parsed.scheme == "s3":
            try:
                with local_copy(normalized_uri) as tmp_path:
                    yield Path(tmp_path)
            except Exception as exc:  # pragma: no cover - best effort for remote policies
                logger.error("PolicyUploader: failed to download %s: %s", normalized_uri, exc)
                yield None
            return

        logger.debug("PolicyUploader: unsupported checkpoint scheme %s", parsed.scheme)
        yield None
