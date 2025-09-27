"""Policy upload management component for wandb and other destinations."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import wandb

from metta.common.wandb.context import WandbRun
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import DistributedHelper, TrainerComponent
from softmax.lib.utils import local_copy
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class UploaderConfig(Config):
    """Configuration for policy uploading."""

    epoch_interval: int = 1000
    """How often to upload policy to wandb (in epochs)."""


class Uploader(TrainerComponent):
    """Manages uploading policies to wandb and other destinations."""

    def __init__(
        self,
        *,
        config: UploaderConfig,
        checkpoint_manager: CheckpointManager,
        distributed_helper: DistributedHelper,
        wandb_run: Optional[WandbRun] = None,
    ) -> None:
        super().__init__(epoch_interval=max(1, config.epoch_interval))
        self._master_only = True
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._distributed = distributed_helper
        self._wandb_run = wandb_run

    def update_wandb_run(self, wandb_run: Optional[WandbRun]) -> None:
        self._wandb_run = wandb_run

    # ------------------------------------------------------------------
    # Callback entry-points
    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        self._maybe_upload(epoch, require_interval=True, reason=f"epoch {epoch}")

    def on_training_complete(self) -> None:  # type: ignore[override]
        self._maybe_upload(self.context.epoch, final=True, reason="final upload")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _can_upload(self) -> bool:
        return self._distributed.should_checkpoint() and self._wandb_run is not None

    def _maybe_upload(
        self,
        epoch: int,
        *,
        require_interval: bool = False,
        final: bool = False,
        reason: str,
    ) -> None:
        if not self._can_upload():
            return
        if require_interval and epoch % self._config.epoch_interval != 0:
            return

        checkpoint_uri = self.context.latest_policy_uri()
        if not checkpoint_uri:
            logger.debug("Uploader: no checkpoint available for %s", reason)
            return

        metadata = {
            "epoch": epoch,
            "agent_step": self.context.agent_step,
        }
        if final:
            metadata["final"] = True

        scores = self.context.latest_eval_scores
        if scores and (scores.category_scores or scores.simulation_scores):
            metadata.update(
                score=scores.avg_simulation_score,
                avg_reward=scores.avg_category_score,
            )

        self._upload(checkpoint_uri, epoch, metadata)

    def _upload(
        self,
        checkpoint_uri: str,
        epoch: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        artifact_name = f"policy-{epoch}"

        with self._materialize_checkpoint(checkpoint_uri) as local_path:
            if local_path is None:
                return None

            assert self._wandb_run is not None
            artifact = wandb.Artifact(name=artifact_name, type="model")
            if metadata:
                artifact.metadata.update(metadata)
            artifact.add_file(str(local_path))
            logger.info("Uploading policy checkpoint artifact %s", artifact_name)
            logged_artifact = self._wandb_run.log_artifact(artifact)
            return getattr(logged_artifact, "id", None)

    @contextmanager
    def _materialize_checkpoint(self, checkpoint_uri: str) -> Iterator[Optional[Path]]:
        """Yield a local file path for the given checkpoint URI."""
        normalized_uri = CheckpointManager.normalize_uri(checkpoint_uri)
        parsed = urlparse(normalized_uri)

        if parsed.scheme in ("", "file"):
            local_path = Path(parsed.path)
            if not local_path.exists():
                logger.warning("Uploader: checkpoint path %s does not exist", local_path)
                yield None
            else:
                yield local_path
            return

        try:
            with local_copy(normalized_uri) as tmp_path:
                yield Path(tmp_path)
        except Exception as exc:  # pragma: no cover - best effort for remote policies
            logger.error("Uploader: failed to materialize %s: %s", normalized_uri, exc)
            yield None
