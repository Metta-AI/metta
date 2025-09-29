"""Torch profiler component for training."""

import gzip
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch.profiler
import wandb

from metta.common.wandb.context import WandbRun
from metta.rl.training import ComponentContext, TrainerComponent
from metta.rl.utils import should_run
from metta.utils.file import http_url, is_public_uri, write_file

logger = logging.getLogger(__name__)


class TorchProfileSession:
    """Context-managed wrapper around ``torch.profiler`` for periodic traces."""

    def __init__(
        self,
        *,
        master: bool,
        profiler_config: Any,
        wandb_run: WandbRun | None,
        run_dir: Path | None,
    ) -> None:
        self._master = master
        self._profiler_config = profiler_config
        self._run_dir = run_dir or ""
        self._wandb_run = wandb_run
        self._profiler: torch.profiler.profile | None = None
        self._active = False
        self._start_epoch: int | None = None
        self._profile_filename_base: str | None = None
        self._first_profile_epoch = 300  # allow torch warmup cycles before profiling
        self._duration_epochs: int = max(1, int(getattr(self._profiler_config, "duration_epochs", 1)))
        self._epochs_remaining: int = 0

    def prepare_for_epoch(self, epoch: int) -> None:
        """Arm the profiler if the upcoming epoch should be captured."""

        if self._active:
            return

        interval = getattr(self._profiler_config, "interval_epochs", 0)
        if interval <= 0:
            return

        target_epoch = epoch + 1
        force = target_epoch == self._first_profile_epoch
        if should_run(target_epoch, interval, force=force):
            self._setup_profiler(target_epoch)

    def _setup_profiler(self, epoch: int) -> None:
        if self._active:
            logger.warning("Profiler already active; ignoring setup request")
            return
        if self._profiler is not None:
            logger.warning("Profiler instance exists while idle; resetting")
            self._profiler = None

        self._active = True
        self._start_epoch = epoch
        self._epochs_remaining = self._duration_epochs
        run_basename = os.path.basename(self._run_dir) if self._run_dir else "unknown_run"
        self._profile_filename_base = f"trace_{run_basename}_epoch_{self._start_epoch}"
        logger.info("Torch profiler scheduled for epoch %s", epoch)

    def __enter__(self):
        if not self._active:
            return self

        if self._profiler is None:
            logger.info(
                "Starting torch profiler for epoch %s (duration=%s epochs)",
                self._start_epoch,
                self._duration_epochs,
            )
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
            )
            self._profiler.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._active or self._profiler is None:
            self._active = False
            return False

        try:
            self._profiler.step()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to advance torch profiler step")

        self._epochs_remaining -= 1
        if self._epochs_remaining > 0:
            logger.info(
                "Continuing torch profiler for epoch %s (%s epochs remaining)",
                self._start_epoch,
                self._epochs_remaining,
            )
            return False

        logger.info("Stopping torch profiler for epoch %s", self._start_epoch)
        try:
            self._profiler.stop()
            self._log_profile_summary(self._profiler)
            self._save_profile(self._profiler)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to save torch profile")
        finally:
            self._profiler = None
            self._active = False
            self._profile_filename_base = None
            self._epochs_remaining = 0

        return False

    def _log_profile_summary(self, prof: torch.profiler.profile) -> None:
        try:
            cpu_table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
            logger.info("Torch profiler summary (CPU) for epoch %s\n%s", self._start_epoch, cpu_table)
            if torch.cuda.is_available():
                cuda_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25)
                logger.info("Torch profiler summary (CUDA) for epoch %s\n%s", self._start_epoch, cuda_table)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to log torch profiler summary")

    @property
    def is_active(self) -> bool:
        return self._active

    # Internal helpers -------------------------------------------------
    def _save_profile(self, prof: torch.profiler.profile) -> None:
        if self._profile_filename_base is None:
            logger.error("Profiler filename unset; skipping save")
            return

        output_filename_json = f"{self._profile_filename_base}.json"
        output_filename_gz = f"{output_filename_json}.gz"
        temp_dir = tempfile.mkdtemp(prefix="torch_profile_")
        temp_json_path = os.path.join(temp_dir, output_filename_json)
        final_gz_path = os.path.join(temp_dir, output_filename_gz)
        upload_path = os.path.join(self._profiler_config.profile_dir, output_filename_gz)

        try:
            self._export_profile(prof, temp_json_path)
            self._compress_trace(temp_json_path, final_gz_path)
            write_file(upload_path, final_gz_path, content_type="application/gzip")
            upload_url = http_url(upload_path)

            if is_public_uri(upload_url) and self._wandb_run:
                link_summary = {
                    "torch_traces/link": wandb.Html(
                        f'<a href="{upload_url}">Torch Trace (Epoch {self._start_epoch})</a>'
                    )
                }
                self._wandb_run.log(link_summary)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _export_profile(self, prof: torch.profiler.profile, output_path: str) -> None:
        logger.info("Exporting torch profile to %s", output_path)
        prof.export_chrome_trace(output_path)

    def _compress_trace(self, input_path: str, output_path: str) -> None:
        logger.info("Compressing torch profile to %s", output_path)
        with open(input_path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        try:
            os.remove(input_path)
        except OSError:
            logger.debug("Unable to delete temporary torch profile %s", input_path)


class TorchProfiler(TrainerComponent):
    """Manages torch profiling during training."""

    def __init__(
        self,
        *,
        profiler_config: Any,
        wandb_run: Optional[WandbRun] = None,
        run_dir: Optional[Path] = None,
        is_master: bool = True,
    ) -> None:
        interval = getattr(profiler_config, "interval_epochs", 0)
        super().__init__(epoch_interval=max(1, interval) if interval else 0)
        self._config = profiler_config
        self._wandb_run = wandb_run
        self._run_dir = run_dir
        self._is_master = is_master
        self._session: Optional[TorchProfileSession] = None
        self._original_train_epoch = None
        self._master_only = True

    def register(self, context: ComponentContext) -> None:  # type: ignore[override]
        super().register(context)
        interval = getattr(self._config, "interval_epochs", 0)
        if not interval:
            return

        if self._session is None:
            run_dir = self._run_dir
            self._session = TorchProfileSession(
                master=self._is_master,
                profiler_config=self._config,
                wandb_run=self._wandb_run,
                run_dir=run_dir,
            )

        original_train_epoch = context.get_train_epoch_callable()

        def wrapped_train_epoch():
            if self._session is None:
                return original_train_epoch()
            current_epoch = self.context.epoch
            self._session.prepare_for_epoch(current_epoch)
            with self._session:
                return original_train_epoch()

        context.set_train_epoch_callable(wrapped_train_epoch)
        self._original_train_epoch = original_train_epoch

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        # Profiling sessions are scoped to the epoch via the context manager, so
        # no additional teardown is required here. This hook is retained for
        # compatibility with the TrainerComponent interface.
        return

    def on_training_complete(self) -> None:  # type: ignore[override]
        if self._original_train_epoch is not None:
            self.context.set_train_epoch_callable(self._original_train_epoch)
