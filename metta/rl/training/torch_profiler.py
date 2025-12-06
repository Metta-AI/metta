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
from mettagrid.util.file import http_url, is_public_uri, write_file

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
        # Default to profiling after an initial warmup unless overridden via env.
        env_first_epoch = os.environ.get("TORCH_PROFILER_FIRST_EPOCH")
        self._first_profile_epoch = max(1, int(env_first_epoch)) if env_first_epoch else 300

    def on_epoch_end(self, epoch: int) -> None:
        if should_run(epoch, getattr(self._profiler_config, "interval_epochs", 0), force=False):
            self._setup_profiler(epoch)

    def start_if_due(self, epoch: int, interval: int) -> None:
        """Arm the profiler ahead of an epoch when the schedule permits."""
        if self._active:
            return
        if should_run(epoch, interval, force=False):
            self._setup_profiler(epoch)

    def _setup_profiler(self, epoch: int) -> None:
        if self._active:
            logger.warning("Profiler already active; ignoring setup request")
            return
        if self._profiler is not None:
            logger.warning("Profiler instance exists while idle; resetting")
            self._profiler = None

        self._active = True
        self._start_epoch = epoch
        run_basename = os.path.basename(self._run_dir) if self._run_dir else "unknown_run"
        self._profile_filename_base = f"trace_{run_basename}_epoch_{self._start_epoch}"
        logger.info("Torch profiler armed for epoch %s", epoch)

    def __enter__(self):
        if not self._active:
            return self

        logger.info("Starting torch profiler for epoch %s", self._start_epoch)
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=self._profiler_config.active_steps,
                repeat=1,
            ),
        )
        self._profiler.start()
        self.step()  # Prime the schedule so rollout is captured as well.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._active or self._profiler is None:
            self._active = False
            return False

        logger.info("Stopping torch profiler for epoch %s", self._start_epoch)
        self._profiler.stop()
        self._save_profile(self._profiler)
        self._profiler = None
        self._active = False
        self._profile_filename_base = None

        return False

    def step(self) -> None:
        """Advance the profiler schedule for per-minibatch stepping."""
        if not self._active or self._profiler is None:
            return
        self._profiler.step()

    # Internal helpers -------------------------------------------------
    def _save_profile(self, prof: torch.profiler.profile) -> None:
        if self._profile_filename_base is None:
            logger.error("Profiler filename unset; skipping save", exc_info=True)
            return

        output_filename_json = f"{self._profile_filename_base}.json"
        output_filename_gz = f"{output_filename_json}.gz"
        upload_path = os.path.join(self._profiler_config.profile_dir, output_filename_gz)

        with tempfile.TemporaryDirectory(prefix="torch_profile_") as temp_dir:
            temp_json_path = os.path.join(temp_dir, output_filename_json)
            final_gz_path = os.path.join(temp_dir, output_filename_gz)

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

    def _export_profile(self, prof: torch.profiler.profile, output_path: str) -> None:
        logger.info("Exporting torch profile to %s", output_path)
        prof.export_chrome_trace(output_path)

    def _compress_trace(self, input_path: str, output_path: str) -> None:
        logger.info("Compressing torch profile to %s", output_path)
        with open(input_path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(input_path)


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
        self._epoch_counter = 0

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
        context.profiler_step = self._session.step if self._session else None

        def wrapped_train_epoch():
            if self._session is None:
                return original_train_epoch()
            # Arm the profiler ahead of the epoch so short runs capture traces.
            self._epoch_counter += 1
            self._session.start_if_due(self._epoch_counter, interval)
            with self._session:
                return original_train_epoch()

        context.set_train_epoch_callable(wrapped_train_epoch)
        self._original_train_epoch = original_train_epoch

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        if self._session:
            self._session.on_epoch_end(epoch)

    def on_training_complete(self) -> None:  # type: ignore[override]
        self.context.profiler_step = None
        if self._original_train_epoch is not None:
            self.context.set_train_epoch_callable(self._original_train_epoch)
