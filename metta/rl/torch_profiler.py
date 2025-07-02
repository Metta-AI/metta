import gzip
import logging
import os

import torch.profiler
import wandb

from metta.mettagrid.util.file import http_url, write_file
from metta.rl.trainer_config import TorchProfilerConfig

logger = logging.getLogger(__name__)


class TorchProfiler:
    """
    Creates a torch profiler object that can be used as context wherever
    traces are needed.

    Profiles are saved as json.gz files locally in
    train_dir/<your_run>/torch_traces/. By default, profiles are also uploaded
    to S3 if AWS credentials are configured. A link to download the S3 file is
    dropped into wandb under 'torch_traces'. To view traces, go to
    chrome://tracing (or arc://tracing if you happen to use that browser
    which is fine) and select 'load'. Navigate traces using WASD on your
    keyboard.

    Set profiler.interval_epochs in the config to zero to turn it off.

    Upload behavior is configured via trainer.profiler config:
    - trainer.profiler.interval_epochs: profiling interval (0 to disable)
    - trainer.profiler.upload_dir: upload location (supports s3:// or local paths)
    - Setting upload_dir to null disables uploads
    - S3 uploads will be skipped if AWS credentials are not configured

    Future work could include support for TensorBoard.
    """

    def __init__(self, master: bool, profiler_config: TorchProfilerConfig, wandb_run, run_dir: str):
        self._master = master
        self._profiler_config = profiler_config
        self._wandb_run = wandb_run
        self._run_dir = run_dir

        self._profiler = None
        self._active = False
        self._start_epoch = None
        self._profile_filename_base = None

        # Hardcoding _first_profile_epoch to keep cfgs clean.
        # It just needs to be greater than pytorch warmup cycles.
        # We may need to revisit if compiling models under "max-autotune".
        self._first_profile_epoch = 300

    def on_epoch_end(self, epoch):
        should_profile_this_epoch = False
        if not self._active:
            should_profile_this_epoch = (
                self._profiler_config.interval_epochs != 0
                and (epoch % self._profiler_config.interval_epochs == 0 or epoch == self._first_profile_epoch)
                and self._master
            )
        if should_profile_this_epoch:
            self.setup_profiler(epoch)

    def setup_profiler(self, epoch):
        """Prepare the profiler to start on the next context entry."""
        if self._active:
            logger.warning("Profiler setup called while already active. Profiling will occur for the current setup.")
            return  # don't re-setup profiler if we're already active

        if self._profiler is not None:
            logger.warning(
                "Profiler object exists during setup. This might indicate an incomplete previous profile cycle."
            )
            self._profiler = None

        self._active = True
        self._start_epoch = epoch
        self._profile_filename_base = f"trace_{os.path.basename(self._run_dir)}_epoch_{self._start_epoch}"
        logger.info(f"Torch profiler armed for epoch {epoch}. Will start profiling on context entry.")

    def __enter__(self):
        """Enter the context manager, starting the profiler if active."""
        if not self._active:
            return self

        logger.info(f"Entering profiler context for epoch {self._start_epoch}. Starting profiling.")
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
        """Exit the context manager, stopping the profiler and saving the trace."""
        if not self._active or self._profiler is None:
            self._active = False
            return False

        logger.info(f"Exiting profiler context for epoch {self._start_epoch}. Stopping profiling.")
        try:
            self._profiler.stop()
            self._save_profile(self._profiler)
        except Exception as e:
            logger.error(f"Error stopping or saving profile for epoch {self._start_epoch}: {e}", exc_info=True)
        finally:
            self._profiler = None
            self._active = False
            self._profile_filename_base = None

        return False

    def _save_profile(self, prof):
        """Orchestrates exporting, compressing, uploading, and logging the profile trace."""
        if self._profile_filename_base is None:
            logger.error("Profile filename base not set. Cannot save trace.")
            return

        output_filename_json = f"{self._profile_filename_base}.json"
        output_filename_gz = f"{output_filename_json}.gz"

        # Use temp directory for intermediate files
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="torch_profile_")
        temp_json_path = os.path.join(temp_dir, output_filename_json)  # for uncompressed, to be deleted
        final_gz_path = os.path.join(temp_dir, output_filename_gz)  # compressed path

        try:
            self._export_profile(prof, temp_json_path)  # temp_json_path is a Chrome trace format file
            self._compress_trace(temp_json_path, final_gz_path)  # final_gz_path is a gzip compressed file

            # Upload to configured location if enabled
            if self._profiler_config.enabled:
                upload_url = self._upload(final_gz_path, output_filename_gz)
                if upload_url and self._wandb_run:
                    self._wandb_log_trace_link(upload_url)
            else:
                logger.info(f"Profile saved locally at {final_gz_path} (upload disabled)")

        except Exception as e:
            logger.error(f"Error handling profile trace for epoch {self._start_epoch}: {e}", exc_info=True)

    # _save_profile() helper methods -----------------------------------------------------
    def _export_profile(self, prof, output_path):
        """Exports the profile trace to `output_path` in Chrome trace format."""
        logger.info(f"Exporting profile trace to temporary file {output_path}...")
        prof.export_chrome_trace(output_path)

    def _compress_trace(self, input_path, output_path):
        """Compresses the raw JSON trace using gzip and removes the original file."""
        logger.info(f"Compressing trace to {output_path}...")
        with open(input_path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
            f_out.writelines(f_in)
        logger.info(f"Successfully saved compressed profile trace to {output_path} for epoch {self._start_epoch}.")
        try:
            os.remove(input_path)
        except OSError as e:
            logger.warning(f"Could not remove temporary trace file {input_path}: {e}")

    def _upload(self, local_path, filename_gz):
        """Uploads the compressed trace to configured location and returns URL."""
        if not self._profiler_config.upload_dir:
            return None

        # Construct the full upload path
        upload_path = os.path.join(self._profiler_config.upload_dir, filename_gz)

        try:
            # Use write_file which handles local/s3/wandb paths
            write_file(upload_path, local_path, content_type="application/gzip")
            # Convert to HTTP URL if it's an S3 path
            return http_url(upload_path)
        except Exception as e:
            logger.error(f"Failed to upload profile trace: {e}")
            return None
        finally:
            # Clean up temp directory
            import shutil

            temp_dir = os.path.dirname(local_path)
            if temp_dir.startswith("/tmp/torch_profile_") or temp_dir.startswith("/var/folders/"):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

    def _wandb_log_trace_link(self, upload_url):
        """Logs the upload link to the profile trace in WandB."""
        link_summary = {
            "torch_traces/link": wandb.Html(f'<a href="{upload_url}">Torch Trace (Epoch {self._start_epoch})</a>')
        }
        self._wandb_run.log(link_summary)
