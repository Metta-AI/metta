import logging
import os
import gzip
import wandb
import boto3
import torch.profiler

logger = logging.getLogger(__name__)

class TorchProfiler:
    def __init__(self, run_dir, wandb_run):
        self.run_dir = run_dir
        self.wandb_run = wandb_run
        self.profile_dir = os.path.join(self.run_dir, "torch_traces")
        os.makedirs(self.profile_dir, exist_ok=True)
        self.s3_client = boto3.client("s3")
        self.profiler = None
        self.active = False
        self._start_epoch = None
        self._profile_filename_base = None

    def setup_profiler(self, epoch):
        """Prepare the profiler to start on the next context entry."""
        if self.active:
            logger.warning("Profiler setup called while already active. Profiling will occur for the current setup.")
            return # don't re-setup profiler if we're already active

        if self.profiler is not None:
             logger.warning("Profiler object exists during setup. This might indicate an incomplete previous profile cycle.")
             self.profiler = None

        self.active = True
        self._start_epoch = epoch
        self._profile_filename_base = f"trace_{os.path.basename(self.run_dir)}_epoch_{self._start_epoch}"
        logger.info(f"Torch profiler armed for epoch {epoch}. Will start profiling on context entry.")

    def __enter__(self):
        """Enter the context manager, starting the profiler if active."""
        if not self.active:
            return self

        if self.profiler is not None: # shouldn't happen if setup/exit logic is correct
            logger.error("Profiler context entered but profiler object already exists. Attempting to stop previous profile.")
            try:
                self.profiler.stop()
            except Exception as e:
                logger.error(f"Error stopping lingering profiler: {e}")

        logger.info(f"Entering profiler context for epoch {self._start_epoch}. Starting profiling.")
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        )
        self.profiler.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, stopping the profiler and saving the trace."""
        if not self.active or self.profiler is None:
            self.active = False 
            return False

        logger.info(f"Exiting profiler context for epoch {self._start_epoch}. Stopping profiling.")
        try:
            self.profiler.stop()
            self._save_profile(self.profiler)
        except Exception as e:
            logger.error(f"Error stopping or saving profile for epoch {self._start_epoch}: {e}", exc_info=True)
        finally:
            self.profiler = None
            self.active = False
            self._profile_filename_base = None

        return False

    def _save_profile(self, prof):
        """Saves the profiling results to a compressed JSON file and uploads to S3.
        You can find it in S3 at torch_traces/<run_dir>/ or locally at train_dir/<run_dir>/torch_traces/"""
        if self._profile_filename_base is None:
            logger.error("Profile filename base not set. Cannot save trace.")
            return

        output_filename_json = f"{self._profile_filename_base}.json"
        output_filename_gz = f"{output_filename_json}.gz"
        temp_json_path = os.path.join(self.profile_dir, output_filename_json) # Temp path for uncompressed
        final_gz_path = os.path.join(self.profile_dir, output_filename_gz) # Final compressed path

        try:
            logger.info(f"Exporting profile trace to temporary file {temp_json_path}...")
            prof.export_chrome_trace(temp_json_path) # Export uncompressed first

            logger.info(f"Compressing trace to {final_gz_path}...")
            with open(temp_json_path, 'rb') as f_in, gzip.open(final_gz_path, 'wb') as f_out:
                f_out.writelines(f_in)
            logger.info(f"Successfully saved compressed profile trace to {final_gz_path} for epoch {self._start_epoch}.")

            # clean up uncompressed file only after successful compression
            try:
                os.remove(temp_json_path)
            except OSError as e:
                logger.warning(f"Could not remove temporary trace file {temp_json_path}: {e}")

            # S3 upload:
            s3_bucket = "softmax-public"
            s3_path = f"torch_traces/{os.path.basename(self.run_dir)}/{output_filename_gz}"
            logger.info(f"Uploading profile trace to S3: s3://softmax-public/{s3_path}")
            self.s3_client.upload_file(final_gz_path, s3_bucket, s3_path)
            logger.info(f"Successfully uploaded profile trace to S3.")

            # Log the link to WandB
            link = f"https://{s3_bucket}.s3.us-east-1.amazonaws.com/{s3_path}"
            link_summary = {"torch_traces/link": wandb.Html(f'<a href="{link}">Torch Trace (Epoch {self._start_epoch})</a>')}
            self.wandb_run.log(link_summary)

        except Exception as e:
            logger.error(f"Error exporting/compressing/uploading profile trace for epoch {self._start_epoch}: {e}", exc_info=True)
            # Attempt cleanup of temp file even on error
            if os.path.exists(temp_json_path):
                 try:
                     os.remove(temp_json_path)
                 except OSError as e_rem:
                     logger.warning(f"Could not remove temporary trace file {temp_json_path} after error: {e_rem}")
 